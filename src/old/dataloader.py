import pandas as pd
import yaml
import numpy as np


def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_energy_data(cfg):

    # 1. HELPER: Load CSV efficiently
    def load_price_csv(filepath, col_name):
        df = pd.read_csv(
            filepath,
            sep=";",
            usecols=[1, 2],
            names=["datetime", col_name],
            header=None,
            skiprows=1,
            decimal=",",
            parse_dates=["datetime"],
            dayfirst=True,
        )
        # Drop duplicates and sort
        df = df.drop_duplicates(subset="datetime").set_index(
            "datetime").sort_index()
        # Resample to hourly to ensure grid is perfect (fills missing hours with NaN for now)
        df = df.resample("h").asfreq()
        print(df.head())
        return df

    # 2. LOAD DATA
    # Load 5 years of prices
    strom = load_price_csv(cfg["data"]["electricity_price"], "price_el")
    gas = load_price_csv(cfg["data"]["gas_price"], "price_gas")

    # Load 1 year of heat
    waerme = pd.read_excel(
        cfg["data"]["heat_demand"],
        usecols=[0, 1],
        names=["datetime", "demand_th"],
        header=None,
        skiprows=1,
        parse_dates=["datetime"],
    )
    waerme = (
        waerme.drop_duplicates(subset="datetime").set_index(
            "datetime").sort_index()
    )
    waerme = waerme.resample("h").asfreq()  # Ensure hourly grid
    print(waerme.head())

    # Align prices to the exact timeframe of the heat data
    strom_1y = strom.reindex(waerme.index)
    gas_1y = gas.reindex(waerme.index)

    dataset_1y = pd.concat([strom_1y, gas_1y, waerme], axis=1)

    # Basic cleanup for the 1-year set
    # Fill small gaps if any existed in prices
    dataset_1y = dataset_1y.ffill().bfill()

    # Save
    dataset_1y.to_csv("data/interim/data_1year_strict.csv", index=True)
    print(f"-> Saved 'data_1year_strict.csv' with shape {dataset_1y.shape}")

    full_range = strom.index.union(gas.index)
    dataset_5y = pd.DataFrame(index=full_range)
    dataset_5y = dataset_5y.join(strom).join(gas)

    # Create Time Features for Mapping
    # We need to match Heat to Prices based on Month-Day-Hour
    dataset_5y["month"] = dataset_5y.index.month
    dataset_5y["day"] = dataset_5y.index.day
    dataset_5y["hour"] = dataset_5y.index.hour

    waerme_source = waerme.copy()
    waerme_source["month"] = waerme_source.index.month
    waerme_source["day"] = waerme_source.index.day
    waerme_source["hour"] = waerme_source.index.hour

    # Reset index to allow merge on columns
    dataset_5y_reset = dataset_5y.reset_index()

    # MERGE: Left join ensures we keep all 5 years of timestamps.
    # We match on [month, day, hour].
    merged = pd.merge(
        dataset_5y_reset,
        waerme_source[["month", "day", "hour", "demand_th"]],
        on=["month", "day", "hour"],
        how="left",
    )

    # Restore Index
    merged = merged.set_index("datetime")
    # Remove helper columns
    merged = merged.drop(columns=["month", "day", "hour"])

    nan_count_before = merged.isna().sum().sum()
    merged = merged.ffill().bfill()
    nan_count_after = merged.isna().sum().sum()

    if nan_count_after > 0:
        print(
            f"WARNING: There are still {nan_count_after} NaNs. Check your price data coverage."
        )
    else:
        print(
            f"SUCCESS: Filled {nan_count_before} missing values (Leap days/Price gaps). Zero NaNs remaining."
        )

    # Calculate expected rows
    # A standard year has 8760 hours. A leap year has 8784.
    # We calculate exact hours between start and end.
    start_date = pd.Timestamp("2026-01-01 00:00:00")
    end_date = merged.index.max()
    expected_rows = (end_date - start_date).total_seconds() / 3600 + 1

    actual_rows = len(merged)

    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"Expected Rows (Math): {int(expected_rows)}")
    print(f"Actual Rows (DF):     {actual_rows}")

    if int(expected_rows) == actual_rows:
        print("-> CHECK PASSED: Row count matches time duration perfectly.")
    else:
        print(
            "-> CHECK FAILED: There might be duplicate hours or missing hours in the index."
        )

    # Save
    merged.to_csv("data/interim/data_5year_synthetic.csv", index=True)
    print(f"-> Saved 'data_5year_synthetic.csv'")

    return dataset_1y, merged


if __name__ == "__main__":
    cfg = load_config()
    dfs = process_energy_data(cfg)
    print(dfs[0].head())
    print(dfs[1].head())

# Usage
# dfs = process_energy_data(load_config())
