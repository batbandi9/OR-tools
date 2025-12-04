"""
Data Loading and Configuration Utilities
"""

import pandas as pd
import yaml


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(filepath: str, cfg: dict) -> pd.DataFrame:
    """
    Load and filter data based on configuration.

    Args:
        filepath: Path to the CSV data file
        cfg: Configuration dictionary

    Returns:
        Filtered DataFrame
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["datetime"], index_col="datetime")

    # Filter by month if specified
    target_months = cfg["settings"].get("month", [])
    if target_months:
        if not isinstance(target_months, list):
            target_months = [target_months]
        print(f"Filtering for Month(s): {target_months}...")
        df = df[df.index.month.isin(target_months)]

    # Filter by day if specified
    target_day = cfg["settings"].get("day")
    if target_day is not None:
        print(f"Filtering for Day: {target_day}...")
        df = df[df.index.day == target_day]

    # Filter by hour if specified
    target_hour = cfg["settings"].get("hour")
    if target_hour is not None:
        print(f"Filtering for Hour: {target_hour}...")
        df = df[df.index.hour == target_hour]

    if df.empty:
        raise ValueError(
            "DataFrame is empty after filtering! Check your dates.")

    print(f"Data ready: {len(df)} hours to optimize.")
    return df


def process_energy_data(cfg: dict) -> tuple:
    """
    Process raw energy data files into clean datasets.

    Args:
        cfg: Configuration dictionary with data paths

    Returns:
        Tuple of (1-year dataset, 5-year synthetic dataset)
    """

    def load_price_csv(filepath, col_name):
        """Load and clean price CSV file."""
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
        df = df.drop_duplicates(subset="datetime").set_index(
            "datetime").sort_index()
        df = df.resample("h").asfreq()
        return df

    # Load price data
    strom = load_price_csv(cfg["data"]["electricity_price"], "price_el")
    gas = load_price_csv(cfg["data"]["gas_price"], "price_gas")

    # Load heat demand
    waerme = pd.read_excel(
        cfg["data"]["heat_demand"],
        usecols=[0, 1],
        names=["datetime", "demand_th"],
        header=None,
        skiprows=1,
        parse_dates=["datetime"],
    )
    waerme = waerme.drop_duplicates(
        subset="datetime").set_index("datetime").sort_index()
    waerme = waerme.resample("h").asfreq()

    # Create 1-year dataset
    strom_1y = strom.reindex(waerme.index)
    gas_1y = gas.reindex(waerme.index)
    dataset_1y = pd.concat([strom_1y, gas_1y, waerme], axis=1)
    dataset_1y = dataset_1y.ffill().bfill()

    # Save 1-year dataset
    dataset_1y.to_csv("data/interim/data_1year_strict.csv", index=True)
    print(f"Saved 'data_1year_strict.csv' with shape {dataset_1y.shape}")

    # Create 5-year synthetic dataset
    full_range = strom.index.union(gas.index)
    dataset_5y = pd.DataFrame(index=full_range)
    dataset_5y = dataset_5y.join(strom).join(gas)

    # Time features for mapping
    dataset_5y["month"] = dataset_5y.index.month
    dataset_5y["day"] = dataset_5y.index.day
    dataset_5y["hour"] = dataset_5y.index.hour

    waerme_source = waerme.copy()
    waerme_source["month"] = waerme_source.index.month
    waerme_source["day"] = waerme_source.index.day
    waerme_source["hour"] = waerme_source.index.hour

    # Merge heat demand
    dataset_5y_reset = dataset_5y.reset_index()
    merged = pd.merge(
        dataset_5y_reset,
        waerme_source[["month", "day", "hour", "demand_th"]],
        on=["month", "day", "hour"],
        how="left",
    )
    merged = merged.set_index("datetime")
    merged = merged.drop(columns=["month", "day", "hour"])
    merged = merged.ffill().bfill()

    # Save 5-year dataset
    merged.to_csv("data/interim/data_5year_synthetic.csv", index=True)
    print(f"Saved 'data_5year_synthetic.csv'")

    return dataset_1y, merged


if __name__ == "__main__":
    cfg = load_config()
    dfs = process_energy_data(cfg)
    print(dfs[0].head())
