import pandas as pd
import numpy as np

START_YEAR = 2026
END_YEAR = 2040

SCENARIO_FACTORS = {
    # Year:  (Electricity,    Gas,    Heat)
    2026: (1.0, 1.0, 1.0),  # Base Year
    2027: (1.1, 1.1, 1.0),
    2028: (1.2, 1.3, 1.0),
    2029: (1.1, 1.4, 0.95),
    2030: (1.0, 1.5, 0.95),
    2031: (0.9, 1.6, 0.90),
    2032: (0.9, 1.7, 0.90),
    2033: (0.8, 1.8, 0.85),
    2034: (0.8, 1.9, 0.85),
    2035: (0.8, 2.0, 0.90),
    2036: (0.7, 2.1, 0.90),
    2037: (0.7, 2.2, 0.90),
    2038: (0.7, 2.3, 0.85),
    2039: (0.7, 2.4, 0.80),
    2040: (0.7, 2.5, 0.80),
}


# Helper to get factors for years not explicitly defined (Linear Interpolation optional, here we just use nearest or default)
def get_factors(year):
    # Returns (el_factor, gas_factor, heat_factor)
    # If year is in dict, return it. If not, return default (1,1,1) or last known.
    return SCENARIO_FACTORS.get(year, (1.0, 1.0, 1.0))


def generate_long_term_scenario():

    # 1. Load the "Seed" Data (The clean 1-year profile you made)
    seed_path = "data/interim/data_1year_strict.csv"
    try:
        base_df = pd.read_csv(seed_path, parse_dates=["datetime"], index_col="datetime")
    except FileNotFoundError:
        print(
            f"Error: Could not find {seed_path}. Please run the previous step to generate the 1-year file first."
        )
        return

    # Add time features to base for mapping
    base_df["month"] = base_df.index.month
    base_df["day"] = base_df.index.day
    base_df["hour"] = base_df.index.hour

    # Remove duplicates in base just in case (keep first occurrence of every hour)
    base_df = base_df.drop_duplicates(subset=["month", "day", "hour"])

    all_years_data = []

    # 2. Loop through every year
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing Year: {year}...", end="")

        # A. Create the target timeframe for this specific year
        # freq='h' ensures we get 8760 hours (or 8784 for leap years)
        current_index = pd.date_range(
            start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:00", freq="h"
        )

        # B. Create a temporary DataFrame
        yearly_df = pd.DataFrame(index=current_index)
        yearly_df.index.name = "datetime"

        # Extract features for merging
        yearly_df["month"] = yearly_df.index.month
        yearly_df["day"] = yearly_df.index.day
        yearly_df["hour"] = yearly_df.index.hour

        # C. Map the Base Data onto this Year
        # We merge based on Month-Day-Hour.
        # If 'year' is a leap year but base wasn't, Feb 29 will be NaN initially.
        merged = pd.merge(
            yearly_df.reset_index(),
            base_df[["month", "day", "hour", "price_el", "price_gas", "demand_th"]],
            on=["month", "day", "hour"],
            how="left",
        )

        # Set index back
        merged = merged.set_index("datetime")

        # D. Handle Leap Years (Fill Feb 29th gap if it exists)
        merged = merged.ffill().bfill()

        # E. Apply Scaling Factors
        f_el, f_gas, f_heat = get_factors(year)

        merged["price_el"] = merged["price_el"] * f_el
        merged["price_gas"] = merged["price_gas"] * f_gas
        merged["demand_th"] = merged["demand_th"] * f_heat

        # Drop helper columns
        merged = merged.drop(columns=["month", "day", "hour", "index"], errors="ignore")

        all_years_data.append(merged)
        print(f" Done. (Factors: El={f_el}, Gas={f_gas}, Heat={f_heat})")

    # 3. Combine All Years
    print("\nCombining data...")
    final_scenario = pd.concat(all_years_data)

    # 4. Final Verification
    print(f"Final Shape: {final_scenario.shape}")
    print(f"Date Range: {final_scenario.index.min()} to {final_scenario.index.max()}")

    # Check for NaNs
    if final_scenario.isna().sum().sum() > 0:
        print("WARNING: NaNs detected. Filling with forward fill...")
        final_scenario = final_scenario.ffill()

    # 5. Save
    output_filename = f"data/interim/scenario_{START_YEAR}_{END_YEAR}.csv"
    final_scenario.to_csv(output_filename, index=True)
    print(f"-> Saved Successfully: {output_filename}")

    return final_scenario


# Run it
if __name__ == "__main__":
    df = generate_long_term_scenario()
