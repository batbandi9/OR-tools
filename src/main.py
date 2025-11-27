import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.CHP_Boiler import CHPOptimizer 
from src.dataloader import load_config

def main():

    # 1. Load Configuration
    try:
        cfg = load_config()
    except FileNotFoundError:
        print("Error: config/model_config.yaml not found.")
        return

    # 2. Load Data
    input_file = "data/interim/data_1year_strict.csv"
    
    try:
        print(f"Loading raw data from {input_file}...")
        df = pd.read_csv(input_file, parse_dates=["datetime"], index_col="datetime")
        
        target_months = cfg["settings"].get("month", [])
        
        if target_months:
            # Check if user passed a list or a single int
            if not isinstance(target_months, list): 
                target_months = [target_months]
            
            print(f"Filtering for Month(s): {target_months} ...")
            df = df[df.index.month.isin(target_months)]
            
        if df.empty:
            print("Error: DataFrame is empty after filtering! Check your dates.")
            return

        print(f"Data ready: {len(df)} hours to optimize.")

    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    # 3. Run Optimization
    optimizer = CHPOptimizer(df, cfg)
    
    try:
        results = optimizer.optimize()
        
        # 4. Save Results
        if results is not None:
            os.makedirs("results", exist_ok=True)
            
            # Add the original demand to the results for comparison
            results["heat_demand"] = df["demand_th"]
            results["electricity_price"] = df["price_el"]
            results["gas_price"] = df["price_gas"]
            results_rounded = results.round(3)
            
            output_path = "results/chp_shoulder_results.xlsx"
            results_rounded.to_excel(output_path)
            
            print(f"\nSUCCESS! Results saved to: {output_path}")
            print(results[["chp_heat_out", "heat_demand"]].head(10))
            
            total_gas = results["chp_gas_in"].sum()
            print(f"Total Gas Consumed: {total_gas:,.2f} MWh")
        else:
            print("Optimization finished but returned no results.")

    except Exception as e:
        print(f"Optimization Crash: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()