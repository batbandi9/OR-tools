"""
Main Entry Point for Energy System Optimization
Runs both OR-Tools and PyPSA models and compares results.
"""

import os
import sys

# Add project root (OR_tools folder) to path BEFORE imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change working directory to project root for relative paths
os.chdir(project_root)

# Now import from src (after path is set)
from src.models.pypsa_model import PyPSAOptimizer
from src.models.ortools_model import ORToolsOptimizer
from src.utils.dataloader import load_config, load_data
from src.utils.plotting import plot_network, plot_results_comparison, plot_results_timeseries, plot_energy_balance, plot_daily_profile
from src.utils.analysis import compare_results, calculate_kpis


def main():
    """Run the full optimization workflow."""

    print("ENERGY SYSTEM OPTIMIZATION")

    # 1. Load Configuration
    try:
        cfg = load_config()
        print("Configuration loaded")
    except FileNotFoundError:
        print("Error: config/model_config.yaml not found.")
        return

    # 2. Load Data
    try:
        df = load_data("data/interim/data_1year_strict.csv", cfg)
        print("Data loaded")
    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    # Create results folder
    os.makedirs("results", exist_ok=True)

    # 3. Run OR-Tools Optimization
    print("Running OR-Tools Model...")

    try:
        optimizer_ortools = ORToolsOptimizer(df, cfg)
        results_ortools = optimizer_ortools.optimize()

        if results_ortools is not None:
            results_ortools["heat_demand"] = df["demand_th"]
            results_ortools["electricity_price"] = df["price_el"]
            results_ortools["gas_price"] = df["price_gas"]
            results_ortools.round(3).to_csv("results/ortools_results.csv")
            print("OR-Tools results saved to: results/ortools_results.csv")
            calculate_kpis(results_ortools, cfg)
            plot_results_timeseries(results_ortools, "OR-Tools", save_path="results/ortools_timeseries.png")
            plot_energy_balance(results_ortools, save_path="results/ortools_energy_balance.png")

    except Exception as e:
        print(f"OR-Tools Error: {e}")
        import traceback
        traceback.print_exc()
        results_ortools = None

    # 4. Run PyPSA Optimization
    print("Running PyPSA Model...")

    try:
        optimizer_pypsa = PyPSAOptimizer(df, cfg)
        optimizer_pypsa.build_model()
        plot_network(optimizer_pypsa.network, save_path="results/network_diagram.png")
        solver_name = cfg["settings"].get("solver", "scip").lower()
        optimizer_pypsa.solve(solver_name=solver_name)
        results_pypsa = optimizer_pypsa.get_results()

        if results_pypsa is not None:
            results_pypsa["heat_demand"] = df["demand_th"]
            results_pypsa["electricity_price"] = df["price_el"]
            results_pypsa["gas_price"] = df["price_gas"]
            results_pypsa.round(3).to_csv("results/pypsa_results.csv")
            print("PyPSA results saved to: results/pypsa_results.csv")
            calculate_kpis(results_pypsa, cfg)
            plot_results_timeseries(results_pypsa, "PyPSA", save_path="results/pypsa_timeseries.png")
            plot_energy_balance(results_pypsa, save_path="results/pypsa_energy_balance.png")
    except Exception as e:
        print(f"PyPSA Error: {e}")
        import traceback
        traceback.print_exc()
        results_pypsa = None

    # 5. Compare Results
    if results_ortools is not None and results_pypsa is not None:
        print("Comparing Models...")
        comparison = compare_results(results_ortools, results_pypsa)
        comparison.to_csv("results/model_comparison.csv")
        print("Comparison saved to: results/model_comparison.csv")
        plot_results_comparison(results_ortools, results_pypsa, save_path="results/comparison_plot.png")

    print("OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    main()
