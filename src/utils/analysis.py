"""
Post-Analysis and Comparison Utilities
"""

import pandas as pd
import numpy as np


def compare_results(
    results_ortools: pd.DataFrame,
    results_pypsa: pd.DataFrame,
    tolerance: float = 0.01
) -> pd.DataFrame:
    """
    Compare OR-Tools and PyPSA results.
    
    Args:
        results_ortools: OR-Tools optimization results
        results_pypsa: PyPSA optimization results
        tolerance: Tolerance for considering values as matching
        
    Returns:
        DataFrame with comparison metrics
    """
    # Common columns
    common_cols = [
        "chp_gas_in", "chp_heat_out", "chp_el_out",
        "boiler_gas_in", "boiler_heat_out"
    ]
    
    comparison = {}
    
    for col in common_cols:
        if col in results_ortools.columns and col in results_pypsa.columns:
            ortools_sum = results_ortools[col].sum()
            pypsa_sum = results_pypsa[col].sum()
            diff = abs(ortools_sum - pypsa_sum)
            diff_pct = (diff / ortools_sum * 100) if ortools_sum != 0 else 0
            
            comparison[col] = {
                "OR-Tools Total": ortools_sum,
                "PyPSA Total": pypsa_sum,
                "Difference": diff,
                "Diff %": diff_pct,
                "Match": diff_pct < tolerance * 100
            }
    
    df_comparison = pd.DataFrame(comparison).T
    
    
    print("MODEL COMPARISON: OR-Tools vs PyPSA")
    print(df_comparison.round(2))
    
    return df_comparison


def calculate_kpis(results: pd.DataFrame, cfg: dict) -> dict:
    """
    Calculate key performance indicators from results.
    
    Args:
        results: Optimization results DataFrame
        cfg: Configuration dictionary
        
    Returns:
        Dictionary of KPIs
    """
    kpis = {}
    
    # Total gas consumption
    kpis["total_gas_mwh"] = results["chp_gas_in"].sum() + results["boiler_gas_in"].sum()
    
    # Total heat production
    kpis["total_heat_mwh"] = results["chp_heat_out"].sum() + results["boiler_heat_out"].sum()
    
    # Total electricity production
    kpis["total_elec_mwh"] = results["chp_el_out"].sum()
    
    # CHP utilization (hours running / total hours)
    if "chp_status" in results.columns:
        kpis["chp_utilization_pct"] = results["chp_status"].mean() * 100
    else:
        kpis["chp_utilization_pct"] = (results["chp_gas_in"] > 0).mean() * 100
    
    # Average CHP load when running
    running_mask = results["chp_gas_in"] > 0
    if running_mask.any():
        kpis["avg_chp_load_mwh"] = results.loc[running_mask, "chp_gas_in"].mean()
    else:
        kpis["avg_chp_load_mwh"] = 0
    
    # Boiler share of heat
    total_heat = kpis["total_heat_mwh"]
    if total_heat > 0:
        kpis["boiler_heat_share_pct"] = results["boiler_heat_out"].sum() / total_heat * 100
    else:
        kpis["boiler_heat_share_pct"] = 0
    
    print("\n--- Key Performance Indicators ---")
    for key, value in kpis.items():
        print(f"  {key}: {value:,.2f}")
    
    return kpis


def hourly_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze results by hour of day.
    
    Args:
        results: Optimization results DataFrame
        
    Returns:
        DataFrame with hourly averages
    """
    hourly = results.groupby(results.index.hour).mean()
    print("\n--- Hourly Average Analysis ---")
    print(hourly.round(2))
    return hourly


def monthly_analysis(results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze results by month.
    
    Args:
        results: Optimization results DataFrame
        
    Returns:
        DataFrame with monthly totals
    """
    monthly = results.groupby(results.index.month).sum()
    print("\n--- Monthly Total Analysis ---")
    print(monthly.round(2))
    return monthly
