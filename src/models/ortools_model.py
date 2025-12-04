"""
OR-Tools based Energy System Optimizer
CHP + Boiler model using Google OR-Tools
"""

from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import os


class ORToolsOptimizer:
    """OR-Tools optimizer for CHP + Boiler energy system."""

    def __init__(self, data: pd.DataFrame, config: dict):
        self.data = data
        self.cfg = config
        self.solver = None
        self.results = None

        self.c_chp = config["chp"]
        self.c_boiler = config["boiler"]
        self.c_eco = config["economics"]

    def optimize(self) -> pd.DataFrame:
        """Run the full optimization workflow."""
        self._build_model()
        self._solve()
        return self.results

    def _build_model(self) -> None:
        """Build the OR-Tools optimization model."""
        # Setup solver
        solver_name = self.cfg["settings"]["solver"]
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if not self.solver:
            raise ValueError(f"{solver_name} not available.")

        T = len(self.data)

        # Economics: Gas cost + CO2 tax
        gas_cost_total = self.data["price_gas"].values + (
            self.cfg["data"]["co2_price"] * self.c_eco["co2_intensity_gas"]
        )
        price_el = self.data["price_el"].values
        demand = self.data["demand_th"].values

        # Variables: CHP
        self.v_chp_gas = [
            self.solver.NumVar(0, self.c_chp["p_gas_max"], f"chp_gas_{t}")
            for t in range(T)
        ]
        self.v_chp_status = [
            self.solver.IntVar(0, 1, f"chp_on_{t}")
            for t in range(T)
        ]

        # Variables: Boiler
        self.v_boiler_gas = [
            self.solver.NumVar(0, self.c_boiler["p_gas_max"], f"bl_gas_{t}")
            for t in range(T)
        ]

        objective = self.solver.Objective()

        for t in range(T):
            # CHP min/max constraints (linked to status)
            self.solver.Add(
                self.v_chp_gas[t] <= self.c_chp["p_gas_max"] *
                self.v_chp_status[t]
            )
            self.solver.Add(
                self.v_chp_gas[t] >= self.c_chp["p_gas_min"] *
                self.v_chp_status[t]
            )

            # Heat balance constraint
            q_chp = self.v_chp_gas[t] * self.c_chp["eta_th"]
            q_boiler = self.v_boiler_gas[t] * self.c_boiler["eta_th"]
            self.solver.Add(q_chp + q_boiler == demand[t])

            # Objective coefficients
            coeff_chp = (
                (price_el[t] * self.c_chp["eta_el"])
                - gas_cost_total[t]
                - self.c_chp["marginal_cost"]
            )

            if coeff_chp > 0:
                print(f"Time {t}: CHP profitable with coeff {coeff_chp:.2f}")
            coeff_boiler = -gas_cost_total[t] - self.c_boiler["marginal_cost"]

            objective.SetCoefficient(self.v_chp_gas[t], coeff_chp)
            objective.SetCoefficient(self.v_boiler_gas[t], coeff_boiler)

        objective.SetMaximization()
        print(f"OR-Tools model built: {T} time steps.")

        # Export LP file
        self._export_lp_file()

    def _export_lp_file(self) -> None:
        """Export the model as LP file for debugging."""
        try:
            os.makedirs("results", exist_ok=True)
            lp_text = self.solver.ExportModelAsLpFormat(False)
            with open("results/ortools_model.lp", "w") as f:
                f.write(lp_text)
            print("Exported LP file: results/ortools_model.lp")
        except Exception as e:
            print(f"Could not export LP file: {e}")

    def _solve(self) -> None:
        """Solve the optimization problem."""
        print(f"Solving with {self.cfg['settings']['solver']}...")
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print(
                f"Optimal. Profit: {self.solver.Objective().Value():,.2f} EUR")
            self._extract_results()
        else:
            print("No optimal solution found.")

    def _extract_results(self) -> None:
        """Extract solution values into a DataFrame."""
        chp_gas = [v.solution_value() for v in self.v_chp_gas]
        boiler_gas = [v.solution_value() for v in self.v_boiler_gas]
        status = [v.solution_value() for v in self.v_chp_status]

        self.results = pd.DataFrame(
            {
                "chp_gas_in": chp_gas,
                "chp_el_out": np.array(chp_gas) * self.c_chp["eta_el"],
                "chp_heat_out": np.array(chp_gas) * self.c_chp["eta_th"],
                "chp_status": status,
                "boiler_gas_in": boiler_gas,
                "boiler_heat_out": np.array(boiler_gas) * self.c_boiler["eta_th"],
            },
            index=self.data.index,
        )

        # Print summary
        print("\n--- OR-Tools Results Summary ---")
        print(self.results.sum())

    def get_results(self) -> pd.DataFrame:
        """Return the results DataFrame."""
        return self.results
