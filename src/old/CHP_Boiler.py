from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np


class CHPOptimizer:
    def __init__(self, data, config):
        self.data = data
        self.cfg = config
        self.solver = None
        self.results = None

        # Load parameters exactly from Config
        self.c_chp = config["chp"]
        self.c_boiler = config["boiler"]
        self.c_eco = config["economics"]

    def optimize(self):
        self._build_model()
        self._solve()
        return self.results

    def _build_model(self):
        # 1. Setup Solver
        solver_name = self.cfg["settings"]["solver"]
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if not self.solver:
            raise ValueError(f"{solver_name} not available.")

        T = len(self.data)

        # 2. Economics Vectors
        # Gas Price + CO2 Tax (EUR/MWh_gas)
        gas_cost_total = self.data["price_gas"].values + (
            self.cfg["data"]["co2_price"] * self.c_eco["co2_intensity_gas"]
        )
        price_el = self.data["price_el"].values
        demand = self.data["demand_th"].values

        # 3. Variables

        # CHP (Input Limit: 2.556 from config)
        self.v_chp_gas = [
            self.solver.NumVar(0, self.c_chp["p_gas_max"], f"chp_gas_{t}")
            for t in range(T)
        ]
        self.v_chp_status = [self.solver.IntVar(
            0, 1, f"chp_on_{t}") for t in range(T)]

        # BOILER (Input Limit: 19.13 from config)
        self.v_boiler_gas = [
            self.solver.NumVar(0, self.c_boiler["p_gas_max"], f"bl_gas_{t}")
            for t in range(T)
        ]

        objective = self.solver.Objective()

        for t in range(T):

            # CHP Min/Max Status
            self.solver.Add(
                self.v_chp_gas[t] <= self.c_chp["p_gas_max"] *
                self.v_chp_status[t]
            )
            self.solver.Add(
                self.v_chp_gas[t] >= self.c_chp["p_gas_min"] *
                self.v_chp_status[t]
            )

            # Heat Balance: (CHP Gas * 0.458) + (Boiler Gas * 0.836) == Demand
            q_chp = self.v_chp_gas[t] * self.c_chp["eta_th"]
            q_boiler = self.v_boiler_gas[t] * self.c_boiler["eta_th"]
            self.solver.Add(q_chp + q_boiler == demand[t])

            # CHP Profit Coefficient:
            coeff_chp = (
                (price_el[t] * self.c_chp["eta_el"])
                - gas_cost_total[t]
                - (self.c_chp["marginal_cost"])  # * self.c_chp["eta_el"])
            )

            # Boiler Cost Coefficient:

            coeff_boiler = -gas_cost_total[t] - (
                self.c_boiler["marginal_cost"]
            )  # * self.c_boiler["eta_th"])

            objective.SetCoefficient(self.v_chp_gas[t], coeff_chp)
            objective.SetCoefficient(self.v_boiler_gas[t], coeff_boiler)

        objective.SetMaximization()
        print(f"Model built: {T} time steps.")

        # Export LP file for comparison
        try:
            import os

            os.makedirs("results", exist_ok=True)
            lp_text = self.solver.ExportModelAsLpFormat(False)
            with open("results/ortools_model.lp", "w") as f:
                f.write(lp_text)
            print("Exported LP file: results/ortools_model.lp")
        except Exception as e:
            print(f"Could not export LP file: {e}")

    def _solve(self):
        print(f"Solving with {self.cfg['settings']['solver']}...")
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print(
                f"Optimal. Profit: {self.solver.Objective().Value():,.2f} EUR")
            self._extract_results()
        else:
            print("No optimal solution found.")

    def _extract_results(self):
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
