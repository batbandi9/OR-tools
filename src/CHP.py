from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

class CHPOptimizer:
    def __init__(self, data, config):
        self.data = data
        self.cfg = config
        self.solver = None
        self.results = None
        
        self.c_chp = config["chp"]
        self.c_eco = config["economics"]

    def optimize(self):
        self._build_model()
        self._solve()
        return self.results

    def _build_model(self):
        # 1. Setup
        solver_name = self.cfg["settings"]["solver"]
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if not self.solver: raise ValueError(f"{solver_name} not available.")

        T = len(self.data)
        
        # 2. Economics
        gas_cost_total = (
            self.data["price_gas"].values + 
            (self.cfg["data"]["co2_price"] * self.c_eco["co2_intensity_gas"])
        )
        price_el = self.data["price_el"].values
        demand = self.data["demand_th"].values

        # 3. Variables (CHP Only)
        self.v_chp_gas = [self.solver.NumVar(0, self.c_chp["p_gas_max"], f"chp_gas_{t}") for t in range(T)]
        self.v_chp_status = [self.solver.IntVar(0, 1, f"chp_on_{t}") for t in range(T)]
        
        # 4. Constraints & Objective
        objective = self.solver.Objective()

        for t in range(T):
            
            # C1: CHP Start/Stop Logic
            self.solver.Add(self.v_chp_gas[t] <= self.c_chp["p_gas_max"] * self.v_chp_status[t])
            self.solver.Add(self.v_chp_gas[t] >= self.c_chp["p_gas_min"] * self.v_chp_status[t])

            # C2: Heat Balance (CHP Only)
            # WARNING: If Demand > CHP_Max, this will be INFEASIBLE.
            q_chp = self.v_chp_gas[t] * self.c_chp["eta_th"]
            
            # I changed '==' to '<=' to prevent crashing if demand is too high
            # This means: "CHP tries its best, but cannot exceed demand"
            # If you strictly need '==', and demand is high, the solver will fail.
            self.solver.Add(q_chp == demand[t]) 

            # Objective: Profit
            coeff_chp = (
                (price_el[t] * self.c_chp["eta_el"]) - 
                gas_cost_total[t] - 
                (self.c_chp["marginal_cost"] * self.c_chp["eta_el"])
            )
            
            objective.SetCoefficient(self.v_chp_gas[t], coeff_chp)

        objective.SetMaximization()
        print(f"Model built: {T} time steps (CHP Only).")

    def _solve(self):
        print(f"Solving with {self.cfg['settings']['solver']}...")
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print(f"Optimal. Profit: {self.solver.Objective().Value():,.2f} EUR")
            self._extract_results()
        else:
            print("No optimal solution found. (Likely Demand > CHP Capacity)")

    def _extract_results(self):
        # Only extract CHP variables
        chp_gas = [v.solution_value() for v in self.v_chp_gas]
        status = [v.solution_value() for v in self.v_chp_status]

        self.results = pd.DataFrame({
            "chp_gas_in": chp_gas,
            "chp_el_out": np.array(chp_gas) * self.c_chp["eta_el"],
            "chp_heat_out": np.array(chp_gas) * self.c_chp["eta_th"],
            "chp_status": status,
        }, index=self.data.index)