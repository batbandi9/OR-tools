"""
PyPSA-based Energy System Optimizer
CHP + Boiler model using PyPSA framework
"""

import pypsa
import pandas as pd


class PyPSAOptimizer:
    """PyPSA optimizer for CHP + Boiler energy system."""

    def __init__(self, data: pd.DataFrame, cfg: dict) -> None:
        self.data = data
        self.cfg = cfg
        self.network = None
        self.results = None

    def build_model(self) -> None:
        """Build the PyPSA network with all components."""
        self.network = pypsa.Network()

        # Add carriers
        for carrier in ["gas", "electricity", "heat", "chp", "boiler"]:
            self.network.add("Carrier", carrier)

        # Add buses
        self.network.add("Bus", "gas", carrier="gas")
        self.network.add("Bus", "electricity", carrier="electricity")
        self.network.add("Bus", "heat", carrier="heat")
        self.network.set_snapshots(self.data.index)

        # Gas cost including CO2 price
        co2_price = self.cfg["data"]["co2_price"]
        co2_intensity = self.cfg["economics"]["co2_intensity_gas"]
        gas_cost_total = self.data["price_gas"] + (co2_price * co2_intensity)

        # Gas supply generator
        self.network.add(
            "Generator",
            "Gas_Supply",
            bus="gas",
            p_nom_extendable=True,
            marginal_cost=gas_cost_total,
            carrier="gas",
        )

        # Electricity market (for selling)
        self.network.add(
            "Generator",
            "Market_Sale",
            bus="electricity",
            p_nom_extendable=True,
            sign=-1,
            marginal_cost=-self.data["price_el"],
            carrier="electricity",
        )

        # CHP unit
        self.network.add(
            "Link",
            "CHP",
            bus0="gas",
            bus1="heat",
            bus2="electricity",
            efficiency=self.cfg["chp"]["eta_th"],
            efficiency2=self.cfg["chp"]["eta_el"],
            p_nom=self.cfg["chp"]["p_gas_max"],
            p_min_pu=self.cfg["chp"]["p_gas_min"] /
            self.cfg["chp"]["p_gas_max"],
            marginal_cost=self.cfg["chp"].get("marginal_cost"),
            committable=True,
            carrier="chp",
        )

        # Boiler unit
        self.network.add(
            "Link",
            "Boiler",
            bus0="gas",
            bus1="heat",
            efficiency=self.cfg["boiler"].get("eta_th", 0.836),
            p_nom=self.cfg["boiler"]["p_gas_max"],
            marginal_cost=self.cfg["boiler"].get("marginal_cost"),
            carrier="boiler",
        )

        # Heat demand load
        self.network.add(
            "Load",
            "Heat_Load",
            bus="heat",
            p_set=self.data["demand_th"],
            carrier="heat",
        )

    def add_custom_constraints(self) -> None:
        """Add custom constraints to the model (call after create_model)."""
        m = self.network.model

        # Constraint: CHP gas input >= Boiler gas input
        chp_p = m["Link-p"].sel(name="CHP")
        boiler_p = m["Link-p"].sel(name="Boiler")
        m.add_constraints(chp_p - boiler_p >= 0, name="CHP_gas_geq_Boiler_gas")

    def solve(self, solver_name: str = "scip") -> None:
        """Solve the optimization problem."""
        # Create the optimization model
        self.network.optimize.create_model()

        # Add custom constraints
        self.add_custom_constraints()

        # Export readable model
        self.export_readable_model("results/pypsa_model_readable.txt")

        # Solve
        self.network.optimize.solve_model(solver_name=solver_name)
        self._extract_results()

    def export_readable_model(self, filepath: str) -> None:
        """Export the model in a human-readable format."""
        m = self.network.model

        print(m)

        with open(filepath, "w", encoding="utf-8") as f:
            # === PyPSA MODEL - HUMAN READABLE FORMAT ===
            f.write("PyPSA MODEL - Human Readable Format\n")
            f.write("\nOBJECTIVE FUNCTION\n")
            f.write("Objective:\n")
            f.write("----------\n")
            f.write(str(m.objective) + "\n\n")

            # === SUMMARY SECTION ===
            f.write("\n\n" + "="*70)
            f.write("\nSUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write("Linopy MILP model\n")
            f.write("=================\n\n")

            # Variables with dimensions
            f.write("Variables:\n")
            f.write("----------\n")
            for var_name in m.variables:
                v = m.variables[var_name]
                dims_str = ", ".join(v.dims) if v.dims else ""
                f.write(f" * {var_name} ({dims_str})\n")

            f.write("\nConstraints:\n")
            f.write("------------\n")
            for const_name in m.constraints:
                c = m.constraints[const_name]
                dims_str = ", ".join(c.dims) if c.dims else ""
                f.write(f" * {const_name} ({dims_str})\n")

            # Extract and list all unique names
            all_names = set()
            f.write("\n\nUnique Component Names Found:\n")
            f.write("-----------------------------\n")

            for var_name in m.variables:
                var = m.variables[var_name]
                if 'name' in var.dims:
                    names = var.coords['name'].values
                    for n in names:
                        all_names.add(n)

            for const_name in m.constraints:
                const = m.constraints[const_name]
                if 'name' in const.dims:
                    names = const.coords['name'].values
                    for n in names:
                        all_names.add(n)

            for i, name in enumerate(sorted(all_names), 1):
                f.write(f"{i}. {name}\n")

         # === VARIABLES ===
            f.write("VARIABLES\n\n")
            for var_name in m.variables:
                v = m.variables[var_name]
                is_binary = v.attrs.get("binary", False)
                is_integer = v.attrs.get("integer", False)
                vtype = "BINARY" if is_binary else (
                    "INTEGER" if is_integer else "CONTINUOUS")
                f.write(f"--- {var_name} [{vtype}] ---\n")
                f.write(str(v) + "\n\n")

            # === BINARY VARIABLES SUMMARY ===
            f.write("BINARY VARIABLES (Summary)\n")
            has_binary = False
            for var_name in m.variables:
                v = m.variables[var_name]
                if v.attrs.get("binary", False):
                    has_binary = True
                    f.write(f"  - {var_name}: shape={v.shape}\n")
            if not has_binary:
                f.write("  (No binary variables)\n")

            # === CONSTRAINTS ===
            f.write("\nCONSTRAINTS\n\n")
            for const_name in m.constraints:
                f.write(f"--- {const_name} ---\n")
                f.write(str(m.constraints[const_name]) + "\n\n")

            f.write("END OF MODEL\n")
        print(f"Exported readable model to: {filepath}")

    def _extract_results(self) -> None:
        """Extract optimization results into a DataFrame.

        Note: PyPSA uses sign convention where:
        - p0 (input) is positive
        - p1, p2 (outputs) are negative
        We flip the signs to match OR-Tools convention (all positive).
        """
        snaps = self.network.snapshots
        links_t = self.network.links_t

        self.results = pd.DataFrame(
            {
                # Gas inputs are positive in PyPSA
                "chp_gas_in": links_t.p0.get("CHP", pd.Series([0.0] * len(snaps))).values,
                # Outputs are negative in PyPSA, flip to positive
                "chp_el_out": -links_t.p2.get("CHP", pd.Series([0.0] * len(snaps))).values,
                "chp_heat_out": -links_t.p1.get("CHP", pd.Series([0.0] * len(snaps))).values,
                "boiler_gas_in": links_t.p0.get("Boiler", pd.Series([0.0] * len(snaps))).values,
                "boiler_heat_out": -links_t.p1.get("Boiler", pd.Series([0.0] * len(snaps))).values,
            },
            index=snaps,
        )

        # Print summary
        print("\n--- PyPSA Results Summary ---")
        print(self.results.sum())

    def get_results(self) -> pd.DataFrame:
        """Return the results DataFrame."""
        return self.results
