from OR_tools.src.old.dataloader import load_config
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class PyPSAOptimizer:

    def __init__(self, data, cfg) -> None:
        self.data = data
        self.cfg = cfg
        self.network = None
        self.results = None

    def build_model(self) -> None:

        self.network = pypsa.Network()
        for carrier in ["gas", "electricity", "heat", "chp", "boiler"]:
            self.network.add("Carrier", carrier)
        self.network.add("Bus", "gas", carrier="gas")
        self.network.add("Bus", "electricity", carrier="electricity")
        self.network.add("Bus", "heat", carrier="heat")
        self.network.set_snapshots(self.data.index)

        # Gas cost including CO2 price (same as OR-Tools)
        co2_price = self.cfg["data"]["co2_price"]
        co2_intensity = self.cfg["economics"]["co2_intensity_gas"]
        gas_cost_total = self.data["price_gas"] + (co2_price * co2_intensity)

        self.network.add(
            "Generator",
            "Gas_Supply",
            bus="gas",
            p_nom_extendable=True,
            marginal_cost=gas_cost_total,
            carrier="gas",
        )
        self.network.add(
            "Generator",
            "Market_Sale",
            bus="electricity",
            p_nom_extendable=True,
            sign=-1,
            marginal_cost=-self.data["price_el"],
            carrier="electricity",
        )
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
        self.network.add(
            "Load",
            "Heat_Load",
            bus="heat",
            p_set=self.data["demand_th"],
            carrier="heat",
        )

    def plot_network(self):
        """Plot the full network with all components (buses, links, generators, loads) using networkx."""
        n = self.network
        G = nx.DiGraph()
        pos = {
            "gas": (0, 1),
            "heat": (1, 2),
            "electricity": (2, 1),
            "Gas_Supply": (-1, 1),
            "Market_Sale": (3, 1),
            "CHP": (1, 1),
            "Boiler": (0.5, 1.5),
            "Heat_Load": (1, 3),
        }
        for bus in n.buses.index:
            G.add_node(bus, node_type="bus")
        for gen in n.generators.index:
            G.add_node(gen, node_type="generator")
            bus = n.generators.loc[gen, "bus"]
            G.add_edge(gen, bus, edge_type="generator")
        for load in n.loads.index:
            G.add_node(load, node_type="load")
            bus = n.loads.loc[load, "bus"]
            G.add_edge(bus, load, edge_type="load")
        for link in n.links.index:
            G.add_node(link, node_type="link")
            bus0 = n.links.loc[link, "bus0"]
            bus1 = n.links.loc[link, "bus1"]
            G.add_edge(bus0, link, edge_type="link_in")
            G.add_edge(link, bus1, edge_type="link_out")
            if "bus2" in n.links.columns:
                bus2 = n.links.loc[link, "bus2"]
                if pd.notna(bus2) and bus2 != "":
                    G.add_edge(link, bus2, edge_type="link_out2")
        fig, ax = plt.subplots(figsize=(12, 8))
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            node_type = G.nodes[node].get("node_type", "unknown")
            if node_type == "bus":
                node_colors.append("#3498db")
                node_sizes.append(2000)
            elif node_type == "generator":
                node_colors.append("#2ecc71")
                node_sizes.append(1500)
            elif node_type == "load":
                node_colors.append("#e74c3c")
                node_sizes.append(1500)
            elif node_type == "link":
                node_colors.append("#f39c12")
                node_sizes.append(1800)
            else:
                node_colors.append("#95a5a6")
                node_sizes.append(1000)
        edge_colors = []
        for u, v in G.edges():
            edge_type = G.edges[u, v].get("edge_type", "unknown")
            if edge_type == "generator":
                edge_colors.append("#2ecc71")
            elif edge_type == "load":
                edge_colors.append("#e74c3c")
            elif edge_type in ["link_in", "link_out", "link_out2"]:
                edge_colors.append("#f39c12")
            else:
                edge_colors.append("#7f8c8d")
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=node_sizes, edge_color=edge_colors,
                font_size=10, font_weight="bold", arrows=True,
                arrowsize=20, connectionstyle="arc3,rad=0.1")
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#3498db', markersize=15, label='Bus'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#2ecc71', markersize=15, label='Generator'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#e74c3c', markersize=15, label='Load'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
                       markersize=15, label='Link (CHP/Boiler)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        plt.title(
            "PyPSA Network: CHP + Boiler System\n(Gas → CHP/Boiler → Heat + Electricity)", fontsize=14)
        plt.tight_layout()
        plt.savefig("results/network_diagram.png", dpi=150)
        print("Network diagram saved to: results/network_diagram.png")
        plt.show()

    def solve(self, solver_name="scip") -> None:
        # Create the optimization model first (linopy model)
        self.network.optimize.create_model()

        # Get the linopy model
        m = self.network.model

        # Add constraint: CHP gas input >= Boiler gas input for every snapshot
        # In PyPSA/linopy, Link-p uses 'name' dimension (not 'Link')
        chp_p = m["Link-p"].sel(name="CHP")
        boiler_p = m["Link-p"].sel(name="Boiler")

        # Add constraint: CHP >= Boiler (for gas input)
        m.add_constraints(chp_p - boiler_p >= 0, name="CHP_gas_geq_Boiler_gas")

        # Export readable model before solving
        self.export_readable_model("results/pypsa_model_readable.txt")

        # Now solve the model
        self.network.optimize.solve_model(solver_name=solver_name)
        self.extract_results()

    def export_readable_model(self, filepath) -> None:
        """Export the model in a human-readable format (objective, constraints, variables)."""
        m = self.network.model

        with open(filepath, "w", encoding="utf-8") as f:
            # Header
            f.write("PyPSA MODEL - Human Readable Format\n")

            # Objective Function

            f.write("OBJECTIVE FUNCTION\n")
            f.write(str(m.objective) + "\n\n")

            # Variables
            f.write("VARIABLES\n")
            for name in m.variables:
                v = m.variables[name]
                is_binary = v.attrs.get("binary", False)
                is_integer = v.attrs.get("integer", False)
                if is_binary:
                    vtype = "BINARY"
                elif is_integer:
                    vtype = "INTEGER"
                else:
                    vtype = "CONTINUOUS"
                f.write(f"\n--- {name} [{vtype}] ---\n")
                f.write(str(v) + "\n")

            # Binary Variables (separate section for clarity)
            f.write("BINARY VARIABLES (Summary)\n")

            has_binary = False
            for name in m.variables:
                v = m.variables[name]
                if v.attrs.get("binary", False):
                    has_binary = True
                    f.write(f"  - {name}: shape={v.shape}\n")
            if not has_binary:
                f.write("  (No binary variables)\n")

            # Constraints
            f.write("CONSTRAINTS\n")
            for name in m.constraints:
                f.write(f"\n--- {name} ---\n")
                f.write(str(m.constraints[name]) + "\n")

            f.write("END OF MODEL\n")

        print(f"Exported readable model to: {filepath}")

    def extract_results(self) -> None:
        snaps = self.network.snapshots
        links_t = self.network.links_t
        self.results = pd.DataFrame(
            {
                "chp_gas_in": (
                    links_t.p0["CHP"].values
                    if "CHP" in links_t.p0.columns
                    else [0.0] * len(snaps)
                ),
                "chp_el_out": (
                    links_t.p2["CHP"].values
                    if "CHP" in links_t.p2.columns
                    else [0.0] * len(snaps)
                ),
                "chp_heat_out": (
                    links_t.p1["CHP"].values
                    if "CHP" in links_t.p1.columns
                    else [0.0] * len(snaps)
                ),
                "boiler_gas_in": (
                    links_t.p0["Boiler"].values
                    if "Boiler" in links_t.p0.columns
                    else [0.0] * len(snaps)
                ),
                "boiler_heat_out": (
                    links_t.p1["Boiler"].values
                    if "Boiler" in links_t.p1.columns
                    else [0.0] * len(snaps)
                ),
            },
            index=snaps,
        )
        print(self.results.sum())
        print("--- Carrier Statistics ---")
        for carrier in ["gas", "electricity", "heat", "chp", "boiler"]:
            mask = self.network.links.carrier == carrier
            if mask.any():
                total = self.network.links_t.p0.loc[:, mask].sum().sum()
                print(f"Total {carrier} input: {total:.2f} MWh")
            gen_mask = self.network.generators.carrier == carrier
            if gen_mask.any():
                total_gen = self.network.generators_t.p.loc[:, gen_mask].sum(
                ).sum()
                print(f"Total {carrier} generation: {total_gen:.2f} MWh")

    def get_results(self) -> pd.DataFrame:
        return self.results


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(
        "data/interim/data_1year_strict.csv", index_col=0, parse_dates=True
    )
    target_months = cfg["settings"].get("month", [])
    if target_months:
        if not isinstance(target_months, list):
            target_months = [target_months]
        print(f"Filtering for Month(s): {target_months} ...")
        df = df[df.index.month.isin(target_months)]
        if df.empty:
            print("Error: DataFrame is empty after filtering! Check your dates.")
            exit(1)
        print(f"Data ready: {len(df)} hours to optimize.")
    optimizer = PyPSAOptimizer(df, cfg)
    optimizer.build_model()
    optimizer.plot_network()  # Show network visualization
    solver_name = cfg["settings"].get("solver", "scip").lower()
    optimizer.solve(solver_name=solver_name)
    res = optimizer.get_results()
    if res is not None:
        res.to_csv("results/pypsa_results.csv")
        print("Saved results/pypsa_results.csv")
