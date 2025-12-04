"""
Plotting and Visualization Utilities
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_network(network, save_path: str = "results/network_diagram.png") -> None:
    """
    Plot the PyPSA network as a graph using networkx.

    Args:
        network: PyPSA Network object
        save_path: Path to save the diagram
    """
    n = network
    G = nx.DiGraph()

    # Node positions
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

    # Add buses
    for bus in n.buses.index:
        G.add_node(bus, node_type="bus")

    # Add generators
    for gen in n.generators.index:
        G.add_node(gen, node_type="generator")
        bus = n.generators.loc[gen, "bus"]
        G.add_edge(gen, bus, edge_type="generator")

    # Add loads
    for load in n.loads.index:
        G.add_node(load, node_type="load")
        bus = n.loads.loc[load, "bus"]
        G.add_edge(bus, load, edge_type="load")

    # Add links
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

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Node colors and sizes
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

    # Edge colors
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

    # Draw graph
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=node_sizes, edge_color=edge_colors,
            font_size=10, font_weight="bold", arrows=True,
            arrowsize=20, connectionstyle="arc3,rad=0.1")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#3498db', markersize=15, label='Bus'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#2ecc71', markersize=15, label='Generator'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#e74c3c', markersize=15, label='Load'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#f39c12', markersize=15, label='Link'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.title("PyPSA Network: CHP + Boiler System", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Network diagram saved to: {save_path}")
    plt.show()


def plot_results_comparison(
    results_ortools: pd.DataFrame,
    results_pypsa: pd.DataFrame,
    save_path: str = "results/comparison.png"
) -> None:
    """
    Plot comparison of OR-Tools vs PyPSA results.

    Args:
        results_ortools: OR-Tools optimization results
        results_pypsa: PyPSA optimization results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # CHP Gas Input
    axes[0].plot(results_ortools.index, results_ortools["chp_gas_in"],
                 label="OR-Tools", alpha=0.7)
    axes[0].plot(results_pypsa.index, results_pypsa["chp_gas_in"],
                 label="PyPSA", alpha=0.7, linestyle="--")
    axes[0].set_ylabel("CHP Gas Input (MWh)")
    axes[0].legend()
    axes[0].set_title("Model Comparison: OR-Tools vs PyPSA")

    # Boiler Gas Input
    axes[1].plot(results_ortools.index, results_ortools["boiler_gas_in"],
                 label="OR-Tools", alpha=0.7)
    axes[1].plot(results_pypsa.index, results_pypsa["boiler_gas_in"],
                 label="PyPSA", alpha=0.7, linestyle="--")
    axes[1].set_ylabel("Boiler Gas Input (MWh)")
    axes[1].legend()

    # Heat Output
    axes[2].plot(results_ortools.index, results_ortools["chp_heat_out"],
                 label="OR-Tools CHP Heat", alpha=0.7)
    axes[2].plot(results_pypsa.index, results_pypsa["chp_heat_out"],
                 label="PyPSA CHP Heat", alpha=0.7, linestyle="--")
    axes[2].set_ylabel("Heat Output (MWh)")
    axes[2].legend()
    axes[2].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plot saved to: {save_path}")
    plt.show()


def plot_daily_profile(results: pd.DataFrame, title: str = "Daily Profile", save_path: str = None) -> None:
    """Plot daily average profile of results."""
    daily = results.groupby(results.index.hour).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    daily[["chp_gas_in", "boiler_gas_in"]].plot(ax=ax, kind="bar", width=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Gas Input (MWh)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Daily profile saved to: {save_path}")
    plt.show()


def plot_results_timeseries(
    results: pd.DataFrame,
    model_name: str = "Model",
    save_path: str = None
) -> None:
    """
    Plot timeseries of optimization results.

    Args:
        results: Optimization results DataFrame
        model_name: Name of the model for title
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Gas Input
    axes[0].plot(results.index, results["chp_gas_in"],
                 label="CHP Gas", color="#e74c3c")
    axes[0].plot(results.index, results["boiler_gas_in"],
                 label="Boiler Gas", color="#3498db")
    axes[0].set_ylabel("Gas Input (MWh)")
    axes[0].legend()
    axes[0].set_title(f"{model_name} Results")
    axes[0].grid(True, alpha=0.3)

    # Heat Output
    axes[1].plot(results.index, results["chp_heat_out"],
                 label="CHP Heat", color="#e74c3c")
    axes[1].plot(results.index, results["boiler_heat_out"],
                 label="Boiler Heat", color="#3498db")
    if "heat_demand" in results.columns:
        axes[1].plot(results.index, results["heat_demand"],
                     label="Demand", color="#2ecc71", linestyle="--")
    axes[1].set_ylabel("Heat (MWh)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Electricity
    axes[2].plot(results.index, results["chp_el_out"],
                 label="CHP Electricity", color="#9b59b6")
    if "electricity_price" in results.columns:
        ax2 = axes[2].twinx()
        ax2.plot(results.index, results["electricity_price"],
                 label="Price", color="#f39c12", alpha=0.5)
        ax2.set_ylabel("Price (EUR/MWh)", color="#f39c12")
    axes[2].set_ylabel("Electricity (MWh)")
    axes[2].legend(loc="upper left")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Timeseries plot saved to: {save_path}")
    plt.show()


def plot_energy_balance(results: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot energy balance summary (bar chart of totals).

    Args:
        results: Optimization results DataFrame
        save_path: Path to save the plot (optional)
    """
    totals = results[["chp_gas_in", "chp_heat_out", "chp_el_out",
                      "boiler_gas_in", "boiler_heat_out"]].sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c", "#ff6b6b", "#c0392b", "#3498db", "#5dade2"]
    totals.plot(kind="bar", ax=ax, color=colors)

    ax.set_ylabel("Total Energy (MWh)")
    ax.set_title("Energy Balance Summary")
    ax.set_xticklabels(["CHP Gas In", "CHP Heat Out", "CHP Elec Out",
                        "Boiler Gas In", "Boiler Heat Out"], rotation=45)

    # Add value labels on bars
    for i, v in enumerate(totals):
        ax.text(i, v + 50, f"{v:,.0f}", ha="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Energy balance plot saved to: {save_path}")
    plt.show()
