import sys
print(sys.executable)

import openpyxl
print(openpyxl.__version__)

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import scipy
print(scipy.__version__)


def run_data_preprocessing():

    # ======================================================
    # Load Dataset
    # ======================================================

    df = pd.read_excel("data/main_data.xlsx")

    print("Dataset loaded successfully!\n")

    print(df.head())

    print(df.columns)

    print(df.dtypes)

    print(df.shape)

    print(df.isnull().sum())

    print(df.duplicated().sum())

    # ======================================================
    # Reporter Countries
    # ======================================================

    reporters = [
        "China",
        "Japan",
        "Rep. of Korea",
        "Singapore",
        "USA"
    ]

    years = [1993, 2003, 2013, 2023]

    # ======================================================
    # Network Graph Figure
    # ======================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(24,8)
    )
    axes = axes.flatten()

    for ax, year in zip(axes, years):

        year_df = df[df["Year"] == year]

        G = nx.DiGraph()

        for _, row in year_df.iterrows():

            G.add_edge(
                row["Reporter"],
                row["Partner"],
                weight=row["Export_Value"]
            )

        # -----------------------------
        # Layout
        # -----------------------------

        pos = {}

        radius_center = 1.2

        angles = np.linspace(
            0,
            2*np.pi,
            len(reporters),
            endpoint=False
        )

        for reporter, angle in zip(reporters, angles):

            pos[reporter] = (
                radius_center*np.cos(angle),
                radius_center*np.sin(angle)
            )

        partners = sorted(
            [n for n in G.nodes() if n not in reporters]
        )

        radius_outer = 3.8

        angles_outer = np.linspace(
            0,
            2*np.pi,
            len(partners),
            endpoint=False
        )

        for partner, angle in zip(partners, angles_outer):

            pos[partner] = (
                radius_outer*np.cos(angle),
                radius_outer*np.sin(angle)
            )

        # -----------------------------
        # Node Colors
        # -----------------------------

        node_colors = [
            "tomato" if node in reporters else "skyblue"
            for node in G.nodes()
        ]

        # -----------------------------
        # Node Sizes
        # -----------------------------

        export_totals = (
            year_df
            .groupby("Reporter")["Export_Value"]
            .sum()
            .to_dict()
        )

        max_export = max(export_totals.values())

        node_sizes = []

        for node in G.nodes():

            if node in reporters:

                node_sizes.append(
                    1500 + (export_totals[node]/max_export)*2500
                )

            else:

                node_sizes.append(900)

        # -----------------------------
        # Edge Widths
        # -----------------------------

        weights = list(
            nx.get_edge_attributes(
                G,
                "weight"
            ).values()
        )

        max_weight = max(weights)

        edge_widths = [
            1 + (w/max_weight)*5
            for w in weights
        ]

        # -----------------------------
        # Draw
        # -----------------------------

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=1.5,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            width=edge_widths,
            arrows=True,
            arrowsize=18,
            alpha=0.7,
            connectionstyle="arc3,rad=0.08",
            ax=ax
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=9,
            font_weight="bold",
            ax=ax
        )

        ax.set_title(
            f"Trade Network - {year}",
            fontsize=16,
            fontweight="bold"
        )

        ax.axis("off")

    legend_elements = [

        Line2D(
            [0],[0],
            marker='o',
            color='w',
            label='Reporter Country',
            markerfacecolor='tomato',
            markeredgecolor='black',
            markersize=14
        ),

        Line2D(
            [0],[0],
            marker='o',
            color='w',
            label='Trading Partner',
            markerfacecolor='skyblue',
            markeredgecolor='black',
            markersize=14
        ),

        Line2D(
            [0],[0],
            color='gray',
            lw=4,
            label='Trade Flow'
        )

    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        "figures/trade_network_comparison.png",
        dpi=500,
        bbox_inches="tight"
    )

    plt.close()

    print("Data preprocessing completed.")

    return df


if __name__ == "__main__":
    run_data_preprocessing()