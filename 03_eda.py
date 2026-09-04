import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# ==========================================
# LOAD DATASETS
# ==========================================

master_df = pd.read_excel(
    "output/master_network_metrics.xlsx"
)

final_df = pd.read_excel(
    "output/final_ai_dataset.xlsx"
)

print(final_df[["Country", "Year", "In Degree"]].head(20))

reporters = [
    "China",
    "Japan",
    "Rep. of Korea",
    "Singapore",
    "USA"
]

print(master_df.head())
print(final_df.head())

####################################################
# EDA PART 1
# Uses master_network_metrics.xlsx
####################################################

# ==========================================
# EDA 1 - Descriptive Statistics
# ==========================================

print("\n========================================")
print("DESCRIPTIVE STATISTICS")
print("========================================\n")

#stats = master_df.drop(
stats = final_df.drop(
    columns=["Year", "Country"]
).describe()

print(stats)

stats.to_excel(
    "output/descriptive_statistics.xlsx"
)


# ==========================================
# EDA 2 - Full Correlation Matrix
# ==========================================

full_corr = final_df.drop(
    columns=["Year","Country", "In Degree"] #there is no info for In-Degree because it does not change partners at all.
# it remains 4 and hence has no variation. therefore, pandas changes it into 0 or NaN. this is why we removed it
).corr()

full_corr.to_excel(
    "output/final_correlation_matrix.xlsx"
)

print(full_corr)


# ==========================================
# EDA 3 - Full Correlation Heatmap
# ==========================================

plt.figure(figsize=(12,10))

plt.imshow(
    full_corr,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(full_corr.columns)),
    full_corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(full_corr.columns)),
    full_corr.columns
)

plt.title(
    "Correlation Between Network and Economic Variables",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "figures/final_correlation_heatmap.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 4 - Weighted Degree Over Time
# ==========================================

reporters = [
    "China",
    "Japan",
    "Rep. of Korea",
    "Singapore",
    "USA"
]

plt.figure(figsize=(12, 7))

for country in reporters:

    country_data = master_df[
        master_df["Country"] == country
    ]

    plt.plot(
        country_data["Year"],
        country_data["Weighted Degree"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

plt.title(
    "Weighted Degree of Reporter Countries (2000–2024)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year", fontsize=12)

plt.ylabel("Weighted Degree", fontsize=12)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/weighted_degree_over_time.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 5 - In Strength vs Out Strength
# (shows whether countries are primarily export-oriented or import-oriented)
# ==========================================

reporters = [
    "China",
    "Japan",
    "Rep. of Korea",
    "Singapore",
    "USA"
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# -------------------------
# In Strength
# -------------------------

for country in reporters:

    country_data = master_df[
        master_df["Country"] == country
    ]

    axes[0].plot(
        country_data["Year"],
        country_data["In Strength"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

axes[0].set_title(
    "In Strength (2000–2024)",
    fontsize=15,
    fontweight="bold"
)

axes[0].set_xlabel("Year")
axes[0].set_ylabel("In Strength")

axes[0].grid(alpha=0.3)

# -------------------------
# Out Strength
# -------------------------

for country in reporters:

    country_data = master_df[
        master_df["Country"] == country
    ]

    axes[1].plot(
        country_data["Year"],
        country_data["Out Strength"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

axes[1].set_title(
    "Out Strength (2000–2024)",
    fontsize=15,
    fontweight="bold"
)

axes[1].set_xlabel("Year")
axes[1].set_ylabel("Out Strength")

axes[1].grid(alpha=0.3)

# -------------------------
# Shared Legend
# -------------------------

handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    fontsize=10,
    frameon=False
)

plt.tight_layout(rect=[0,0.08,1,1])

plt.savefig(
    "figures/in_out_strength.png",
    dpi=500,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# EDA 6 - PageRank Over Time
# to see the influence in a directed network
# ==========================================

plt.figure(figsize=(12,7))

for country in reporters:

    country_data = master_df[
        master_df["Country"] == country
    ]

    plt.plot(
        country_data["Year"],
        country_data["PageRank"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

plt.title(
    "PageRank of Reporter Countries (2000–2024)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year")
plt.ylabel("PageRank")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/pagerank_over_time.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 7 - Eigenvector Centrality
# ==========================================

plt.figure(figsize=(12,7))

for country in reporters:

    country_data = master_df[
        master_df["Country"] == country
    ]

    plt.plot(
        country_data["Year"],
        country_data["Eigenvector"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

plt.title(
    "Eigenvector Centrality (2000–2024)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year")
plt.ylabel("Eigenvector Centrality")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/eigenvector_over_time.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 8 - Boxplots
# Before doing PCA or clustering, it's good practice to inspect the distributions and potential outliers
# ==========================================

# ==========================================
# EDA 8A - Network Variables Boxplot
# ==========================================

network_metrics = final_df[
    [
        "Degree",
        "Out Degree",
        "Weighted Degree",
        "In Strength",
        "Out Strength",
        "Betweenness",
        "Closeness",
        "Eigenvector",
        "PageRank",
        "Clustering"
    ]
]

plt.figure(figsize=(12,6))

plt.boxplot(
    network_metrics.values,
    tick_labels=network_metrics.columns,
    patch_artist=True
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.title(
    "Distribution of Network Variables",
    fontsize=16,
    fontweight="bold"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/network_variables_boxplot.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 8B - Economic Variables Boxplot
# ==========================================

economic_variables = final_df[
    [
        "GDP",
        "Imports",
        "Exchange_Rate"
    ]
]

plt.figure(figsize=(8,6))

plt.boxplot(
    economic_variables.values,
    tick_labels=economic_variables.columns,
    patch_artist=True
)

plt.title(
    "Distribution of Economic Variables",
    fontsize=16,
    fontweight="bold"
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/economic_variables_boxplot.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 8C - Standardized Boxplot (only use this one in paper)
# ==========================================

metrics = final_df.drop(
    columns=["Year", "Country", "In Degree"]
)

scaler = StandardScaler()

scaled = scaler.fit_transform(metrics)

scaled_df = pd.DataFrame(
    scaled,
    columns=metrics.columns
)

plt.figure(figsize=(15,7))

plt.boxplot(
    scaled_df.values,
    tick_labels=scaled_df.columns,
    patch_artist=True
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.title(
    "Distribution of Standardized Analytical Variables",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Standardized Value (Z-score)")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/standardized_variables_boxplot.png",
    dpi=500
)

plt.show()

scaled_summary = scaled_df.describe().T[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
]

print(scaled_summary)

scaled_summary.to_excel(
    "output/standardized_variables_summary.xlsx"
)


from sklearn.preprocessing import MinMaxScaler

# Normalize the original variables
scaler = MinMaxScaler()

normalized = scaler.fit_transform(metrics)

normalized_df = pd.DataFrame(
    normalized,
    columns=metrics.columns
)

# Summary statistics
normalized_summary = normalized_df.describe().T[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
]

print(normalized_summary)

normalized_summary.to_excel(
    "output/normalized_variables_summary.xlsx"
)




# ==========================================
# EDA 9 - GDP Trends (figure not needed)
# ==========================================

reporters = [
    "China",
    "Japan",
    "Rep. of Korea",
    "Singapore",
    "USA"
]

plt.figure(figsize=(12,7))

for country in reporters:

    country_data = final_df[
        final_df["Country"] == country
    ]

    plt.plot(
        country_data["Year"],
        country_data["GDP"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

plt.title(
    "GDP of Reporter Countries (2000–2024)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year")
plt.ylabel("GDP")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/gdp_over_time.png",
    dpi=500
)

plt.show()

# ==========================================
# EDA 10 - Imports Over Time (figure not needed)
# ==========================================

plt.figure(figsize=(12,7))

for country in reporters:

    country_data = final_df[
        final_df["Country"] == country
    ]

    plt.plot(
        country_data["Year"],
        country_data["Imports"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

plt.title(
    "Imports of Reporter Countries (2000–2024)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year")
plt.ylabel("Imports")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/imports_over_time.png",
    dpi=500
)

plt.show()


# ==========================================
# EDA 11 - Exchange Rate (figure not needed)
# ==========================================

plt.figure(figsize=(12,7))

for country in reporters:

    country_data = final_df[
        final_df["Country"] == country
    ]

    plt.plot(
        country_data["Year"],
        country_data["Exchange_Rate"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=country
    )

plt.title(
    "Exchange Rate (2000–2024)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year")
plt.ylabel("Exchange Rate")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/exchange_rate_over_time.png",
    dpi=500
)

plt.show()