import pandas as pd
import networkx as nx

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_excel("data/main_data.xlsx")

reporters = [
    "China",
    "Japan",
    "Rep. of Korea",
    "Singapore",
    "USA"
]

# ==========================================
# COUNTRY-LEVEL NETWORK METRICS
# ==========================================

results = []

years = sorted(df["Year"].unique())

for year in years:

    year_df = df[df["Year"] == year]

    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()

    for _, row in year_df.iterrows():

        G_directed.add_edge(
            row["Reporter"],
            row["Partner"],
            weight=row["Export_Value"]
        )

        G_undirected.add_edge(
            row["Reporter"],
            row["Partner"],
            weight=row["Export_Value"]
        )

    # -----------------------------
    # Network Metrics
    # -----------------------------

    degree = nx.degree_centrality(G_undirected)

    weighted_degree = dict(
        G_undirected.degree(weight="weight")
    )

    betweenness = nx.betweenness_centrality(
        G_undirected,
        weight="weight",
        normalized=True
    )

    closeness = nx.closeness_centrality(
        G_undirected
    )

    eigenvector = nx.eigenvector_centrality(
        G_undirected,
        weight="weight",
        max_iter=1000
    )

    clustering = nx.clustering(
        G_undirected,
        weight="weight"
    )

    pagerank = nx.pagerank(
        G_directed,
        weight="weight"
    )

    in_degree = dict(
        G_directed.in_degree()
    )

    out_degree = dict(
        G_directed.out_degree()
    )

    in_strength = dict(
        G_directed.in_degree(weight="weight")
    )

    out_strength = dict(
        G_directed.out_degree(weight="weight")
    )

    # -----------------------------
    # Save Results
    # -----------------------------

    for country in G_directed.nodes():

        results.append({

            "Year": year,

            "Country": country,

            "Degree": degree[country],

            "In Degree": in_degree[country],

            "Out Degree": out_degree[country],

            "Weighted Degree": weighted_degree[country],

            "In Strength": in_strength[country],

            "Out Strength": out_strength[country],

            "Betweenness": betweenness[country],

            "Closeness": closeness[country],

            "Eigenvector": eigenvector[country],

            "PageRank": pagerank[country],

            "Clustering": clustering[country]

        })

# ==========================================
# CREATE MASTER DATASET
# ==========================================

master_df = pd.DataFrame(results)

master_df = master_df.round({

    "Degree":4,
    "In Degree":0,
    "Out Degree":0,
    "Weighted Degree":0,
    "In Strength":0,
    "Out Strength":0,
    "Betweenness":4,
    "Closeness":4,
    "Eigenvector":4,
    "PageRank":4,
    "Clustering":4

})

master_df = master_df[[

    "Year",
    "Country",
    "Degree",
    "In Degree",
    "Out Degree",
    "Weighted Degree",
    "In Strength",
    "Out Strength",
    "Betweenness",
    "Closeness",
    "Eigenvector",
    "PageRank",
    "Clustering"

]]

master_df = master_df.sort_values(
    ["Year", "Country"]
)

print("\nMissing Values")
print(master_df.isnull().sum())

print("\nDuplicate Rows")
print(master_df.duplicated().sum())

print("\nSummary Statistics")
print(master_df.describe())

master_df.to_excel(
    "output/master_network_metrics.xlsx",
    index=False
)

print(master_df.head())

# ==========================================
# MERGE ECONOMIC VARIABLES
# ==========================================

gdp_df = pd.read_excel("data/gdp_data.xlsx")
imports_df = pd.read_excel("data/import_data.xlsx")
exchange_df = pd.read_excel("data/exchange_rate.xlsx")

master_df = master_df.merge(
    gdp_df,
    on=["Year", "Country"],
    how="left"
)

master_df = master_df.merge(
    imports_df,
    on=["Year", "Country"],
    how="left"
)

master_df = master_df.merge(
    exchange_df,
    on=["Year", "Country"],
    how="left"
)

master_df = master_df[
    master_df["Country"].isin(reporters)
].reset_index(drop=True)

print(master_df.shape)
print(master_df.isnull().sum())

# ==========================================
# Singapore Connectivity Check
# ==========================================

singapore = master_df[
    master_df["Country"] == "Singapore"
]

print(
    singapore[
        ["Year", "Out Degree", "Degree"]
    ]
)

# ==========================================
# SAVE FINAL AI DATASET
# ==========================================
print("\nColumns in final dataset:")
print(master_df.columns.tolist())

print("\nFirst five rows:")
print(master_df.head())

print("\nGDP columns:")
print(gdp_df.columns.tolist())

print("\nImports columns:")
print(imports_df.columns.tolist())

print("\nExchange columns:")
print(exchange_df.columns.tolist())

print(master_df.columns.tolist())

print(master_df[["Country", "Year", "In Degree"]].head(20))

print(master_df["In Degree"].describe())

master_df.to_excel(
    "output/final_ai_dataset.xlsx",
    index=False
)

print("\nFinal AI dataset saved successfully.")
print(master_df.head())

