#does the Behavioral Role Framework actually tell us something meaningful?
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    clusters = pd.read_excel(
        "output/clustered_countries.xlsx"
    )

    features = pd.read_excel(
        "output/final_ai_dataset.xlsx"
    )

    print("Datasets loaded successfully.\n")

    return clusters, features

# ==========================================
# MERGE DATASETS
# ==========================================

def merge_data(clusters, features):

    clusters = clusters[
        ["Year", "Country", "Cluster"]
    ]

    df = pd.merge(

        features,

        clusters,

        on=["Year", "Country"],

        how="inner"

    )

    print("Merged dataset shape:")

    print(df.shape)

    return df

# ======================================
# FIRST EVALUATION: BEHAVIORAL ROLE DISTRIBUTION
# ======================================
# ==========================================
# ROLE DISTRIBUTION
# ==========================================

def role_distribution(df):

    distribution = (

        df["Cluster"]

        .value_counts()

        .sort_index()

        .reset_index()

    )

    distribution.columns = [

        "Behavioral Role",

        "Frequency"

    ]

    distribution["Percentage"] = (

        distribution["Frequency"]

        / distribution["Frequency"].sum()

        * 100

    )

    print("=" * 50)

    print("BEHAVIORAL ROLE DISTRIBUTION")

    print("=" * 50)

    print(distribution)

    distribution.to_excel(

        "output/role_distribution.xlsx",

        index=False

    )

    return distribution

# ==========================================
# ROLE DISTRIBUTION PIE CHART
# ==========================================

def plot_role_distribution(distribution):

    plt.figure(figsize=(7,7))

    plt.pie(

        distribution["Frequency"],

        labels=[

            f"Role {int(i)}"

            for i in distribution["Behavioral Role"]

        ],
#        wedgeprops={"edgecolor": "white", "linewidth": 2},

        autopct="%1.1f%%",

        startangle=90,

        counterclock=False,

        textprops={"fontsize":14}

    )

    plt.title(

        "Distribution of Behavioral Roles",

        fontsize=18,

        fontweight="bold"

    )

    plt.tight_layout()

    plt.savefig(

        "figures/role_distribution.png",

        dpi=600,

        bbox_inches="tight"

    )

    plt.show()



# ===================================================
# EVALUATION 2: BEHAVIORAL ROLE TRANSITION MATRIX
# ===================================================
# ==========================================
# TRANSITION MATRIX
# ==========================================

def transition_matrix(df):

    df = df.sort_values(

        ["Country", "Year"]

    )

    df["Next_Cluster"] = (

        df

        .groupby("Country")["Cluster"]

        .shift(-1)

    )

    transitions = df.dropna(

        subset=["Next_Cluster"]

    )

    matrix = pd.crosstab(

        transitions["Cluster"],

        transitions["Next_Cluster"],

        rownames=["Current Role"],

        colnames=["Next Role"]

    )

    print("=" * 50)

    print("BEHAVIORAL ROLE TRANSITION MATRIX")

    print("=" * 50)

    print(matrix)

    matrix.to_excel(

        "output/transition_matrix.xlsx"

    )

    return matrix

# ==========================================
# TRANSITION HEATMAP
# ==========================================

def plot_transition_matrix(matrix):

    plt.figure(figsize=(8,7))

    im = plt.imshow(
        matrix,
        cmap="turbo",
        interpolation="nearest"
    )

    cbar = plt.colorbar(im)
    cbar.set_label(
        "Number of Transitions",
        fontsize=16
    )

    plt.xticks(
        range(len(matrix.columns)),
        [f"Role {int(i)}" for i in matrix.columns],
        fontsize=14
    )

    plt.yticks(
        range(len(matrix.index)),
        [f"Role {int(i)}" for i in matrix.index],
        fontsize=14
    )

    plt.xlabel(
        "Next Role",
        fontsize=18
    )

    plt.ylabel(
        "Current Role",
        fontsize=18
    )

    plt.title(
        "Behavioral Role Transition Matrix",
        fontsize=22,
        fontweight="bold",
        pad=20
    )

    max_value = matrix.values.max()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            value = matrix.iloc[i, j]

          #  color = "white" if value > max_value / 2 else "black"
            color="white"
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=color
            )

    plt.tight_layout()

    plt.savefig(
        "figures/transition_heatmap.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


# ===============================================
# EVALUATION 3: COUNTRY BEHAVIORAL STABILITY
# ===============================================
# ==========================================
# COUNTRY STABILITY
# ==========================================

def country_stability(df):

    df = df.sort_values(

        ["Country", "Year"]

    )

    stability = []

    for country in df["Country"].unique():

        temp = df[

            df["Country"] == country

        ]

        role_changes = (

            temp["Cluster"]

            != temp["Cluster"].shift()

        ).sum() - 1

        stability.append({

            "Country": country,

            "Role Changes": role_changes

        })

    stability = pd.DataFrame(stability)

    print("=" * 50)

    print("COUNTRY BEHAVIORAL STABILITY")

    print("=" * 50)

    print(stability)

    stability.to_excel(

        "output/country_stability.xlsx",

        index=False

    )

    return stability

# only table is used for evaluation 3. no visualization


# ==================================================
# EVALUATION 4: BEHAVIORAL ROLE PROFILES
# ==================================================
# ==========================================
# BEHAVIORAL ROLE PROFILES
# ==========================================

def role_profiles(df):

    numeric_cols = df.select_dtypes(include="number").columns

    numeric_cols = numeric_cols.drop(["Cluster", "Year"])

    profiles = (

        df.groupby("Cluster")[numeric_cols]

        .mean()

        .round(3)

    )

    profiles.index.name = "Behavioral Role"

    print("=" * 50)
    print("BEHAVIORAL ROLE PROFILES")
    print("=" * 50)
    print(profiles)

    profiles.to_excel(
        "output/behavioral_role_profiles.xlsx"
    )

    return profiles

# ================================================
# EVALUATION 5: COUNTRY BEHAVIORAL TIMELINE
# ================================================
# ==========================================
# BEHAVIORAL TIMELINE
# ==========================================

def behavioral_timeline(df):

    df = df.sort_values(["Country", "Year"])

    timeline = (

        df.groupby("Country")["Cluster"]

        .apply(lambda x: "".join(x.astype(int).astype(str)))

        .reset_index()

    )

    timeline.columns = [

        "Country",

        "Behavioral Timeline"

    ]

    print("=" * 50)
    print("BEHAVIORAL TIMELINE")
    print("=" * 50)
    print(timeline)

    timeline.to_excel(

        "output/behavioral_timeline.xlsx",

        index=False

    )

    return timeline



# ==========================================
# MAIN
# ==========================================

def main():

    clusters, features = load_data()

    df = merge_data(

        clusters,

        features

    )

    distribution = role_distribution(df)

    plot_role_distribution(

        distribution

    )

    matrix = transition_matrix(df)

    plot_transition_matrix(matrix)

    stability = country_stability(df)

    profiles = role_profiles(df)

    timeline = behavioral_timeline(df)

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()