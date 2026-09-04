import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# LOAD PCA SCORES
# ==========================================

def load_data():

    df = pd.read_excel(
        "output/pca_scores.xlsx"
    )

    print("PCA scores loaded successfully.")

    return df


# ==========================================
# SPLIT DATA
# ==========================================

def split_data(df):

    identifiers = df[
        ["Year", "Country"]
    ]

    features = df.drop(
        columns=["Year", "Country"]
    )

    return identifiers, features


# ==========================================
# ELBOW METHOD
# ==========================================

def elbow_method(features):

    inertia = []

    K = range(2,11)

    for k in K:

        model = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10

        )

        model.fit(features)

        inertia.append(
            model.inertia_
        )

    plt.figure(figsize=(8,6))

    plt.plot(

        K,

        inertia,

        marker="o"

    )

    plt.xlabel("Number of Clusters")

    plt.ylabel("Within Cluster Sum of Squares")

    plt.title("Elbow Method")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.axvline(
        x=3,
        linestyle="--",
        linewidth=2,
        color="red"
    )

    plt.savefig(
        "figures/elbow_method.png",
        dpi=500
    )

    plt.show()


# ==========================================
# SILHOUETTE SCORE
# ==========================================

def silhouette_analysis(features):

    scores = []

    K = range(2,11)

    for k in K:

        model = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10

        )

        labels = model.fit_predict(features)

        score = silhouette_score(

            features,

            labels

        )

        scores.append(score)

    plt.figure(figsize=(8,6))

    plt.plot(

        K,

        scores,

        marker="o"

    )

    plt.xlabel("Number of Clusters")

    plt.ylabel("Silhouette Score")

    plt.title("Silhouette Analysis")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.axvline(
        x=3,
        linestyle="--",
        linewidth=2,
        color="red"
    )

    plt.savefig(

        "figures/silhouette_scores.png",

        dpi=500

    )

    plt.show()

    recommended_k = K[scores.index(max(scores))]

    print("\nSilhouette Analysis Recommendation: K =", recommended_k)
    print("Selected K = 3 for the final Behavioral Role Framework")

    return recommended_k


# ==========================================
# FINAL CLUSTERING
# ==========================================

def perform_clustering(

        features,

        k

):

    model = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    labels = model.fit_predict(features)

    return labels


# ==========================================
# SAVE RESULTS
# ==========================================

def save_clusters(

        identifiers,

        features,

        labels

):

    clustered = pd.concat(

        [

            identifiers,

            features

        ],

        axis=1

    )

    clustered["Cluster"] = labels

    clustered.to_excel(

        "output/clustered_countries.xlsx",

        index=False

    )

    print(clustered.head())

    return clustered


# ==========================================
# CLUSTER VISUALIZATION (1993, 2003, 2013, 2023)
# ==========================================

def plot_clusters(clustered):

    years_to_plot = [1993, 2003, 2013, 2023]

    for year in years_to_plot:

        year_data = clustered[
            clustered["Year"] == year
        ].copy()

        plt.figure(figsize=(10,7))

        scatter = plt.scatter(
            year_data["PC1"],
            year_data["PC2"],
            c=year_data["Cluster"],
            cmap="viridis",
            s=120,
            edgecolor="black"
        )

        for _, row in year_data.iterrows():

            plt.text(
                row["PC1"] + 0.05,
                row["PC2"] + 0.05,
                row["Country"],
                fontsize=9
            )

        plt.title(
            f"Behavioral Role Clusters ({year})",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.grid(alpha=0.3)

        plt.colorbar(
            scatter,
            label="Cluster"
        )

        plt.tight_layout()

        plt.savefig(
            f"figures/behavior_clusters_{year}.png",
            dpi=500
        )

        plt.show()


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_data()

    identifiers, features = split_data(df)

    elbow_method(features)

    #Generate the silhouette plot for evaluation
    silhouette_analysis(features)

    #select K=3 for the final Behavioral Role Framework
    best_k = 3

    labels = perform_clustering(

        features,

        best_k

    )

    clustered = save_clusters(

        identifiers,

        features,

        labels

    )

    plot_clusters(clustered)


if __name__ == "__main__":

    main()