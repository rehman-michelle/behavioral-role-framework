import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# ==========================================
# LOAD FEATURE MATRIX
# ==========================================

def load_data():

    df = pd.read_excel(
        "output/feature_matrix.xlsx"
    )

    print("Feature matrix loaded successfully.\n")

    return df


# ==========================================
# SPLIT IDENTIFIERS & FEATURES
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
# RUN PCA
# ==========================================

def run_pca(features):

    pca = PCA()

    principal_components = pca.fit_transform(features)

    explained_variance = pca.explained_variance_ratio_

    cumulative_variance = explained_variance.cumsum()

    return (
        pca,
        principal_components,
        explained_variance,
        cumulative_variance
    )


# ==========================================
# PRINT PCA SUMMARY
# ==========================================

def print_results(explained_variance,
                  cumulative_variance):

    print("=" * 60)
    print("EXPLAINED VARIANCE")
    print("=" * 60)

    for i, value in enumerate(explained_variance):

        print(
            f"PC{i+1}: {value:.4f}"
        )

    print("\n")

    print("=" * 60)
    print("CUMULATIVE VARIANCE")
    print("=" * 60)

    for i, value in enumerate(cumulative_variance):

        print(
            f"PC{i+1}: {value:.4f}"
        )


# ==========================================
# SAVE PCA SCORES (KEEP ONLY FIRST 4 PCs)
# ==========================================

def save_scores(
        identifiers,
        principal_components
):

    # Keep only the first four principal components
    principal_components = principal_components[:, :4]

    pc_columns = [
        "PC1",
        "PC2",
        "PC3",
        "PC4"
    ]

    scores = pd.DataFrame(
        principal_components,
        columns=pc_columns
    )

    scores = pd.concat(
        [
            identifiers.reset_index(drop=True),
            scores
        ],
        axis=1
    )

    scores.to_excel(
        "output/pca_scores.xlsx",
        index=False
    )

    print("\nPCA scores saved successfully!")
    print(scores.head())

# ==========================================
# SAVE PCA LOADINGS (FIRST 4 PCs ONLY)
# ==========================================

def save_loadings(
        pca,
        features
):

    loadings = pd.DataFrame(

        pca.components_[:4].T,

        index=features.columns,

        columns=[
            "PC1",
            "PC2",
            "PC3",
            "PC4"
        ]

    )

    loadings.to_excel(
        "output/pca_loadings.xlsx"
    )

    print("PCA loadings saved successfully!")
    print(loadings)


# ==========================================
# SAVE EXPLAINED VARIANCE (FIRST 4 PCs)
# ==========================================

def save_variance(
        explained_variance,
        cumulative_variance
):

    variance_table = pd.DataFrame({

        "Principal Component": [
            "PC1",
            "PC2",
            "PC3",
            "PC4"
        ],

        "Explained Variance":
            explained_variance[:4],

        "Cumulative Variance":
            cumulative_variance[:4]

    })

    variance_table.to_excel(
        "output/pca_variance.xlsx",
        index=False
    )

    print("Variance table saved successfully!")
    print(variance_table)


# ==========================================
# SCREE PLOT
# ==========================================

def scree_plot(explained_variance):

    plt.figure(figsize=(8,6))

    plt.plot(

        range(
            1,
            len(explained_variance)+1
        ),

        explained_variance,

        marker="o",

        linewidth=2

    )

    plt.xlabel("Principal Component")

    plt.ylabel("Explained Variance")

    plt.title("Scree Plot")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        "figures/scree_plot.png",

        dpi=500

    )

    plt.show()


# ==========================================
# CUMULATIVE VARIANCE PLOT
# ==========================================

def cumulative_plot(cumulative_variance):

    plt.figure(figsize=(8,6))

    plt.plot(

        range(
            1,
            len(cumulative_variance)+1
        ),

        cumulative_variance,

        marker="o",

        linewidth=2

    )

    plt.axhline(

        y=0.90,

        linestyle="--"

    )

    plt.xlabel("Principal Component")

    plt.ylabel("Cumulative Variance")

    plt.title("Cumulative Explained Variance")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        "figures/cumulative_variance.png",

        dpi=500

    )

    plt.show()


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_data()

    identifiers, features = split_data(df)

    (
        pca,
        principal_components,
        explained_variance,
        cumulative_variance
    ) = run_pca(features)

    print_results(

        explained_variance,

        cumulative_variance

    )

    save_scores(

        identifiers,

        principal_components

    )

    save_loadings(

        pca,

        features

    )

    save_variance(

        explained_variance,

        cumulative_variance

    )

    scree_plot(
        explained_variance
    )

    cumulative_plot(
        cumulative_variance
    )


if __name__ == "__main__":

    main()

# according to the result, you will only use PC1, PC2, PC3, and PC4.
# PC1 explains trade strength (after looking at the pca_loadings.xlsx)
# PC2 explains network connectivity
# PC3 explains strategic brokerage
# PC4 explains structural influence

