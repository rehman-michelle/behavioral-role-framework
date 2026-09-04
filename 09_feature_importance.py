import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# ==========================================
# LOAD DATASETS
# ==========================================

def load_data():

    features = pd.read_excel(
        "output/final_ai_dataset.xlsx"
    )

    clusters = pd.read_excel(
        "output/clustered_countries.xlsx"
    )

    print("Datasets loaded successfully.\n")

    return features, clusters


# ==========================================
# MERGE FEATURES WITH CLUSTERS
# ==========================================

def merge_data(features, clusters):

    clusters = clusters[
        ["Year", "Country", "Cluster"]
    ]

    df = pd.merge(

        features,

        clusters,

        on=["Year", "Country"],

        how="inner"

    )

    print("\nMerged dataset shape:")

    print(df.shape)

    return df


# ==========================================
# CREATE TARGET VARIABLE
# ==========================================

def create_target(df):

    df = df.sort_values(

        ["Country", "Year"]

    )

    df["Next_Role"] = (

        df
        .groupby("Country")["Cluster"]
        .shift(-1)

    )

    df = df.dropna()

    df["Next_Role"] = df["Next_Role"].astype(int)

    return df


# ==========================================
# FEATURE MATRIX
# ==========================================

def create_features():

    feature_columns = [

        "Degree",

        "Out Degree",

        "Weighted Degree",

        "In Strength",

        "Out Strength",

        "Betweenness",

        "Closeness",

        "Eigenvector",

        "PageRank",

        "Clustering",

        "GDP",

        "Imports",

        "Exchange_Rate"

    ]

    return feature_columns


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

def train_model(X, y):

    model = RandomForestClassifier(

        n_estimators=300,

        random_state=42

    )

    model.fit(

        X,

        y

    )

    return model


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def feature_importance(model, X):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    print("=" * 50)

    print("RANDOM FOREST FEATURE IMPORTANCE")

    print("=" * 50)

    print(importance)

    importance.to_excel(

        "output/feature_importance.xlsx",

        index=False

    )

    plt.figure(figsize=(10, 6))

    plt.barh(

        importance["Feature"],

        importance["Importance"]

    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(

        "figures/feature_importance.png",

        dpi=300

    )

    plt.show()

    print("\nFeature importance saved successfully.")


# ==========================================
# MAIN
# ==========================================

def main():

    features, clusters = load_data()

    df = merge_data(

        features,

        clusters

    )

    df = create_target(df)

    feature_columns = create_features()

    X = df[feature_columns]

    y = df["Next_Role"]

    model = train_model(

        X,

        y

    )

    feature_importance(

        model,

        X

    )


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()