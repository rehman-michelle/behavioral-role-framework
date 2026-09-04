import pandas as pd
from sklearn.preprocessing import StandardScaler


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    df = pd.read_excel(
        "output/final_ai_dataset.xlsx"
    )

    print("Dataset loaded successfully.\n")

    return df


# ==========================================
# DATA QUALITY CHECK
# ==========================================

def check_data(df):

    print("=" * 50)
    print("DATA QUALITY CHECK")
    print("=" * 50)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


# ==========================================
# CREATE FEATURE MATRIX
# ==========================================

def create_feature_matrix(df):

    feature_columns = [

        "Degree",
        #"In Degree",
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

    features = df[feature_columns]

    identifiers = df[["Year", "Country"]]

    return identifiers, features


# ==========================================
# STANDARDIZE FEATURES
# ==========================================

def standardize_features(features):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(

        scaled,

        columns=features.columns

    )

    return scaled_df


# ==========================================
# CHECK STANDARDIZATION
# ==========================================

def check_scaled_data(scaled_df):

    print("\n")
    print("=" * 50)
    print("STANDARDIZED DATA SUMMARY")
    print("=" * 50)

    print(scaled_df.describe())


# ==========================================
# SAVE OUTPUTS
# ==========================================

def save_outputs(identifiers, scaled_df):

    scaled_df.to_excel(

        "output/scaled_features.xlsx",

        index=False

    )

    feature_matrix = pd.concat(

        [

            identifiers.reset_index(drop=True),

            scaled_df.reset_index(drop=True)

        ],

        axis=1

    )

    feature_matrix.to_excel(

        "output/feature_matrix.xlsx",

        index=False

    )

    print("\nFiles saved successfully!")

    print("→ output/scaled_features.xlsx")

    print("→ output/feature_matrix.xlsx")


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_data()

    check_data(df)

    identifiers, features = create_feature_matrix(df)

    scaled_df = standardize_features(features)

    check_scaled_data(scaled_df)

    save_outputs(

        identifiers,

        scaled_df

    )


if __name__ == "__main__":

    main()