import pandas as pd

# ==========================================
# LOAD CLUSTER RESULTS
# ==========================================

def load_data():

    df = pd.read_excel(
        "output/clustered_countries.xlsx"
    )

    print("Clustered dataset loaded.")

    return df


# ==========================================
# SORT DATA
# ==========================================

def sort_data(df):

    df = df.sort_values(
        ["Country", "Year"]
    ).reset_index(drop=True)

    return df


# ==========================================
# CREATE BEHAVIOUR SEQUENCES
# ==========================================

def create_sequences(df):

    sequences = {}

    for country in df["Country"].unique():

        country_df = df[
            df["Country"] == country
        ].sort_values("Year")

        sequences[country] = list(
            country_df["Cluster"]
        )

    return sequences


# ==========================================
# PRINT SEQUENCES
# ==========================================

def print_sequences(sequences):

    print("\nBehaviour Sequences\n")

    for country, sequence in sequences.items():

        print(country)

        print(sequence)

        print()


# ==========================================
# SAVE SEQUENCES
# ==========================================

def save_sequences(sequences):

    sequence_df = pd.DataFrame({

        "Country": list(sequences.keys()),

        "Behaviour_Sequence":
            [str(seq) for seq in sequences.values()]

    })

    print(sequence_df)

    sequence_df.to_excel(

        "output/behaviour_sequences.xlsx",

        index=False

    )
    test = pd.read_excel("output/behaviour_sequences.xlsx")
    print(test)

    print("Behaviour sequences saved.")


# ==========================================
# CREATE TRANSITION TABLE
# ==========================================

def create_transition_table(sequences):

    rows = []

    for country, sequence in sequences.items():

        for i in range(len(sequence)-1):

            rows.append({

                "Country": country,

                "Year_t": 1992 + i,

                "Current_Role": sequence[i],

                "Next_Role": sequence[i+1]

            })

    transition_df = pd.DataFrame(rows)

    print(transition_df.head())
    print(transition_df.shape)

    transition_df.to_excel(

        "output/behaviour_transitions.xlsx",

        index=False

    )
    test2 = pd.read_excel("output/behaviour_transitions.xlsx")
    print(test2.head())

    print("Transition table saved.")

    return transition_df


# ==========================================
# MAIN
# ==========================================

def main():

    df = load_data()

    df = sort_data(df)

    sequences = create_sequences(df)

    print_sequences(sequences)

    save_sequences(sequences)

    create_transition_table(sequences)


if __name__ == "__main__":

    main()