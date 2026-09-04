import pandas as pd


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    profiles = pd.read_excel(

        "output/behavioral_role_profiles.xlsx",

        index_col=0

    )

    return profiles

# ==========================================
# ASSIGN BEHAVIORAL ROLES
# ==========================================

def assign_behavioral_roles(profiles):
    assignment = pd.DataFrame({

        "Behavioral Role": [0, 1, 2],

         "Assigned Taxonomy": [

            "Stable Hub",

            "Community Anchor",

            "Bridge"

        ],

        "Primary Characteristics": [

            "Highest GDP, imports, in-strength, and out-strength.",

            "Highest eigenvector centrality, PageRank, and clustering coefficient.",

            "Highest betweenness centrality, acting as an intermediary."

        ],

        "Meaning": [

            "Dominant economic trade hub with consistently high trade activity.",

            "Highly influential and well-connected within the trade network.",

            "Connects different parts of the trade network by acting as an intermediary."

        ]

     })

    return assignment



    # ==========================================
    # Stable Hub
    # ==========================================

    stable_scores = (
        (profiles["GDP"] == profiles["GDP"].max()).astype(int) +
        (profiles["Imports"] == profiles["Imports"].max()).astype(int) +
        (profiles["In Strength"] == profiles["In Strength"].max()).astype(int) +
        (profiles["Out Strength"] == profiles["Out Strength"].max()).astype(int)
    )

    stable_role = stable_scores.idxmax()

    assignment[stable_role] = {
        "Assigned Taxonomy": "Stable Hub",
        "Primary Characteristics":
            "Highest GDP, imports, and trade strengths."
    }

    # ==========================================
    # Community Anchor
    # ==========================================

    community_scores = (
        (profiles["Eigenvector"] == profiles["Eigenvector"].max()).astype(int) +
        (profiles["PageRank"] == profiles["PageRank"].max()).astype(int) +
        (profiles["Clustering"] == profiles["Clustering"].max()).astype(int)
    )

    community_role = community_scores.idxmax()

    assignment[community_role] = {
        "Assigned Taxonomy": "Community Anchor",
        "Primary Characteristics":
            "Highest eigenvector centrality, PageRank, and clustering coefficient."
    }

    # ==========================================
    # Bridge
    # ==========================================

    bridge_role = profiles["Betweenness"].idxmax()

    assignment[bridge_role] = {
        "Assigned Taxonomy": "Bridge",
        "Primary Characteristics":
            "Highest betweenness centrality, acting as an intermediary."
    }

    # ==========================================
    # Convert to DataFrame
    # ==========================================

    assignment_df = (
        pd.DataFrame.from_dict(
            assignment,
            orient="index"
        )
        .reset_index()
        .rename(columns={"index": "Behavioral Role"})
        .sort_values("Behavioral Role")
    )

    return assignment_df


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(assignment):

    print("=" * 60)
    print("BEHAVIORAL ROLE ASSIGNMENT")
    print("=" * 60)

    print(assignment)

    assignment.to_excel(

        "output/behavioral_role_assignment.xlsx",

        index=False

    )


# ==========================================
# MAIN
# ==========================================

def main():

    profiles = load_data()

    assignment = assign_behavioral_roles(profiles)

    save_results(assignment)


if __name__ == "__main__":

    main()