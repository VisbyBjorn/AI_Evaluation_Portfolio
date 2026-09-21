from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TFIDF_FILE = PROJECT_ROOT / "results" / "Evalcase03_RetrievalRun01_results.xlsx"
EMBEDDING_FILE = PROJECT_ROOT / "results" / "Evalcase03_EmbeddingRun01_results.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "results" / "Evalcase03_RetrievalComparison_v1.0.xlsx"


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

tfidf = pd.read_excel(TFIDF_FILE)
embedding = pd.read_excel(EMBEDDING_FILE)

rows = []


# ---------------------------------------------------------
# Compare question by question
# ---------------------------------------------------------

for _, t_row in tfidf.iterrows():

    test_id = t_row["test_id"]

    matching_rows = embedding[
        embedding["test_id"] == test_id
    ]

    if matching_rows.empty:
        raise ValueError(
            f"No embedding result found for test_id: {test_id}"
        )

    e_row = matching_rows.iloc[0]

    tfidf_docs = [
        str(t_row[f"retrieved_{i}"])
        for i in range(1, 6)
    ]

    embedding_docs = [
        str(e_row[f"retrieved_{i}"])
        for i in range(1, 6)
    ]

    tfidf_all = str(t_row["all_gold_at_5"])
    embedding_all = str(e_row["all_gold_at_5"])

    if tfidf_all == "No" and embedding_all == "Yes":
        change = "Improved"
    elif tfidf_all == "Yes" and embedding_all == "No":
        change = "Regressed"
    else:
        change = "Unchanged"

    rows.append({
        "test_id": test_id,
        "gold_sources": t_row["gold_sources"],

        "tfidf_1": tfidf_docs[0],
        "tfidf_2": tfidf_docs[1],
        "tfidf_3": tfidf_docs[2],
        "tfidf_4": tfidf_docs[3],
        "tfidf_5": tfidf_docs[4],
        "tfidf_hit_at_3": t_row["hit_at_3"],
        "tfidf_hit_at_5": t_row["hit_at_5"],
        "tfidf_all_gold_at_5": tfidf_all,
        "tfidf_missing": t_row["missing_gold_sources"],

        "embedding_1": embedding_docs[0],
        "embedding_2": embedding_docs[1],
        "embedding_3": embedding_docs[2],
        "embedding_4": embedding_docs[3],
        "embedding_5": embedding_docs[4],
        "embedding_hit_at_3": e_row["hit_at_3"],
        "embedding_hit_at_5": e_row["hit_at_5"],
        "embedding_all_gold_at_5": embedding_all,
        "embedding_missing": e_row["missing_gold_sources"],

        "change": change,
    })


comparison = pd.DataFrame(rows)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def yes_count(series):
    return (series.astype(str) == "Yes").sum()


summary = pd.DataFrame([
    {
        "method": "TF-IDF",
        "hit_at_3": yes_count(tfidf["hit_at_3"]),
        "hit_at_5": yes_count(tfidf["hit_at_5"]),
        "all_gold_at_5": yes_count(tfidf["all_gold_at_5"]),
    },
    {
        "method": "Embeddings",
        "hit_at_3": yes_count(embedding["hit_at_3"]),
        "hit_at_5": yes_count(embedding["hit_at_5"]),
        "all_gold_at_5": yes_count(embedding["all_gold_at_5"]),
    },
])


# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl",
) as writer:

    comparison.to_excel(
        writer,
        sheet_name="Question comparison",
        index=False,
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("Retrieval comparison complete.")

for _, row in summary.iterrows():
    print(
        f"{row['method']}: "
        f"Hit@3 {row['hit_at_3']}/15, "
        f"Hit@5 {row['hit_at_5']}/15, "
        f"All Gold @5 {row['all_gold_at_5']}/15"
    )

print(f"Results saved to: {OUTPUT_FILE}")