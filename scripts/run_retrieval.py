from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_FILE = PROJECT_ROOT / "data" / "Evalcase03_RetrievalCorpus_v1.0.xlsx"
RUN_FILE = PROJECT_ROOT / "data" / "Evalcase03_RetrievalRun01.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "results" / "Evalcase03_RetrievalRun01_results.xlsx"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

corpus = pd.read_excel(CORPUS_FILE)
run = pd.read_excel(RUN_FILE)


# ---------------------------------------------------------
# Prepare corpus text
# ---------------------------------------------------------

corpus["search_text"] = (
    corpus["title"].fillna("").astype(str)
    + " "
    + corpus["status"].fillna("").astype(str)
    + " "
    + corpus["customer"].fillna("").astype(str)
    + " "
    + corpus["content"].fillna("").astype(str)
)


# ---------------------------------------------------------
# Build TF-IDF index
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
)

document_matrix = vectorizer.fit_transform(
    corpus["search_text"].tolist()
)


# ---------------------------------------------------------
# Prepare result columns
# ---------------------------------------------------------

for col in [
    "retrieved_1",
    "retrieved_2",
    "retrieved_3",
    "retrieved_4",
    "retrieved_5",
    "hit_at_3",
    "hit_at_5",
    "missing_gold_sources",
]:
    run[col] = run[col].astype("object")

run["all_gold_at_5"] = ""


# ---------------------------------------------------------
# Retrieval and evaluation
# ---------------------------------------------------------

for index, row in run.iterrows():

    question = str(row["question"])

    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        document_matrix
    )[0]

    ranked_indices = scores.argsort()[::-1][:5]

    top_docs = (
        corpus.iloc[ranked_indices]["doc_id"]
        .astype(str)
        .tolist()
    )

    # Save retrieved documents
    for rank, doc_id in enumerate(top_docs, start=1):
        run.at[index, f"retrieved_{rank}"] = doc_id

    # Parse gold sources
    gold_sources = [
        source.strip()
        for source in str(row["gold_sources"]).split("|")
        if source.strip()
    ]

    top_3 = top_docs[:3]
    top_5 = top_docs[:5]

    # At least one gold source in Top 3
    run.at[index, "hit_at_3"] = (
        "Yes"
        if any(source in top_3 for source in gold_sources)
        else "No"
    )

    # At least one gold source in Top 5
    run.at[index, "hit_at_5"] = (
        "Yes"
        if any(source in top_5 for source in gold_sources)
        else "No"
    )

    # Gold sources missing from Top 5
    missing = [
        source
        for source in gold_sources
        if source not in top_5
    ]

    run.at[index, "missing_gold_sources"] = (
        " | ".join(missing)
        if missing
        else ""
    )

    # All gold sources retrieved in Top 5
    run.at[index, "all_gold_at_5"] = (
        "Yes"
        if all(source in top_5 for source in gold_sources)
        else "No"
    )


# ---------------------------------------------------------
# Place All Gold Sources @5 next to Hit@5
# ---------------------------------------------------------

cols = list(run.columns)

if "all_gold_at_5" in cols:
    cols.remove("all_gold_at_5")

position = cols.index("hit_at_5") + 1
cols.insert(position, "all_gold_at_5")

run = run[cols]


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

run.to_excel(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

hit_at_3 = (run["hit_at_3"] == "Yes").sum()
hit_at_5 = (run["hit_at_5"] == "Yes").sum()
all_gold_at_5 = (run["all_gold_at_5"] == "Yes").sum()
total = len(run)

print("TF-IDF retrieval complete.")
print(f"Questions: {total}")
print(f"Hit@3: {hit_at_3}/{total} ({hit_at_3 / total:.1%})")
print(f"Hit@5: {hit_at_5}/{total} ({hit_at_5 / total:.1%})")
print(
    f"All Gold Sources @5: "
    f"{all_gold_at_5}/{total} ({all_gold_at_5 / total:.1%})"
)
print(f"Results saved to: {OUTPUT_FILE}")