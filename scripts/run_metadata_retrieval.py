from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_FILE = PROJECT_ROOT / "data" / "Evalcase03_RetrievalCorpus_v1.0.xlsx"
RUN_FILE = PROJECT_ROOT / "data" / "Evalcase03_MetadataRun01.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "results" / "Evalcase03_MetadataRun01_results.xlsx"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

corpus = pd.read_excel(CORPUS_FILE)
run = pd.read_excel(RUN_FILE)


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
# Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

document_texts = [
    "passage: " + text
    for text in corpus["search_text"].tolist()
]

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# Metadata adjustment
# ---------------------------------------------------------

def metadata_adjustment(row, question):
    adjustment = 0.0

    status = str(row["status"]).lower()
    customer = str(row["customer"]).lower()
    question_lower = question.lower()

    if "gällande" in status:
        adjustment += 0.10

    if "kundspecifikt avtal" in status:
        adjustment += 0.10

    if customer and customer != "nan":
        if customer in question_lower:
            adjustment += 0.05

    negative_terms = [
        "utkast",
        "ej accepterat",
        "ej godkänt",
        "ersatt",
    ]

    if any(term in status for term in negative_terms):
        adjustment -= 0.10

    weak_terms = [
        "intern informell sammanfattning",
        "intern prognos",
        "marknadsmaterial",
        "historiskt exempel",
    ]

    if any(term in status for term in weak_terms):
        adjustment -= 0.05

    return adjustment


# ---------------------------------------------------------
# Retrieval and evaluation
# ---------------------------------------------------------

for index, row in run.iterrows():

    question = str(row["question"])

    query_embedding = model.encode(
        "query: " + question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    embedding_scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0].cpu().numpy()

    final_scores = []

    for i, score in enumerate(embedding_scores):
        meta_bonus = metadata_adjustment(
            corpus.iloc[i],
            question
        )

        final_scores.append(
            float(score) + meta_bonus
        )

    ranked_indices = sorted(
        range(len(final_scores)),
        key=lambda i: final_scores[i],
        reverse=True
    )[:5]

    top_docs = (
        corpus.iloc[ranked_indices]["doc_id"]
        .astype(str)
        .tolist()
    )

    for rank, doc_id in enumerate(top_docs, start=1):
        run.at[index, f"retrieved_{rank}"] = doc_id

    gold_sources = [
        source.strip()
        for source in str(row["gold_sources"]).split("|")
        if source.strip()
    ]

    top_3 = top_docs[:3]
    top_5 = top_docs[:5]

    run.at[index, "hit_at_3"] = (
        "Yes"
        if any(source in top_3 for source in gold_sources)
        else "No"
    )

    run.at[index, "hit_at_5"] = (
        "Yes"
        if any(source in top_5 for source in gold_sources)
        else "No"
    )

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

print("Metadata-aware retrieval complete.")
print(f"Questions: {total}")
print(f"Hit@3: {hit_at_3}/{total} ({hit_at_3 / total:.1%})")
print(f"Hit@5: {hit_at_5}/{total} ({hit_at_5 / total:.1%})")
print(
    f"All Gold Sources @5: "
    f"{all_gold_at_5}/{total} ({all_gold_at_5 / total:.1%})"
)
print(f"Results saved to: {OUTPUT_FILE}")