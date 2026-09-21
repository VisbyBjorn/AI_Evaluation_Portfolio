from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_FILE = PROJECT_ROOT / "data" / "Evalcase03_RetrievalCorpus_v1.0.xlsx"
RESULTS_FILE = PROJECT_ROOT / "results" / "Evalcase03_RetrievalRun01_results.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "examples" / "Run01_Prompts"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

corpus = pd.read_excel(CORPUS_FILE)
results = pd.read_excel(RESULTS_FILE)

corpus_lookup = corpus.set_index("doc_id").to_dict("index")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Create isolated prompts
# ---------------------------------------------------------

for _, row in results.iterrows():

    test_id = str(row["test_id"])
    question = str(row["question"])

    retrieved_ids = [
        str(row[f"retrieved_{i}"])
        for i in range(1, 6)
        if pd.notna(row[f"retrieved_{i}"])
    ]

    parts = []

    parts.append(
        "INSTRUCTION\n"
        "Answer the question using only the documents below.\n"
        "Do not use external knowledge and do not assume anything "
        "that is not supported by the documents.\n"
        "If the available evidence is insufficient for a confident "
        "answer, state that clearly.\n"
        "Prioritize current, customer-specific and authoritative "
        "sources over older, informal or preliminary documents.\n"
    )

    parts.append(
        f"\nQUESTION\n{question}\n"
    )

    parts.append(
        "\nDOCUMENTS\n"
    )

    for doc_id in retrieved_ids:

        doc = corpus_lookup.get(doc_id)

        if doc is None:
            parts.append(
                f"\n[{doc_id}]\n"
                "Document not found in corpus.\n"
            )
            continue

        customer = (
            doc["customer"]
            if pd.notna(doc["customer"])
            else ""
        )

        parts.append(
            f"\n[{doc_id}]\n"
            f"Title: {doc['title']}\n"
            f"Date: {doc['date']}\n"
            f"Status: {doc['status']}\n"
            f"Customer: {customer}\n"
            f"Content: {doc['content']}\n"
        )

    prompt_text = "\n".join(parts)

    output_file = OUTPUT_DIR / f"{test_id}.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(prompt_text)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("Prompt generation complete.")
print(f"Prompts created: {len(results)}")
print(f"Output directory: {OUTPUT_DIR}")