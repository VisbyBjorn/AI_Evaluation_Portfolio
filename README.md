# AI Evaluation Portfolio

A small reproducible evaluation project focused on retrieval quality, source authority, temporal context, and document-grounded AI answers.

## Project overview

This repository contains a controlled portfolio case built around a synthetic SaaS documentation environment.

The project evaluates how different retrieval approaches perform when relevant information is distributed across current policies, customer-specific agreements, drafts, internal notes, and outdated documents.

The main evaluation questions include:

- Does the system retrieve the right documents?
- Does it retrieve all decision-critical sources?
- Does it confuse current and outdated information?
- Does it use the correct source authority?
- Does it mix information across customers or contexts?
- Does it abstain when the available evidence is insufficient?

## Dataset

The evaluation corpus contains:

- 18 primary source documents
- 32 distractor documents
- 50 documents in total
- 15 retrieval evaluation questions
- a larger Golden Set of 45 questions

The dataset is synthetic and was created specifically for evaluation purposes.

## Retrieval methods tested

Four retrieval variants were compared:

1. TF-IDF
2. Multilingual embeddings
3. Embeddings with simple metadata adjustments
4. Embeddings with CrossEncoder reranking

## Results

| Method | Hit@3 | Hit@5 | All Gold Sources @5 |
|---|---:|---:|---:|
| TF-IDF | 80.0% | 86.7% | 46.7% |
| Embeddings | 93.3% | 93.3% | 53.3% |
| Embeddings + metadata | 66.7% | 80.0% | 40.0% |
| Embeddings + reranking | 86.7% | 93.3% | 46.7% |

The embedding-based baseline achieved the strongest overall retrieval performance on this test set.

More complex retrieval variants did not automatically improve performance.

Further optimization on the same evaluation set was intentionally stopped to reduce the risk of overfitting.

## Answer evaluation

For the TF-IDF retrieval run, each question was evaluated in an isolated context containing only the five retrieved documents.

Results:

- PASS: 15/15
- PARTIAL: 0
- FAIL: 0
- Critical failures: 0

A PASS did not require complete retrieval. The answer was evaluated based on whether the model used the available evidence correctly and abstained when the evidence was insufficient.

## Metrics

The project uses:

- Hit@3
- Hit@5
- All Gold Sources @5
- Missing Gold Sources
- PASS / PARTIAL / FAIL
- Critical failure tracking

All Gold Sources @5 is used as a stricter retrieval metric because finding one relevant document is not always sufficient for questions requiring multiple sources.

## Repository structure

```text
AI_Evaluation_Portfolio
│
├── data
├── scripts
├── results
├── docs
├── examples
├── README.md
└── requirements.txt
```

## Tools

The project uses:

* Python
* pandas
* scikit-learn
* TF-IDF
* cosine similarity
* SentenceTransformers
* multilingual embeddings
* CrossEncoder reranking

## Limitations

This is a controlled portfolio evaluation, not a production RAG benchmark.

The corpus is small and manually constructed.

The project does not currently include:

* production vector databases
* document chunking
* access control
* large-scale retrieval
* production observability
* automated LLM API evaluation pipelines

Results should therefore be interpreted only within this evaluation setup.

## Purpose

The purpose of this project is to demonstrate a practical and structured approach to AI quality evaluation, including retrieval measurement, source analysis, error analysis, and reproducible experimentation.
