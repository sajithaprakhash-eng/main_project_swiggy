# Swiggy Instamart Analytics, Guardrailed Triage, & RAG Refund Agent Pipeline

## Overview
This repository contains a comprehensive data analytics and automated customer support pipeline built across four major parts (**Parts A through D**). It integrates SQL-driven exploratory data analysis, machine learning customer segmentation (K-Means RFM), an LLM support ticket classifier equipped with input guardrails and PII redaction, and a retrieval-augmented generation (RAG) ReAct refund agent featuring state checkpointing, strict tool-calling safety maps, and fallback handlers.

---
## Setup and run
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run cells from top to bottom. The notebook calls the same modules used by the scripts.

## Repository map
- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers and agent tool calls.
- `src/data_pipeline.py`: audit, clean, SQLite load, validation.
- `src/customer_models.py`: RFM, K-means, PCA, lapse model with train/test separation.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, routing.
- `src/policy_rag.py`: chunking, TF-IDF retrieval and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger and bounded tool loop.
- `sql/queries.sql`: seven SQL learning patterns.
- `outputs/`: generated artifacts; `state/`: agent checkpoint examples.

## How students should learn the data-fetching flow
1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

## Reproduce
To replicate the complete analysis and run all pipelines locally, follow these steps:

1. **Clone the repository and enter the directory:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
   Install the required dependencies:

Bash
pip install -r requirements.txt
Run the unified end-to-end execution script:

Bash
python run_all.py

**Part A: SQL & Data Exploration**

Explored orders, products, and customer databases via SQLite database queries documented in sql/queries.sql.

Extracted critical operational metrics, including order status breakdowns, delivery turnaround times, and category-wise sales volumes.

**Part B: Customer Segmentation & Lapse Prediction**

RFM Clustering: Computed Recency, Frequency, and Monetary metrics for all customers relative to a 2025-10-01 snapshot date. Utilized scikit-learn's StandardScaler and KMeans (
𝐾=4
K=4) to cluster customers into actionable segments.

Segment Definitions: Identified High-Value Champions (is_premium = True) based on peak monetary contributions, alongside At-Risk Churners, Frequent Loyalists, and Low-Activity Occasionals.

Output Artifacts: Customer segment profiles are saved and tracked in outputs/customer_segments.csv.

**Part C: Support Ticket Triage with Guardrails**

Input Guardrails: Automatically filtered out empty or whitespace-only tickets (skipped_empty) and oversized tickets exceeding 400 estimated tokens (flagged_long), preventing unnecessary LLM calls and budget drain.

PII Redaction: Sanitized sensitive information using exact regular expressions for emails ([EMAIL]), phone numbers ([PHONE]), and order IDs ([ORDER_ID]). A total of 278 substitutions were executed, verifying zero unredacted traces remained.

Model Accuracy & Metrics:

Zero-shot accuracy: 85.7%

Few-shot accuracy: 100%

Validation breakdown: 135 classified_json, 12 classified_keyword, and 9 flagged_invalid_output.

Token Budget Demo: Executed a zero-shot run with TOKEN_BUDGET = 6000 which successfully triggered budget exhaustion constraints, logging results to outputs/triage_budget_demo.csv.

Production Export: Final route destinations mapped against customer segments and issue severity are stored in outputs/triage_results.csv (164 rows).

**Part D: Policy RAG Assistant and Refund Agent**

Chunking & Retrieval: Chunked 5 policy text files into 40-word segments with a 10-word overlap (15 total chunks). Indexed the corpus using scikit-learn's TfidfVectorizer and ranked chunks via cosine_similarity.

-Evaluation Summary:

Evaluated 14 questions via policy_eval.csv (12 answerable, 2 unanswerable).

Overlap=10 run achieved 12/12 expected document matches, 12/12 fact matches, and 2/2 unanswerable fallbacks.

Zero-overlap run highlighted chunk boundary splitting limitations on specific cross-boundary answers.

Scenario Execution Results:
| Scenario ID | Status | Refund Amount (INR) | Model Calls | Tokens Used |
| :--- | :--- | :--- | :--- | :--- |
| S1 | REFUND | 128.25 | 5 | 3420 |
| S2 | REFUND | 135.00 | 5 | 3510 |
| S3 | ESCALATE | 1999.00 | 5 | 3680 |
| S4 | DENIED | 0.00 | 4 | 2890 |
| S5 | DENIED | 0.00 | 3 | 2150 |
| S6 | NOT_FOUND | 0.00 | 2 | 1420 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 4850 |

Safety Demo Statuses:

Token limit demo (token_limit=200): FALLBACK_TOKEN_LIMIT

Timeout demo (time_limit_seconds=0): FALLBACK_TIMEOUT

Crash and Resume Demonstration:

Simulated a system crash for scenario S1 immediately following the second tool execution.

Successfully resumed via resume_agent("s1_crash_resume_test").

Tools executed across the crash and resume in strict non-repeated order: get_order 
→
→ search_policy 
→
→ calculate_refund 
→
→ issue_refund, yielding final status REFUND with 128.25 INR.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

Limitations and Next Steps
TF-IDF Limitations: TF-IDF relies entirely on exact keyword overlap and struggles with semantic paraphrasing or synonyms. Future iterations will incorporate dense vector embeddings (e.g., Sentence-Transformers) and specialized vector databases.

ReAct Latency: Multi-step tool invocation loops introduce higher token overhead and latency compared to structured plan-and-execute architectures. Implementing upfront step planning can eliminate redundant conversational round-trips for standard refund workflows.