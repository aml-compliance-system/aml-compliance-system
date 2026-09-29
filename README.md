# FINDETECT

**AI-powered Anti-Money Laundering (AML) detection pipeline** that combines multiple machine learning models, customer behavior analysis, transaction graph context, and an LLM-generated explanation layer to score financial transactions and produce audit-ready risk reports.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [How It Works](#how-it-works)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Evaluation Results](#evaluation-results)
9. [Configuration](#configuration)
10. [Data & Privacy](#data--privacy)
11. [Limitations & Disclaimer](#limitations--disclaimer)
12. [Contributing](#contributing)
13. [License](#license)
14. [Contact](#contact)

---

## Overview

Financial institutions and regulators must identify suspicious activity across very large volumes of transactions. Rules-based systems generate many false positives, and single-model approaches miss patterns that only appear when several signals are combined.

FINDETECT addresses this by running every transaction batch through a multi-stage pipeline:

- Several independent detection models look at the data from different angles (supervised, unsupervised, and behavioral).
- A **meta-learner** fuses their outputs into one risk score per account per day. It is a ranking score, not a calibrated probability (see [Limitations](#limitations--disclaimer)).
- A **risk agent** and an **LLM** turn the score into a human-readable explanation.
- A final **evaluation** step measures how well the pipeline performed.

The pipeline is orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph), which makes each stage explicit, testable, and easy to extend.

**Intended audience:** compliance analysts, financial institutions, government and regulatory bodies, and developers/students building on the project.

---

## Key Features

- **Multi-model detection**: XGBoost (supervised), anomaly detection (unsupervised), and behavioral modeling run in parallel.
- **Customer deviation analysis**: compares each transaction against the customer's own historical behavior.
- **Graph context extraction**: adds relationship and network features between accounts.
- **Meta-learner fusion**: learns how to weight the individual models instead of relying on fixed rules; includes fusion-weight optimization.
- **Explainable output**: a locally hosted LLM (via Ollama) generates a plain-language rationale for each risk decision, keeping data on-premises.
- **Automated reporting and evaluation**: produces reports and performance metrics at the end of each run.
- **Threshold analysis**: risk scores can be exported and analyzed to choose operating thresholds.
- **Batch processing**: run a single day or all days in a dataset.
- **Web dashboard**: analysts run the pipeline from the browser, review flagged accounts, and chat with a local AI investigator about each case (FastAPI backend + HTML frontend).
- **Local analytical storage** using DuckDB and Parquet for fast processing without a separate database server.

---

## How It Works

The pipeline is defined in `LangGraph.py` as a directed graph with a parallel fan-out and fan-in.

```mermaid
flowchart TD
    A[Preprocess] --> B[Customer Deviation]
    B --> C[XGBoost Model]
    B --> D[Anomaly Model]
    B --> E[Behavioral Model]
    C --> F[Graph Extraction]
    D --> F
    E --> F
    F --> G[Meta-Learner]
    G --> H[Risk Agent]
    H --> I[LLM API]
    I --> J[Report]
    J --> K[Evaluation]
```

| Stage | Purpose |
|---|---|
| **Preprocess** | Loads the prepared feature dataset (Parquet) for the selected day. |
| **Customer Deviation** | Measures how far each transaction deviates from the customer's own historical behavior. |
| **XGBoost model** | Supervised model that outputs a fraud probability per transaction. |
| **Autoencoder anomaly model** | Unsupervised neural network that scores how unusual a transaction is by its reconstruction error. |
| **Behavioral risk model** | Rule-based scoring of amount drift, frequency drift, peer-group outliers (robust MAD scaling) and short-term velocity. |
| **Graph Extraction** | Builds a daily transaction graph and adds context: exposure to high-risk accounts up to **3 hops**, **bidirectional flows** (layering indicator) and **fan-in / fan-out** patterns (money-mule indicator). Very high-degree accounts such as merchants are shielded from false seeding. |
| **Meta-Learner** | A trained stacking model that fuses the three model scores, graph features and customer-deviation features into one risk score per account per day. If the trained meta-learner is not found, the pipeline falls back to the weighted-average risk score. |
| **Risk Agent** | Applies the investigation threshold and signal filters, then sorts flagged accounts into `TIER_1_HIGH`, `TIER_2_MEDIUM` and `TIER_3_LOW` queues. |
| **LLM API** | Builds a structured case for the highest-risk accounts (top 25 per run) and drafts a Suspicious Activity Report (SAR) using a local LLM served by Ollama. The default limit is 25 reports per run because of local hardware limits; it can be raised on stronger hardware (`MAX_REPORTS` in `nodes.py`). |
| **Report** | Assembles the final results. |
| **Evaluation** | If ground-truth labels (`Is Laundering`) are present, computes precision, recall and F1 for each model and the Risk Agent. |

The three detection models run **in parallel**.

Shared data moves between stages through a custom state object (`AMLPipelineState`), which uses a `merge_dicts` reducer so that parallel model branches can safely write their results without overwriting each other.

---

## Project Structure

```
FINDETECT/
├── frontend/                        # Web dashboard (HTML): case list and AI investigation chat
├── models/                          # Trained models (incl. autoencoder_final64/ and meta_learner_calibrated.pkl)
├── Data/processed/                  # 64 input Parquet files: block_d_numerical_bucket_000 to 063 (not committed to Git)
├── outputs/                         # Pipeline results, one folder per run (generated)
├── pre_model/                       # Pre-trained / preliminary model artifacts
├── threshold_report/                # Threshold analysis outputs
├── AMLPipelineState.py              # Shared pipeline state (with merge_dicts reducer)
├── config.py                        # Central configuration
├── customer_deviation.py            # Customer behavior deviation logic
├── LangGraph.py                     # Pipeline graph definition and compilation
├── LLM.py                           # LLM integration for explanations
├── main.py                          # FastAPI backend for the web dashboard
├── make_sample.py                   # Generates sample data
├── meta_learner.py                  # Meta-learner model
├── nodes.py                         # Node implementations for each pipeline stage
├── optimize_fusion_weights.py       # Optimizes weights used to fuse model outputs
├── train_meta_learner_fixed.py      # Trains the meta-learner
├── run_full_pipeline.py             # Command-line runner (--input file and/or --date YYYY-MM-DD)
├── run_all_days.py                  # Runs the pipeline across multiple days
├── pick_threshold.py                # Picks the investigation threshold for a chosen recall
├── pick_tiers.py                    # Derives investigation, watchlist and tier cutoffs from a run
├── eval_days.parquet                # Account/day results of the last evaluated run (generated)
├── risk_scores_for_threshold_analysis.parquet  # Exported scores for threshold analysis
├── data.duckdb                      # Local DuckDB database
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.10 or higher *(update if you require a different version)*
- The packages listed in `requirements.txt`, including LangGraph, XGBoost, TensorFlow, DuckDB, pandas, FastAPI, Uvicorn and the `ollama` Python client
- [Ollama](https://ollama.com) installed and running locally, with a model pulled for the explanation step *(see [Configuration](#configuration))*

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/aml-compliance-system/aml-compliance-system.git
cd aml-compliance-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama (https://ollama.com) and pull the model used for explanations
ollama pull qwen2.5

dataset: [https://drive.google.com/drive/u/0/folders/15Jb5T64Ua99DMZWRbz8ptRSEMY2KBLKQ]
# pip install gdown if you want directly download file from drive
# gdown --folder 15Jb5T64Ua99DMZWRbz8ptRSEMY2KBLKQ
```


---

## Usage

### 1. Prepare the input data and models

The pipeline expects:

- **Feature dataset:** 64 Parquet files named `block_d_numerical_bucket_*.parquet` in `Data/processed/` (or another folder set with `FINDETECT_FINAL_DATA_DIR`).
- **Autoencoder package** in `models/autoencoder_final64/` containing:
  - `autoencoder_final64_fixed.keras`
  - `preprocessing_and_scoring_final64.joblib`
  - `metadata_final64.json`

The script stops with a clear error message if either is missing.

### 2. Start Ollama

Make sure the Ollama service is running and the model has been pulled (`ollama pull qwen2.5`).

### 3. Run the pipeline from the command line

Use `--input` to analyze a single Parquet file, or `--date` to analyze one day from the 64-file dataset. You can also combine them to analyze one day of a single file.

```bash
# Whole file (no date needed)
python run_full_pipeline.py --input "Data/processed/prod_sample_oct23_25.parquet"

# One day of the full 64-file dataset
python run_full_pipeline.py --date 2022-09-01

# One day of a single file
python run_full_pipeline.py --input "Data/processed/prod_sample_oct23_25.parquet" --date 2022-10-24
```

`--date` must be a real date in `YYYY-MM-DD` format that exists in your data. The script prepares the input with DuckDB, runs the full LangGraph pipeline, and saves the results.

### 4. Run across multiple days

```bash
python run_all_days.py
```

### 5. Optional utilities

```bash
python make_sample.py               # generate sample data
python optimize_fusion_weights.py   # optimize fusion weights
```

### 6. Train the meta-learner

The meta-learner is trained on **account/day** rows built with the same aggregation used at inference (maximum of each model score per account per day, plus graph and customer-deviation features).

```bash
python train_meta_learner_fixed.py \
    --train-window <START>:<END> \
    --input-glob "Data/processed/block_d_numerical_bucket_*.parquet" \
    --neg-sample-rate 0.5
```

| Argument | Meaning |
|---|---|
| `--train-window` | Date window to train on (`YYYY-MM-DD:YYYY-MM-DD`, end date included). It must **not overlap** the window you evaluate on. |
| `--neg-sample-rate` | Fraction of non-suspicious transactions kept, to fit in memory. All suspicious transactions are always kept. The reported results use `0.5`. `1.0` disables sampling. |

Things to know:

- The trained model is saved to `models/meta_learner_calibrated.pkl`, with its sampling rate in a `.json` file next to it. **Back up the file before retraining**, because a new run overwrites it.
- The graph features (`hop1_count`, `bidir_count`, and so on) are computed on the sampled transactions, so a lower sampling rate gives a thinner graph in training than at inference. Higher rates match inference better but use more memory.
- The graph features depend on the seed threshold and the fusion weights in `config.py`. **Retrain the meta-learner whenever you change either.**
- The scores are not probabilities. Sampling and calibration shift their scale, so choose thresholds from data (next section), not from a fixed number such as 0.5.

### 7. Choose the alert thresholds

1. Run the pipeline on a **tuning window**. The evaluation step writes `eval_days.parquet` (one row per account/day with its score and the true label).
2. Run `python pick_threshold.py` to see the precision you get at each recall level, or `python pick_tiers.py --recall 0.75` to derive the investigation, watchlist and tier cutoffs together.
3. Put the chosen values in `config.py`.
4. Run the pipeline on a **different window** without changing anything and report those numbers.

Choosing the thresholds and reporting results on the same window makes the results look better than they will be on new data.

### 8. Use the pipeline programmatically

```python
from LangGraph import app

final_state = app.invoke({"final_parquet_path": "path/to/daily_file.parquet"})
risk_scores = final_state["risk_scores"]   # pandas DataFrame, one row per account/day
```

### 9. Run the web dashboard

The dashboard lets an analyst run the pipeline, review flagged accounts, and chat with an AI investigator about each case. It has two parts that run in two separate terminals, both started from the project folder.

**Terminal 1: the API (port 8000)**

```bash
uvicorn main:app --reload
```

`--reload` restarts the server when Python files change, which is convenient during development. For a deployment, run without it.

**Terminal 2: the frontend (port 5600)**

```bash
cd frontend
python -m http.server 5600
```

Then open `http://127.0.0.1:5600/`, choose the dashboard page, and click **Run Risk Agent**. Make sure Ollama is running first, since it writes the SAR drafts and answers the chat.

The dashboard analyzes `Data/processed/prod_sample_oct23_25.parquet` by default. The file is set in the dashboard page (`final_parquet_path`) and can be changed to any Parquet file with the same feature columns.

Web runs use the same preparation and output code as the command line, so the results and saved files (`risk_scores.parquet`, `sar_reports.jsonl`, `summary.json`) are identical.

| Endpoint | Purpose |
|---|---|
| `POST /pipeline/analyze` | Runs the pipeline. Body: `{"final_parquet_path": "...", "date": "YYYY-MM-DD"}` (`date` is optional). |
| `POST /chat` | Asks the local LLM questions about a flagged case. |
| `GET /case/{case_id}` | Returns the stored details and SAR draft for a flagged account. |

**Important notes**

- **Serve the frontend with `python -m http.server`, not VS Code Live Server.** Live Server (port 5500) reloads the page whenever a file in the project changes. Each run writes files to `outputs/` and `.tmp/`, so the page would reload and the cards would disappear.
- **Pick a free port.** Port 5600 is used in these instructions, but any free port works. Avoid 5500 (Live Server) and 8080 (often used by other software such as PostgreSQL).
- If `uvicorn` is not installed, run `pip install fastapi uvicorn`.

### Output

Results are written to `outputs/FINDETECT_Pipeline_Output/<run>/`, where `<run>` is the date, or the input file name when no date is given (web runs add a timestamp):

| File | Content |
|---|---|
| `risk_scores.parquet` | Risk score, risk label, investigation flag and tier for every account on that day. |
| `sar_reports.jsonl` | LLM-drafted SAR reports (one JSON record per line) for the highest-risk cases. |
| `summary.json` | Run summary: number of account/days, investigation cases, high-risk seeds and directly exposed receivers. |

---

## Evaluation Results

FINDETECT is evaluated at two levels: each detection model is scored per **transaction**, and the final Risk Agent is scored per **account/day**, which is the level at which a compliance analyst actually reviews alerts. Because the grains differ, the two tables should not be compared row by row.

Suspicious activity is extremely rare in this data (about 0.18% of transactions and 0.53% of account/days), so accuracy is not a useful metric. Precision, recall and F1 on the suspicious class are reported instead.

**Evaluation data:** `Data/processed/prod_sample_oct23_25.parquet`, 2,761,250 transactions from 23 to 25 October 2022 (4,833 suspicious transactions). At account/day level that is 789,075 account/days, of which 4,184 are suspicious. The meta-learner was trained on an earlier window that does not overlap this one.

### Individual models (transaction level)

Decision thresholds used for evaluation: XGBoost probability >= 0.50, autoencoder anomaly score >= 0.98, behavioral score >= 0.40. They were chosen from threshold sweeps (printed by the evaluation step) on this same data, so these three rows are somewhat optimistic.

| Model | Precision | Recall | F1-Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.1234 | 0.0368 | 0.0567 | 178 | 1,264 | 4,655 | 2,755,153 |
| Autoencoder (anomaly) | 0.0093 | 0.1202 | 0.0173 | 581 | 61,674 | 4,252 | 2,694,743 |
| Behavioral Risk | 0.0509 | 0.1386 | 0.0744 | 670 | 12,503 | 4,163 | 2,743,914 |

The models behave very differently. XGBoost is the most precise at high scores (about 77% precision at a score of 0.75, with very few cases caught), the behavioral model is moderately precise and catches more, and the autoencoder is only slightly better than random (about 5x the base rate at its strictest cutoff). No single model is sufficient on its own, which is the reason for the fusion step.

### Final Risk Agent (account/day level)

| Metric | Value |
|---|---|
| Precision | 0.1165 (11.7%) |
| Recall | 0.7517 (75.2%) |
| F1-Score | 0.2017 |
| True Positives | 3,145 |
| False Positives | 23,849 |
| False Negatives | 1,039 |
| True Negatives | 761,042 |

Of 789,075 account/days, the Risk Agent flagged 26,994 (about 3.4%) and caught 3,145 of the 4,184 truly suspicious ones. That is roughly 22 times more precise than reviewing account/days at random (base rate 0.53%), while recovering about three quarters of the suspicious activity.

### Choosing an operating point

The numbers above are a **high-recall** setting (about 75% recall). Raising the investigation threshold sends fewer accounts to review and makes each alert more likely to be real, but misses more suspicious accounts. Lowering it does the reverse. Across the configurations tested, precision at a fixed recall level was similar, so the threshold is mainly a choice about review capacity: how many alerts per day your analysts can work through. Use `pick_threshold.py` and `pick_tiers.py` to see the trade-off on your data.

### Reading these results

- The combined Risk Agent reaches far higher recall (75.2%) than any individual model (at most 14% at the thresholds above), which shows the benefit of fusing models and graph context.
- Precision is still low in absolute terms (11.7%), so most alerts are false positives and need human review.
- The ceiling is set by the base models. If none of them scores a suspicious account highly, the fusion step cannot recover it. Better input features would help more than further threshold changes.
- These figures come from one dataset and one configuration, and the thresholds were tuned on the same window. Re-evaluate on a separate window and on your own data before operational use.

---

## Configuration

Thresholds, model paths, fusion weights and penalties are defined in `config.py`. Paths can also be overridden with environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `FINDETECT_FINAL_DATA_DIR` | Folder with the 64 Parquet feature files | `Data/processed` |
| `AUTOENCODER_MODEL_DIR` | Folder with the autoencoder package | `models/autoencoder_final64` |
| `FINDETECT_OUTPUT_DIR` | Where results are saved | `outputs/FINDETECT_Pipeline_Output` |
| `FINDETECT_TEMP_DIR` | Temporary daily Parquet files | `.tmp/findetect_daily` |

Thresholds and weights that shape the results:

| Variable | Purpose |
|---|---|
| `FINDETECT_XGB_WEIGHT`, `FINDETECT_ANOMALY_WEIGHT`, `FINDETECT_BEHAVIORAL_WEIGHT` | Weights that combine the three model scores into `base_risk` (must sum to 1.0). |
| `FINDETECT_SEED_THRESHOLD` | `base_risk` above which an account becomes a high-risk seed for graph exposure. |
| `FINDETECT_INVESTIGATION_THRESHOLD` | Meta-learner score at or above which an account is sent for investigation. |
| `FINDETECT_MEDIUM_RISK_THRESHOLD` | Score at or above which an unflagged account is labelled Medium Risk (watchlist). |
| `FINDETECT_TIER_1_THRESHOLD`, `FINDETECT_TIER_2_THRESHOLD` | Score cutoffs for `TIER_1_HIGH` and `TIER_2_MEDIUM`. Everything else that is flagged is `TIER_3_LOW`. |

The cutoffs must satisfy `MEDIUM <= INVESTIGATION <= TIER_2 <= TIER_1`, otherwise a tier ends up empty. Changing the fusion weights or the seed threshold changes the graph features, so retrain the meta-learner afterwards (see [Train the meta-learner](#6-train-the-meta-learner)).

Example (Windows PowerShell):

```powershell
$env:FINDETECT_FINAL_DATA_DIR = "D:\data\processed"
python run_full_pipeline.py --date 2022-09-01
```

The Ollama model is `qwen2.5` by default, and Ollama listens on `http://localhost:11434`. Because the LLM runs locally, no API key is required. If you add other secrets later, keep them in environment variables or a local `.env` file excluded through `.gitignore`.

---

## Data & Privacy

Financial data is highly sensitive. When deploying FINDETECT:

- Run it inside your organization's secure environment; data is stored locally in DuckDB/Parquet files.
- The LLM explanation step runs **locally through [Ollama](https://ollama.com)**, so transaction details are not sent to an external API or third-party service. This makes FINDETECT suitable for environments with strict data-residency requirements.
- The dashboard API is intended for local use. It accepts requests from any origin and any file path under your control; before exposing it on a network, restrict the allowed origins and input folders, and add authentication.
- Do not commit real customer data to the repository. Use `make_sample.py` for demos and testing.
- Follow applicable regulations for your jurisdiction (for example GDPR or local AML/KYC requirements).

---

## Limitations & Disclaimer

- FINDETECT is a **decision-support tool**. It does not replace human compliance analysts or make final determinations of illicit activity.
- Model performance depends on data quality and how closely the data matches the training data. Results should be validated on your own data before operational use.
- SAR drafts are generated for the top 25 highest-risk cases per run by default, a limit chosen for local hardware. Production deployments with more compute can raise it so that every flagged case gets a draft. All flagged cases are still saved in `risk_scores.parquet`.
- LLM-generated explanations are meant to assist review and may contain inaccuracies; they should not be treated as evidence.
- The risk score is a ranking score. It is not a calibrated probability of laundering, so do not read a score of 0.4 as a 40% chance.
- Thresholds should be tuned and approved by your compliance team, on a window that is separate from the one used to report results.
- Results were measured on one three-day window of one dataset. Performance on other periods may differ.

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes with clear messages
4. Push the branch and open a Pull Request

Please keep changes focused, and include tests or sample results where possible.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

**Team:** FINDETECT team SDA
**Email:**
- [jawadaljabbar@gmail.com](mailto:jawadaljabbar@gmail.com)
- [shooq.otibii@gmail.com](mailto:shooq.otibii@gmail.com)
- [Dalalalsolami2d@gmail.com](mailto:Dalalalsolami2d@gmail.com)
- [shahad.a.aldossarii@gmail.com ](shahad.a.aldossarii@gmail.com )

