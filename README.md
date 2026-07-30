# LLM Cold-Start Recommender

University of Michigan MADS Capstone project for Team Cold Start. The project explores
using language models to improve recommendations for new users and items with little or
no interaction history.

The data pipeline uses the Amazon Reviews 2023 "Books" and "Movies and TV" datasets.
It cleans and combines ratings, then creates standard and cold-start evaluation splits.

## Quick start

1. Use Python 3 and install the core dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Get the data. Either:
   - **Fast path (recommended):** download the four `data/filtered/*.parquet` files
     from the links under **Data and outputs** into `data/filtered/`, then skip to
     step 4.
   - **From raw:** place the raw files at the `data/raw/...` paths listed under
     **Data and outputs**, then run step 3.
3. Run `notebooks/data_filtering.ipynb`.  
4. Run `notebooks/hyperparameter_tuning.ipynb`. Skippable — `outputs/hyperparams.json` is committed.
5. Run `notebooks/steel_thread.ipynb` It reads `outputs/hyperparams.json`; if that file is missing it falls back to untuned `lambda=1`.

The notebooks use project-relative paths and can be run from VS Code or Jupyter.

## Pipeline

```text
data/raw/*.jsonl.gz
    -> data_filtering.ipynb
data/filtered/*.parquet
    -> recsys/load.py :: load_dataset()
Dataset  (interaction matrix + warm / cold-val / cold-test splits, built in memory)
```

## Project structure

```text
.
|-- data/raw/                           # local raw data (not committed)
|-- data/filtered/                      # local filtered data (not committed)
|-- data/cache/                         # local cache data (not committed)
|-- design_documents/
|   |-- README.md                       # index: which doc answers which question
|   |-- 01-getting-started.md           # zero -> running evaluation; what each notebook writes
|   |-- 02-module-dependencies.md       # import graph; what breaks if you edit a file
|   |-- 03-adding-a-model.md            # the RetrievalModel contract and its implementors
|   |-- 04-cbhcf-score-composition.md   # ALS + content -> AdditiveItemBlock; where lambda comes from
|   |-- 05-what-metrics-mean.md         # Mode A/B control flow, ceilings, floors, pools
|   |-- 06-provider-equity.md           # equity_metrics.py as an integration spec (not yet wired in)
|   `-- initial_pipeline_design/        # RETIRED MovieLens steel thread; design record only
|-- notebooks/
|   |-- data_filtering.ipynb            # filtering reviews datasets and metadata as well
|   |-- hyperparameter_tuning.ipynb     # selects CBHCF's lambda on the cold-item VALIDATION set
|   `-- steel_thread.ipynb              # how-to for recsys modules
|-- outputs/
|   `-- hyperparams.json                # Tuned lambda for CBHCF model - output of `hyperparameter_tuning.ipynb`
|-- recsys/
|   |-- __init__.py                     # pins the BLAS thread pool to 1 (implicit does its own parallelism)
|   |-- load.py                         # Amazon Books loader; the 80/10/10 warm / cold-val / cold-test split (of items with ≥ min_interactions=25)
|   |-- protocol.py                     # `RetrievalModel` -- the fit/recommend/fold_in contract `eval.py` relies on
|   |-- pop.py                          # popularity and activity baseline models (the floors)
|   |-- cf.py                           # ALS collaborative filtering baseline
|   |-- content.py                      # item content vectors: role-based fields, BM25F weighted blocks
|   |-- cbhcf.py                        # content-based hybrid CF -- ALS score + weighted content score
|   |-- scores.py                       # score sources: the interface `gpu_retrieval` consumes instead of factors
|   |-- gpu_retrieval.py                # exact GPU top-K, candidate-pool AUC, and the Mode B duals
|   |-- eval.py                         # evaluation harness: metrics, within-item ceiling, both warm-up sweeps
|   `-- equity_metrics.py               # provider-side fairness metrics — Gini, catalog share, equity ratio
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Data and outputs

| File | Purpose | Link |
|---|---|---|
| `data/raw/books/Books.jsonl.gz` | Book reviews & ratings | https://amazon-reviews-2023.github.io |
| `data/raw/books/meta_Books.jsonl.gz` | Movie and TV metadata | https://amazon-reviews-2023.github.io |
| `data/raw/movies/Movies_and_TV.jsonl.gz` | Movie and TV reviews & ratings | https://amazon-reviews-2023.github.io |
| `data/raw/movies/meta_Movies_and_TV.jsonl.gz` | Movie and TV metadata | https://amazon-reviews-2023.github.io |
| `data/raw/5-core/Books.csv.gz` | Book ratings | https://amazon-reviews-2023.github.io/data_processing/5core.html |
| `data/raw/5-core/Movies_and_TV.csv.gz` | Movie and TV ratings | https://amazon-reviews-2023.github.io/data_processing/5core.html |
| `data/filtered/books_meta_5core_common.parquet` | Book metadata for filtered subset | https://drive.google.com/file/d/1_xSLj0kM2yeNt4FX7aAHcvY19fFaOnaR/view?usp=sharing |
| `data/filtered/books_5core_common.parquet` | Book reviews & ratings for filtered subset | https://drive.google.com/file/d/1YBsJEZDu1QlFG8WJxFJadgnSoJBzoewZ/view?usp=sharing |
| `data/filtered/movies_meta_5core_common.parquet` | Movies metadata for filtered subset | https://drive.google.com/file/d/1iDKH3So891ohCIk977BrUOW_hpSbktQA/view?usp=sharing |
| `data/filtered/movies_5core_common.parquet` | Movies reviews & ratings for filtered subset | https://drive.google.com/file/d/1kvN5UuoBenbZFNuUq387WZSOgDfXpLQv/view?usp=sharing |
| `data/cache/common_user_ids.pkl` | Intersection of `user_id`s from Books & Movies datasets - output of `data_filtering.ipynb` shared to save recompute | https://drive.google.com/file/d/1K-emjnRk_G3AjZ9RhB17ksI4EKZ5XMxr/view?usp=sharing | 
| `outputs/hyperparams.json` | Tuned lambda for CBHCF model | Generated by step 4 |

Raw and filtered data are excluded from Git because they are large. The fast path in
**Quick start**  downloads the filtered parquet files directly; the raw files are only 
needed to re-run `data_filtering.ipynb` yourself.

No sample data is currently committed. A fresh sample derived from
`data/filtered/*.parquet` is planned.

## design_documents folder
Two things live here: **current component documentation** for `recsys/`, and a **retired
design record** from the project's first architecture.

### Component documentation (current)

Six Mermaid diagrams, each named for the question it answers rather than the component it
covers. Every node is a real symbol and every table row points at `file:line`, so a diagram
can be checked against the code rather than trusted. Start at
[`README.md`](design_documents/README.md), which indexes them.

| Doc | Question it answers |
|---|---|
| [01-getting-started.md](design_documents/01-getting-started.md) | How do I get from zero to a running evaluation? |
| [02-module-dependencies.md](design_documents/02-module-dependencies.md) | What breaks if I edit this file? |
| [03-adding-a-model.md](design_documents/03-adding-a-model.md) | How do I add a new model? |
| [04-cbhcf-score-composition.md](design_documents/04-cbhcf-score-composition.md) | How does CBHCF combine ALS and content? |
| [05-what-metrics-mean.md](design_documents/05-what-metrics-mean.md) | What does this accuracy number mean? |
| [06-provider-equity.md](design_documents/06-provider-equity.md) | Who gets exposure, and how do we turn it on? |

Docs 05 and 06 each end with a **Drift** section (places where prose contradicts the code,
reported but not fixed) and an **Open questions** section (things not determinable from
source). Read `06` before touching `recsys/equity_metrics.py` — it documents a blocking
id-format mismatch that fails silently.

### initial_pipeline_design
This folder contains:
1. `build_solution_architecture.py`
    - .py file to generate solution architecture diagram `solution_architecture_generated.html` 
2. `diagram_lib.py`
    - Helper functions py to generate `solution_architecture_generated.html` 
3. `solution_architecture_generated.html`
    - Solution architecture diagram generated by `build_solution_architecture.py`

These were used during the initial design of the project to establish extreme baseline
cases and validate the initial steel thread architecture.

Note: the generated diagram describes the original MovieLens steel thread
(`u.data`, 943 users / 1,682 items), not the current Amazon Reviews pipeline.
It is preserved as a design record, not as current documentation.

## Notes

- The full dataset contains about 17 million interactions and requires substantial disk
  space and processing time.
