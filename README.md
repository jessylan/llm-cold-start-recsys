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

2. Place the raw files in the locations shown under **Project structure**.
3. Run `notebooks/data_filtering.ipynb`. 
4. Run `notebooks/hyperparameter_tuning.ipynb`.
5. Run `notebooks/baseline_steel_thread_cf.ipynb` after step `4` completes, or this will fallback to untuned `lambda=1`.


The notebooks use project-relative paths and can be run from VS Code or Jupyter.

## Pipeline

```text
Raw Amazon data
    -> cleaning and standardization
    -> unified interactions
    -> train, validation, test, and cold-start splits
    -> modeling dataset
```

`load.csv` and `load_sample.csv` include:

- `split`: `train`, `validation`, `test`, `cold_start_user_test`, or
  `cold_start_item_test`
- `is_sparse_user_test`: identifies test interactions for users with only 1-5 training
  interactions

## Project structure

```text
.
|-- data/raw/                           # local raw data (not committed)
|-- data/filtered/                      # local filtered data (not committed)
|-- data/cache/                         # local cache data (not committed)
|-- design_documents/
|   `-- initial_pipeline_design/        # baseline and architecture artifacts
|-- notebooks/
|   |-- baseline_steel_thread_cf.ipynb  # how-to for recsys modules
|   |-- hyperparameter_tuning.ipynb     # selects CBHCF's lambda on the cold-item VALIDATION set
|   |-- data_cleaning.ipynb
|   |-- data_filtering.ipynb            # filtering reviews datasets and metadata as well
|   `-- train_test_split.ipynb
|-- outputs/
|   |-- clean_sample.csv
|   |-- load_sample.csv
|   `-- metadata_sample.csv
|-- recsys/
|   |-- __init__.py                     # pins the BLAS thread pool to 1 (implicit does its own parallelism)
|   |-- load.py                         # Amazon Books loader; the 80/10/10 warm / cold-val / cold-test split
|   |-- protocol.py                     # `RetrievalModel` -- the fit/recommend/fold_in contract `eval.py` relies on
|   |-- pop.py                          # popularity and activity baseline models (the floors)
|   |-- cf.py                           # ALS collaborative filtering baseline
|   |-- content.py                      # item content vectors: role-based fields, BM25F weighted blocks
|   |-- cbhcf.py                        # content-based hybrid CF -- ALS score + weighted content score
|   |-- scores.py                       # score sources: the interface `gpu_retrieval` consumes instead of factors
|   |-- gpu_retrieval.py                # exact GPU top-K, candidate-pool AUC, and the Mode B duals
|   `-- eval.py                         # evaluation harness: metrics, within-item ceiling, both warm-up sweeps
|-- .gitignore
`-- README.md
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
| `data/cache/common_user_ids.pkl` | Intersection of `user_id`s from Books & Movies datasets | https://drive.google.com/file/d/1K-emjnRk_G3AjZ9RhB17ksI4EKZ5XMxr/view?usp=sharing | 
| `clean.csv` | Full cleaned interactions (local/shared storage) | *n/a* |
| `load.csv` | Full modeling dataset (local/shared storage) | *n/a* |
| `metadata.csv` | Full unified item table: movie metadata + book ID stubs (local/shared storage) | *n/a* |
| `clean_sample.csv` | Git-friendly sample of cleaned interactions | *n/a* |
| `load_sample.csv` | Git-friendly sample with split labels | *n/a* |
| `metadata_sample.csv` | Git-friendly sample of the item table (see limitation below) | *n/a* |

Raw data and full generated outputs are excluded from Git because they are large. The
sample CSVs are committed so collaborators can test the workflow and inspect the
schema immediately. `clean_sample.csv`/`load_sample.csv` contain 100,000 rows: 10,000
movie and 10,000 book rows from each standard and cold-start split. `metadata_sample.csv`
is a 100,000-row random sample of `metadata.csv`.

## design_documents folder
### initial_pipeline_design
This folder contains:
1. `build_solution_architecture.py`
    - .py file to generate solution architecture diagram `solution_architecture_generated.html` 
2. `diagram_lib.py`
    - Helper functions py to generate `solution_architecture_generated.html` 
3. `requirements.txt` 
    - Requirements file
4. `solution_architecture_generated.html`
    - Solution architecture diagram generated by `build_solution_architecture.py`

These are files used during the initial design of the project to establish extreme baseline cases and validate the steel thread architecture. 

## Notes

- The full dataset contains about 17 million interactions and requires substantial disk
  space and processing time.
- Confirm notebook column mappings before running the pipeline with a different schema.
- Keep the split labels unchanged so evaluation remains consistent across the project.
- **Notebook/output naming is out of sync.** As committed, `data_cleaning.ipynb` saves
  `processed_items.csv` / `processed_interactions.csv`, and `train_test_split.ipynb`
  saves `train_interactions.csv`, `validation_interactions.csv`, etc. — not the
  `clean.csv` / `load.csv` names currently in `outputs/`. Those came from an
  uncommitted local run. Running the notebooks fresh today will not reproduce
  `clean.csv` / `load.csv` under those names; reconcile before the next full run.
- `metadata.csv` / `metadata_sample.csv` were generated the same way `clean.csv` /
  `load.csv` were (a local script mirroring the notebook's item-table logic), not by
  running `data_cleaning.ipynb` directly, since it doesn't save the item table under
  that filename either.

