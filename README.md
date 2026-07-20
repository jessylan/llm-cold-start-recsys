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
4. Run `notebooks/data_cleaning.ipynb`. # NOTE - may need rework after adding `data_filtering.ipynb` - to be addressed soon
5. Run `notebooks/train_test_split.ipynb`. # NOTE - may need rework after adding `data_filtering.ipynb` - to be addressed soon
6. Use `outputs/load_sample.csv` for shared development and testing. Use the full
   `outputs/load.csv` only for full-scale local runs.

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
|   |-- data_cleaning.ipynb
|   |-- data_filtering.ipynb            # filtering reviews datasets and metadata as well
|   `-- train_test_split.ipynb
|-- outputs/
|   |-- clean_sample.csv
|   |-- load_sample.csv
|   `-- metadata_sample.csv
|-- recsys/
|   |-- __init__.py                     # 
|   |-- cf.py                           # pure collaborative filtering baseline model
|   |-- eval.py                         # evaluation harness
|   |-- load.py                         # load the data (currently MovieLens-100k)
|   |-- pop.py                          # popularity and activity baseline models
|   `-- protocol.py                     # defines the `RetrievalModel` class which `eval.py` uses to ensure models are correctly built 
|-- .gitignore
`-- README.md
```

## Data and outputs

| File | Purpose | Link |
|---|---|---|
| `data/raw/books/Books.csv.gz` | Book ratings | *missing* |
| `data/raw/books/Books.jsonl.gz` | Book reviews & ratings | https://amazon-reviews-2023.github.io |
| `data/raw/books/meta_Books.jsonl.gz` | Movie and TV metadata | https://amazon-reviews-2023.github.io |
| `data/raw/movies/Movies_and_TV.csv.gz` | Movie and TV ratings | *missing* |
| `data/raw/movies/Movies_and_TV.jsonl.gz` | Movie and TV reviews & ratings | https://amazon-reviews-2023.github.io |
| `data/raw/movies/meta_Movies_and_TV.jsonl.gz` | Movie and TV metadata | https://amazon-reviews-2023.github.io |
| `data/filtered/books_meta_common.parquet` | Book metadata for filtered subset | https://drive.google.com/file/d/1ok7IMYRSVeK8-HXwaJw7kzw2Mk3qYavu/view?usp=sharing |
| `data/filtered/books_reviews_common.parquet` | Book reviews & ratings for filtered subset | https://drive.google.com/file/d/1bzgYlx1W1bt3i7aY8ALoctoGhomRuRLZ/view?usp=sharing |
| `data/filtered/movies_meta_common.parquet` | Movies metadata for filtered subset | https://drive.google.com/file/d/1-Gyhxr615g30-Qa6fdG0fjAe4JGK7FMS/view?usp=sharing |
| `data/filtered/movies_reviews_common.parquet` | Movies reviews & ratings for filtered subset | https://drive.google.com/file/d/1Uz88BunWZNWFVz86O6iAcq3vqkjxbwj2/view?usp=sharing |
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

### Known limitation: books have no metadata yet

`metadata.csv`/`metadata_sample.csv` only have real `title`/`genre`/`description` for
**movies**, pulled from `meta_Movies_and_TV.jsonl.gz`. Book rows are blank stubs (one
per rated `parent_asin` in `Books.csv.gz`) because **no book metadata file is in the
pipeline yet**. The official book metadata file, `meta_Books.jsonl.gz`, exists but is
4.9 GB compressed (vs. 271 MB for movies) and hasn't been downloaded or wired in. Until
that happens, books have no title/genre/description for content-based features.

`genre` and `description` are `" | "`-joined plain text, flattened from the raw JSON
arrays Amazon's schema stores them as (not a Python list's string repr).

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

