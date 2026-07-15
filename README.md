# LLM Cold-Start Recommender

University of Michigan MADS Capstone project for Team Cold Start. The project explores
using language models to improve recommendations for new users and items with little or
no interaction history.

The data pipeline uses the Amazon Reviews 2023 5-core Books and Movies and TV datasets.
It cleans and combines ratings, then creates standard and cold-start evaluation splits.

## Quick start

1. Use Python 3 and install the core dependencies:

   ```bash
   pip install pandas numpy matplotlib jupyter ipykernel
   ```

2. Place the raw files in the locations shown under **Project structure**.
3. Run `notebooks/data_cleaning.ipynb`.
4. Run `notebooks/train_test_split.ipynb`.
5. Use `outputs/load_sample.csv` for shared development and testing. Use the full
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
|-- data/raw/                         # local raw data (not committed)
|-- design_documents/
|   `-- initial_pipeline_design/      # baseline and architecture artifacts
|-- notebooks/
|   |-- data_cleaning.ipynb
|   `-- train_test_split.ipynb
|-- outputs/
|   |-- clean_sample.csv
|   |-- load_sample.csv
|   `-- metadata_sample.csv
|-- .gitignore
`-- README.md
```

## Data and outputs

| File | Purpose |
|---|---|
| `Books.csv.gz` | Book ratings |
| `Movies_and_TV.csv.gz` | Movie and TV ratings |
| `meta_Movies_and_TV.jsonl.gz` | Movie and TV metadata |
| `clean.csv` | Full cleaned interactions (local/shared storage) |
| `load.csv` | Full modeling dataset (local/shared storage) |
| `metadata.csv` | Full unified item table: movie metadata + book ID stubs (local/shared storage) |
| `clean_sample.csv` | Git-friendly sample of cleaned interactions |
| `load_sample.csv` | Git-friendly sample with split labels |
| `metadata_sample.csv` | Git-friendly sample of the item table (see limitation below) |

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

## Design documents

`design_documents/initial_pipeline_design/` contains the baseline collaborative-filtering
notebook and its Jupytext source, architecture-generation scripts, requirements, and the
generated solution architecture. These artifacts document the initial steel-thread and
extreme-baseline design work.

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
