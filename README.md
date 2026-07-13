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
|   `-- load_sample.csv
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
| `clean_sample.csv` | Git-friendly sample of cleaned interactions |
| `load_sample.csv` | Git-friendly sample with split labels |

Raw data and full generated outputs are excluded from Git because they are large. The
two sample CSVs are committed so collaborators can test the workflow immediately. The
sample contains 100,000 rows: 10,000 movie and 10,000 book rows from each standard and
cold-start split.

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
