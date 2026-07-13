# Cold-Start Recommender: Data Preparation

Data preparation for a generative recommendation system focused on the cold-start
problem: recommending items to new users and recommending new items with little or no
interaction history.

The pipeline uses the Amazon Reviews 2023 5-core Books and Movies and TV datasets. It
cleans and combines the ratings, then creates standard and cold-start evaluation splits.

## Quick start

1. Use Python 3 and install the dependencies:

   ```bash
   pip install pandas numpy matplotlib jupyter ipykernel
   ```

2. Place the raw files in the locations shown under **Project structure**.
3. Run `notebooks/data_cleaning.ipynb` first.
4. Run `notebooks/train_test_split.ipynb` second.
5. Use `outputs/load_sample.csv` for shared development and testing. Use the full
   `outputs/load.csv` only for full-scale local runs.

The notebooks use project-relative paths, so they can be run from VS Code or Jupyter.

## Pipeline

```text
Raw Amazon data
    -> cleaning and standardization
    -> unified interactions (clean.csv)
    -> train, validation, test, and cold-start splits
    -> modeling handoff (load.csv / load_sample.csv)
```

`load.csv` and `load_sample.csv` contain the cleaned interaction columns plus:

- `split`: `train`, `validation`, `test`, `cold_start_user_test`, or
  `cold_start_item_test`
- `is_sparse_user_test`: identifies test interactions for users with only 1-5 training
  interactions

## Project structure

```text
.
|-- data/
|   `-- raw/
|       |-- books/
|       |   `-- Books.csv.gz
|       `-- movies/
|           |-- Movies_and_TV.csv.gz
|           `-- meta_Movies_and_TV.jsonl.gz
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
two sample CSVs are committed so everyone can develop and test the workflow immediately.

## Notes

- The full dataset contains about 17 million interactions, so a complete pipeline run
  can take several minutes and requires substantial disk space.
- The collaboration sample contains 100,000 rows: 10,000 movie and 10,000 book rows
  from each train, validation, test, cold-user, and cold-item split.
- Before rerunning the cleaning notebook with different data, confirm that its column
  mappings match the source schema.
- Keep the split labels unchanged so evaluation remains consistent across the project.
