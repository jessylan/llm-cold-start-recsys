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
5. Use `outputs/load.csv` for downstream modeling and analysis.

The notebooks use project-relative paths, so they can be run from VS Code or Jupyter.

## Pipeline

```text
Raw Amazon data
    -> cleaning and standardization
    -> unified interactions (clean.csv)
    -> train, validation, test, and cold-start splits
    -> modeling handoff (load.csv)
```

`load.csv` contains the cleaned interaction columns plus:

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
|   |-- clean.csv
|   `-- load.csv
|-- .gitignore
`-- README.md
```

## Data and outputs

| File | Purpose |
|---|---|
| `Books.csv.gz` | Book ratings |
| `Movies_and_TV.csv.gz` | Movie and TV ratings |
| `meta_Movies_and_TV.jsonl.gz` | Movie and TV metadata |
| `clean.csv` | Cleaned, combined interactions |
| `load.csv` | Final modeling dataset with split labels |

Raw data and generated outputs are intentionally excluded from Git because they are
large. Obtain them from the team's shared storage or regenerate them locally.

## Notes

- The full dataset contains about 17 million interactions, so a complete pipeline run
  can take several minutes and requires substantial disk space.
- Before rerunning the cleaning notebook with different data, confirm that its column
  mappings match the source schema.
- Keep the split labels unchanged so evaluation remains consistent across the project.
