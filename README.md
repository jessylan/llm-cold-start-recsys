# Netflix Cold Start — Data Preparation

Capstone project ("Team Cold Start"): data preparation for a generative recommendation
system focused on the **cold-start problem** — recommending items to new users and
recommending new items that have little or no interaction history.

> **Note on naming:** the project is called "netflix-cold-start", but the raw data is the
> **Amazon Reviews 2023** dataset (McAuley Lab): the official 5-core **Books** and
> **Movies_and_TV** ratings files plus the Movies_and_TV metadata file. The team's earlier
> data summary (`Codex/2026-07-09/data/processed/data_summary.md`) confirms this. There is
> no separate Netflix dataset in this project.

## Folder structure

```
netflix-cold-start/
├── README.md
├── .gitignore
├── netflix-cold-start.code-workspace
├── notebooks/
│   ├── data_cleaning.ipynb                 <- run first: load + clean, saves cleaned tables
│   ├── train_test_split.ipynb              <- run second: splits + leakage checks
│   └── archive/
│       └── cold_start_data_prep_original.ipynb   <- untouched backup, do not edit
├── data/
│   └── raw/
│       ├── movies/
│       │   ├── Movies_and_TV.csv.gz        <- movie ratings (user_id, parent_asin, rating, timestamp)
│       │   └── meta_Movies_and_TV.jsonl.gz <- movie metadata (title, description, categories, ...)
│       └── books/
│           └── Books.csv.gz                <- book ratings (user_id, parent_asin, rating, timestamp)
└── outputs/                                <- all generated files are written here
```

## How to run (VS Code)

1. Open `netflix-cold-start.code-workspace` in Visual Studio Code
   (File → Open Workspace from File…).
2. Open `notebooks/data_cleaning.ipynb` and run it top to bottom. It loads and cleans
   the raw data and saves `processed_items.csv` / `processed_interactions.csv` to `outputs/`.
3. Then open `notebooks/train_test_split.ipynb` and run it top to bottom. It loads the
   cleaned tables from `outputs/` and saves the train/validation/test and cold-start
   split files back to `outputs/`.
4. Select a Python kernel that has the dependencies below installed
   (VS Code will prompt; the Jupyter extension is required for notebooks).
   The first cells of each notebook set up all paths relative to the project folder,
   so the notebooks work no matter where the kernel's working directory is.

## Data

- Raw movie inputs live in `data/raw/movies/`, raw book inputs in `data/raw/books/`.
  These are copies; the originals remain in `C:\Users\Seema\Documents\Codex\2026-07-09\data\raw\`.
- Files are gzip-compressed; pandas reads them directly, no unzipping needed.
- The ratings files are large (9.5M book rows, 7.4M movie rows) — loading and cleaning
  the full data takes several minutes.

## Known caveat (needs a team decision)

The notebook was written as a schema-agnostic template: its "Identify important columns"
section (§6) still contains placeholder column names (`show_id`, `isbn`, `listed_in`, …)
that do **not** match the Amazon Reviews 2023 schema. The loading, path, and structure
cells all work, and the column-check cell will print clear warnings, but the cleaning
sections need the §6 variables updated to the real columns before the notebook can run
end to end. The actual columns found in each file are documented in a note inside
`data_cleaning.ipynb`, right above §6.

## Dependencies

Determined from the notebook's imports (Python 3):

- `pandas`
- `numpy`
- `matplotlib`
- `jupyter` / `ipykernel` (to run the notebook)

Note: the default Python on this machine has pandas and numpy but **not matplotlib**
(the original notebook's only recorded run failed on `import matplotlib`). Install it,
or pick a kernel/environment that has it, before running.
