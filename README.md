<!-- This file was created with the assistance of Generative AI -->
# LLM Cold-Start Recommender

University of Michigan MADS Capstone project for Team Cold Start. The project explores
using language models to improve recommendations for new users and items with little or
no interaction history.

The data pipeline uses the Amazon Reviews 2023 "Books" and "Movies and TV" datasets.
It cleans and combines ratings, then creates standard and cold-start evaluation splits.

## Quick start

1. Use Python 3.12 and install the dependencies then remove one package vLLM pulls in that breaks Intervention A:
   ```bash
   pip install -r requirements.txt && pip uninstall -y torchcodec
   ```
   `requirements.txt` describes a Linux + CUDA environment: the interventions are GPU jobs
   with no CPU fallback. The baseline steel thread will still run without a GPU (`implicit`
   fits on CPU); only retrieval and the interventions need one.
2. Get the data. Either:
   - **Fast path (recommended):** download the four `data/filtered/*.parquet` files
     from the links under **Data and outputs** into `data/filtered/`, then skip to
     step 4.
   - **From raw:** place the raw files at the `data/raw/...` paths listed under
     **Data and outputs**, then run step 3.
3. Run `notebooks/data_filtering.ipynb`.  
4. Run `notebooks/hyperparameter_tuning.ipynb`. Skippable - `outputs/hyperparams.json` is committed.
5. **(Optional, GPU only)** Intervention A - sentence embeddings in place of TF-IDF for the
   item's own prose. Run `notebooks/intervention_a_encoding.ipynb` **once** to build
   `data/embeddings/*.npz` (~12 GB, keyed by `parent_asin`, so it is split-independent);
   everything downstream reads those files and never touches a GPU encoder again. The three
   selection notebooks - `intervention_a_model_selection`, `..._description_variants`,
   `..._weight_sweep` - choose the encoder and field weights on the cold-item **validation**
   set, and write their results to `outputs/intervention_a_*.json`, which is committed. Skip
   them and `steel_thread.ipynb` uses the committed configuration.
   See [design_documents/07-intervention-a-embeddings.md](design_documents/07-intervention-a-embeddings.md).
6. **(Optional, GPU only)** Intervention B - ColdLLM-style synthetic interactions for
   cold-start items. Run `notebooks/intervention_b_coldllm.ipynb`. It is self-contained: it
   computes its own CBHCF baseline and does not need step 7 to have run. Needs a CUDA GPU
   with `vllm` installed separately, and generation is a multi-hour job the first time
   (scores are cached afterwards). See
   [design_documents/08-intervention-b-coldllm.md](design_documents/08-intervention-b-coldllm.md).
7. Run `notebooks/steel_thread.ipynb`. It reads `outputs/hyperparams.json`; if that file is
   missing it falls back to untuned `lambda=1`. It reports whichever arms have artifacts
   available and **skips the rest with a printed reason**, so it runs on its own if you
   skipped steps 5 and 6, and adds the Intervention A and B curves if you did not.
8. Run `notebooks/report_figures.ipynb` to regenerate the report figures and the poster
   figure. Needs only matplotlib -- no GPU, no `data/`. It reads `outputs/figure_curves.json`
   and, for the equity figure's per-k curves, `outputs/baseline_cf_20260815_012902.json`.

The notebooks use project-relative paths and can be run from VS Code or Jupyter. 
Run only one GPU notebook at a time. `steel_thread.ipynb` and `intervention_b_coldllm.ipynb`
are pinned to the same card, and under WSL an over-subscribed GPU pages into host RAM instead
of failing, which takes down the whole VM rather than one kernel.

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
|-- data/embeddings/                    # Intervention A sentence embeddings (not committed)
|-- design_documents/
|   |-- README.md                       # index: which doc answers which question
|   |-- 01-getting-started.md           # zero -> running evaluation; what each notebook writes
|   |-- 02-module-dependencies.md       # import graph; what breaks if you edit a file
|   |-- 03-adding-a-model.md            # the RetrievalModel contract and its implementors
|   |-- 04-cbhcf-score-composition.md   # ALS + content -> AdditiveItemBlock; where lambda comes from
|   |-- 05-what-metrics-mean.md         # Mode A/B control flow, ceilings, floors, pools
|   |-- 06-provider-equity.md           # equity_metrics.py: provider exposure, wired into Section 9c and run
|   |-- 07-intervention-a-embeddings.md # sentence embeddings in place of TF-IDF prose
|   |-- 08-intervention-b-coldllm.md    # LLM-simulated interactions for cold items
|   `-- initial_pipeline_design/        # RETIRED MovieLens steel thread; design record only
|-- notebooks/
|   |-- data_filtering.ipynb                        # filtering reviews datasets and metadata as well
|   |-- hyperparameter_tuning.ipynb                 # selects CBHCF's lambda on the cold-item VALIDATION set
|   |-- steel_thread.ipynb                          # how-to for recsys modules; the reported baseline and both interventions
|   |-- intervention_a_encoding.ipynb               # encodes the catalogue ONCE -> data/embeddings/*.npz (GPU)
|   |-- intervention_a_model_selection.ipynb        # picks the sentence encoder on the cold-item VALIDATION set
|   |-- intervention_a_description_variants.ipynb   # should `description` be embedded, and does its structure matter?
|   |-- intervention_a_weight_sweep.ipynb           # field weights, same search budget for both arms
|   |-- intervention_b_coldllm.ipynb                # Intervention B: ColdLLM-style synthetic interactions (needs vLLM + GPU)
|   `-- report_figures.ipynb                        # report Figs 2-4 and the poster figure, from figure_curves.json (matplotlib only, no GPU)
|-- outputs/
|   |-- hyperparams.json                # tuned CBHCF lambda and the Intervention A configuration
|   |-- baseline_cf_20260815_012902.json  # the reported run: every curve, ceiling, floor, and equity sweep from steel_thread.ipynb
|   `-- figure_curves.json              # per-k curves + ceilings lifted from that run's printed outputs; drives report_figures.ipynb
|-- recsys/
|   |-- __init__.py                     # pins the BLAS thread pool to 1 (implicit does its own parallelism)
|   |-- load.py                         # Amazon Books loader; the 80/10/10 warm / cold-val / cold-test split (of items with ≥ min_interactions=25)
|   |-- protocol.py                     # `RetrievalModel` -- the fit/recommend/fold_in contract `eval.py` relies on
|   |-- pop.py                          # popularity and activity baseline models (the floors)
|   |-- cf.py                           # ALS collaborative filtering baseline
|   |-- content.py                      # item content vectors: role-based fields, BM25F weighted blocks
|   |-- item_space.py                   # ItemSpace: the dense+sparse item representation CBHCF wraps, TF-IDF or embeddings
|   |-- cbhcf.py                        # content-based hybrid CF -- ALS score + weighted content score
|   |-- intervention_a.py               # Intervention A: sentence-embedding item space (needs sentence-transformers + GPU)
|   |-- scores.py                       # score sources: the interface `gpu_retrieval` consumes instead of factors
|   |-- gpu_retrieval.py                # exact GPU top-K, candidate-pool AUC, and the Mode B duals
|   |-- eval.py                         # evaluation harness: metrics, within-item ceiling, both warm-up sweeps
|   |-- equity_metrics.py               # provider-side fairness metrics - Gini, catalog share, equity ratio
|   `-- coldllm.py                      # Intervention B: ColdLLM-style synthetic interaction generation (vLLM, GPU-only)
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
| `outputs/hyperparams.json` | Tuned CBHCF lambda and the Intervention A configuration | Generated by steps 4 and 5 |
| `outputs/baseline_cf_20260815_012902.json` | Reported results of `steel_thread.ipynb`. Every number in the report traces here. | Generated by step 7 |
| `outputs/figure_curves.json` | Extracted from `steel_thread.ipynb` cell output. Minimum input to run `report_figures.ipynb` | Committed; regenerate only if the steel thread is re-run |

Raw and filtered data are excluded from Git because they are large. The fast path in
**Quick start**  downloads the filtered parquet files directly; the raw files are only 
needed to re-run `data_filtering.ipynb` yourself.

The Intervention A embeddings (`data/embeddings/`, ~12 GB) and every ColdLLM score cache are
also excluded: they are regenerable from committed code and configuration, and syncing them
costs more than rebuilding them.

No sample data is currently committed. A fresh sample derived from
`data/filtered/*.parquet` is planned.

## design_documents folder
Two things live here: **current component documentation** for `recsys/`, and a **retired
design record** from the project's first architecture.

### Component documentation (current)

Eight Mermaid diagrams, each named for the question it answers rather than the component it
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
| [07-intervention-a-embeddings.md](design_documents/07-intervention-a-embeddings.md) | How does Intervention A replace TF-IDF with sentence embeddings? |
| [08-intervention-b-coldllm.md](design_documents/08-intervention-b-coldllm.md) | How does Intervention B generate synthetic interactions for cold items? |

Docs 05 and 06 each end with a **Drift** section (places where prose contradicts the code,
reported but not fixed) and an **Open questions** section (things not determinable from
source). Read `06` before touching `recsys/equity_metrics.py` - it records why the module is
measured over the eval-user set rather than the full population, which is not obvious from the
code and is what a naive widening would break. Its Drift section is now empty: the id-format
mismatch it used to document was fixed rather than merely reported.

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

## Use of Generative AI

Open-weight models are part of the method: Intervention A encodes item text with pretrained
sentence-embedding models (`Snowflake/snowflake-arctic-embed-l-v2.0`), and Intervention B
prompts a locally served LLM (`Qwen2.5-7B-Instruct-AWQ` via vLLM) zero-shot to generate
synthetic interactions for cold-start items. Both run locally, unmodified - no fine-tuning
and no third-party APIs. Separately, the team used AI coding assistants to help write code
and documentation; all such output was reviewed and tested by a team member. No reported
result was produced by an AI assistant - every number comes from running the committed code.