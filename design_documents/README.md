# `recsys/` documentation

EIght diagrams, each named for the question it answers. Every node is a real symbol; every
table row points at `file:line`. Read them in this order on your first afternoon.

| Doc | Question it answers | Read it when |
|---|---|---|
| [01-getting-started.md](01-getting-started.md) | How do I get from zero to a running evaluation? | Day one. Which files to download, which notebooks to run, what each writes. |
| [02-module-dependencies.md](02-module-dependencies.md) | What breaks if I edit this file? | Before your first change to anything in `recsys/`. |
| [03-adding-a-model.md](03-adding-a-model.md) | How do I add a new model? | You want a new retrieval method to appear on the warm-up curve. |
| [04-cbhcf-score-composition.md](04-cbhcf-score-composition.md) | How does CBHCF combine ALS and content? | You are touching the hybrid, the content space, or lambda. |
| [05-what-metrics-mean.md](05-what-metrics-mean.md) | What does this accuracy number mean? | You have a number and need to know what it measured against. |
| [06-provider-equity.md](06-provider-equity.md) | Who gets exposure, and how do we turn it on? | You are picking up the provider-equity work. Written as an integration spec — the module is not wired in yet. |
| [07-intervention-a-embeddings.md](07-intervention-a-embeddings.md) | How does Intervention A replace TF-IDF with sentence embeddings? | You are touching `recsys/intervention_a.py`, `recsys/item_space.py`, or any of the four `intervention_a_*` notebooks. |
| [08-intervention-b-coldllm.md](08-intervention-b-coldllm.md) | How does Intervention B generate synthetic interactions for cold items? | You are touching `recsys/coldllm.py` or the ColdLLM notebook, or comparing its two prompting strategies. |

Two cross-cutting sections live at the bottom of the docs that own them:

- **Drift** — every place the README, a docstring, or notebook prose contradicts the code.
  Reported, not fixed. See [05](05-what-metrics-mean.md#drift),
  [06](06-provider-equity.md#drift) and [07](07-intervention-a-embeddings.md#drift).

## Orientation in one paragraph

`recsys/` evaluates how quickly a **cold item** becomes retrievable as it accumulates
interactions. [load.py](../recsys/load.py) builds a `Dataset` that holds a warm training
split plus a population of structurally cold items, each with a *revealable* pool and a
permanently reserved test set. A model is fit once on the warm data, then **folded in** at
reveal level `k = 0, 1, 2, ... 20` — only the cold items' representations move — and
evaluated at each level against the same fixed test set. The resulting curve is the
"warm-up curve." Everything else in the package exists to make that curve computable at
Amazon-Books scale (487,790 items) and comparable across methods.

## Conventions you will trip over

- **`k` is a reveal level, not a list length.** The top-K list length is `K` (capital,
  usually 100 for Mode A and 10 for Mode B). Both appear in the same signatures.
- **Mode A vs Mode B.** Mode A ranks *items* for a user. Mode B ranks *users* for a cold
  item. They are duals, with separate ceilings and separate floors.
- **Notebooks run from `notebooks/`,** so their paths are `../data/...`. The module
  defaults in [load.py:151](../recsys/load.py:151) and
  [load.py:366](../recsys/load.py:366) are repo-root-relative and will not resolve from a
  notebook. Every notebook passes its path explicitly; do the same.
- **`design_documents/initial_pipeline_design/` is a retired design record** describing the
  original MovieLens steel thread. It is not documentation of the current system.
  `notebooks/steel_thread.ipynb` — despite sharing the name — *is* the current baseline
  runner and is unrelated to it.
