# What breaks if I edit this file?

Thirteen modules, and the import graph is deliberately shallow: **seven of the thirteen are
imported by nothing else in `recsys/`, and nothing imports `eval.py`.** Most coupling runs
through two contracts rather than through imports — `protocol.RetrievalModel` (what a model
must be) and `scores.ItemBlockScores` / `scores.UserBlockScores` (what a score source must
be).

The graph has two kinds of edge, and the difference matters when you are tracing a change:

- **Solid** = module-level `import` at the top of the file. A cycle here would be a real
  circular-import error.
- **Dashed** = function-local import, deferred to call time. `cf`, `cbhcf`, and `eval` all
  reach `gpu_retrieval` this way, which is how `cf.py` and `gpu_retrieval.py` can depend on
  each other without a cycle — and how a CPU-only environment can import the package
  without `torch` installed.

```mermaid
flowchart BT
    load["load.py<br/>Dataset, load_dataset<br/>382 lines"]
    protocol["protocol.py<br/>RetrievalModel<br/>42 lines"]
    scores["scores.py<br/>ItemBlockScores, UserBlockScores<br/>290 lines"]
    gpu["gpu_retrieval.py<br/>topk / AUC kernels<br/>498 lines"]
    cf["cf.py<br/>ALSModel<br/>314 lines"]
    cbhcf["cbhcf.py<br/>CBHCFModel<br/>483 lines"]
    content["content.py<br/>ContentSpace<br/>293 lines"]
    ispace["item_space.py<br/>ItemSpace, as_item_space<br/>205 lines"]
    ia["intervention_a.py<br/>build_space, encoders<br/>808 lines"]
    coldllm["coldllm.py<br/>ColdLLM simulation<br/>1,020 lines"]
    pop["pop.py<br/>Popularity, Activity<br/>123 lines"]
    evalm["eval.py<br/>the sweeps<br/>624 lines"]
    equity["equity_metrics.py<br/>provider equity<br/>177 lines"]

    protocol -->|"Dataset type hint"| load
    cf -->|"dataset_fingerprint"| load
    evalm -->|"RetrievalModel"| protocol
    equity -->|"RetrievalModel"| protocol
    gpu -->|"Factor/Dense blocks,<br/>as_gpu, gather_rows"| scores
    cbhcf -->|"DenseItemBlock, AdditiveItemBlock,<br/>scale_over_users"| scores
    cbhcf -->|"as_item_space"| ispace
    ia -->|"ContentSpace,<br/>field extraction"| content
    ia -->|"ItemSpace<br/>(dense + sparse blocks)"| ispace

    cf -.->|"topk_recommend,<br/>cold_merge_recommend"| gpu
    cf -.->|"FactorItemBlock,<br/>FactorUserBlock"| scores
    cbhcf -.->|"topk_recommend,<br/>cold_merge_recommend"| gpu
    evalm -.->|"mode_a_auc, mode_a_auc_sweep,<br/>mode_b_topk_users, mode_b_auc"| gpu

    classDef leaf stroke-dasharray: 4 3
    classDef orphan stroke-width:3px
    class pop,coldllm leaf
    class equity orphan
```

`coldllm.py` has **no edge in either direction** — it is the largest module in the package and
imports nothing from `recsys` at all. That is not an accident; see "the couplings that imports
do not show" below, because it is coupled to `load.py` anyway.

## Reading the graph

| Module | file | Nothing in `recsys/` imports it? | Who actually uses it |
|---|---|---|---|
| [load.py](../recsys/load.py) | 382 lines | — | `protocol` (type hint), `cf` (fingerprint), and every notebook. **The most load-bearing file in the package.** |
| [protocol.py](../recsys/protocol.py) | 42 lines | — | `eval`, `equity_metrics`. Pure contract; no runtime behavior. |
| [scores.py](../recsys/scores.py) | 290 lines | — | `gpu_retrieval` (top-level), `cbhcf` (top-level), `cf` (deferred). |
| [gpu_retrieval.py](../recsys/gpu_retrieval.py) | 498 lines | — | `cf`, `cbhcf`, `eval` — all deferred. |
| [item_space.py](../recsys/item_space.py) | 205 lines | — | `cbhcf` and `intervention_a`, both top-level. The seam that lets one CBHCF wrap either a TF-IDF matrix or a dense+sparse embedding space. |
| [content.py](../recsys/content.py) | 293 lines | — | `intervention_a` (top-level); `cbhcf` consumes its *output matrix*, not the module. |
| [cf.py](../recsys/cf.py) | 314 lines | **yes** | Notebooks, and `cbhcf` holds an instance (never imports the module). |
| [cbhcf.py](../recsys/cbhcf.py) | 483 lines | **yes** | Notebooks only. |
| [intervention_a.py](../recsys/intervention_a.py) | 808 lines | **yes** | Notebooks only. Builds an `ItemSpace` that `cbhcf` then wraps exactly as it wraps TF-IDF. |
| [coldllm.py](../recsys/coldllm.py) | 1,020 lines | **yes — and it imports nothing from `recsys` either** | Notebooks only. Fully standalone by design: `vllm` is imported lazily inside `VLLMColdLLMSimulator.__init__`, so filtering, the work queue, assembly and `SyntheticAugmentedDataset` all run in an environment with no engine installed. |
| [pop.py](../recsys/pop.py) | 123 lines | **yes** | Notebooks only. `eval` handles Popularity/Activity duck-typed, never by import. |
| [eval.py](../recsys/eval.py) | 624 lines | **yes** | Notebooks only. It is the top of the stack. |
| [equity_metrics.py](../recsys/equity_metrics.py) | 177 lines | **yes — and by nothing else either** | Nothing. No module, no notebook. See [06](06-provider-equity.md). |

Seven modules are import-leaves (`cf`, `cbhcf`, `intervention_a`, `coldllm`, `pop`, `eval`,
`equity_metrics`) — that is by design: the notebooks are the wiring layer. `equity_metrics.py`
is different in kind: it is a leaf that **nothing calls at all**, including notebooks. That is
intentional and tracked as future work, not an oversight.

## The couplings that imports do not show

An import graph under-reports this package. Four things bind modules together without an
edge above, and they are where changes actually propagate.

**1. `protocol.RetrievalModel` is structural, not nominal.** It is a
`@runtime_checkable` `Protocol` ([protocol.py:13](../recsys/protocol.py:13)), so
`isinstance()` checks method *existence only* — not signatures, not behavior.
`eval._check_model` ([eval.py:23](../recsys/eval.py:23)) and
`equity_metrics._check_model` ([equity_metrics.py:120](../recsys/equity_metrics.py:120))
both rely on it. Consequence: a class can pass the check and still fail at runtime. See
[03](03-adding-a-model.md) for the two live cases.

**2. Private attributes are a de-facto second interface.** `eval.py` reaches into
`model._auc_candidate_ids`, `model._auc_global_to_local`, `model._auc_warm_global_to_local`,
`model._cold_global_to_local`, and `model._device`
([eval.py:126](../recsys/eval.py:126), [eval.py:167](../recsys/eval.py:167),
[eval.py:181](../recsys/eval.py:181), [eval.py:226](../recsys/eval.py:226),
[eval.py:319](../recsys/eval.py:319)). `CBHCFModel` has to mirror all of them —
`_POOL_ATTRS` ([cbhcf.py:121](../recsys/cbhcf.py:121)) exists solely to copy that set off
the wrapped `ALSModel`. **Rename a `_pool` attribute in `cf.py` and you must update
`cbhcf._POOL_ATTRS` and `eval.py` together.**

**3. Duck-typed branches.** `eval.py` detects the Activity floor with
`hasattr(models[0], "user_activity")` ([eval.py:543](../recsys/eval.py:543)), and
`ceiling_reference` / `evaluate_at_k` decide whether to compute AUC with
`getattr(model, "_auc_candidate_ids", None) is not None`. Adding an attribute with one of
those names to an unrelated model silently changes which branch runs.

**4. `coldllm.SyntheticAugmentedDataset` is coupled to `load.py` with no import.** It is a
duck-typed wrapper, not a subclass ([coldllm.py:963](../recsys/coldllm.py:963)): it overrides
`revealed_item_users_at_k` ([coldllm.py:1015](../recsys/coldllm.py:1015)) and delegates
everything else through `__getattr__` ([coldllm.py:1006](../recsys/coldllm.py:1006)). Because
every field `load.dataset_fingerprint` reads is delegated, an augmented dataset would otherwise
be **indistinguishable** from the plain one — and `cf.ALSModel.fold_in` memoizes cold-item
factors on that fingerprint, so a model folded against both would silently reuse the first
one's factors. The fix is a salt, `_fingerprint_salt`
([coldllm.py:997](../recsys/coldllm.py:997)), which `load.dataset_fingerprint` appends *only
when present* ([load.py:140](../recsys/load.py:140)) so plain Datasets keep the exact tuple
every cache key and `outputs/hyperparams.json` already store. **Add a field to
`dataset_fingerprint` and you must check it is either delegated or overridden here.**

## Change-impact cheat sheet

| If you edit… | Re-check |
|---|---|
| `load.Dataset` fields | `dataset_fingerprint` ([load.py:107](../recsys/load.py:107)) — it enumerates fields explicitly; every `eval.py` sweep; `cf.py`'s fold cache key; `coldllm.SyntheticAugmentedDataset.__getattr__`. |
| `load.load_dataset` split params | Every committed artifact keyed on the fingerprint, including `outputs/hyperparams.json`, plus the cached ColdLLM export `data/cache/coldllm_synthetic_*.pkl`, which refuses to load against a different split. |
| `load.dataset_fingerprint`'s tuple | `coldllm._fingerprint_salt` ([coldllm.py:997](../recsys/coldllm.py:997)) — the salt is appended, so anything that assumes a fixed tuple length breaks on augmented datasets only. |
| `protocol.RetrievalModel` | `pop.py`, `cf.py`, `cbhcf.py` (all three must still satisfy it), plus both `_check_model` error messages. |
| any `_auc_*` / `_cold_*` / `_warm_*` attribute in `cf.py` | `cbhcf._POOL_ATTRS`, and the five `eval.py` sites above. |
| a `scores.*Block` signature | `gpu_retrieval` (every kernel takes `source=`), `cf`'s four `*_source()` methods, `cbhcf`'s five. |
| `gpu_retrieval` kernel signatures | `cf.recommend`/`recommend_cached`/`build_warm_cache`, the CBHCF equivalents, and six call sites in `eval.py`. |
| `content.ContentSpace.transform` | **The row-L2-normalization is load-bearing.** `cbhcf` relies on `T @ T.T` being exactly cosine similarity ([cbhcf.py:31](../recsys/cbhcf.py:31)); breaking it silently corrupts every content score rather than raising. `intervention_a` builds on the same guarantee. |
| `item_space.ItemSpace` / `as_item_space` | Both consumers: `cbhcf.py:121` (wraps whatever it is handed) and `intervention_a.py`. This is the seam that makes Intervention A a drop-in for TF-IDF — changing it changes both arms of the comparison at once, which destroys the controlled contrast. |
| `coldllm` selection rules or prompt text | Nothing in `recsys/` — but every cached score in `data/cache/coldllm_*` and the run keys in `notebooks/intervention_b_coldllm.ipynb`. Scores are stored graded and selection is applied offline, so changing a *selection rule* is cheap and changing a *prompt* is a full regeneration. |
