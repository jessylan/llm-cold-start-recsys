# How does Intervention B generate synthetic interactions for cold items?

A two-stage funnel, following ColdLLM ([arXiv:2402.09176](https://arxiv.org/abs/2402.09176)):
**Filtering Simulation** narrows every cold item's candidate pool from all users down to the
top-K most content-similar, then **Refining Simulation** asks an LLM a yes/no for each
surviving (item, user) pair. Positive answers become a synthetic interactions matrix, added to
`ref_train` before fitting ALS — giving a cold item something to learn from *before* it has
accumulated a single real interaction, which is exactly the case
[05-what-metrics-mean.md](05-what-metrics-mean.md) and `steel_thread.ipynb`'s own fold-in
sweep show ALS's `recalculate_item` cannot help with at all (its closed-form solve returns the
exact zero vector at `k=0`).

**Two Refining prompting strategies share one code path and one notebook run**, forking only at
the Refining stage: **B1 (direct)** asks the bare yes/no question with choice-constrained
decoding; **B2 (reasoning)** asks for a one-sentence justification first, with regex-constrained
decoding. Everything upstream (dataset, cached CBHCF baseline, item metadata, Filtering) and
downstream (the comparison) is shared and computed once — see
`notebooks/intervention_b_coldllm.ipynb`'s Section headers.

```mermaid
flowchart TD
    DOCS["docs: role -> per-item text<br/>(Section 3)"] -->|"flatten to one string/item"| META["item_metadata<br/>feeds LLM prompts only"]
    DOCS -->|"content.fit_content_space<br/>+ .transform (warm-fit, inductive)"| TFIDF["item_content: n_items x n_terms<br/>L2-normalized rows -- SAME matrix CBHCF uses"]

    TRAIN["dataset.ref_train"] -->|"_row_scaled<br/>(same op as cbhcf._row_scaled)"| HIST["row-scaled history"]
    HIST -->|"@ item_content"| PROF["user_content_profile<br/>(n_users x n_terms), sparse"]
    TFIDF --> PROF

    TFIDF --> FC["filter_candidates<br/>sparse inner product = cosine sim, top_k=50<br/>(Section 4 -- NO LLM, runs ONCE)"]
    PROF --> FC
    COLD["dataset.cold_item_ids"] --> FC
    FC --> CAND["candidates: item_idx -> top-K user_idx<br/>SHARED by both prompting strategies"]

    CAND --> RC["refine_candidates<br/>loop over STRATEGIES (Section 5, LLM via vLLM)"]
    META --> RC
    TRAIN --> RC

    RC -->|"reasoning=False<br/>choice=['yes','no'], max_tokens=5"| SB1["synthetic_matrices['direct']"]
    RC -->|"reasoning=True<br/>regex: reasoning text + Answer: yes/no<br/>max_tokens=200"| SB2["synthetic_matrices['reasoning']"]

    SB1 --> FIT1["ref_train + synthetic<br/>cf.ALSModel.fit (Section 6)"]
    SB2 --> FIT2["ref_train + synthetic<br/>cf.ALSModel.fit (Section 6)"]

    FIT1 --> CB1["cbhcf.CBHCFModel<br/>reuses item_content + CBHCF_LAMBDA<br/>build_content_cache(path=...)"]
    FIT2 --> CB2["cbhcf.CBHCFModel<br/>build_content_cache(reuse_from=CB1)<br/>-- no recompute, no disk reload"]
    TFIDF --> CB1
    TFIDF --> CB2

    AUG1["SyntheticAugmentedDataset<br/>(dataset, synthetic['direct'])"] --> SWEEP1["ev.sweep"]
    AUG2["SyntheticAugmentedDataset<br/>(dataset, synthetic['reasoning'])"] --> SWEEP2["ev.sweep"]
    CB1 --> SWEEP1
    CB2 --> SWEEP2

    SWEEP1 --> CURVES["curves['direct'] / curves['reasoning']<br/>vs. cached cbhcf_curve"]
    SWEEP2 --> CURVES
    CURVES --> CMP["Section 7: per-metric delta<br/>reasoning - direct, at k=0 and averaged"]
```

## Node reference

| Node | Source | Purpose |
|---|---|---|
| `_row_scaled` | [coldllm.py:84](../recsys/coldllm.py:84) | Row-normalizes `ref_train` to a user profile; all-zero rows stay zero. Identical operation to `cbhcf._row_scaled` ([cbhcf.py:63](../recsys/cbhcf.py:63)), duplicated rather than imported so this module doesn't depend on cbhcf.py's private helper. |
| `user_content_profile` | [coldllm.py:93](../recsys/coldllm.py:93) | `_row_scaled(train_matrix) @ item_content` — a user's TF-IDF content profile, the exact computation CBHCF's own user profile uses. |
| `filter_candidates` | [coldllm.py:99](../recsys/coldllm.py:99) | Sparse-dense inner product (= cosine similarity, since `item_content`'s rows are L2-normalized) + `argpartition`, matching the paper's own Eq. 17 brute-force approach. |
| `VLLMColdLLMSimulator` | [coldllm.py:31](../recsys/coldllm.py:31) | One vLLM `task="generate"` engine, built at construction and used only by Refining Simulation. |
| `yes_probability` | [coldllm.py:41](../recsys/coldllm.py:41) | The two decoding strategies — see **Two prompting strategies** below. |
| `_build_prompt` | [coldllm.py:121](../recsys/coldllm.py:121) | `reasoning` flag toggles whether a "give a one-sentence reason first" instruction is appended. |
| `_parse_final_answer` | [coldllm.py:144](../recsys/coldllm.py:144) | Pulls the **last** `Answer: yes/no` match out of a reasoning completion (in case the reasoning text itself echoes the word). |
| `refine_candidates` | [coldllm.py:154](../recsys/coldllm.py:154) | Stage 2 end to end: builds one prompt per surviving pair, thresholds `yes_probability`'s output, returns a sparse synthetic matrix. |
| `SyntheticAugmentedDataset` | [coldllm.py:198](../recsys/coldllm.py:198) | Wraps `dataset` so `fold_in()` sees synthetic interactions at every `k`, including `k=0` (where `fold_in`'s own closed-form solve would otherwise return zero). |
| `revealed_item_users_at_k` override | [coldllm.py:236](../recsys/coldllm.py:236) | The **only** method overridden — see **Keeping synthetic data out of the exclusion set** below. |
| `build_content_cache` / `reuse_from` | [cbhcf.py:164](../recsys/cbhcf.py:164) | The content half of CBHCF's score is independent of the collaborative model, so it's built once and shared across both strategies' `CBHCFModel` instances. |
| `sweep` | [eval.py:240](../recsys/eval.py:240) | The same warm-up sweep machinery `steel_thread.ipynb` uses, run here against `SyntheticAugmentedDataset` instead of the plain `Dataset`. |

## Two prompting strategies

Both live behind one `reasoning` flag threaded through `_build_prompt` /
`yes_probability` / `refine_candidates` — nothing about Filtering, the CBHCF wrap, or the
sweep machinery differs between them.

| | B1 — direct | B2 — reasoning |
|---|---|---|
| Prompt | Bare yes/no question | Same question + "give a one-sentence reason, then answer on a final line" |
| Decoding constraint | `GuidedDecodingParams(choice=["yes", "no"])` — the *entire* completion is one of those two strings | `GuidedDecodingParams(regex=...)` — free text up to ~600 chars, followed by a mandatory `Answer: yes`/`Answer: no` line |
| `max_tokens` | 5 | 200 |
| Parsing | None needed — the output IS the answer | `_parse_final_answer` extracts the last `Answer:` match |
| Cost | Cheaper — same call count as B2, far fewer tokens per call | Meaningfully slower for the same call count |

Neither produces a smooth confidence score — both resolve to a hard 0.0/1.0 per prompt. The
paper's own Refining model is LoRA-fine-tuned; this implementation is zero-shot for both
strategies, so there's no calibrated probability to threshold either way.

## Keeping synthetic data out of the exclusion set

`refine_candidates()`'s output is data, not a scorer — it is meant to be added to whatever
matrix an existing model (`cf.ALSModel`, `cbhcf.CBHCFModel`) is fit or folded on, the same slot
`Dataset.revealed_matrix_at_k(k)` already occupies. It must **never** be folded into whatever
matrix `eval.py` treats as "already interacted" for `recommend()` / `ndcg_at_k` /
`recall_and_hit_rate_at_k` / `auc_at_full`. Unlike real revealed history — guaranteed disjoint
from `dataset.test_matrix` by `load.py`'s reveal/reserve split — a synthetic interaction has no
such guarantee, so masking with it would risk silently excluding the exact test-set items being
measured.

This is why `SyntheticAugmentedDataset` overrides **only** `revealed_item_users_at_k`
([coldllm.py:236](../recsys/coldllm.py:236)), which `fold_in()` uses to recalculate cold-item
factors, and deliberately leaves `revealed_matrix_at_k` untouched — that method is what
`eval.py` separately uses to build the real-only exclusion matrix.

## GPU / vLLM requirements

Only Refining Simulation needs a CUDA GPU with `vllm` installed separately
(`pip install vllm`, version matched to that machine's CUDA toolkit) — it is not pinned in
`requirements.txt`, which targets the Mac/CPU baseline environment.
