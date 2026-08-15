# What does this accuracy number mean?

Before reading a number off a curve, you need three facts about it: **which mode** produced
it, **which candidate pool** it ranked against, and **which reference** it should be
compared to. Get any of those wrong and the number is uninterpretable — an AUC of 0.93 is
excellent against a degree-matched pool and unremarkable against the full catalog.

- **Mode A** ranks *items* for a user. "Did the cold item reach this user's top-K?"
- **Mode B** ranks *users* for a cold item. "Did the item's ranking surface the users who
  actually wanted it?"

They are duals, computed by separate code paths, with separate ceilings and separate floors.
`K` differs too: 100 for Mode A, 10 for Mode B. And `k` — lowercase — is the reveal level,
`0..20`, which is the x-axis of every curve. Both appear in the same signatures.

```mermaid
flowchart TD
    DS(["load.Dataset"]) --> MODE{"which mode?"}

    MODE -->|"Mode A: user -&gt; items"| A0{"which sweep?"}
    MODE -->|"Mode B: item -&gt; users"| B0["build_mode_b_context<br/>fixed 5-user downsample per cold item"]

    A0 -->|"generic, any model"| SW["eval.sweep<br/>loops k, calls evaluate_at_k"]
    A0 -->|"GPU warm_cold models"| SWC["eval.sweep_mode_a_cached<br/>warm top-N cached once per seed"]

    SW --> EAK["eval.evaluate_at_k<br/>train_k = ref_train + revealed_matrix_at_k(k)"]
    EAK --> FOLD["model.fold_in(dataset, k)"]
    FOLD --> REC["folded.recommend(eval_users, train_k, N=K)"]
    REC --> MM["mode_a_metrics_at_k<br/>-&gt; mode_a_metrics_from_recs"]

    SWC --> BWC["model.build_warm_cache<br/>once per seed - warm block is k-invariant"]
    BWC --> KLOOP{"for k in k_levels"}
    KLOOP --> FOLD2["model.fold_in(dataset, k)"]
    FOLD2 --> RCC["folded.recommend_cached<br/>score ~2.7k cold items, merge with warm cache"]
    RCC --> CORE["eval._mode_a_metrics_core<br/>held-out structure built ONCE"]
    CORE --> KLOOP

    MM --> METRICS(["NDCG@100, Precision@100,<br/>Recall@100, HitRate@100"])
    CORE --> METRICS

    EAK -.->|"if model._auc_candidate_ids"| AUC1["gpu_retrieval.mode_a_auc"]
    KLOOP -.->|"with_auc=True"| AUC2["gpu_retrieval.mode_a_auc_sweep<br/>ONE warm sort reused across all k"]
    AUC1 --> POOLA(["AUC over the DEGREE-MATCHED pool:<br/>warm with ref_train degree &gt;= 25, PLUS cold"])
    AUC2 --> POOLA

    B0 --> SWB["eval.sweep_item_to_user_gpu"]
    SWB --> ISACT{"hasattr(model,<br/>'user_activity')?"}
    ISACT -->|"yes - Activity floor"| MBA["eval._mode_b_topk_activity<br/>numpy, no matmul"]
    ISACT -->|"no - factor model"| MBF["model.fold_in(dataset, k)<br/>-&gt; folded.mode_b_source()"]
    MBF --> MBT["gpu_retrieval.mode_b_topk_users<br/>rank ALL users per cold item"]
    MBF -.->|"with_auc, factor models only"| MBAUC["gpu_retrieval.mode_b_auc<br/>full user population, NOT degree-matched"]
    MBA --> MBM["eval._mode_b_metrics_from_topk"]
    MBT --> MBM
    MBM --> METRICSB(["NDCG@10, Precision@10,<br/>Recall@10, HitRate@10"])

    METRICS --> CMP{"compare against what?"}
    METRICSB --> CMP
    CMP -->|"Mode A upper ref"| C1["eval.ceiling_reference<br/>fold_in_ceiling: item's OWN fully-warm state"]
    CMP -->|"Mode A floor"| C2["eval.pop_auc_curve<br/>eval.pop_ceiling_auc"]
    CMP -->|"Mode B upper ref"| C3["eval.mode_b_reference"]
    CMP -->|"Mode B theoretical max"| C4["eval.mode_b_ceiling<br/>NDCG=1.0, AUC=1.0, Prec=min(K,r_i)/K"]
    CMP -->|"Mode B floor"| C5["eval.activity_auc_curve"]
```

## Node reference

| Node | Source | Purpose |
|---|---|---|
| `sweep` | [eval.py:240](../recsys/eval.py:240) | Generic Mode A sweep. Works for any `RetrievalModel`; averages across a list of independently-fit seeds at each `k`. Returns `(curve, n_eval_per_k)`. **The Popularity path**, not a parity check for the cached sweep — see [Design decisions 1](#design-decisions-on-record). |
| `evaluate_at_k` | [eval.py:214](../recsys/eval.py:214) | One reveal level: build `train_k`, fold, recommend, score. |
| `sweep_mode_a_cached` | [eval.py:261](../recsys/eval.py:261) | The fast path every score-source model runs: ALS, CBHCF, Intervention A, Intervention B. Produces the **identical** NDCG/Precision/Recall/HitRate curve as `sweep` — verified bit-for-bit — at ~L-fold less recommend work, L = number of k-levels. Requires `candidates='warm_cold'`, which is why Popularity cannot use it. |
| `mode_a_metrics_at_k` | [eval.py:38](../recsys/eval.py:38) | Macro-averaged NDCG/Precision/Recall/HitRate from a **single** `recommend()` call, restricted to users who actually have a held-out test item. Users with zero test items contribute nothing to a macro average, so skipping them is exact, not an approximation. |
| `_mode_a_metrics_core` | [eval.py:77](../recsys/eval.py:77) | The metric math, factored out so a sweep can build the held-out structure `(rows, items, rel)` once from the fixed test set and reuse it across every `(seed, k)`. |
| `mode_a_auc` | [gpu_retrieval.py:153](../recsys/gpu_retrieval.py:153) | Per-user Mann-Whitney rank-sum AUC over the candidate pool. **Average ranks** via two-sided `searchsorted`, so ties are handled correctly. |
| `mode_a_auc_sweep` | [gpu_retrieval.py:254](../recsys/gpu_retrieval.py:254) | Sweep fast path: the warm score row is frozen within a seed, so it is sorted **once** and reused for every `k`; only the small cold block re-sorts. Chunk-outer / k-inner on purpose, so a chunk's sorted warm scores never exist for all users at once. |
| `ceiling_reference` | [eval.py:107](../recsys/eval.py:107) | The Mode A upper reference. |
| `cold_item_degree_buckets` | [eval.py:343](../recsys/eval.py:343) | Equal-count tail/torso/head buckets by **total** interaction count (ceiling-pool size + the fixed `test_size`). |
| `ceiling_metrics_by_bucket` | [eval.py:367](../recsys/eval.py:367) | Stratified HitRate@K and NDCG@K at the ceiling. Answers "even fully warm, are low-degree items retrieved worse?" |
| `build_mode_b_context` | [eval.py:428](../recsys/eval.py:428) | Draws a fixed random 5-user sample from each cold item's reserved test pool, once, reused across every `k` and seed — so Precision/Recall are not dominated by a few high-`r_i` items. |
| `sweep_item_to_user_gpu` | [eval.py:525](../recsys/eval.py:525) | Mode B sweep. Scores only the cold-item columns, never the dense `n_users x n_items` matrix. |
| `mode_b_topk_users` | [gpu_retrieval.py:379](../recsys/gpu_retrieval.py:379) | Per cold item, rank all users, return the top-N non-revealed. Items as rows so top-K runs along the contiguous axis. |
| `mode_b_auc` | [gpu_retrieval.py:420](../recsys/gpu_retrieval.py:420) | Per-cold-item AUC over the **full** user population. Positives are the full reserved test set, not the downsampled context. |
| `mode_b_ceiling` | [eval.py:449](../recsys/eval.py:449) | Theoretical maximum, not a model result: NDCG and AUC are 1.0 by construction; Precision@K = `min(K, r_i)/K` and Recall@K = `min(K, r_i)/r_i`. |
| `mode_b_reference` | [eval.py:594](../recsys/eval.py:594) | Mode B analog of `ceiling_reference`. Factor models only. |
| `pop_auc_curve` / `pop_ceiling_auc` | [eval.py:172](../recsys/eval.py:172), [eval.py:158](../recsys/eval.py:158) | The Popularity floor's AUC, on the same degree-matched basis as ALS. |
| `activity_auc_curve` | [eval.py:195](../recsys/eval.py:195) | The Mode B floor's AUC, over all users. |
| `_rank1` | [eval.py:152](../recsys/eval.py:152) | `(n, 1)` factors whose dot with an all-ones vector reproduces a value — lets the rank-1 Popularity/Activity models reuse the AUC kernels with no special-casing. |

## The three things that most change a number's meaning

### 1. Which pool the AUC ranked against

**Mode A AUC is degree-matched. Mode B AUC is not.** This asymmetry is deliberate.

Mode A ranks a partially-revealed cold item against warm competitors whose `ref_train`
degree is at least `auc_min_degree = 25` ([cf.py:130](../recsys/cf.py:130)), plus the cold
items. Without that restriction, a Popularity floor scores a misleadingly high AUC simply by
beating the enormous low-degree tail. Degree-matching removes that inflation. **Top-K
metrics do not use the degree-matched pool** — they keep the full `warm_cold` pool
([eval.py:375](../recsys/eval.py:375)). So NDCG@100 and AUC on the same plot rank against
*different* pools, on purpose.

Mode B's ranked set is the whole user population, with no degree matching
([eval.py:598](../recsys/eval.py:598)).

Report Mode A AUC as **"AUC (candidate pool)"**. The pool is the entire deterministic
candidate set retrieved from, not a random negative sample, so the Krichene–Rendle critique
of sampled metrics does not apply ([gpu_retrieval.py:163](../recsys/gpu_retrieval.py:163)).

### 2. Which reference the curve is heading toward

`ceiling_reference` is **not** a different population of warm items. It is the *same* cold
items, folded in with all of their pre-test history (`dataset.ceiling_pool`), evaluated on
the *same* last-5 reserved test. That makes it degree-matched by construction — it is the
curve's own asymptote rather than a cross-population comparison.

For an item with exactly `min_interactions = 25` total interactions, the ceiling equals the
`k=20` curve endpoint. For higher-degree items it uses the extra interactions between
`n_reveal` and `item_total - test_size`, so the reference sits **above** the endpoint. That
gap is the data-abundance headroom, not a modeling failure.

### 3. Whether AUC is `NaN`, and why

`evaluate_at_k` and `ceiling_reference` both gate AUC on
`getattr(model, "_auc_candidate_ids", None) is not None`. Models without candidate pools —
Popularity — return `NaN`, and their AUC is filled separately by `pop_auc_curve` /
`pop_ceiling_auc` so it lands on the same basis as ALS. `_mode_b_metrics_from_topk`
([eval.py:479](../recsys/eval.py:479)) always returns `AUC: NaN`, because Mode B AUC needs
the full-set ranking and comes from `mode_b_auc` instead. The `ActivityModel` floor's Mode B
AUC stays `NaN` throughout — it ranks by global activity, not factors.

A per-user or per-item `NaN` from the AUC kernels themselves means something different:
that user had no in-pool positive, or no negative. Both kernels return `NaN` in that case
and the sweeps aggregate with `np.nanmean`.

### A detail that surprises people

An eval user with no warm history gets an all-zero ALS factor, so **every candidate scores
identically** and the user's positive ties with the entire pool. The correct AUC there is
exactly 0.5, and only average ranks produce it — ordinal `argsort` ranks would return an
arbitrary value in `[0, 1]`. That is why both AUC kernels use two-sided `searchsorted`
rather than a sort position, verified against `scipy.rankdata`.

## Why `K` differs between the modes

Mode A reports at `K = 100`, Mode B at `K_MODE_B = 10`. Both are set in
`steel_thread.ipynb` and both have a rationale recorded there — in notebook markdown rather
than in the modules, which is why neither is visible from `eval.py`.

**Mode A, `K = 100`** — a retrieval-stage cutoff. A cold item competes against ~247k warm
items, so a strict top-10 is hopeless; K=100 asks whether the item surfaced into the
candidate pool at all. The question is recall-shaped, so the cutoff is deliberately loose.

**Mode B, `K = 10`** — Mode B inverts the ranked axis: each cold item is ranked over the
user population (384,339 users), not the item catalogue. The binding constraint is therefore
*not* pool size but the positive count. The leave-last-5-out split fixes `r_i` at exactly 5
for every cold item, so at K=10 the ceilings are `min(K, r_i)/K = 0.5` for Precision and
`min(K, r_i)/r_i = 1.0` for Recall. At K=100 the Precision ceiling would collapse to 0.05.

The two arguments cite the same fact — an enormous pool — and move K in opposite directions,
because they ask different questions: *was this item retrieved anywhere* versus *are these
the right users to show it to*.

### Is `K = 10` still right at Amazon-Books scale?

Yes. The notebook records that the value "was tuned on the earlier small-scale
proof-of-concept" (the retired MovieLens steel thread, 943 users) and asks for it to be
revisited once Mode B ran on GPU at full scale. That run exists —
`outputs/baseline_cf_20260815_012902.json`, 10 seeds, 384,339 users, 2,727 cold items — and
the value holds. Mode B HitRate@10 at `k = 20`:

| arm | HitRate@10 | seed std |
|---|---|---|
| Activity floor | 0.0037 | 0.0000 |
| ALS | 0.0192 | 0.0005 |
| CBHCF | 0.0779 | 0.0005 |

A 21x floor-to-CBHCF spread against a seed std of 0.0005. The saturation risk that motivated
a small K on a 943-user population did not reappear as its opposite — a floor-crushed metric
— at 384k users. K=10 discriminates, and the open action is resolved.

What K=10 does *not* separate is the content arms from one another: CBHCF, Intervention A,
and Intervention B fall within roughly 1.5 seed-sigma of each other on Mode B top-K. That is
not a cutoff artifact — Mode A at K=100 shows the same compression, with Intervention B's
reasoning and random arms identical to four decimal places — so it is a property of those
arms, not of `K`. Choosing a different `K` will not surface a difference that Mode A at 10x
the depth also cannot find.

## Curve shape

Every sweep returns `(curve, n_eval_per_k)` where
`curve[metric] = {"mean": [...], "std": [...]}`, one entry per `k`. `mean` and `std` are
**across seeds** at that `k`, not across users — with `N_SEEDS = 10` in the notebook, `std`
is the ALS initialization spread. Deterministic models (Popularity, Activity) report
`std = 0` exactly.

---

## Drift

Places where prose contradicts the code. Reported, not fixed.

**None outstanding.** All three items this section has carried were **fixed** rather than left
reported. Two belonged to `recsys/equity_metrics.py` and were tracked against its integration
rather than against the evaluation harness: `load_provider_metadata`'s citation of the deleted
`data_cleaning.ipynb` (and the `book_`/`movie_` id convention it described), and `_check_model`'s
reference to a `score_matrix` method the protocol does not declare. Both went in the Step 2
rewrite — neither string survives in the module. The third was `steel_thread.ipynb`'s Mode B
markdown, which still asked for `K_MODE_B` to be revisited once Mode B ran on GPU at full scale;
that run settled it and the notebook prose was corrected to match
[Is `K = 10` still right at Amazon-Books scale?](#is-k--10-still-right-at-amazon-books-scale).
See [06](06-provider-equity.md#drift), which is likewise empty.

## Design decisions on record

Choices that look arbitrary or accidental from the source alone, and are not.

1. **Both Mode A sweeps are live; they are not redundant.** `steel_thread.ipynb` calls
   `ev.sweep` and `ev.sweep_mode_a_cached`, but for different models, not as a parity check.
   `sweep_mode_a_cached` requires models prepared with `candidates='warm_cold'`
   ([eval.py:267](../recsys/eval.py:267)), which `PopularityModel` is not — and Popularity's
   AUC has to be filled separately by `pop_auc_curve` regardless. So `_pop_curve` takes
   `ev.sweep` and every score-source model (ALS, CBHCF, Intervention A, Intervention B) takes
   `sweep_mode_a_cached`. A run's stdout shows this directly: only `[sweep_mode_a_cached]`
   timing lines appear, one per seed per arm.
2. **The `bench_*` parity gates are deliberately not in the repo.** `bench_7`, `bench_8`,
   `bench_8b`, `bench_9`, `bench_14`, `bench_15`, and `bench_16` are cited across
   `gpu_retrieval.py`, `eval.py`, and `scores.py` as the correctness checks on the fast
   paths. They exist and were run; they are development-time verification scripts against a
   local GPU environment, not part of the shipped pipeline, and were left out of the upload
   on purpose. A citation to one is a record of what was checked, not a broken path.
3. **`METRICS` and `METRICS_MODE_B` are identical lists on purpose**
   ([eval.py:19-20](../recsys/eval.py:19)), for two reasons. They are decoupled so either mode
   can gain or drop a metric without editing the other — today's agreement is not a promise.
   And the shared strings name different quantities: Mode B's `AUC` ranks over the full user
   population rather than a degree-matched item pool, and its `Precision` carries a 0.5 ceiling
   by construction. Identical members, non-identical semantics.
4. **`auc_min_degree = 25` matching `min_interactions = 25` is intended coupling.** A warm
   competitor should clear the same degree bar that made an item eligible to be a cold item, so
   the AUC pool compares cold items against items of comparable standing rather than against the
   whole long tail. **This intent is not enforced.** They are two independent literals in two
   modules ([cf.py:130](../recsys/cf.py:130), [load.py:152](../recsys/load.py:152)) with no
   comment linking them, so changing one silently redefines what the degree-matched pool means
   without any error. Deriving one from the other is an open code change, tracked here because
   the coupling is a property of the evaluation design rather than of either module.

## Open questions

Things not determinable from the source.

**None outstanding.** The five questions this section carried have all been answered and moved:
four to [Design decisions on record](#design-decisions-on-record) above, and the choice of `K`
per mode to [Why `K` differs between the modes](#why-k-differs-between-the-modes).
