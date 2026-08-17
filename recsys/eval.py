# This file was created with the assistance of Generative AI.
"""Model-agnostic evaluation: the metric functions, the warm-item reference score, and the two
cold-start "warm-up curve" sweeps (Mode A -- does the item surface in a user's personal top-K;
Mode B -- does the item's ranking surface the users who actually want it).

Every function here takes a `model` (or a list of `model`s, for averaging across seeds)
satisfying protocol.RetrievalModel and a `load.Dataset` -- never a model-specific branch. What
makes this possible: `train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)` is pure
Dataset arithmetic, and `model.fold_in(dataset, k)` already encapsulates whatever
model-specific cold-start mechanism applies (a partial update for ALS, a full refit for
Popularity) behind one return value satisfying the same Protocol.
"""
import time

import numpy as np
import scipy.sparse as sparse

from recsys.protocol import RetrievalModel

METRICS = ["NDCG", "Precision", "Recall", "HitRate", "AUC"]
METRICS_MODE_B = ["NDCG", "Precision", "Recall", "HitRate", "AUC"]


def _check_model(model):
    """Fails fast, with a clear message, on e.g. a raw implicit.als.AlternatingLeastSquares
    passed instead of cf.ALSModel -- it has fit()/recommend() but not fold_in(), which would
    otherwise only surface as an AttributeError deep inside a sweep."""
    if not isinstance(model, RetrievalModel):
        raise TypeError(
            f"{model!r} does not satisfy protocol.RetrievalModel "
            "(needs fit, recommend, and fold_in)"
        )


# ---------------------------------------------------------------------------
# Metric primitives -- model-agnostic, operate on (model, train_matrix, test_matrix) directly
# ---------------------------------------------------------------------------

def mode_a_metrics_at_k(model, train_matrix, test_matrix, K=100):
    """Macro-averaged NDCG@K, Precision@K, Recall@K, and HitRate@K, all from a SINGLE
    model.recommend() call restricted to users who actually have a held-out test item.

    Two savings folded into one function, since they compose. (1) A user with zero test items
    contributes nothing to any of these macro-averaged metrics -- scoring them via recommend()
    is pure wasted work, exact to skip (not an approximation). On the cold-item test set the
    eval-user count is bounded by dataset.test_matrix.nnz (test_size per cold item), a small
    fraction of n_users -- so the saving scales with that ratio, not a fixed factor.
    (2) NDCG used to come from a second, separate call to implicit.evaluation.ndcg_at_k,
    which duplicates exactly the recommend() call this function already needs -- computing NDCG
    directly here (same DCG/IDCG formula used in the Mode B metrics)
    means ONE recommend() call serves all four metrics instead of two. Confirmed empirically
    that implicit's ndcg_at_k already does (1) internally but our old precision_recall_hit_at_k
    did not -- this function does both, combined."""
    test_csr = test_matrix.tocsr()
    eval_users = np.flatnonzero(np.diff(test_csr.indptr))
    if len(eval_users) == 0:
        return {"NDCG": float("nan"), "Precision": float("nan"), "Recall": float("nan"),
                "HitRate": float("nan"), "n_eval": 0}

    rec_ids, _ = model.recommend(eval_users, train_matrix[eval_users], N=K, filter_already_liked_items=True)

    return mode_a_metrics_from_recs(rec_ids, eval_users, test_csr, K)


def mode_a_metrics_from_recs(rec_ids, eval_users, test_csr, K):
    """Vectorized NDCG@K / Precision@K / Recall@K / HitRate@K from precomputed top-K recs
    (rec_ids: (len(eval_users), K) global item ids, row-aligned to eval_users; test_csr: CSR).

    Iterates each eval user's sparse held-out items (nnz ~1.2/user on the cold set) instead of
    users x K -- ~80x fewer ops than the per-user Python loop, identical numbers (verified bench_9).
    Shared by both mode_a_metrics_at_k (generic sweep) and sweep_mode_a_cached, so the metric has a
    single definition."""
    sub = test_csr[eval_users]                       # (n_eval, n_items); row r == eval_users[r]'s tests
    rows, items = sub.nonzero()                       # (nnz,): row aligned to eval_users, item global id
    return _mode_a_metrics_core(rec_ids, rows, items, np.diff(sub.indptr), K)


def _mode_a_metrics_core(rec_ids, rows, items, rel, K):
    """Metric math given the precomputed held-out structure (rows: eval-local user row per held-out
    item; items: its global id; rel: held-out count per eval user). Factored out so a sweep can
    build (rows, items, rel) ONCE from the fixed test set and reuse it across every (seed, k)."""
    n_eval = len(rel)
    match = rec_ids[rows] == items[:, None]          # (nnz, K): did each held-out item land in top-K?
    hitmask = match.any(axis=1)
    ranks = match.argmax(axis=1)                      # 0-indexed rank of the hit (valid where hitmask)

    n_hit = np.bincount(rows[hitmask], minlength=n_eval)          # hits per eval user
    dcg_contrib = np.zeros(len(rows))
    dcg_contrib[hitmask] = 1.0 / np.log2(ranks[hitmask] + 2)
    dcg = np.bincount(rows, weights=dcg_contrib, minlength=n_eval)
    ideal_gains = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(K) + 2))])  # ideal_gains[m]=IDCG@m
    idcg = ideal_gains[np.minimum(rel, K)]
    ndcg = np.where(idcg > 0, dcg / idcg, 0.0)

    return {
        "NDCG": float(ndcg.mean()),
        "Precision": float((n_hit / K).mean()),
        "Recall": float((n_hit / rel).mean()),
        "HitRate": float((n_hit > 0).mean()),
        "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# Warm-item reference score (Section 6) -- "what does normal look like on this system?"
# ---------------------------------------------------------------------------

def ceiling_reference(models, dataset, K=100):
    """Within-item warm reference (replaces the cross-population Section-6 line): each cold item
    folded in with ALL its pre-test interactions (dataset.ceiling_pool), evaluated on the SAME
    last-test_size reserved test the warm-up curve uses. It's the curve's own asymptote -- same
    items, same test -- so it's degree-matched by construction, not a different item population.

    Top-K uses the full warm_cold pool (model.recommend); AUC uses the degree-matched pool
    (model._auc_*). Models without a candidate pool (e.g. Popularity) leave AUC NaN here -- fill it
    via pop_ceiling_auc (the notebook does exactly this in Section 6)."""
    ceiling_train = dataset.ref_train + dataset.ceiling_matrix()
    per_seed = {m: [] for m in METRICS}
    n_eval = None
    for model in models:
        _check_model(model)
        folded = model.fold_in_ceiling(dataset)
        mode_a = mode_a_metrics_at_k(folded, ceiling_train, dataset.test_matrix, K=K)
        for m in ["NDCG", "Precision", "Recall", "HitRate"]:
            per_seed[m].append(mode_a[m])
        n_eval = mode_a["n_eval"]
        if getattr(model, "_auc_candidate_ids", None) is not None:
            from recsys import gpu_retrieval
            ev_u = np.flatnonzero(np.diff(dataset.test_matrix.tocsr().indptr))
            a = gpu_retrieval.mode_a_auc(
                None, None,                     # scores come from `source`, not from factors
                model._auc_candidate_ids, model._auc_global_to_local,
                ev_u, ceiling_train[ev_u], dataset.test_matrix[ev_u],
                device=model._device, source=folded.auc_pool_source())
            per_seed["AUC"].append(float(np.nanmean(a)))
        else:
            per_seed["AUC"].append(float("nan"))
    return {
        "mean": {m: float(np.mean(per_seed[m])) for m in METRICS},
        "std": {m: float(np.std(per_seed[m])) for m in METRICS},
        "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# Popularity / Activity AUC -- the floors' AUC, on the SAME basis as ALS (warm_cold pool for Mode A,
# all users for Mode B) so they're directly comparable on the plots. Both are rank-1 score models
# (Popularity scores item i as popularity[i], user-independent; Activity scores user u as
# user_activity[u], item-independent), so they reuse the ALS AUC kernels unchanged via _rank1
# factors. Deterministic -> std 0. Heavy integer-count ties handled by the kernels' average ranks.
# ---------------------------------------------------------------------------

def _rank1(vals):
    """(n, 1) float32 factors whose dot with an all-ones vector reproduces `vals` -- lets the rank-1
    Popularity/Activity score models reuse mode_a_auc*/mode_b_auc with no special-casing."""
    return np.ascontiguousarray(vals, dtype=np.float32)[:, None]


def pop_ceiling_auc(pop_ceiling_model, dataset, pool_model, device=None):
    """Popularity within-item ceiling AUC over the DEGREE-MATCHED pool -- the Popularity dual of
    ceiling_reference's AUC. `pop_ceiling_model` is Popularity refit on ref_train + ceiling_matrix
    (i.e. base_pop_model.fold_in_ceiling(dataset)); evaluated on the last-test_size test."""
    from recsys import gpu_retrieval
    device = device or pool_model._device
    ceiling_train = dataset.ref_train + dataset.ceiling_matrix()
    ev_u = np.flatnonzero(np.diff(dataset.test_matrix.tocsr().indptr))
    a = gpu_retrieval.mode_a_auc(_rank1(np.ones(dataset.n_users)), _rank1(pop_ceiling_model.popularity),
                                 pool_model._auc_candidate_ids, pool_model._auc_global_to_local,
                                 ev_u, ceiling_train[ev_u], dataset.test_matrix[ev_u], device=device)
    return float(np.nanmean(a))


def pop_auc_curve(pop_model, dataset, pool_model, k_levels, device=None):
    """Popularity Mode A AUC warm-up curve over pool_model's DEGREE-MATCHED pool (same basis as ALS
    Mode A AUC). Rank-1: user factors = ones; warm factors = warm popularity (frozen -- warm items are
    never revealed); cold factors at k = the cold items' revealed-user count at that level (==
    Popularity's fold-in popularity, since cold items have no ref_train interactions). Degree-matching
    the pool (warm competitors with degree >= auc_min_degree) is what removes Popularity's
    "beat the low-degree tail" AUC inflation. Returns {"mean", "std"} with std all 0 (exact)."""
    from recsys import gpu_retrieval
    device = device or pool_model._device
    warm_ids, cold_ids = pool_model._auc_warm_ids, pool_model._cold_ids
    eval_users = np.flatnonzero(np.diff(dataset.test_matrix.tocsr().indptr))
    cold_fac_by_k = [_rank1(np.asarray((dataset.revealed_matrix_at_k(k)[:, cold_ids] > 0).sum(axis=0)).ravel())
                     for k in k_levels]
    auc = gpu_retrieval.mode_a_auc_sweep(
        _rank1(np.ones(dataset.n_users)), _rank1(pop_model.popularity[warm_ids]), cold_fac_by_k,
        pool_model._auc_warm_global_to_local, pool_model._cold_global_to_local,
        eval_users, dataset.ref_train[eval_users],
        [dataset.revealed_matrix_at_k(k)[eval_users] for k in k_levels],
        dataset.test_matrix[eval_users], device=device)
    means = [float(np.nanmean(auc[ki])) for ki in range(len(k_levels))]
    return {"mean": means, "std": [0.0] * len(k_levels)}


def activity_auc_curve(activity_model, dataset, k_levels, device="cuda:0"):
    """Activity Mode B AUC warm-up curve over all users (same basis as ALS Mode B AUC). Rank-1: cold
    factors = ones; user factors = user_activity (frozen -- fold_in is a no-op), so only the
    revealed-user exclusion changes with k. Returns {"mean", "std"} with std all 0 (exact)."""
    from recsys import gpu_retrieval
    cold_ids = dataset.cold_item_ids
    cold_test_csc = dataset.test_matrix[:, cold_ids].tocsc()
    uf, ones_cf = _rank1(activity_model.user_activity), _rank1(np.ones(len(cold_ids)))
    means = []
    for k in k_levels:
        rev = dataset.revealed_matrix_at_k(k)[:, cold_ids].tocsc()
        means.append(float(np.nanmean(gpu_retrieval.mode_b_auc(uf, ones_cf, cold_test_csc, rev, device=device))))
    return {"mean": means, "std": [0.0] * len(k_levels)}


# ---------------------------------------------------------------------------
# Mode A -- user-centric warm-up curve (Sections 7 & 9)
# ---------------------------------------------------------------------------

def evaluate_at_k(model, dataset, k, K=100):
    """Folds `model` in at reveal level k and evaluates against the fixed reserved test set.
    Works identically whether `model.fold_in` did a partial update (ALS) or a full refit
    (Popularity) -- this function only calls the folded model's recommend().

    AUC is the degree-matched candidate-pool average-rank AUC (GPU, model._auc_* pool), matching
    the Section-9 sweep -- so e.g. the Section-7 k=0 ALS AUC is a real number, not NaN. Models
    without a candidate pool (Popularity) return NaN here; their AUC is filled via pop_auc_curve."""
    _check_model(model)
    train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
    folded = model.fold_in(dataset, k)
    mode_a = mode_a_metrics_at_k(folded, train_k, dataset.test_matrix, K=K)
    if getattr(model, "_auc_candidate_ids", None) is not None:
        from recsys import gpu_retrieval
        ev_u = np.flatnonzero(np.diff(dataset.test_matrix.tocsr().indptr))
        a = gpu_retrieval.mode_a_auc(
            None, None,                         # scores come from `source`, not from factors
            model._auc_candidate_ids, model._auc_global_to_local,
            ev_u, train_k[ev_u], dataset.test_matrix[ev_u],
            device=model._device, source=folded.auc_pool_source())
        auc = float(np.nanmean(a))
    else:
        auc = float("nan")
    return {**mode_a, "AUC": auc}


def sweep(models, dataset, k_levels, K=100):
    """Runs evaluate_at_k across k_levels for `models` (independently-fit instances of one
    method), averaging across them at each k. Returns (curve, n_eval_per_k) where
    curve[metric] = {"mean": [...], "std": [...]}, one entry per k in k_levels."""
    curve = {m: {"mean": [], "std": []} for m in METRICS}
    n_eval_per_k = []
    for k in k_levels:
        per_seed = {m: [] for m in METRICS}
        n_eval = None
        for model in models:
            result = evaluate_at_k(model, dataset, k, K=K)
            for m in METRICS:
                per_seed[m].append(result[m])
            n_eval = result["n_eval"]
        for m in METRICS:
            curve[m]["mean"].append(float(np.mean(per_seed[m])))
            curve[m]["std"].append(float(np.std(per_seed[m])))
        n_eval_per_k.append(n_eval)
    return curve, n_eval_per_k


def sweep_mode_a_cached(models, dataset, k_levels, K=100, verbose=True, with_auc=True,
                        on_recs=None):
    """Cached Mode-A warm-up sweep for GPU warm_cold ALS models (report S8(b)). Builds each seed's
    warm top-N once (warm factors are frozen across the reveal sweep) and reuses it for every k,
    scoring only the cold items (len(dataset.cold_item_ids)) per level. Produces the identical NDCG/Precision/Recall/HitRate
    curve as sweep() -- verified bit-for-bit by bench_8 (metrics) and bench_8b (recommended ids) --
    at ~L-fold less recommend work (L = number of k-levels). Requires models prepared with
    candidates='warm_cold' (build_warm_cache / recommend_cached on cf.ALSModel).

    AUC (with_auc=True) is the candidate-pool average-rank AUC from gpu_retrieval.mode_a_auc_sweep,
    which likewise reuses ONE warm score-sort per user across all k (only the cold block re-ranks per
    level). It's the informative Mode A metric where HitRate/NDCG saturate near zero: cold items
    ranked against the DEGREE-MATCHED pool (model._auc_warm_ids -- warm items with ref_train degree
    >= auc_min_degree -- plus cold), no dense matrix, no sampling (so the Krichene-Rendle
    sampled-metric critique doesn't apply). Degree-matching is reserved for AUC; the top-K path above
    still uses the full warm_cold pool. Verified exact vs scipy.rankdata in bench_14 (pre-degree-match).
    Costs ~2.5x the top-K sweep; pass with_auc=False to skip it. Same (curve, n_eval_per_k) shape as sweep().

    `on_recs(ki, si, rec)` -- optional callback, handed the top-K ids this sweep already computed
    for k_levels[ki] and models[si], before they are reduced to metrics. It exists so a second
    measurement over the SAME recommendation lists costs no extra retrieval: `equity_metrics`
    passes `ExposureAccumulator.add` here, and its provider-exposure curve is then guaranteed to
    describe exactly the lists this NDCG curve describes. Deliberately generic -- eval.py must not
    learn what a provider is. Its cost shows up as its own `on_recs` timing bucket rather than
    hiding inside `recommend`."""
    from recsys import gpu_retrieval
    t = {"setup": 0.0, "warm": 0.0, "foldin": 0.0, "recommend": 0.0, "metric": 0.0, "auc": 0.0,
         "on_recs": 0.0}
    _t = time.perf_counter()
    test_csr = dataset.test_matrix.tocsr()
    eval_users = np.flatnonzero(np.diff(test_csr.indptr))
    warm_liked = dataset.ref_train[eval_users]                      # k-invariant warm already-liked
    train_k_eval = [(dataset.ref_train + dataset.revealed_matrix_at_k(k))[eval_users] for k in k_levels]
    # Held-out test structure is fixed across the whole sweep -- build it once and reuse per eval.
    _sub = test_csr[eval_users]
    m_rows, m_items = _sub.nonzero()
    m_rel = np.diff(_sub.indptr)
    if with_auc:
        # AUC inputs (seed-invariant): warm/cold exclusions split (ref_train is all-warm, revealed is
        # all-cold), held-out cold positives. The per-k cold factors are captured from the fold-ins
        # the top-K loop already does, so AUC adds no extra fold_in work.
        warm_excl = dataset.ref_train[eval_users]
        cold_excl_by_k = [dataset.revealed_matrix_at_k(k)[eval_users] for k in k_levels]
        test_eval = dataset.test_matrix[eval_users]
    t["setup"] = time.perf_counter() - _t

    results = [[None] * len(models) for _ in k_levels]
    auc_seed = [[float("nan")] * len(models) for _ in k_levels]
    for si, model in enumerate(models):
        model.load_factors_gpu()                                    # this seed's factors resident on GPU
        _t = time.perf_counter()
        model.build_warm_cache(eval_users, warm_liked, K)           # once per seed
        t["warm"] += time.perf_counter() - _t
        cold_by_k = []
        for ki, k in enumerate(k_levels):
            _t = time.perf_counter(); folded = model.fold_in(dataset, k); t["foldin"] += time.perf_counter() - _t
            if with_auc:
                cold_by_k.append(folded.cold_block_source())
            _t = time.perf_counter(); rec = folded.recommend_cached(eval_users, train_k_eval[ki], K); t["recommend"] += time.perf_counter() - _t
            if on_recs is not None:
                _t = time.perf_counter(); on_recs(ki, si, rec); t["on_recs"] += time.perf_counter() - _t
            _t = time.perf_counter(); results[ki][si] = _mode_a_metrics_core(rec, m_rows, m_items, m_rel, K); t["metric"] += time.perf_counter() - _t
        if with_auc:
            _t = time.perf_counter()
            # Degree-matched pool for AUC (reserve degree-matching for AUC only): rank against
            # _auc_warm_ids (warm degree >= auc_min_degree) + cold, not the full warm pool. warm_excl
            # is the full ref_train; items outside the degree-matched pool map to -1 and are ignored.
            auc_arr = gpu_retrieval.mode_a_auc_sweep(
                None, None, None,               # scores come from the sources below, not from factors
                model._auc_warm_global_to_local, model._cold_global_to_local,
                eval_users, warm_excl, cold_excl_by_k, test_eval, device=model._device,
                warm_source=model.warm_auc_source(), cold_sources_by_k=cold_by_k)
            t["auc"] += time.perf_counter() - _t
            for ki in range(len(k_levels)):
                auc_seed[ki][si] = float(np.nanmean(auc_arr[ki]))
        model.free_factors_gpu()                                    # free VRAM before the next seed

    curve = {m: {"mean": [], "std": []} for m in METRICS}
    n_eval_per_k = []
    for ki in range(len(k_levels)):
        for m in ["NDCG", "Precision", "Recall", "HitRate"]:
            vals = [results[ki][si][m] for si in range(len(models))]
            curve[m]["mean"].append(float(np.mean(vals)))
            curve[m]["std"].append(float(np.std(vals)))
        avals = [auc_seed[ki][si] for si in range(len(models))]
        curve["AUC"]["mean"].append(float(np.mean(avals)) if with_auc else float("nan"))
        curve["AUC"]["std"].append(float(np.std(avals)) if with_auc else float("nan"))
        n_eval_per_k.append(results[ki][0]["n_eval"])
    if verbose:
        # `on_recs` is omitted when unused, so a sweep without the hook prints exactly what it
        # always did.
        print("[sweep_mode_a_cached] "
              + "  ".join(f"{k}={v:.1f}s" for k, v in t.items() if k != "on_recs" or v > 0)
              + f"  total={sum(t.values()):.1f}s")
    return curve, n_eval_per_k


def cold_item_degree_buckets(dataset, n_buckets=3):
    """Assign each cold item to an equal-count degree bucket by TOTAL interaction count: bucket 0 =
    lowest-degree (tail) ... n_buckets-1 = highest (head). Returns (item_bucket, labels, bucket_info):
    item_bucket is a length-n_items int array giving each cold item's bucket (-1 for non-cold items);
    labels names the buckets; bucket_info[b] = {n_items, deg_min, deg_median, deg_max} describes the
    total-degree range each bucket spans (ceiling-pool size + the fixed test_size)."""
    cold = dataset.cold_item_ids
    test_per_item = int(round(dataset.test_matrix.nnz / max(len(cold), 1)))       # = test_size (5), fixed by the split
    deg = np.array([len(dataset.ceiling_pool[i]) for i in cold]) + test_per_item  # TOTAL interactions per cold item
    order = np.argsort(deg, kind="stable")                       # ascending total degree
    bucket_of_cold = np.empty(len(cold), dtype=np.int64)
    for b, idx in enumerate(np.array_split(order, n_buckets)):   # ~equal counts per bucket
        bucket_of_cold[idx] = b
    item_bucket = np.full(dataset.n_items, -1, dtype=np.int64)
    item_bucket[cold] = bucket_of_cold
    labels = ["tail", "torso", "head"] if n_buckets == 3 else [f"q{b + 1}" for b in range(n_buckets)]
    bucket_info = []
    for b in range(n_buckets):
        db = deg[bucket_of_cold == b]
        bucket_info.append({"n_items": int(db.size), "deg_min": int(db.min()),
                            "deg_median": int(np.median(db)), "deg_max": int(db.max())})
    return item_bucket, labels, bucket_info


def ceiling_metrics_by_bucket(models, dataset, item_bucket, K=100, n_buckets=3):
    """Head/torso/tail stratified retrieval at the within-item ceiling: per-bucket HitRate@K AND
    NDCG@K, averaged over seeds. Answers the long-tail question -- 'even fully warm, are low-degree
    cold items retrieved worse (and ranked lower)?' -- the tail-breakdown secondary panel. Each
    held-out interaction is one (user, cold item) on the last-5 test, so per-interaction HitRate@K =
    fraction retrieved into top-K, and NDCG@K = its rank-discounted version (single relevant item ->
    IDCG=1, so NDCG = mean of 1/log2(rank+2) over hits). Both stratified by the item's degree bucket.

    Top-K over the full warm_cold pool (degree-matching is reserved for AUC, not top-K)."""
    ceiling_train = dataset.ref_train + dataset.ceiling_matrix()
    test_csr = dataset.test_matrix.tocsr()
    eval_users = np.flatnonzero(np.diff(test_csr.indptr))
    sub = test_csr[eval_users]
    rows, items = sub.nonzero()                                  # per held-out interaction: eval-local user row, item id
    buckets = item_bucket[items]
    hit_seed = {b: [] for b in range(n_buckets)}
    ndcg_seed = {b: [] for b in range(n_buckets)}
    for model in models:
        _check_model(model)
        folded = model.fold_in_ceiling(dataset)
        rec_ids, _ = folded.recommend(eval_users, ceiling_train[eval_users], N=K, filter_already_liked_items=True)
        match = (rec_ids[rows] == items[:, None])                # (n_interactions, K)
        hit = match.any(axis=1)                                  # did each held-out item land in its user's top-K
        ranks = match.argmax(axis=1)                             # 0-indexed rank of the hit (valid where hit)
        dcg = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)       # per-interaction NDCG (IDCG=1, single relevant)
        for b in range(n_buckets):
            mask = buckets == b
            hit_seed[b].append(float(hit[mask].mean()) if mask.any() else float("nan"))
            ndcg_seed[b].append(float(dcg[mask].mean()) if mask.any() else float("nan"))
    return {b: {"hitrate": float(np.mean(hit_seed[b])), "hitrate_std": float(np.std(hit_seed[b])),
                "ndcg": float(np.mean(ndcg_seed[b])), "ndcg_std": float(np.std(ndcg_seed[b])),
                "n_interactions": int((buckets == b).sum())} for b in range(n_buckets)}


def top_n_recommendations(model, dataset, user_index, N=5, titles=None):
    """Thin wrapper around Section 8's example -- top-N recommendations for one user, from
    ref_train. `titles` (load.load_titles()'s output) is optional, for display only."""
    _check_model(model)
    ids, scores = model.recommend(user_index, dataset.ref_train[user_index], N=N, filter_already_liked_items=True)
    results = []
    for item_idx, score in zip(ids, scores):
        raw_item_id = dataset.index_to_item[int(item_idx)]
        title = titles.get(raw_item_id, "<unknown>") if titles is not None else None
        results.append({"item_index": int(item_idx), "raw_item_id": raw_item_id, "score": float(score), "title": title})
    return results


# ---------------------------------------------------------------------------
# Mode B -- item-to-user warm-up curve (Section 10): same fold-in, transposed ranking axis
# ---------------------------------------------------------------------------

class ModeBContext:
    """Section-10-specific derived state (not a Dataset field, since it's not needed by
    Sections 1-9): a fixed, randomly downsampled subset of each cold item's reserved test
    users, so Precision@K/Recall@K aren't dominated by the handful of high-r_i items."""

    def __init__(self, downsampled_test_matrix, downsample_size):
        self.downsampled_test_matrix = downsampled_test_matrix
        self.downsample_size = downsample_size


def build_mode_b_context(dataset, downsample_size=5, seed=42):
    """Draws a fixed, random `downsample_size`-user sample from each cold item's reserved test
    pool -- discarding the rest -- for NDCG/Precision/Recall/HitRate. Every eligible item is
    guaranteed at least `downsample_size` reserved users by construction. Drawn once, reused
    identically across every k and every seed model."""
    downsample_rng = np.random.default_rng(seed)
    full_test_csc = dataset.test_matrix.tocsc()
    rows, cols = [], []
    for item_idx in dataset.cold_item_ids:
        start, end = full_test_csc.indptr[item_idx], full_test_csc.indptr[item_idx + 1]
        item_test_users = full_test_csc.indices[start:end]
        chosen = downsample_rng.choice(item_test_users, size=downsample_size, replace=False)
        for u in chosen:
            rows.append(u)
            cols.append(item_idx)
    downsampled_test_matrix = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(dataset.n_users, dataset.n_items)
    )
    return ModeBContext(downsampled_test_matrix, downsample_size)


def mode_b_ceiling(dataset, ctx, K=10):
    """Best possible score for every Mode B metric under a hypothetically perfect ranking.
    NDCG/HitRate/AUC ceilings are trivially 1.0 by construction. Precision@K and Recall@K
    actually depend on r_i (the item's reserved-test-user count): min(K, r_i)/K and
    min(K, r_i)/r_i respectively -- on the Downsampled Test Set, r_i is fixed at
    ctx.downsample_size for every item, so these are the same number for every item."""
    downsampled_csc = ctx.downsampled_test_matrix.tocsc()
    max_precisions, max_recalls, max_hitrates = [], [], []
    for item_idx in dataset.cold_item_ids:
        start, end = downsampled_csc.indptr[item_idx], downsampled_csc.indptr[item_idx + 1]
        r_i = end - start
        if r_i == 0:
            continue
        max_precisions.append(min(K, r_i) / K)
        max_recalls.append(min(K, r_i) / r_i)
        max_hitrates.append(1.0 if r_i >= 1 else 0.0)
    return {
        "NDCG": 1.0,
        "Precision": float(np.mean(max_precisions)),
        "Recall": float(np.mean(max_recalls)),
        "HitRate": float(np.mean(max_hitrates)),
        "AUC": 1.0,
    }


# ---------------------------------------------------------------------------
# Mode B on GPU -- cold-column scoring, so it runs at Amazon-Books scale (the old dense
# score_matrix() Mode B path did not fit at this scale and has been removed; report S8 Mode B).
# ---------------------------------------------------------------------------

def _mode_b_metrics_from_topk(topk_users, cold_item_ids, downsampled_csc, K):
    """NDCG/Precision/Recall/HitRate@K on the Downsampled Test Set from each cold item's ordered
    top-K users (topk_users: (n_cold, K), -1-padded). Same formulas as the Mode B metrics, fed
    precomputed rankings. AUC is NaN here -- it needs the full-set ranking (see mode_b_auc)."""
    ndcgs, precisions, recalls, hits = [], [], [], []
    for col, item_idx in enumerate(cold_item_ids):
        d0, d1 = downsampled_csc.indptr[item_idx], downsampled_csc.indptr[item_idx + 1]
        relevant = set(downsampled_csc.indices[d0:d1].tolist())
        if not relevant:
            continue
        ranked = [u for u in topk_users[col].tolist() if u >= 0]     # drop -1 padding
        ranked_set = set(ranked)
        n_hit = len(relevant & ranked_set)
        precisions.append(n_hit / len(ranked_set) if ranked_set else 0.0)
        recalls.append(n_hit / len(relevant))
        hits.append(1.0 if n_hit > 0 else 0.0)
        ideal_hits = min(len(relevant), K)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
        dcg = sum(1.0 / np.log2(r + 2) for r, u in enumerate(ranked) if u in relevant)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return {"NDCG": float(np.mean(ndcgs)), "Precision": float(np.mean(precisions)),
            "Recall": float(np.mean(recalls)), "HitRate": float(np.mean(hits)),
            "AUC": float("nan"), "n_eval": len(recalls)}


def _mode_b_topk_activity(user_activity, cold_revealed_csc, K):
    """ActivityModel's Mode B ranking: users ordered by raw activity, the SAME global ranking for
    every cold item, minus that item's revealed users. Numpy (no matmul -- the score is
    item-independent). Vectorized like pop.recommend, on the transposed axis."""
    ranked = np.argsort(-user_activity)                              # global user ranking by activity
    n_cold = cold_revealed_csc.shape[1]
    rev = cold_revealed_csc.tocsc()
    max_rev = int(np.diff(rev.indptr).max()) if n_cold else 0
    M = min(K + max_rev, len(ranked))                                # >= K unrevealed guaranteed in top-M
    topM = ranked[:M]
    pos = np.full(len(ranked), -1, dtype=np.int64); pos[topM] = np.arange(M)
    out = np.full((n_cold, K), -1, dtype=np.int64)
    for j in range(n_cold):
        loc = pos[rev.indices[rev.indptr[j]:rev.indptr[j + 1]]]
        loc = loc[loc >= 0]
        valid = np.ones(M, dtype=bool); valid[loc] = False
        take = np.nonzero(valid & (np.cumsum(valid) <= K))[0]
        out[j, :len(take)] = topM[take]
    return out


def sweep_item_to_user_gpu(models, dataset, ctx, k_levels, K=10, verbose=True, with_auc=True):
    """GPU Mode B sweep -- the scale-safe item-to-user warm-up sweep. Scores only the
    cold-item columns (never the dense n_users x n_items matrix), so it runs on Amazon Books. Same
    (curve, n_eval_per_k) shape. Handles GPU ALS models (via gpu_retrieval.mode_b_topk_users) and
    the ActivityModel floor (via _mode_b_topk_activity).

    AUC (with_auc=True, factor/ALS models only) is the full-user-population average-rank AUC from
    gpu_retrieval.mode_b_auc -- the non-saturating companion to the four Downsampled Test Set
    metrics, reusing the folded cold factors this sweep already computes (no extra fold-in). Mode B
    has no cross-k cache (the cold-item query folds), so each level re-ranks the users; costs ~2.8x
    the top-K sweep (bench_15). The ActivityModel floor's AUC stays NaN (ranked by global activity,
    not factors); pass with_auc=False to skip it entirely."""
    from recsys import gpu_retrieval
    t = {"setup": 0.0, "foldin": 0.0, "score": 0.0, "metric": 0.0, "auc": 0.0}
    _t = time.perf_counter()
    downsampled_csc = ctx.downsampled_test_matrix.tocsc()
    cold_ids = dataset.cold_item_ids
    cold_revealed_by_k = [dataset.revealed_matrix_at_k(k)[:, cold_ids].tocsc() for k in k_levels]
    is_activity = hasattr(models[0], "user_activity")
    do_auc = with_auc and not is_activity
    cold_test_csc = dataset.test_matrix[:, cold_ids].tocsc() if do_auc else None   # positives, k-invariant
    t["setup"] = time.perf_counter() - _t

    results = [[None] * len(models) for _ in k_levels]
    auc_seed = [[float("nan")] * len(models) for _ in k_levels]
    for si, model in enumerate(models):
        if not is_activity:
            model.load_factors_gpu(items=False)                     # Mode B only needs user factors resident
        for ki, k in enumerate(k_levels):
            if is_activity:
                _t = time.perf_counter()
                topk_users = _mode_b_topk_activity(model.user_activity, cold_revealed_by_k[ki], K)
                t["score"] += time.perf_counter() - _t
            else:
                _t = time.perf_counter(); folded = model.fold_in(dataset, k); t["foldin"] += time.perf_counter() - _t
                b_src = folded.mode_b_source()                      # reused by top-K and AUC
                _t = time.perf_counter()
                topk_users = gpu_retrieval.mode_b_topk_users(
                    None, None, cold_revealed_by_k[ki], K, device=model._device, source=b_src)
                t["score"] += time.perf_counter() - _t
                if do_auc:
                    _t = time.perf_counter()
                    a = gpu_retrieval.mode_b_auc(None, None, cold_test_csc,
                                                 cold_revealed_by_k[ki], device=model._device,
                                                 source=b_src)
                    t["auc"] += time.perf_counter() - _t
                    auc_seed[ki][si] = float(np.nanmean(a))
            _t = time.perf_counter()
            results[ki][si] = _mode_b_metrics_from_topk(topk_users, cold_ids, downsampled_csc, K)
            t["metric"] += time.perf_counter() - _t
        if not is_activity:
            model.free_factors_gpu()

    curve = {m: {"mean": [], "std": []} for m in METRICS_MODE_B}
    n_eval_per_k = []
    for ki in range(len(k_levels)):
        for m in METRICS_MODE_B:
            vals = ([auc_seed[ki][si] for si in range(len(models))] if m == "AUC"
                    else [results[ki][si][m] for si in range(len(models))])
            allnan = all(np.isnan(v) for v in vals)
            curve[m]["mean"].append(float("nan") if allnan else float(np.nanmean(vals)))
            curve[m]["std"].append(float("nan") if allnan else float(np.nanstd(vals)))
        n_eval_per_k.append(results[ki][0]["n_eval"])
    if verbose:
        print("[sweep_item_to_user_gpu] " + "  ".join(f"{k}={v:.1f}s" for k, v in t.items())
              + f"  total={sum(t.values()):.1f}s")
    return curve, n_eval_per_k


def mode_b_reference(models, dataset, ctx, K=10, with_auc=True):
    """Mode B within-item warm reference: each cold item folded in with ALL its pre-test interactions
    (ceiling), ranking users, evaluated on the same last-test_size test -- the Mode B analog of
    ceiling_reference. GPU cold-column path (no dense matrix). AUC is over the full user population
    (Mode B AUC is deliberately NOT degree-matched). ALS/factor models only (the Activity floor is
    k-invariant, so its 'ceiling' is just its flat curve value). Returns {"mean", "std"}."""
    from recsys import gpu_retrieval
    cold_ids = dataset.cold_item_ids
    downsampled_csc = ctx.downsampled_test_matrix.tocsc()
    ceiling_rev_csc = dataset.ceiling_matrix()[:, cold_ids].tocsc()      # all pre-test users -> excluded from ranking
    cold_test_csc = dataset.test_matrix[:, cold_ids].tocsc() if with_auc else None
    per_seed = {m: [] for m in METRICS_MODE_B}
    for model in models:
        _check_model(model)
        model.load_factors_gpu(items=False)                              # user factors resident
        folded = model.fold_in_ceiling(dataset)
        b_src = folded.mode_b_source()
        topk_users = gpu_retrieval.mode_b_topk_users(
            None, None, ceiling_rev_csc, K, device=model._device, source=b_src)
        met = _mode_b_metrics_from_topk(topk_users, cold_ids, downsampled_csc, K)
        for m in ["NDCG", "Precision", "Recall", "HitRate"]:
            per_seed[m].append(met[m])
        if with_auc:
            a = gpu_retrieval.mode_b_auc(None, None, cold_test_csc,
                                         ceiling_rev_csc, device=model._device, source=b_src)
            per_seed["AUC"].append(float(np.nanmean(a)))
        else:
            per_seed["AUC"].append(float("nan"))
        model.free_factors_gpu()
    return {"mean": {m: float(np.mean(per_seed[m])) for m in METRICS_MODE_B},
            "std": {m: float(np.std(per_seed[m])) for m in METRICS_MODE_B}}
