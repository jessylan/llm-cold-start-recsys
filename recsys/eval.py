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
import numpy as np
import scipy.sparse as sparse
from scipy.stats import rankdata
from implicit.evaluation import ndcg_at_k

from recsys.protocol import RetrievalModel

METRICS = ["NDCG", "Recall", "HitRate", "AUC"]
METRICS_MODE_B = ["NDCG", "Precision", "Recall", "HitRate", "AUC"]


def _check_model(model):
    """Fails fast, with a clear message, on e.g. a raw implicit.als.AlternatingLeastSquares
    passed instead of cf.ALSModel -- it has fit()/recommend() but not score_matrix()/fold_in(),
    which would otherwise only surface as an AttributeError deep inside a sweep."""
    if not isinstance(model, RetrievalModel):
        raise TypeError(
            f"{model!r} does not satisfy protocol.RetrievalModel "
            "(needs fit, recommend, score_matrix, and fold_in)"
        )


# ---------------------------------------------------------------------------
# Metric primitives -- model-agnostic, operate on (model, train_matrix, test_matrix) directly
# ---------------------------------------------------------------------------

def recall_and_hit_rate_at_k(model, train_matrix, test_matrix, K=100):
    """Macro-averaged Recall@K and HitRate@K. Recall@K is the fraction of a user's held-out
    items that appear in their top-K; HitRate@K is whether at least one held-out item appears
    in their top-K. Returns (recall, hit_rate, n_eval_users)."""
    n_users = train_matrix.shape[0]
    user_ids = np.arange(n_users)
    rec_ids, _ = model.recommend(user_ids, train_matrix[user_ids], N=K, filter_already_liked_items=True)

    test_csr = test_matrix.tocsr()
    recalls, hits = [], []
    for u in range(n_users):
        start, end = test_csr.indptr[u], test_csr.indptr[u + 1]
        relevant = set(test_csr.indices[start:end].tolist())
        if not relevant:
            continue
        recommended = set(rec_ids[u].tolist())
        n_hit = len(relevant & recommended)
        recalls.append(n_hit / len(relevant))
        hits.append(1.0 if n_hit > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(hits)), len(recalls)


def auc_at_full(score_matrix, train_matrix, test_matrix):
    """Full-catalog pairwise AUC: for each user with a held-out item, the probability a random
    positive (their held-out item(s)) outranks a random negative (every item not already in
    train_matrix for that user), via the Mann-Whitney rank-sum shortcut -- ties get
    average-rank credit. score_matrix is the full (n_users x n_items) score array; no K is
    involved."""
    train_csr = train_matrix.tocsr()
    test_csr = test_matrix.tocsr()
    n_users_local, n_items_local = score_matrix.shape
    aucs = []
    for u in range(n_users_local):
        pos_start, pos_end = test_csr.indptr[u], test_csr.indptr[u + 1]
        positive_ids = test_csr.indices[pos_start:pos_end]
        if len(positive_ids) == 0:
            continue
        train_start, train_end = train_csr.indptr[u], train_csr.indptr[u + 1]
        excluded_ids = train_csr.indices[train_start:train_end]

        candidate_mask = np.ones(n_items_local, dtype=bool)
        candidate_mask[excluded_ids] = False
        candidate_ids = np.flatnonzero(candidate_mask)
        is_positive = np.isin(candidate_ids, positive_ids)
        n_pos = int(is_positive.sum())
        n_neg = len(candidate_ids) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        ranks = rankdata(score_matrix[u, candidate_ids])
        rank_sum_pos = ranks[is_positive].sum()
        aucs.append((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Warm-item reference score (Section 6) -- "what does normal look like on this system?"
# ---------------------------------------------------------------------------

def reference_scores(models, dataset, K=100):
    """Evaluates `models` (a list of independently-fit instances of one method -- len=1 for a
    deterministic method like Popularity, len=N_SEEDS for a stochastic method like ALS) against
    dataset.ref_test, using each model's own natively-fit representations. No fold-in involved.
    Returns {"mean": {...}, "std": {...}, "n_eval": int}."""
    per_seed = {m: [] for m in METRICS}
    n_eval = None
    for model in models:
        _check_model(model)
        per_seed["NDCG"].append(ndcg_at_k(model, dataset.ref_train, dataset.ref_test, K=K, show_progress=False))
        recall, hit, n_eval = recall_and_hit_rate_at_k(model, dataset.ref_train, dataset.ref_test, K=K)
        per_seed["Recall"].append(recall)
        per_seed["HitRate"].append(hit)
        per_seed["AUC"].append(auc_at_full(model.score_matrix(), dataset.ref_train, dataset.ref_test))
    return {
        "mean": {m: float(np.mean(per_seed[m])) for m in METRICS},
        "std": {m: float(np.std(per_seed[m])) for m in METRICS},
        "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# Mode A -- user-centric warm-up curve (Sections 7 & 9)
# ---------------------------------------------------------------------------

def evaluate_at_k(model, dataset, k, K=100):
    """Folds `model` in at reveal level k and evaluates against the fixed reserved test set.
    Works identically whether `model.fold_in` did a partial update (ALS) or a full refit
    (Popularity) -- this function only calls the folded model's recommend()/score_matrix()."""
    _check_model(model)
    train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
    folded = model.fold_in(dataset, k)
    ndcg = ndcg_at_k(folded, train_k, dataset.test_matrix, K=K, show_progress=False)
    recall, hit, n_eval = recall_and_hit_rate_at_k(folded, train_k, dataset.test_matrix, K=K)
    auc = auc_at_full(folded.score_matrix(), train_k, dataset.test_matrix)
    return {"NDCG": ndcg, "Recall": recall, "HitRate": hit, "AUC": auc, "n_eval": n_eval}


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


def evaluate_item_to_user_at_k(model, dataset, ctx, k, K=10):
    """Mode B: for each cold item, ranks users by predicted affinity (model.fold_in(dataset, k)
    read via score_matrix(), sliced to the cold-item columns), excluding users already in the
    item's revealed pool. NDCG/Precision/Recall/HitRate use the Downsampled Test Set (ctx);
    AUC uses the full reserved test set and needs no K at all.

    Generalizes both of the original notebook's item_to_user_scores (ALS) and
    item_to_user_activity_baseline (Activity) into one function: ALS's fold_in returns a
    partially-updated model, Activity's fold_in returns self unchanged (see pop.ActivityModel);
    this function only ever calls score_matrix()/fold_in(), identically either way.

    Ranks by sorting each item's full (unmasked) score column once, then filtering out
    already-revealed users while preserving that order -- NOT by masking revealed users to
    -inf before sorting. The two are only equivalent when ties are impossible: masking first
    can change which pivot values a non-stable sort (numpy's default) sees, which can shuffle
    the relative order of exactly-tied entries and occasionally admit a different top-K set.
    ALS's real-valued dot products are effectively never tied, so this made no difference for
    ALS, but ActivityModel's integer interaction counts have many exact ties, where it did.
    Sort-then-filter matches the original notebook's Activity-baseline approach exactly (a
    single global ranking, filtered per item) and is what's used here for every model.
    """
    _check_model(model)
    item_ids = dataset.cold_item_ids
    folded = model.fold_in(dataset, k)
    scores = folded.score_matrix()[:, item_ids]  # n_users x n_cold

    downsampled_csc = ctx.downsampled_test_matrix.tocsc()
    full_csc = dataset.test_matrix.tocsc()
    ndcgs, precisions, recalls, hits, aucs = [], [], [], [], []
    for col, item_idx in enumerate(item_ids):
        d_start, d_end = downsampled_csc.indptr[item_idx], downsampled_csc.indptr[item_idx + 1]
        relevant_downsampled = set(downsampled_csc.indices[d_start:d_end].tolist())
        f_start, f_end = full_csc.indptr[item_idx], full_csc.indptr[item_idx + 1]
        relevant_full = set(full_csc.indices[f_start:f_end].tolist())
        if not relevant_downsampled:
            continue
        revealed_users = dataset.reveal_pool[item_idx][:k]
        item_scores = scores[:, col]
        revealed_set = set(revealed_users.tolist())
        full_ranking = np.argsort(-item_scores)
        ranked_users = np.array([u for u in full_ranking.tolist() if u not in revealed_set][:K])
        ranked_set = set(ranked_users.tolist())
        n_hit = len(relevant_downsampled & ranked_set)
        precisions.append(n_hit / len(ranked_set) if len(ranked_set) else 0.0)
        recalls.append(n_hit / len(relevant_downsampled))
        hits.append(1.0 if n_hit > 0 else 0.0)

        ideal_hits = min(len(relevant_downsampled), K)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
        dcg = sum(1.0 / np.log2(r + 2) for r, u in enumerate(ranked_users) if u in relevant_downsampled)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        if relevant_full:
            candidate_mask = np.ones(dataset.n_users, dtype=bool)
            candidate_mask[revealed_users] = False
            candidate_ids = np.flatnonzero(candidate_mask)
            is_positive = np.isin(candidate_ids, list(relevant_full))
            n_pos, n_neg = int(is_positive.sum()), len(candidate_ids) - int(is_positive.sum())
            if n_pos > 0 and n_neg > 0:
                ranks = rankdata(item_scores[candidate_ids])
                rank_sum_pos = ranks[is_positive].sum()
                aucs.append((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {
        "NDCG": float(np.mean(ndcgs)),
        "Precision": float(np.mean(precisions)),
        "Recall": float(np.mean(recalls)),
        "HitRate": float(np.mean(hits)),
        "AUC": float(np.mean(aucs)) if aucs else float("nan"),
        "n_eval": len(recalls),
    }


def sweep_item_to_user(models, dataset, ctx, k_levels, K=10):
    """Mode B analog of sweep(): runs evaluate_item_to_user_at_k across k_levels for `models`,
    averaging across them at each k."""
    curve = {m: {"mean": [], "std": []} for m in METRICS_MODE_B}
    n_eval_per_k = []
    for k in k_levels:
        per_seed = {m: [] for m in METRICS_MODE_B}
        n_eval = None
        for model in models:
            result = evaluate_item_to_user_at_k(model, dataset, ctx, k, K=K)
            for m in METRICS_MODE_B:
                per_seed[m].append(result[m])
            n_eval = result["n_eval"]
        for m in METRICS_MODE_B:
            curve[m]["mean"].append(float(np.mean(per_seed[m])))
            curve[m]["std"].append(float(np.std(per_seed[m])))
        n_eval_per_k.append(n_eval)
    return curve, n_eval_per_k


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
