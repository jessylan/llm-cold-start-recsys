"""Exact GPU top-K retrieval, as a drop-in for implicit's model.recommend().

Why this exists: implicit's recommend() is CPU-only here, and its cost at Amazon-Books scale is
dominated by the top-K *selection*, not the matmul (benchmarked: CPU np.argpartition top-K is
~24s/5k users, ~27x the matmul; cupy.argpartition is only ~5.5x faster because it full-sorts;
torch.topk -- a radix *select* -- is ~490x, exact, ~2 min for the full catalog). So we score on
the GPU (user_factors @ item_factors.T) and select with torch.topk.

Candidate set (D1): we score against a restricted item set (the caller's `candidate_ids` --
warm-retained + cold items, ~388k of the 2.8M) rather than all item_factors rows. The ~2.4M
items dropped by the sparsity pre-filter carry near-zero (regularized) factors and effectively
never enter a top-100, so this is near-exact vs implicit's full-catalog scoring; the parity check
(bench_7) quantifies any divergence. Restricting also keeps the score matrix chunk-sized in VRAM.

torch is imported lazily so CPU-only environments (e.g. the Mac) that never select the GPU
backend don't need it installed.
"""
import numpy as np


# Factor args below may be either numpy arrays (host) OR torch tensors already resident on the GPU
# (see cf.ALSModel.load_factors_gpu). The two helpers below let the same code path use a resident
# tensor with no host->device copy, or transfer from numpy when that's all we have.

def _as_gpu(factors, device):
    """Full factor matrix as a float32 GPU tensor: resident torch tensor returned as-is (no copy),
    numpy transferred once."""
    import torch
    if torch.is_tensor(factors):
        return factors
    return torch.as_tensor(np.ascontiguousarray(factors, dtype=np.float32), device=device)


def _gather_rows(factors, idx, device):
    """Rows `idx` (numpy int) of `factors` as a GPU tensor: a device-side gather if `factors` is a
    resident GPU tensor (no host copy of the big matrix), else a transfer of just those rows."""
    import torch
    if torch.is_tensor(factors):
        return factors[torch.as_tensor(idx, device=device)]
    return torch.as_tensor(np.ascontiguousarray(factors[idx], dtype=np.float32), device=device)


def topk_recommend(user_factors, item_factors, candidate_ids, global_to_local,
                   userid, user_items, N=10, filter_already_liked_items=True,
                   device="cuda:0", chunk=4096):
    """Exact top-N over `candidate_ids`, matching implicit.recommend()'s calling convention.

    user_factors, item_factors : implicit's fitted factor arrays (numpy, float32).
    candidate_ids   : sorted int64 array of global item ids to score against.
    global_to_local : int64 array of length n_items; global_to_local[g] = local column of g in
                      candidate_ids, or -1 if g is not a candidate (used to mask already-liked).
    userid          : scalar or 1-D array of user ids.
    user_items      : CSR (len(userid) x n_items), row i = known likes for userid[i] -- same
                      row-alignment implicit requires.
    Returns (ids, scores): global item ids and their scores, shaped (N,) for a scalar userid or
    (len(userid), N) for an array -- identical to implicit.recommend().
    """
    import torch

    single = np.isscalar(userid)
    userid = np.atleast_1d(np.asarray(userid))
    user_items = user_items.tocsr()
    n_query = len(userid)

    V = _gather_rows(item_factors, candidate_ids, device)        # candidate factors (resident gather or transfer)
    n_cand = V.shape[0]
    N_eff = min(N, n_cand)

    out_ids = np.full((n_query, N), -1, dtype=np.int64)
    out_scores = np.full((n_query, N), -np.inf, dtype=np.float32)

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        uf = _gather_rows(user_factors, userid[start:stop], device)   # this chunk's user factors
        scores = uf @ V.T  # (b, n_cand)

        if filter_already_liked_items:
            # Mask each user's already-liked items that fall inside the candidate set, in one
            # vectorized scatter. The per-user Python loop this replaces dominated Mode A at 193k
            # users x hundreds of calls -- gather all (row, col) coords at once from the CSR instead.
            liked = user_items.indices[user_items.indptr[start]:user_items.indptr[stop]]
            loc = global_to_local[liked]
            ur = np.repeat(np.arange(stop - start), np.diff(user_items.indptr[start:stop + 1]))
            keep = loc >= 0
            if keep.any():
                scores[torch.as_tensor(ur[keep], device=device),
                       torch.as_tensor(loc[keep], device=device)] = float("-inf")

        vals, idx = torch.topk(scores, N_eff, dim=1)  # sorted descending, exact select
        idx_np = idx.cpu().numpy()
        out_ids[start:stop, :N_eff] = candidate_ids[idx_np]
        out_scores[start:stop, :N_eff] = vals.cpu().numpy()

    if single:
        return out_ids[0], out_scores[0]
    return out_ids, out_scores


def cold_merge_recommend(user_factors, cold_factors, cold_ids, cold_global_to_local,
                         warm_top_ids, warm_top_scores, userid, user_items, N,
                         device="cuda:0", chunk=4096):
    """Merge a cached warm top-M with freshly-scored cold items -> exact top-N over warm+cold.

    This is the k-sweep fast path (report S8(b)): the warm block of the score is invariant across
    reveal levels within a seed, so warm_top_ids/warm_top_scores (per-user cached top-M, M>=N) are
    computed once; here we only score the ~5k cold items at the current fold-in level, mask each
    user's revealed cold items, and re-select N over the (warm top-M) + (cold) union. Correct
    because a warm item outside the warm top-M can never enter the final top-N.

    warm_top_ids, warm_top_scores : (len(userid), M) global ids and scores from build_warm_cache.
    Returns global ids (len(userid), N).
    """
    import torch

    userid = np.atleast_1d(np.asarray(userid))
    user_items = user_items.tocsr()
    n = len(userid)
    M = warm_top_ids.shape[1]
    Vc = _as_gpu(cold_factors, device)                          # (n_cold, f) -- small, resident or transfer
    n_cold = Vc.shape[0]
    # The merge score matrix is only (chunk x (M + n_cold)) wide -- ~75x narrower than the warm pool,
    # so the incoming (warm-sized) chunk splits users into far more chunks than necessary. Re-size
    # for the actual width so 193k users go through in ~1-2 chunks instead of ~37.
    chunk = min(n, max(chunk, int(6e9 / ((M + n_cold) * 4))))
    out_ids = np.full((n, N), -1, dtype=np.int64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        uf = _gather_rows(user_factors, userid[start:stop], device)   # this chunk's user factors
        cold_s = uf @ Vc.T  # (b, n_cold)

        # mask each user's already-revealed cold items, vectorized (no per-user Python loop -- this
        # loop ran over all 193k eval users on every (seed, k) call and was Mode A's real bottleneck)
        liked = user_items.indices[user_items.indptr[start]:user_items.indptr[stop]]
        loc = cold_global_to_local[liked]
        ur = np.repeat(np.arange(stop - start), np.diff(user_items.indptr[start:stop + 1]))
        keep = loc >= 0
        if keep.any():
            cold_s[torch.as_tensor(ur[keep], device=device),
                   torch.as_tensor(loc[keep], device=device)] = float("-inf")

        w_scores = torch.as_tensor(
            np.ascontiguousarray(warm_top_scores[start:stop], dtype=np.float32), device=device
        )  # (b, M)
        combined = torch.cat([w_scores, cold_s], dim=1)  # (b, M + n_cold): [warm cache | cold]
        _, idx = torch.topk(combined, N, dim=1)
        idx_np = idx.cpu().numpy()  # (b, N), column in [0, M+n_cold)

        w_ids_chunk = warm_top_ids[start:stop]                       # (b, M) global warm ids
        from_warm = idx_np < M
        warm_sel = np.take_along_axis(w_ids_chunk, np.clip(idx_np, 0, M - 1), axis=1)
        cold_sel = cold_ids[np.clip(idx_np - M, 0, n_cold - 1)]
        out_ids[start:stop] = np.where(from_warm, warm_sel, cold_sel)

    return out_ids


def mode_a_auc(user_factors, item_factors, candidate_ids, global_to_local,
               userid, user_items, test_items, device="cuda:0", chunk=512):
    """Per-user pairwise AUC over the candidate pool -- the full-ranking companion to topk_recommend.

    AUC never needed the dense n_users x n_items score matrix (which does not fit at Amazon-Books
    scale): it only needs, per user, where that user's held-out positives rank among the
    candidates. That's the same uf @ V.T score row topk_recommend already computes -- here we take a
    full *rank* of it instead of a top-K select, so it runs at Amazon-Books scale with no dense
    allocation. Ranking is over `candidate_ids` (the warm_cold retrieval pool), NOT the full 2.8M
    catalog -- this is the entire, deterministic candidate set we retrieve from, not a random sample,
    so it sidesteps the Krichene-Rendle sampled-metric inconsistency (that critique is about small
    random negative samples). Report it as "AUC (candidate pool)".

    Mann-Whitney rank-sum, the standard full-ranking AUC but pool-restricted: for each
    user, positives = held-out test items, negatives = candidates not in user_items (already-liked /
    revealed) and not positive. Excluded items are pinned to -inf so they occupy the bottom ranks and
    are subtracted off. Ranks are AVERAGE (rankdata convention) via two-sided searchsorted, so ties
    are handled correctly -- essential here because empty-warm-history users get an all-zero ALS
    factor and thus a fully-tied score row (correct AUC 0.5). Handles Popularity's heavy integer
    ties too. Verified bit-for-bit vs scipy.rankdata in bench_14.

    user_factors, item_factors : implicit's fitted factor arrays (numpy, or resident GPU tensors).
    candidate_ids   : sorted int64 global item ids to rank against (the pool).
    global_to_local : int64 length-n_items map, global id -> column in candidate_ids, -1 if not a candidate.
    userid          : 1-D array of user ids.
    user_items      : CSR (len(userid) x n_items), row i = items to EXCLUDE for userid[i] (already-liked
                      / revealed) -- same row-alignment topk_recommend uses.
    test_items      : CSR (len(userid) x n_items), row i = held-out positives for userid[i].
    Returns (len(userid),) float64 AUC per user; NaN where a user has no in-pool positive or no negative.
    """
    import torch

    userid = np.atleast_1d(np.asarray(userid))
    user_items = user_items.tocsr()
    test_items = test_items.tocsr()
    n_query = len(userid)

    V = _gather_rows(item_factors, candidate_ids, device)        # (C, f) candidate factors
    C = V.shape[0]

    auc = np.full(n_query, np.nan, dtype=np.float64)

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        b = stop - start
        uf = _gather_rows(user_factors, userid[start:stop], device)
        scores = uf @ V.T                                        # (b, C)

        # Exclude each user's already-liked/revealed items (in-pool only), and count them per user:
        # excluded items are pinned to -inf so they sort to the bottom, then subtracted from ranks.
        liked = user_items.indices[user_items.indptr[start]:user_items.indptr[stop]]
        loc = global_to_local[liked]
        ur = np.repeat(np.arange(b), np.diff(user_items.indptr[start:stop + 1]))
        keep = loc >= 0
        n_excl = np.bincount(ur[keep], minlength=b).astype(np.int64)
        if keep.any():
            scores[torch.as_tensor(ur[keep], device=device),
                   torch.as_tensor(loc[keep], device=device)] = float("-inf")

        # Rank each positive among the non-excluded reals with AVERAGE ranks (rankdata convention):
        # rank = (#reals strictly below) + (#reals tied + 1)/2. Ties are common and consequential --
        # a user with no warm history gets an all-zero ALS factor, so every candidate scores
        # identically and the positive ties with the ENTIRE pool (correct AUC = 0.5, which only
        # average ranks give; ordinal argsort ranks would return an arbitrary 0..1). One sort +
        # two-sided searchsorted, matching scipy.rankdata exactly (bench_14).
        sorted_full, _ = torch.sort(scores, dim=1)               # ascending; excluded (-inf) at bottom

        pos = test_items.indices[test_items.indptr[start]:test_items.indptr[stop]]
        ploc = global_to_local[pos]
        pur = np.repeat(np.arange(b), np.diff(test_items.indptr[start:stop + 1]))
        pkeep = ploc >= 0
        pur, ploc = pur[pkeep], ploc[pkeep]
        if len(pur) == 0:
            continue
        pur_t = torch.as_tensor(pur, device=device)
        s_pos = scores[pur_t, torch.as_tensor(ploc, device=device)].unsqueeze(1)    # (P, 1)
        srt = sorted_full[pur_t]                                                     # (P, C) each user's sorted row
        lt = torch.searchsorted(srt, s_pos, right=False).squeeze(1).cpu().numpy()    # # strictly < s_p (incl -inf)
        le = torch.searchsorted(srt, s_pos, right=True).squeeze(1).cpu().numpy()     # # <= s_p
        lt_reals = lt - n_excl[pur]                              # strip the n_excl excluded (-inf) below
        eq = le - lt                                            # ties with s_p (all reals; -inf cancels)
        avg_rank = lt_reals + (eq + 1.0) / 2.0                  # 1-based average rank among reals

        rank_sum = np.bincount(pur, weights=avg_rank, minlength=b)
        n_pos = np.bincount(pur, minlength=b).astype(np.int64)
        n_neg = (C - n_excl) - n_pos
        u_stat = rank_sum - n_pos * (n_pos + 1) / 2.0           # Mann-Whitney U
        valid = (n_pos > 0) & (n_neg > 0)
        block = np.full(b, np.nan)
        block[valid] = u_stat[valid] / (n_pos[valid] * n_neg[valid])
        auc[start:stop] = block

    return auc


def mode_a_auc_sweep(user_factors, warm_factors, cold_factors_by_k, warm_global_to_local,
                     cold_global_to_local, userid, warm_excl_csr, cold_excl_csr_by_k, test_csr,
                     device="cuda:0", chunk=512):
    """Cached Mode A candidate-pool AUC across reveal levels -- the sweep fast path for mode_a_auc.

    Within a seed the WARM factors are frozen across the whole reveal sweep, so each user's warm
    score row is invariant; only the ~5k cold columns change per k. This sorts each user's warm
    scores ONCE and reuses them for every level (searchsorted to count warm negatives below a cold
    positive), re-ranking only the small cold block per k. That turns L full 388k-wide sorts per
    chunk into ONE warm sort + L cheap 5k-wide sorts (L = len(k_levels)) -- the same amortization
    sweep_mode_a_cached does for top-K. Produces the identical per-user AUC as calling mode_a_auc at
    each k (verified bench_14).

    Loop nesting is chunk-outer / k-inner ON PURPOSE: a chunk's (chunk x n_warm) sorted warm scores
    live only while that chunk is processed, never all users at once (which would be ~300 GB here).

    The pool splits cleanly by construction: ref_train holds only warm interactions and
    revealed_matrix_at_k holds only cold reveals, so warm exclusions (ref_train) are frozen across k
    and cold exclusions (revealed) are the only per-k part. Positives are held-out cold items, never
    excluded. Uses AVERAGE ranks (two-sided searchsorted per block), so fully-tied all-zero-factor
    users score a correct 0.5 -- identical to mode_a_auc on the combined pool (bench_14).

    warm_factors       : (n_warm, f) frozen warm candidate factors (numpy or resident tensor).
    cold_factors_by_k  : list over k of (n_cold, f) folded cold candidate factors.
    warm/cold_global_to_local : global id -> warm-local / cold-local column, -1 outside that block.
    userid             : (n_query,) eval users.
    warm_excl_csr      : CSR (n_query x n_items) frozen warm exclusions (ref_train), row-aligned.
    cold_excl_csr_by_k : list over k of CSR (n_query x n_items) cold exclusions (revealed at k).
    test_csr           : CSR (n_query x n_items) held-out cold positives, row-aligned.
    Returns (n_k, n_query) float64 AUC per (level, user); NaN where a user has no in-pool positive/negative.
    """
    import torch

    userid = np.atleast_1d(np.asarray(userid))
    n_query = len(userid)
    n_k = len(cold_factors_by_k)
    warm_excl_csr = warm_excl_csr.tocsr()
    test_csr = test_csr.tocsr()
    cold_excl = [c.tocsr() for c in cold_excl_csr_by_k]

    Vw = _as_gpu(warm_factors, device)                          # (n_warm, f) frozen
    n_warm = Vw.shape[0]
    Vc_by_k = [_as_gpu(cf_k, device) for cf_k in cold_factors_by_k]
    n_cold = Vc_by_k[0].shape[0]

    out = np.full((n_k, n_query), np.nan, dtype=np.float64)

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        b = stop - start
        uf = _gather_rows(user_factors, userid[start:stop], device)      # (b, f)

        # --- warm block: score, exclude ref_train warm (frozen), sort ONCE, reuse across all k ---
        sw = uf @ Vw.T                                                    # (b, n_warm)
        wl = warm_excl_csr.indices[warm_excl_csr.indptr[start]:warm_excl_csr.indptr[stop]]
        wloc = warm_global_to_local[wl]
        wur = np.repeat(np.arange(b), np.diff(warm_excl_csr.indptr[start:stop + 1]))
        wkeep = wloc >= 0
        n_excl_warm = np.bincount(wur[wkeep], minlength=b).astype(np.int64)
        if wkeep.any():
            sw[torch.as_tensor(wur[wkeep], device=device),
               torch.as_tensor(wloc[wkeep], device=device)] = float("-inf")
        sw_sorted, _ = torch.sort(sw, dim=1)                             # (b, n_warm) ascending -- the cache
        del sw
        n_warm_neg = (n_warm - n_excl_warm).astype(np.int64)             # frozen across k

        for ki in range(n_k):
            sc = uf @ Vc_by_k[ki].T                                      # (b, n_cold)
            ce = cold_excl[ki]
            cl = ce.indices[ce.indptr[start]:ce.indptr[stop]]
            cloc = cold_global_to_local[cl]
            cur = np.repeat(np.arange(b), np.diff(ce.indptr[start:stop + 1]))
            ckeep = cloc >= 0
            n_excl_cold = np.bincount(cur[ckeep], minlength=b).astype(np.int64)
            sc_masked = sc.clone()
            if ckeep.any():
                sc_masked[torch.as_tensor(cur[ckeep], device=device),
                          torch.as_tensor(cloc[ckeep], device=device)] = float("-inf")
            sc_sorted, _ = torch.sort(sc_masked, dim=1)                  # (b, n_cold) asc; excluded at bottom

            pos = test_csr.indices[test_csr.indptr[start]:test_csr.indptr[stop]]
            ploc = cold_global_to_local[pos]
            pur = np.repeat(np.arange(b), np.diff(test_csr.indptr[start:stop + 1]))
            pkeep = ploc >= 0
            pur, ploc = pur[pkeep], ploc[pkeep]
            if len(pur) == 0:
                continue
            pur_t = torch.as_tensor(pur, device=device)
            s_pos = sc[pur_t, torch.as_tensor(ploc, device=device)].unsqueeze(1)     # (P, 1) unmasked cold score

            # Average rank among ALL non-excluded reals = warm reals + cold reals, via two-sided
            # searchsorted on each block (rankdata convention, tie-correct). Identical result to
            # mode_a_auc ranking the combined pool -- verified bench_14. Reuses the ONE warm sort
            # (sw_sorted) across every k; only the small cold block is re-sorted per level.
            wsrt = sw_sorted[pur_t]                                                   # (P, n_warm)
            w_lt = torch.searchsorted(wsrt, s_pos, right=False).squeeze(1).cpu().numpy()
            w_le = torch.searchsorted(wsrt, s_pos, right=True).squeeze(1).cpu().numpy()
            csrt = sc_sorted[pur_t]                                                   # (P, n_cold)
            c_lt = torch.searchsorted(csrt, s_pos, right=False).squeeze(1).cpu().numpy()
            c_le = torch.searchsorted(csrt, s_pos, right=True).squeeze(1).cpu().numpy()

            lt_reals = (w_lt - n_excl_warm[pur]) + (c_lt - n_excl_cold[pur])   # reals strictly below s_p
            eq = (w_le - w_lt) + (c_le - c_lt)                                 # reals tied with s_p
            avg_rank = lt_reals + (eq + 1.0) / 2.0                             # 1-based average rank among reals

            rank_sum = np.bincount(pur, weights=avg_rank, minlength=b)
            n_pos = np.bincount(pur, minlength=b).astype(np.int64)
            n_neg = n_warm_neg + (n_cold - n_excl_cold) - n_pos
            u_stat = rank_sum - n_pos * (n_pos + 1) / 2.0                      # Mann-Whitney U
            valid = (n_pos > 0) & (n_neg > 0)
            block = np.full(b, np.nan)
            block[valid] = u_stat[valid] / (n_pos[valid] * n_neg[valid])
            out[ki, start:stop] = block

    return out


def mode_b_topk_users(user_factors, cold_factors, cold_revealed_csc, N, device="cuda:0", item_chunk=256):
    """Mode B (item -> user): for each cold item, rank ALL users by predicted affinity and return
    its ordered top-N non-revealed users.

    This is what re-enables Mode B at Amazon-Books scale: instead of materializing the dense
    n_users x n_items score matrix (7e12 cells, guarded off in eval), it scores only the cold-item
    columns -- user_factors @ cold_factors.T, shape (n_users, n_cold) -- chunked over items so the
    working set stays ~item_chunk columns wide. Revealed users (that item's first-k interactions)
    are masked to -inf so they can't appear in the ranking, matching eval's Mode B semantics.

    user_factors      : (n_users, f) frozen user factors.
    cold_factors      : (n_cold, f) folded cold-item factors at the current reveal level.
    cold_revealed_csc : (n_users, n_cold) CSC; column j = users already revealed for cold item j.
    Returns (n_cold, N) global user ids, ordered by descending score, -1-padded if fewer than N.
    """
    import torch

    n_users = user_factors.shape[0]
    n_cold = cold_factors.shape[0]
    Uf = _as_gpu(user_factors, device)                          # (n_users, f) resident (once/seed) or transferred
    Cf_all = _as_gpu(cold_factors, device)                      # (n_cold, f) -- small
    N_eff = min(N, n_users)
    out = np.full((n_cold, N), -1, dtype=np.int64)

    for s in range(0, n_cold, item_chunk):
        e = min(s + item_chunk, n_cold)
        Cf = Cf_all[s:e]                                        # (c, f)
        # Items as ROWS, users as columns, so the top-K runs along the contiguous (user) axis --
        # top-K along a strided axis is several times slower on the GPU.
        scores = Cf @ Uf.T                                          # (c, n_users)

        rev = cold_revealed_csc[:, s:e].tocoo()                     # row=user, col=item-local
        if rev.nnz:
            scores[torch.as_tensor(rev.col.astype(np.int64), device=device),
                   torch.as_tensor(rev.row.astype(np.int64), device=device)] = float("-inf")

        _, idx = torch.topk(scores, N_eff, dim=1)                    # (c, N_eff) descending, per item
        out[s:e, :N_eff] = idx.cpu().numpy()

    return out


def mode_b_auc(user_factors, cold_factors, cold_test_csc, cold_revealed_csc,
               device="cuda:0", item_chunk=64):
    """Per-cold-item AUC over the full user population (Mode B) -- the full-ranking companion to
    mode_b_topk_users, transposed onto the user axis.

    For each cold item, rank ALL users by predicted affinity (cold_factors @ user_factors.T), exclude
    that item's revealed users, and take the average-rank AUC of its reserved-test users (positives)
    among the non-revealed users -- the standard per-item Mode B AUC, but
    scoring only the cold columns (no dense n_users x n_items matrix) so it runs at Amazon-Books
    scale. Two-sided searchsorted => tie-correct average ranks. Positives are the FULL reserved test
    set (not the downsampled ctx used for the K-based Mode B metrics).

    NOTE -- unlike Mode A's mode_a_auc_sweep there is NO cross-k cache: the ranked set is the users
    (frozen), but the query is the cold-item factor, which folds -- so every (item, reveal level)
    re-sorts the user axis. Chunked over items (item_chunk rows x n_users) to bound VRAM.

    user_factors      : (n_users, f) frozen user factors (numpy or resident tensor).
    cold_factors      : (n_cold, f) folded cold-item factors at the current reveal level.
    cold_test_csc     : (n_users, n_cold) CSC; column j = reserved test users (positives) for item j.
    cold_revealed_csc : (n_users, n_cold) CSC; column j = revealed users (excluded) for item j at k.
    Returns (n_cold,) float64 AUC per cold item; NaN where the item has no positive or no negative.
    """
    import torch

    Uf = _as_gpu(user_factors, device)                          # (n_users, f)
    n_users = Uf.shape[0]
    Cf_all = _as_gpu(cold_factors, device)                      # (n_cold, f)
    n_cold = Cf_all.shape[0]
    test_csc = cold_test_csc.tocsc()
    rev_csc = cold_revealed_csc.tocsc()

    auc = np.full(n_cold, np.nan, dtype=np.float64)

    for s in range(0, n_cold, item_chunk):
        e = min(s + item_chunk, n_cold)
        c = e - s
        scores = Cf_all[s:e] @ Uf.T                             # (c, n_users) items as rows

        # Positives (reserved test users) per local item -> padded (c, max_pos). Capture their scores
        # BEFORE masking revealed (positives are disjoint from revealed by construction).
        tsub = test_csc[:, s:e]
        r_i = np.diff(tsub.indptr).astype(np.int64)             # (c,) positives per item
        max_pos = int(r_i.max()) if c else 0
        if max_pos == 0:
            continue
        pos_user = np.zeros((c, max_pos), dtype=np.int64)
        valid = np.zeros((c, max_pos), dtype=bool)
        for j in range(c):
            us = tsub.indices[tsub.indptr[j]:tsub.indptr[j + 1]]
            pos_user[j, :len(us)] = us
            valid[j, :len(us)] = True
        ps = scores[torch.arange(c, device=device)[:, None],
                    torch.as_tensor(pos_user, device=device)]   # (c, max_pos) positive scores
        ps[~torch.as_tensor(valid, device=device)] = float("inf")   # padding sorts above everything -> masked out

        # Exclude revealed users (pin -inf), count per item.
        rsub = rev_csc[:, s:e].tocoo()
        n_excl = np.bincount(rsub.col, minlength=c).astype(np.int64)
        if rsub.nnz:
            scores[torch.as_tensor(rsub.col.astype(np.int64), device=device),
                   torch.as_tensor(rsub.row.astype(np.int64), device=device)] = float("-inf")

        sorted_u, _ = torch.sort(scores, dim=1)                 # (c, n_users) ascending; revealed at bottom
        w_lt = torch.searchsorted(sorted_u, ps, right=False).cpu().numpy()   # (c, max_pos) # users < positive
        w_le = torch.searchsorted(sorted_u, ps, right=True).cpu().numpy()    # # users <= positive

        lt_reals = w_lt - n_excl[:, None]                       # non-revealed users strictly below each positive
        eq = w_le - w_lt                                        # ties (revealed -inf cancel)
        avg_rank = lt_reals + (eq + 1.0) / 2.0                  # 1-based avg rank among non-revealed
        avg_rank[~valid] = 0.0
        rank_sum = avg_rank.sum(axis=1)                         # (c,)
        n_pos = r_i
        n_neg = n_users - n_excl - n_pos
        u_stat = rank_sum - n_pos * (n_pos + 1) / 2.0           # Mann-Whitney U
        ok = (n_pos > 0) & (n_neg > 0)
        block = np.full(c, np.nan)
        block[ok] = u_stat[ok] / (n_pos[ok] * n_neg[ok])
        auc[s:e] = block

    return auc
