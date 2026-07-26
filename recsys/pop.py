"""Popularity baseline (item-side, Mode A) and its Mode-B dual, global user Activity. Neither
has any per-user personalization signal -- both rank by raw observed interaction volume, on
opposite axes of the same user x item matrix -- which is exactly why they're the right floor
for this comparison: there's no "user representation" here to hold constant, since there isn't
one to begin with.
"""
import numpy as np
import scipy.sparse as sparse


class PopularityModel:
    """Most-popular recommender, exposing implicit's model.recommend() interface. Ranks items
    by raw training-interaction count -- the same global list for every user, personalized only
    by excluding each user's already-seen items.
    """

    def __init__(self, train_matrix=None):
        self.n_users = None
        self.popularity = None
        self.ranked_items = None
        if train_matrix is not None:
            self.fit(train_matrix)

    def fit(self, train_matrix: sparse.csr_matrix) -> "PopularityModel":
        self.n_users = train_matrix.shape[0]
        self.popularity = np.asarray((train_matrix > 0).sum(axis=0)).ravel()
        self.ranked_items = np.argsort(-self.popularity)
        return self

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        """Vectorized top-N (exact match to the old per-user islice loop, ~orders of magnitude
        faster at Amazon-Books scale). The ranking is global -- the same for every user -- so we
        take the global top-M candidates ONCE, where M = N + (max items any user has already seen).
        That guarantees every user still has >= N unseen items inside the top-M, so per user we just
        pick the first N unseen, in rank order, via a cumulative count over a boolean seen-mask --
        no Python loop over users or over the catalog."""
        single = np.isscalar(userid)
        user_ids = np.atleast_1d(userid)
        n_query = len(user_ids)
        n_items = len(self.ranked_items)

        if not filter_already_liked_items:
            top = self.ranked_items[:N]
            out_ids = np.tile(top.astype(np.int64), (n_query, 1))
            out_scores = np.tile(self.popularity[top].astype(np.float32), (n_query, 1))
            return (out_ids[0], out_scores[0]) if single else (out_ids, out_scores)

        rows = sparse.csr_matrix(user_items)
        max_seen = int(np.diff(rows.indptr).max()) if n_query else 0
        M = min(N + max_seen, n_items)                 # >= N unseen guaranteed within the top-M
        topM = self.ranked_items[:M]                    # (M,) global ids, in rank order
        topM_scores = self.popularity[topM].astype(np.float32)
        pos = np.full(n_items, -1, dtype=np.int64)      # global id -> column in topM, or -1
        pos[topM] = np.arange(M)

        out_ids = np.empty((n_query, N), dtype=np.int64)
        out_scores = np.empty((n_query, N), dtype=np.float32)
        chunk = max(500, min(20_000, 50_000_000 // max(M, 1)))  # bound the (chunk x M) mask memory
        for s in range(0, n_query, chunk):
            e = min(s + chunk, n_query)
            sub = rows[s:e]
            b = e - s
            ur = np.repeat(np.arange(b), np.diff(sub.indptr))   # local user row per seen entry
            loc = pos[sub.indices]                              # its column in topM (-1 if outside)
            keep = loc >= 0
            seen_mask = np.zeros((b, M), dtype=bool)
            seen_mask[ur[keep], loc[keep]] = True
            valid = ~seen_mask
            take = valid & (np.cumsum(valid, axis=1) <= N)      # the first N unseen per user
            cols = np.nonzero(take)[1].reshape(b, N)            # exactly N/row (>= N valids guaranteed)
            out_ids[s:e] = topM[cols]
            out_scores[s:e] = topM_scores[cols]

        return (out_ids[0], out_scores[0]) if single else (out_ids, out_scores)

    def fold_in(self, dataset, k):
        """Popularity has no partial-update mechanism for a raw interaction count, so its
        "fold-in" is a full refit on train + the cold items' revealed interactions at k."""
        train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
        return PopularityModel(train_k)

    def fold_in_ceiling(self, dataset):
        """Within-item warm reference: refit on train + every cold item's FULL pre-test interactions
        (the ceiling). The Popularity dual of ALSModel.fold_in_ceiling."""
        train_ceiling = dataset.ref_train + dataset.ceiling_matrix()
        return PopularityModel(train_ceiling)


class ActivityModel:
    """Global-user-activity floor for Mode B -- the direct dual of PopularityModel: rank USERS
    by observed volume (total interactions in ref_train), no personalization, same principle as
    Popularity applied to the opposite axis. Has no per-item ranking signal at all, so it only
    supports the item-to-user (Mode B) evaluation path, not Mode A's recommend().
    """

    def __init__(self, train_matrix=None):
        self.n_items = None
        self.user_activity = None
        self.ranked_users = None
        if train_matrix is not None:
            self.fit(train_matrix)

    def fit(self, train_matrix: sparse.csr_matrix) -> "ActivityModel":
        self.n_items = train_matrix.shape[1]
        self.user_activity = np.asarray((train_matrix > 0).sum(axis=1)).ravel()
        self.ranked_users = np.argsort(-self.user_activity)
        return self

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        raise NotImplementedError(
            "ActivityModel has no per-user item ranking -- it only supports the item-to-user "
            "(Mode B) evaluation path (gpu_retrieval.mode_b_*), not Mode A's recommend()."
        )

    def fold_in(self, dataset, k):
        # No k-dependence: user_activity is frozen (computed once on ref_train, exactly like
        # user_factors elsewhere), and the revealed-user exclusion is applied generically by
        # eval.py's Mode B sweep, not per-model here.
        return self

    def fold_in_ceiling(self, dataset):
        # Same as fold_in: user_activity is frozen; the ceiling exclusion is applied by the caller.
        return self
