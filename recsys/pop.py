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
        single = np.isscalar(userid)
        user_ids = np.atleast_1d(userid)
        rows = sparse.csr_matrix(user_items)

        out_ids = np.zeros((len(user_ids), N), dtype=np.int32)
        out_scores = np.zeros((len(user_ids), N), dtype=np.float32)
        for i in range(len(user_ids)):
            seen = set(rows.indices[rows.indptr[i]:rows.indptr[i + 1]].tolist()) if filter_already_liked_items else set()
            picks = [item for item in self.ranked_items if item not in seen][:N]
            out_ids[i, :len(picks)] = picks
            out_scores[i, :len(picks)] = self.popularity[picks]

        return (out_ids[0], out_scores[0]) if single else (out_ids, out_scores)

    def score_matrix(self):
        # float64, not popularity's native int64 -- callers (e.g. eval.py's Mode B) mask
        # individual entries to -inf in place, which an int array can't hold.
        return np.tile(self.popularity, (self.n_users, 1)).astype(np.float64)

    def fold_in(self, dataset, k):
        """Popularity has no partial-update mechanism for a raw interaction count, so its
        "fold-in" is a full refit on train + the cold items' revealed interactions at k."""
        train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
        return PopularityModel(train_k)


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
            "(Mode B) evaluation path via score_matrix()/fold_in(), not Mode A's recommend()."
        )

    def score_matrix(self):
        # Same activity score for every item -- the direct dual of PopularityModel's
        # np.tile(popularity, (n_users, 1)), transposed onto the user axis. float64, not
        # user_activity's native int64 -- callers mask entries to -inf in place.
        return np.tile(self.user_activity, (self.n_items, 1)).T.astype(np.float64)

    def fold_in(self, dataset, k):
        # No k-dependence: user_activity is frozen (computed once on ref_train, exactly like
        # user_factors elsewhere), and the revealed-user exclusion is applied generically by
        # eval.py's Mode B sweep, not per-model here.
        return self
