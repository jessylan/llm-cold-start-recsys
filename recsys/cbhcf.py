"""Content-based hybrid collaborative filtering (CBHCF): blends a content-based item-item
similarity score with a collaborative model's score, weighted per item by how many training
interactions that item has. A strict cold-start item (0 interactions) scores purely on content;
an item at or past `warmup_threshold` interactions scores purely on the collaborative signal.
"""
import copy

import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    """Rescales a score matrix to [0, 1] so differently-scaled models can be blended additively."""
    lo, hi = matrix.min(), matrix.max()
    return (matrix - lo) / (hi - lo) if hi > lo else np.zeros_like(matrix)


def _sigmoid_ramp(item_counts: np.ndarray, threshold: float, steepness: float = 10.0) -> np.ndarray:
    """Per-item CF weight: exactly 0 at 0 interactions, exactly 1 at >= threshold interactions,
    logistic in between, crossing 0.5 at threshold / 2. The raw logistic is rescaled by its own
    values at the two endpoints so they land on exactly 0 and 1, rather than merely approaching
    them asymptotically."""
    midpoint = threshold / 2
    z = steepness * (item_counts - midpoint) / threshold
    raw = 1.0 / (1.0 + np.exp(-z))
    raw_at_0 = 1.0 / (1.0 + np.exp(steepness * midpoint / threshold))
    raw_at_threshold = 1.0 / (1.0 + np.exp(-steepness * midpoint / threshold))
    weight = (raw - raw_at_0) / (raw_at_threshold - raw_at_0)
    return np.clip(weight, 0.0, 1.0)


class CBHCFModel:
    """Content-based item-item similarity, optionally blended with an already-fit CF model's
    score, to satisfy protocol.RetrievalModel.
    """

    def __init__(self, warmup_threshold: int = 20, sigmoid_steepness: float = 10.0):
        self.warmup_threshold = warmup_threshold
        self.sigmoid_steepness = sigmoid_steepness
        self._train_matrix = None
        self._cf_embeddings = None
        self._similarity = None

    def fit(self, train_matrix: sparse.csr_matrix, *, item_metadata, cf_embeddings=None) -> "CBHCFModel":
        self._train_matrix = sparse.csr_matrix(train_matrix)
        tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(item_metadata)
        self._similarity = cosine_similarity(tfidf_matrix)
        self._cf_embeddings = cf_embeddings
        return self

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        scores = self.score_matrix()
        single = np.isscalar(userid)
        user_ids = np.atleast_1d(userid)
        rows = sparse.csr_matrix(user_items)

        out_ids = np.zeros((len(user_ids), N), dtype=np.int32)
        out_scores = np.zeros((len(user_ids), N), dtype=np.float32)
        for i, u in enumerate(user_ids):
            row_scores = scores[u].copy()
            if filter_already_liked_items:
                seen = rows.indices[rows.indptr[i]:rows.indptr[i + 1]]
                row_scores[seen] = -np.inf
            picks = np.argsort(-row_scores)[:N]
            out_ids[i] = picks
            out_scores[i] = row_scores[picks]

        return (out_ids[0], out_scores[0]) if single else (out_ids, out_scores)

    def _content_score_matrix(self) -> np.ndarray:
        """Vectorized n_users x n_items content score: each user's row is the mean similarity
        between every candidate item and every item already in that user's training row."""
        history_counts = np.asarray(self._train_matrix.sum(axis=1)).ravel()
        raw_scores = self._train_matrix @ self._similarity
        safe_counts = np.where(history_counts > 0, history_counts, 1)
        scores = raw_scores / safe_counts[:, None]
        scores[history_counts == 0] = 0.0
        return scores

    def score_matrix(self) -> np.ndarray:
        cb_scores = self._content_score_matrix()
        if self._cf_embeddings is None:
            return cb_scores

        cf_scores = self._cf_embeddings.user_factors @ self._cf_embeddings.item_factors.T
        item_counts = np.asarray(self._train_matrix.sum(axis=0)).ravel()
        weight = _sigmoid_ramp(item_counts, self.warmup_threshold, self.sigmoid_steepness)
        return (
            _min_max_normalize(cf_scores) * weight[None, :]
            + _min_max_normalize(cb_scores) * (1 - weight)[None, :]
        )

    def fold_in(self, dataset, k) -> "CBHCFModel":
        """Returns a NEW CBHCFModel reflecting reveal-level-k state, without mutating self.
        Content similarity is static (it needs no fold-in), so only `_train_matrix` (which
        drives both the per-user content score and the per-item cf_weight) and `_cf_embeddings`
        are updated. `_cf_embeddings` is delegated to its own fold_in if it has one (e.g.
        cf.ALSModel's recalculate_item); otherwise it's carried over unchanged."""
        train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
        folded_cf = (
            self._cf_embeddings.fold_in(dataset, k)
            if self._cf_embeddings is not None and hasattr(self._cf_embeddings, "fold_in")
            else self._cf_embeddings
        )
        folded = copy.copy(self)
        folded._train_matrix = sparse.csr_matrix(train_k)
        folded._cf_embeddings = folded_cf
        return folded
