"""ALS/CF wrapper around implicit's AlternatingLeastSquares.

Alternating Least Squares learns latent user and item factor vectors by alternately solving,
in closed form, for all user factors with item factors fixed, then all item factors with user
factors fixed. That alternating structure is exactly what cold-start fold-in needs: implicit
exposes `recalculate_item(itemid, item_users)`, which performs *only* the item-side half of
that update -- an exact closed-form solve for one item's factor, holding every user's factor
completely untouched. This is a native property of how ALS is built, not a workaround bolted
onto it, and it's why ALS is the model used here rather than a model whose parameters emerge
from joint stochastic optimization across the whole dataset (which has no equivalently clean
"hold one side fixed" operation).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # must be set before importing implicit,
# anywhere in the process -- see load.py's matching guard for why this can't live in only one
# of the two modules that import implicit.

import copy

from implicit.als import AlternatingLeastSquares


class ALSModel:
    """Wraps implicit.als.AlternatingLeastSquares to satisfy protocol.RetrievalModel."""

    def __init__(self, factors: int = 50, regularization: float = 0.01, iterations: int = 15, random_state: int = 42):
        self._params = dict(factors=factors, regularization=regularization,
                             iterations=iterations, random_state=random_state)
        self._model = None

    def fit(self, train_matrix, show_progress: bool = False) -> "ALSModel":
        self._model = AlternatingLeastSquares(**self._params)
        self._model.fit(train_matrix, show_progress=show_progress)
        return self

    @property
    def user_factors(self):
        return self._model.user_factors

    @property
    def item_factors(self):
        return self._model.item_factors

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        return self._model.recommend(
            userid, user_items, N=N, filter_already_liked_items=filter_already_liked_items
        )

    def score_matrix(self):
        return self.user_factors @ self.item_factors.T

    def fold_in(self, dataset, k):
        """Returns a NEW ALSModel with cold items' item_factors rows replaced by their fold-in
        factor at reveal level k (via recalculate_item's exact closed-form solve), without
        mutating self. User factors are never touched.

        At k=0, recalculate_item's closed-form solve has nothing but regularization to fall
        back on, so the resulting factor is the exact zero vector for every cold item -- a zero
        vector's dot product with any user factor is exactly zero, so the item cannot outrank
        anything.
        """
        item_ids, item_users = dataset.revealed_item_users_at_k(k)
        folded_item_factors = self._model.recalculate_item(item_ids, item_users)
        factors = self.item_factors.copy()
        factors[item_ids] = folded_item_factors

        folded_model = copy.copy(self)
        folded_model._model = copy.copy(self._model)
        folded_model._model.item_factors = factors
        return folded_model
