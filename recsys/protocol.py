# This file was created with the assistance of Generative AI.
"""Shared contract every retrieval model -- Popularity, ALS/CF, and future methods such as
CBHCF or a content-embedding intervention -- must satisfy so eval.py never branches on model
type.
"""
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sparse

from recsys.load import Dataset


@runtime_checkable
class RetrievalModel(Protocol):
    """Structural contract for a fitted retrieval model.

    Shared-embeddings convention: `fit()` here takes only `train_matrix`. A model wanting to
    optionally reuse another model's embeddings (e.g. a future CBHCF reusing CF's learned
    factors) adds its own keyword-only parameter -- e.g.
    `cbhcf.fit(self, train_matrix, *, cf_embeddings=None)` accepting any object exposing
    `.user_factors`/`.item_factors`. eval.py's sweep functions never pass such a kwarg; it's
    notebook-level orchestration only. `cf.ALSModel` already exposes both attributes publicly,
    satisfying this convention today with no extra code.
    """

    def fit(self, train_matrix: sparse.csr_matrix) -> "RetrievalModel":
        """Fit on the given user x item training matrix. Returns self."""
        ...

    def recommend(
        self, userid, user_items, N: int = 10, filter_already_liked_items: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Same calling convention as implicit's model.recommend(): userid may be scalar or an
        array; returns (ids, scores) shaped to match."""
        ...

    def fold_in(self, dataset: Dataset, k: int) -> "RetrievalModel":
        """Return a NEW object satisfying this Protocol, reflecting cold items' reveal-level-k
        state, WITHOUT mutating self. Strategy is implementation-defined: a cheap non-mutating
        partial update (ALS's recalculate_item) and a full refit on train+revealed (Popularity)
        are both valid -- callers must not assume which."""
        ...
