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

import numpy as np
from implicit.als import AlternatingLeastSquares

from recsys.load import dataset_fingerprint


class ALSModel:
    """Wraps implicit.als.AlternatingLeastSquares to satisfy protocol.RetrievalModel."""

    def __init__(self, factors: int = 50, regularization: float = 0.01, iterations: int = 15,
                 random_state: int = 42, use_gpu: bool = False,
                 recommend_backend: str = "cpu", gpu_chunk: int = None):
        self._params = dict(factors=factors, regularization=regularization,
                             iterations=iterations, random_state=random_state, use_gpu=use_gpu)
        self._model = None
        # Exact-GPU recommend() config (see gpu_retrieval.topk_recommend). Default "cpu" keeps
        # implicit's path, so the Mac and every non-ALS model are unaffected. prepare_gpu_recommend()
        # flips this to "gpu" and fills the candidate/index state below; fold_in() propagates it.
        self.recommend_backend = recommend_backend
        self._gpu_chunk = gpu_chunk
        self._candidate_ids = None
        self._global_to_local = None
        self._device = None
        # Warm/cold split + warm top-N cache for the k-sweep fast path (build_warm_cache /
        # recommend_cached, orchestrated by gpu_sweep). Only populated for the "warm_cold" pool.
        self._warm_ids = None
        self._cold_ids = None
        self._warm_global_to_local = None
        self._cold_global_to_local = None
        self._warm_cache = None
        # Degree-matched AUC pool (report: reserve degree-matching for AUC only). Same cold items,
        # but warm competitors restricted to ref_train degree >= auc_min_degree, so a partially
        # revealed cold item is ranked against comparable-degree items -- this removes the Popularity
        # floor's "beat the low-degree tail" AUC inflation. Top-K keeps the FULL warm_cold pool.
        self._auc_warm_ids = None
        self._auc_candidate_ids = None
        self._auc_warm_global_to_local = None
        self._auc_global_to_local = None
        # GPU-resident factor tensors (torch), loaded per seed during a sweep so the many
        # recommend/score calls gather from VRAM instead of re-transferring factors each call.
        self._uf_gpu = None
        self._itf_gpu = None
        # Memoized cold-block fold-ins, keyed by (dataset fingerprint, level). See
        # _cold_factors. Folded models share the dict by copy.copy, which is fine --
        # they never fold again.
        self._fold_cache = {}

    def fit(self, train_matrix, show_progress: bool = False) -> "ALSModel":
        self._model = AlternatingLeastSquares(**self._params)
        self._model.fit(train_matrix, show_progress=show_progress)
        if self._params.get("use_gpu"):
            # GPU fit is only for speed. implicit's GPU factors come back as implicit.gpu.Matrix
            # objects; move them into a fresh CPU model (plain numpy) so the entire downstream path
            # -- our torch.topk recommend AND implicit's recalculate_item fold-in -- runs unchanged
            # on numpy instead of having to special-case GPU-matrix objects everywhere.
            uf = self._to_numpy(self._model.user_factors)
            itf = self._to_numpy(self._model.item_factors)
            cpu = AlternatingLeastSquares(**{**self._params, "use_gpu": False})
            cpu.user_factors = uf
            cpu.item_factors = itf
            self._model = cpu
        return self

    @staticmethod
    def _to_numpy(factors):
        """implicit's GPU factors are implicit.gpu.Matrix (have .to_numpy()); CPU factors are already
        numpy. Normalize to numpy either way."""
        return factors.to_numpy() if hasattr(factors, "to_numpy") else np.asarray(factors)

    @property
    def user_factors(self):
        return self._model.user_factors

    @property
    def item_factors(self):
        return self._model.item_factors

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        if self.recommend_backend == "gpu":
            from recsys import gpu_retrieval
            uf = self._uf_gpu if self._uf_gpu is not None else self.user_factors    # resident tensor or numpy
            itf = self._itf_gpu if self._itf_gpu is not None else self.item_factors
            return gpu_retrieval.topk_recommend(
                uf, itf, self._candidate_ids, self._global_to_local,
                userid, user_items, N=N, filter_already_liked_items=filter_already_liked_items,
                device=self._device, chunk=self._gpu_chunk,
            )
        return self._model.recommend(
            userid, user_items, N=N, filter_already_liked_items=filter_already_liked_items
        )

    def load_factors_gpu(self, device: str = None, items: bool = True) -> "ALSModel":
        """Move this model's user (and, if `items`, item) factors onto the GPU as resident torch
        tensors, so the sweep's many calls gather from VRAM instead of re-transferring from host each
        time. Call once per seed at a sweep's start; free_factors_gpu() at the end. Only ONE seed's
        factors should be resident at a time (all N_SEEDS at once would OOM). Needs the device from
        prepare_gpu_recommend()."""
        import torch
        dev = device or self._device
        self._uf_gpu = torch.as_tensor(np.ascontiguousarray(self.user_factors, dtype=np.float32), device=dev)
        if items:
            self._itf_gpu = torch.as_tensor(np.ascontiguousarray(self.item_factors, dtype=np.float32), device=dev)
        return self

    def free_factors_gpu(self) -> "ALSModel":
        """Drop the GPU-resident factor tensors (frees VRAM). Safe to call if none are loaded."""
        self._uf_gpu = None
        self._itf_gpu = None
        return self

    def prepare_gpu_recommend(self, dataset, candidates: str = "warm_cold", auc_min_degree: int = 25,
                              device: str = "cuda:0", chunk: int = None,
                              vram_budget_gb: float = 8.0) -> "ALSModel":
        """Switch recommend() to the exact-GPU (torch.topk) backend. Call once after fit(), before
        any recommend(); fold_in() carries the config to folded models.

        candidates:
          "warm_cold" (default) -- score only warm (nonzero cols of ref_train) + cold items.
                         The correct candidate pool -- never recommends an item that was
                         never trained. Enables the k-sweep warm cache (build_warm_cache).
          "all"       -- score every item, i.e. also the sparsity-dropped items that keep
                         their random init. This reproduces implicit.recommend() bit-for-bit
                         (bench_7), which is the only reason to use it; several times heavier,
                         and no caching.

        chunk: users scored per GPU batch; if None, auto-sized to ~vram_budget_gb of score matrix.
        """
        import numpy as np
        cold = np.asarray(dataset.cold_item_ids, dtype=np.int64)
        if candidates == "all":
            cand = np.arange(dataset.n_items, dtype=np.int64)
            g2l = cand  # identity: global id == local column
            warm = None
        elif candidates == "warm_cold":
            warm = np.unique(dataset.ref_train.nonzero()[1]).astype(np.int64)
            cand = np.union1d(warm, cold)
            g2l = np.full(dataset.n_items, -1, dtype=np.int64)
            g2l[cand] = np.arange(len(cand), dtype=np.int64)
        else:
            raise ValueError(f"candidates must be 'all' or 'warm_cold', got {candidates!r}")
        self._candidate_ids = cand
        self._global_to_local = g2l
        self._device = device
        self._gpu_chunk = chunk or max(256, int(vram_budget_gb * 1e9 / (len(cand) * 4)))
        self.recommend_backend = "gpu"
        # Warm/cold split for the k-sweep cache (only meaningful for the "warm_cold" pool).
        self._warm_ids = warm
        self._cold_ids = cold
        self._warm_cache = None
        if warm is not None:
            wg = np.full(dataset.n_items, -1, dtype=np.int64); wg[warm] = np.arange(len(warm), dtype=np.int64)
            cg = np.full(dataset.n_items, -1, dtype=np.int64); cg[cold] = np.arange(len(cold), dtype=np.int64)
            self._warm_global_to_local = wg
            self._cold_global_to_local = cg
            # Degree-matched AUC pool: warm competitors with ref_train degree >= auc_min_degree, + cold.
            # AUC (reference + sweep) ranks against this; top-K keeps the full warm pool above.
            deg = np.asarray((dataset.ref_train > 0).sum(axis=0)).ravel()
            auc_warm = warm[deg[warm] >= auc_min_degree]
            self._auc_warm_ids = auc_warm
            self._auc_candidate_ids = np.union1d(auc_warm, cold)
            awg = np.full(dataset.n_items, -1, dtype=np.int64)
            awg[auc_warm] = np.arange(len(auc_warm), dtype=np.int64)
            self._auc_warm_global_to_local = awg
            ag = np.full(dataset.n_items, -1, dtype=np.int64)
            ag[self._auc_candidate_ids] = np.arange(len(self._auc_candidate_ids), dtype=np.int64)
            self._auc_global_to_local = ag
        return self

    def build_warm_cache(self, eval_users, warm_already_liked, N) -> "ALSModel":
        """Cache each eval user's top-N WARM items (global ids + scores), with warm-already-liked
        filtered. Warm factors are frozen across the reveal sweep, so this is built once per seed
        and reused for every k via recommend_cached() -- the report S8(b) optimization."""
        from recsys import gpu_retrieval
        import numpy as np
        if self._warm_ids is None:
            raise RuntimeError("warm cache requires candidates='warm_cold'; call prepare_gpu_recommend first")
        uf = self._uf_gpu if self._uf_gpu is not None else self.user_factors
        itf = self._itf_gpu if self._itf_gpu is not None else self.item_factors
        ids, scores = gpu_retrieval.topk_recommend(
            uf, itf, self._warm_ids, self._warm_global_to_local,
            eval_users, warm_already_liked, N=N, filter_already_liked_items=True,
            device=self._device, chunk=self._gpu_chunk)
        self._warm_cache = {"users": np.asarray(eval_users), "ids": ids, "scores": scores, "N": N}
        return self

    def recommend_cached(self, eval_users, user_items, N):
        """Top-N over warm+cold for eval_users, reusing the base model's warm cache: score only the
        cold items at this (folded) reveal level, merge with the cached warm top-N. Exactly equal to
        recommend() over the warm_cold pool -- a warm item outside the warm top-N can never enter the
        final top-N -- at a fraction of the cost. Call on a folded model. Returns global ids only."""
        from recsys import gpu_retrieval
        cache = self._warm_cache
        if cache is None or cache["N"] < N:
            raise RuntimeError("warm cache missing/too small; call build_warm_cache(eval_users, ..., N) first")
        uf = self._uf_gpu if self._uf_gpu is not None else self.user_factors
        cold_factors = self.item_factors[self._cold_ids]
        return gpu_retrieval.cold_merge_recommend(
            uf, cold_factors, self._cold_ids, self._cold_global_to_local,
            cache["ids"], cache["scores"], eval_users, user_items, N,
            device=self._device, chunk=self._gpu_chunk)

    # --- Score sources (see scores.py) -----------------------------------------------------------
    # eval.py asks a model for these instead of reaching into its factors, so a method whose score is
    # NOT user_factors @ item_factors.T -- CBHCF's per-item blend, and the interventions after it --
    # plugs into the same sweeps. For ALS each one is a thin wrapper around the factor product.
    # Call them on whichever model holds the right item factors: the base model for the frozen warm
    # block, a folded model for the cold block at reveal level k.

    def _resident_user_factors(self):
        """The GPU-resident user factors if load_factors_gpu() has been called, else the numpy
        array. A folded model shares the base's `_uf_gpu` -- fold-in never touches user factors."""
        return self._uf_gpu if self._uf_gpu is not None else self.user_factors

    def cold_block_source(self):
        """Scores over the cold items, at this model's (possibly folded) item factors."""
        from recsys import scores
        return scores.FactorItemBlock.from_block(
            self._resident_user_factors(), self.item_factors[self._cold_ids], self._device)

    def warm_auc_source(self):
        """Scores over the WARM block of the degree-matched AUC pool. Frozen across the reveal
        sweep, which is what lets mode_a_auc_sweep sort it once and reuse it for every k."""
        from recsys import scores
        return scores.FactorItemBlock.from_block(
            self._resident_user_factors(), self.item_factors[self._auc_warm_ids], self._device)

    def auc_pool_source(self):
        """Scores over the full degree-matched AUC pool (warm degree >= auc_min_degree, plus cold)."""
        from recsys import scores
        return scores.FactorItemBlock(
            self._resident_user_factors(), self.item_factors, self._auc_candidate_ids, self._device)

    def mode_b_source(self):
        """Mode B: this model's cold items x the whole user population."""
        from recsys import scores
        return scores.FactorUserBlock(
            self._resident_user_factors(), self.item_factors[self._cold_ids], self._device)

    def clear_fold_cache(self) -> "ALSModel":
        """Drop the memoized fold-in factors. Worth calling once a hyperparameter search is done --
        the cache only pays for itself when the same (dataset, level) is folded repeatedly."""
        self._fold_cache = {}
        return self

    def _cold_factors(self, cache_key, build):
        """Memoized `recalculate_item` output for the cold block, as (item_ids, factors).

        Fold-ins depend only on the dataset and the reveal level -- never on anything downstream --
        so a sweep that re-folds the same levels (a lambda search re-running the sweep per candidate
        value, say) recomputes an identical closed-form solve every time. Only the COLD rows are
        memoized: 2,727 x 64 floats is ~700 KB per level, where caching whole folded models would be
        ~125 MB each. The `build` thunk is not called on a hit, which also skips
        `revealed_item_users_at_k`'s Python-loop matrix construction.
        """
        hit = self._fold_cache.get(cache_key)
        if hit is None:
            item_ids, item_users = build()
            hit = (item_ids, self._model.recalculate_item(item_ids, item_users))
            self._fold_cache[cache_key] = hit
        return hit

    def _fold_in_with(self, item_ids, folded_item_factors):
        """Returns a NEW ALSModel with cold items' item_factors rows replaced by the fold-in factor
        (from recalculate_item's exact closed-form solve), without mutating self. User factors are
        never touched. Shared by fold_in (reveal level k) and fold_in_ceiling."""
        factors = self.item_factors.copy()
        factors[item_ids] = folded_item_factors

        folded_model = copy.copy(self)
        folded_model._model = copy.copy(self._model)
        folded_model._model.item_factors = factors
        # User factors are unchanged, so the folded model shares the base's resident _uf_gpu. But its
        # cold item rows differ, so drop _itf_gpu -- recommend_cached / Mode B use the folded cold
        # factors (numpy) directly, and a folded model's plain recommend() falls back to numpy.
        folded_model._itf_gpu = None
        return folded_model

    def fold_in(self, dataset, k):
        """Fold every cold item in at reveal level k (first-k of its reveal pool).

        At k=0, recalculate_item's closed-form solve has nothing but regularization to fall
        back on, so the resulting factor is the exact zero vector for every cold item -- a zero
        vector's dot product with any user factor is exactly zero, so the item cannot outrank
        anything.
        """
        key = (dataset_fingerprint(dataset), "k", int(k))
        return self._fold_in_with(*self._cold_factors(
            key, lambda: dataset.revealed_item_users_at_k(k)))

    def fold_in_ceiling(self, dataset):
        """Within-item warm reference: fold every cold item in with ALL its pre-test interactions
        (dataset.ceiling_pool) -- the item's own fully-warm state, evaluated on the same
        last-test_size reserved test the warm-up curve uses (degree-matched by construction)."""
        key = (dataset_fingerprint(dataset), "ceiling")
        return self._fold_in_with(*self._cold_factors(key, dataset.ceiling_item_users))
