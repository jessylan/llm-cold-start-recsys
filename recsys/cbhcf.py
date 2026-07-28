"""Content-based hybrid collaborative filtering (CBHCF).

Adds a content score to an already-fit collaborative model's score:

    score(u, i) = cf(u, i) / s_cf  +  lambda * content(u, i) / s_cb

That is the hybrid form the literature actually uses. CTR (Wang & Blei 2011) and CDL (Wang et al.
2015) set an item's latent vector to `v_j = theta_j + eps_j` -- a content representation plus a
collaborative offset -- so content is structurally permanent and the collaborative part grows to
dominate only where the data supports it. DropoutNet keeps the content pathway permanently and uses
dropout on the *preference* input rather than annealing content away. Feature-based production
hybrids (factorization machines, Wide & Deep, the YouTube DNN) carry content features at every
degree. An earlier version of this module drove content's weight to zero once an item passed a
fixed interaction count; no cited method does that, and it is gone.

**The handoff is automatic, not scheduled.** At reveal level 0 a cold item's ALS fold-in factor is
the exact zero vector (Section 7 of the notebook verifies this), so `cf(u, i) = 0` and the item is
ranked on content alone. As interactions accumulate the collaborative term grows on its own and
comes to dominate. There is no threshold, no ramp, and no per-item weight -- the single free
parameter is `lambda`, selected on the cold-item VALIDATION population (`dataset.cold_val`), never
on the reported test set.

**Scaling is multiplicative only.** `s_cf` and `s_cb` equalize the two terms' spread; subtracting a
minimum would map a zero collaborative score to a positive constant and hand every cold item a
spurious collaborative bonus, destroying the property above. Both scales are measured once on the
base model and held fixed across reveal levels.

Three things make this run at Amazon-Books scale, where the original prototype could not (it built
a dense 487,790 x 487,790 item-item similarity, ~950 GB, and a dense n_users x n_items score matrix):

**1. The similarity matrix never exists.** `content.py` guarantees its item x term matrix `T` has
L2-normalized rows, so `cosine_similarity(T) == T @ T.T` and the content score factors as

    cb[users, items] = (history[users] @ T) @ T[items].T

`(H @ T) @ T.T` instead of `H @ (T @ T.T)`. The per-user profile `H @ T` is never materialized for
the whole population either -- at ~3,500 terms per user over 384,339 users that would be ~10 GB
sparse -- it is projected one user chunk at a time.

**2. The user profile is FROZEN on `ref_train`.** The notebook's controlling principle is that user
preferences are fit once and reused identically, with only the *item's* representation allowed to
move as it accumulates history -- exactly what ALS's `recalculate_item` does. Letting a user's
content profile absorb revealed cold items would add a second moving part, confounded with the
reveal axis being plotted (measured: 15.7% of eval users would shift by k=20, growing monotonically
in k). `freeze_user_profile=False` restores the prototype's behaviour for an ablation.

**3. Freezing makes the content scores k-invariant AND seed-invariant.** They are computed once per
run -- and cached to disk -- then replayed for every (seed, reveal level) instead of being
recomputed 210 times. Nothing about the content half depends on `lambda` either, so a lambda sweep
re-scores from the same cache.
"""
import copy
import os
import pickle
import time

import numpy as np
import scipy.sparse as sparse

from recsys import scores as _scores


def _row_scaled(matrix) -> sparse.csr_matrix:
    """Divide each row by its own sum, leaving all-zero rows at zero (a user with no training
    history has no content profile and must score 0, not NaN)."""
    m = sparse.csr_matrix(matrix)
    counts = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, counts, out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    return (sparse.diags(inv) @ m).astype(np.float32)


class CBHCFModel:
    """Content + collaborative additive hybrid satisfying protocol.RetrievalModel.

    Wraps an ALREADY-FIT collaborative model (`cf.ALSModel`). It is never refit here: CBHCF is an
    intervention layered on the baseline, and reusing the exact same ALS fits is what makes the
    ALS-vs-CBHCF gap attributable to the content term and nothing else.

    Usage, once per run:
        m = CBHCFModel(content_weight=LAMBDA).fit(dataset.ref_train, item_content=T, cf_model=als)
        m.prepare_gpu_recommend(dataset)        # adopts the ALS candidate pools
        m.build_content_cache(eval_users, path=...)   # the expensive part; cached to disk
        m.calibrate(eval_users)                 # fixes s_cf / s_cb
    then hand it to eval.py exactly like an ALSModel. `with_content_weight(l)` makes a cheap copy at
    a different lambda, reusing the cache and the scales -- that is what the sweep iterates over.
    """

    def __init__(self, content_weight: float = 1.0, scale_kind: str = "std",
                 freeze_user_profile: bool = True):
        self.content_weight = content_weight
        self.scale_kind = scale_kind
        self.freeze_user_profile = freeze_user_profile
        self._content = None          # item x term, L2-normalized rows (content.ContentSpace output)
        self._history = None          # user x item, row-scaled ref_train -- FROZEN
        self._cf = None               # the wrapped collaborative model
        self._s_cf = self._s_cb = None
        self._cache = None            # precomputed content score blocks (see build_content_cache)
        # Candidate pools, copied from the wrapped CF model so both methods rank over identical
        # sets -- a prerequisite for the comparison, not just a convenience.
        self._candidate_ids = self._cold_ids = self._warm_ids = None
        self._global_to_local = self._cold_global_to_local = self._warm_global_to_local = None
        self._auc_warm_ids = self._auc_candidate_ids = None
        self._auc_warm_global_to_local = self._auc_global_to_local = None
        self._device = None
        self._gpu_chunk = None
        self._warm_cache = None

    # --- fitting ----------------------------------------------------------------------------------

    def fit(self, train_matrix, *, item_content, cf_model=None) -> "CBHCFModel":
        """`item_content` is the (n_items x n_terms) row-L2-normalized matrix from
        `content.ContentSpace.transform`. `cf_model` is an already-fit collaborative model; omit it
        for a pure content-based model."""
        self._content = sparse.csr_matrix(item_content).astype(np.float32)
        self._history = _row_scaled(train_matrix)
        self._cf = cf_model
        if cf_model is not None:
            self._adopt_pools(cf_model)
        return self

    _POOL_ATTRS = ("_candidate_ids", "_cold_ids", "_warm_ids", "_global_to_local",
                   "_cold_global_to_local", "_warm_global_to_local", "_auc_warm_ids",
                   "_auc_candidate_ids", "_auc_warm_global_to_local", "_auc_global_to_local",
                   "_device", "_gpu_chunk")

    def _adopt_pools(self, cf_model):
        for attr in self._POOL_ATTRS:
            setattr(self, attr, getattr(cf_model, attr, None))

    def prepare_gpu_recommend(self, dataset=None, vram_budget_gb: float = 5.0,
                              **kwargs) -> "CBHCFModel":
        """Pools come from the wrapped CF model -- call its `prepare_gpu_recommend` first.

        The chunk size is NOT inherited. ALS's is sized for one score tensor, but the additive
        source materializes TWO of them (collaborative, then content) before summing, so its peak is
        roughly double at the same chunk. Inheriting ALS's ~8,000-user chunk overflows a 24 GB card
        on the warm block (8,000 x 243,960 x 4 B = 7.8 GB each). The factor of 3 below covers both
        operands plus the sum.
        """
        if self._cf is None or getattr(self._cf, "_candidate_ids", None) is None:
            raise RuntimeError("prepare_gpu_recommend the wrapped CF model first; CBHCF reuses its "
                               "candidate pools so both methods rank over identical sets")
        self._adopt_pools(self._cf)
        widest = max(len(self._warm_ids), len(self._candidate_ids))
        self._gpu_chunk = max(256, int(vram_budget_gb * 1e9 / (widest * 4 * 3)))
        return self

    # --- content scores ---------------------------------------------------------------------------

    def content_scores(self, user_ids, item_ids=None, _items_T=None) -> np.ndarray:
        """Exact content scores for `user_ids` x `item_ids`, dense float32.

        `_items_T` lets a caller hoist `T[items].T` out of a chunk loop. That hoist is worth 12.6x:
        re-slicing and transposing a 136M-nonzero matrix per chunk dominated everything else."""
        items_T = _items_T if _items_T is not None else self._content[np.asarray(item_ids)].T.tocsr()
        profile = self._history[np.asarray(user_ids)] @ self._content       # (b x n_terms) sparse
        return np.asarray((profile @ items_T).todense(), dtype=np.float32)

    def _profile_chunk(self, user_ids):
        return (self._history[np.asarray(user_ids)] @ self._content).astype(np.float32)

    # --- the content cache: computed once, cached to disk, replayed for every (seed, k, lambda) ----

    def build_content_cache(self, eval_users, *, dtype="float16", device=None, gpu_device=None,
                            user_chunk=256, path=None, reuse_from=None,
                            verbose=True) -> "CBHCFModel":
        """Precompute the content half of the score. Independent of the reveal level, the ALS seed
        and lambda, so this runs once per run -- and is reloaded from `path` on later runs.

        Two blocks are stored: eval users x the whole candidate pool (Mode A), and every user x the
        cold items (Mode B). Each sub-pool the sweeps rank over (warm, cold, the degree-matched AUC
        pool) is a per-chunk COLUMN VIEW of the Mode A block, not another copy of it.

        `gpu_device` (e.g. "cuda:1") computes Mode A with cuSPARSE SpMM: measured 35 s versus 10.4
        min on CPU, agreeing to 3.9e-07. Note the algorithm choice -- the product is ~99% dense, so
        SpGEMM (sparse x sparse -> sparse) is wrong for it and in fact exhausts cuSPARSE's
        workspace; SpMM (sparse x dense -> dense) is the right shape. Mode B is left on the CPU on
        purpose: its output is only n_cold wide, so densifying the profile (the step that dominates
        the GPU path) would cost far more than the sparse product saves.

        `device` is where the cached tensors live for scoring (None = host memory, which is the safe
        default at ~6 GB for Mode A).

        `reuse_from` takes another CBHCFModel and borrows what it can. The Mode B block depends only
        on the content space and the cold items -- NOT on `eval_users` -- so it is always reusable,
        and rebuilding it per model is pure waste (it is ~2 GB and several minutes). The Mode A
        block is borrowed too when the eval-user set matches.
        """
        import torch
        if self._cold_ids is None:
            raise RuntimeError("call prepare_gpu_recommend first (CBHCF needs the candidate pools)")
        eval_users = np.asarray(eval_users)
        n_users = self._history.shape[0]

        borrowed_b = None
        if reuse_from is not None and reuse_from._cache is not None:
            borrowed_b = reuse_from._cache["mode_b"]
            if np.array_equal(reuse_from._cache.get("eval_users"), eval_users):
                self._cache = dict(reuse_from._cache)      # shallow: tensors shared by reference
                if verbose:
                    print("[cbhcf] content cache reused in full")
                return self._move_cache(device)

        if path and os.path.exists(path):
            with open(path, "rb") as f:
                blob = pickle.load(f)
            if not np.array_equal(blob["eval_users"], eval_users):
                raise ValueError(f"cached content block at {path} was built for a different eval-user "
                                 f"set ({len(blob['eval_users']):,} vs {len(eval_users):,}); delete it")
            self._cache = {"eval_users": eval_users, "row_of": blob["row_of"],
                           "mode_a": torch.as_tensor(blob["mode_a"]),
                           "mode_b": torch.as_tensor(blob["mode_b"])}
            if verbose:
                print(f"[cbhcf] content cache loaded from {path}")
            return self._move_cache(device)

        row_of = np.full(n_users, -1, dtype=np.int64)
        row_of[eval_users] = np.arange(len(eval_users))
        td = getattr(torch, dtype)

        t0 = time.perf_counter()
        mode_a = self._block_gpu(eval_users, self._candidate_ids, td, gpu_device, user_chunk, verbose) \
            if gpu_device else self._block_cpu(eval_users, self._candidate_ids, td, user_chunk, verbose,
                                               "mode A")
        t_a = time.perf_counter() - t0
        # Mode B ranks the WHOLE user population against only the cold items, and wants items as
        # rows. Narrow output -> CPU sparse product, no profile densification. Independent of
        # eval_users, hence borrowable.
        t0 = time.perf_counter()
        if borrowed_b is not None:
            mode_b, t_b = borrowed_b, 0.0
            if verbose:
                print("[cbhcf] mode B: reused (independent of the eval-user set)")
        else:
            mode_b = self._block_cpu(np.arange(n_users), self._cold_ids, td, max(user_chunk, 4096),
                                     verbose, "mode B").T.contiguous()
            t_b = time.perf_counter() - t0
        self._cache = {"eval_users": eval_users, "row_of": row_of, "mode_a": mode_a, "mode_b": mode_b}

        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"eval_users": eval_users, "row_of": row_of,
                             "mode_a": mode_a.cpu().numpy(), "mode_b": mode_b.cpu().numpy()},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"[cbhcf] content cache saved to {path}")
        if verbose:
            gb = sum(v.element_size() * v.nelement() for v in (mode_a, mode_b)) / 1e9
            print(f"[cbhcf] content cache: mode_a {tuple(mode_a.shape)} in {t_a:.0f}s "
                  f"({'gpu' if gpu_device else 'cpu'})  mode_b {tuple(mode_b.shape)} in {t_b:.0f}s "
                  f"(cpu)  [{dtype}, {gb:.2f} GB]")
        return self._move_cache(device)

    def _move_cache(self, device):
        if device is not None:
            self._cache["mode_a"] = self._cache["mode_a"].to(device)
            self._cache["mode_b"] = self._cache["mode_b"].to(device)
        return self

    def _block_cpu(self, users, items, td, chunk, verbose, label):
        import torch
        items_T = self._content[np.asarray(items)].T.tocsr()      # hoisted -- worth 12.6x
        out = torch.empty((len(users), len(items)), dtype=td)
        for s in range(0, len(users), chunk):
            part = self.content_scores(users[s:s + chunk], _items_T=items_T)
            out[s:s + chunk] = torch.as_tensor(part).to(dtype=td)
            if verbose:
                print(f"\r[cbhcf] {label} (cpu): {min(s + chunk, len(users)):,}/{len(users):,}",
                      end="", flush=True)
        if verbose:
            print(f"\r[cbhcf] {label} (cpu): {len(users):,}/{len(users):,}", flush=True)
        return out

    def _block_gpu(self, users, items, td, gpu_device, chunk, verbose):
        """Mode A via cuSPARSE SpMM: out.T = T[items] @ profile.T, with T[items] resident."""
        import cupy as cp
        import cupyx.scipy.sparse as cusp
        import torch
        dev = int(str(gpu_device).split(":")[-1]) if ":" in str(gpu_device) else 0
        with cp.cuda.Device(dev):
            g_T = cusp.csr_matrix(self._content[np.asarray(items)].tocsr().astype(np.float32))
            out = torch.empty((len(users), len(items)), dtype=td)
            for s in range(0, len(users), chunk):
                prof = self._profile_chunk(users[s:s + chunk])
                g_prof_T = cp.asarray(np.asfortranarray(prof.toarray().T))
                block = (g_T @ g_prof_T).T                        # (b x n_items) dense on GPU
                out[s:s + chunk] = torch.as_tensor(cp.asnumpy(block)).to(dtype=td)
                del g_prof_T, block
                cp.get_default_memory_pool().free_all_blocks()
                if verbose:
                    print(f"\r[cbhcf] mode A (gpu): {min(s + chunk, len(users)):,}/{len(users):,}",
                          end="", flush=True)
            if verbose:
                print(f"\r[cbhcf] mode A (gpu): {len(users):,}/{len(users):,}", flush=True)
        return out

    def calibrate(self, eval_users, chunk=1024, s_cb=None) -> "CBHCFModel":
        """Measure the two scales, ONCE, on this (base) model over the full candidate pool. Folded
        models and lambda-variants inherit them, so the score is a fixed linear combination across
        the whole sweep. `s_cb` short-circuits the content pass, which is seed-invariant."""
        self._s_cb = s_cb or _scores.scale_over_users(self._content_view(), eval_users,
                                                      chunk=chunk, kind=self.scale_kind)
        self._s_cf = (1.0 if self._cf is None else
                      _scores.scale_over_users(self._cf_view(), eval_users, chunk=chunk,
                                               kind=self.scale_kind))
        return self

    def with_content_weight(self, content_weight) -> "CBHCFModel":
        """A cheap copy at a different lambda, sharing the content cache, the scales and the wrapped
        CF model by reference. This is what a lambda sweep iterates over -- nothing is recomputed."""
        out = copy.copy(self)
        out.content_weight = content_weight
        out._warm_cache = None          # the warm top-N ranking depends on lambda
        return out

    # --- score sources (the interface eval.py consumes) ----------------------------------------------

    def _cols(self, item_ids):
        """Columns of the Mode A cache holding `item_ids` -- the cache spans the candidate pool, so
        `_global_to_local` is exactly this map."""
        return self._global_to_local[np.asarray(item_ids)]

    def _content_view(self, item_ids=None):
        if self._cache is None:
            raise RuntimeError("call build_content_cache(eval_users) first")
        return _scores.DenseItemBlock(self._cache["mode_a"], self._cache["row_of"], self._device,
                                      col_idx=None if item_ids is None else self._cols(item_ids))

    def _cf_view(self, item_ids=None):
        ids = self._candidate_ids if item_ids is None else np.asarray(item_ids)
        return _scores.FactorItemBlock(self._cf._resident_user_factors(), self._cf.item_factors,
                                       ids, self._device)

    def _combined(self, item_ids=None):
        ids = self._candidate_ids if item_ids is None else np.asarray(item_ids)
        return _scores.AdditiveItemBlock(self._cf_view(ids), self._content_view(ids),
                                         1.0 / self._s_cf, self.content_weight / self._s_cb)

    def cold_block_source(self):
        return self._combined(self._cold_ids)

    def warm_source(self):
        return self._combined(self._warm_ids)

    def warm_auc_source(self):
        return self._combined(self._auc_warm_ids)

    def auc_pool_source(self):
        return self._combined(self._auc_candidate_ids)

    def mode_b_source(self):
        return _scores.AdditiveUserBlock(
            self._cf.mode_b_source(), _scores.DenseUserBlock(self._cache["mode_b"], self._device),
            1.0 / self._s_cf, self.content_weight / self._s_cb)

    # --- retrieval, mirroring cf.ALSModel -------------------------------------------------------------

    def load_factors_gpu(self, device=None, items=True) -> "CBHCFModel":
        self._cf.load_factors_gpu(device=device, items=items)
        return self

    def free_factors_gpu(self) -> "CBHCFModel":
        self._cf.free_factors_gpu()
        return self

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        from recsys import gpu_retrieval
        return gpu_retrieval.topk_recommend(
            None, None, self._candidate_ids, self._global_to_local, userid, user_items, N=N,
            filter_already_liked_items=filter_already_liked_items, device=self._device,
            chunk=self._gpu_chunk, source=self._combined())

    def build_warm_cache(self, eval_users, warm_already_liked, N) -> "CBHCFModel":
        """Cache each eval user's top-N WARM items. Valid across the whole reveal sweep for the same
        reason it is for ALS: warm items are never revealed, so their CF factors and their content
        scores are both frozen -- the warm block's score does not depend on k."""
        from recsys import gpu_retrieval
        ids, sc = gpu_retrieval.topk_recommend(
            None, None, self._warm_ids, self._warm_global_to_local, eval_users, warm_already_liked,
            N=N, filter_already_liked_items=True, device=self._device, chunk=self._gpu_chunk,
            source=self.warm_source())
        self._warm_cache = {"users": np.asarray(eval_users), "ids": ids, "scores": sc, "N": N}
        return self

    def recommend_cached(self, eval_users, user_items, N):
        from recsys import gpu_retrieval
        cache = self._warm_cache
        if cache is None or cache["N"] < N:
            raise RuntimeError("warm cache missing/too small; call build_warm_cache(eval_users, ..., N) first")
        return gpu_retrieval.cold_merge_recommend(
            None, None, self._cold_ids, self._cold_global_to_local, cache["ids"], cache["scores"],
            eval_users, user_items, N, device=self._device, chunk=self._gpu_chunk,
            source=self.cold_block_source())

    # --- fold-in ---------------------------------------------------------------------------------------

    def _folded(self, cf_folded) -> "CBHCFModel":
        out = copy.copy(self)
        out._cf = cf_folded
        return out

    def _unfreeze(self, folded, count_matrix):
        """Ablation path (`freeze_user_profile=False`): let the profile absorb revealed items, as
        the original prototype did. Invalidates the content cache, so it is far slower AND confounds
        the warm-up curve -- see the module docstring."""
        folded._history = _row_scaled(count_matrix)
        folded._cache = None
        return folded

    def fold_in(self, dataset, k) -> "CBHCFModel":
        """Advance to reveal level k without mutating self. Under the additive model only ONE thing
        moves: the wrapped CF model's cold item factors. The content term is entirely static, and
        there is no blend weight to update."""
        folded = self._folded(self._cf.fold_in(dataset, k) if self._cf is not None else None)
        return folded if self.freeze_user_profile else self._unfreeze(
            folded, dataset.ref_train + dataset.revealed_matrix_at_k(k))

    def fold_in_ceiling(self, dataset) -> "CBHCFModel":
        """Within-item ceiling: every cold item with ALL of its pre-test history."""
        folded = self._folded(self._cf.fold_in_ceiling(dataset) if self._cf is not None else None)
        return folded if self.freeze_user_profile else self._unfreeze(
            folded, dataset.ref_train + dataset.ceiling_matrix())


def wrap_seeds(cf_models, train_matrix, item_content, eval_users, *, content_weight=1.0,
               scale_kind="std", freeze_user_profile=True, cache_dtype="float16",
               cache_device=None, gpu_device=None, user_chunk=256, cache_path=None, verbose=True):
    """One CBHCF per collaborative seed, all SHARING a single content cache.

    The ensemble exists to separate signal from ALS initialization noise, so CBHCF must wrap every
    seed rather than one -- otherwise its band would not be comparable to ALS's. But the content
    half depends on neither the seed nor the reveal level, so the cache is built once and shared by
    reference; only the collaborative scale is remeasured per seed, since factor scales differ by
    initialization.
    """
    models, base = [], None
    for i, cf_model in enumerate(cf_models):
        m = CBHCFModel(content_weight=content_weight, scale_kind=scale_kind,
                       freeze_user_profile=freeze_user_profile)
        m.fit(train_matrix, item_content=item_content, cf_model=cf_model)
        m.prepare_gpu_recommend()
        if base is None:
            m.build_content_cache(eval_users, dtype=cache_dtype, device=cache_device,
                                  gpu_device=gpu_device, user_chunk=user_chunk, path=cache_path,
                                  verbose=verbose)
            m.calibrate(eval_users)
            base = m
        else:
            m.build_content_cache(eval_users, reuse_from=base, device=cache_device, verbose=False)
            m.calibrate(eval_users, s_cb=base._s_cb)
        if verbose:
            print(f"[cbhcf] seed {i}: s_cf {m._s_cf:.5f}  s_cb {m._s_cb:.5f}  "
                  f"lambda {m.content_weight}", flush=True)
        models.append(m)
    return models
