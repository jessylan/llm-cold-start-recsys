# This file was created with the assistance of Generative AI.
"""Score sources -- the one thing `gpu_retrieval` needs from a retrieval method.

Every sweep in this project reduces to "give me a block of scores and I'll rank it". Until now that
block was always computed inline as `user_factors @ item_factors.T`, which quietly made
*bilinearity* a requirement for participating in the evaluation harness. ALS and the rank-1
Popularity/Activity duals satisfy it; CBHCF does not (its score is a per-item weighted blend of two
min-max-normalized score matrices), and neither will several of the interventions planned after it.

So the factor product moves behind two small interfaces, one per access pattern the retrieval
algorithms actually use:

  ItemBlockScores  -- (chunk of users) x (one FIXED block of items).  Mode A: top-K over the
                      candidate pool, the warm/cold split of the cached sweep, candidate-pool AUC.
  UserBlockScores  -- (chunk of items) x (ALL users).                 Mode B: ranking users for a
                      cold item, and its AUC.

`gpu_retrieval`'s functions keep their existing factor arguments and build a Factor* source
internally when no `source=` is passed, so the numeric path for ALS is unchanged op-for-op -- the
bench_8b / bench_14 / bench_15 parity gates are the check on that.

Why the split is by ACCESS PATTERN rather than one general `score(users, items)`: the expensive
setup is hoisted per block, not per chunk. `ItemBlockScores` transfers/gathers its item block once
and is then called for every user chunk; `UserBlockScores` holds the whole user matrix resident and
is sliced by item. A single general method would re-do that work on every call.

IMPORTANT -- callers mask scores in place (pinning already-liked items to -inf before a top-K or a
sort). Every `user_scores`/`item_scores` implementation must therefore return a tensor the caller
may freely mutate. The factor sources get this for free (a matmul allocates); the Dense* sources
must clone, or the precomputed block would be destroyed on first use.
"""
from typing import Protocol, runtime_checkable

import numpy as np


# Factor arguments may be numpy arrays (host) or torch tensors already resident on the GPU (see
# cf.ALSModel.load_factors_gpu). These two helpers let one code path use a resident tensor with no
# host->device copy, or transfer from numpy when that is all we have. They live here rather than in
# gpu_retrieval so the score sources can use them without a circular import; gpu_retrieval re-exports
# them under their original names.

def as_gpu(factors, device):
    """Full factor matrix as a float32 GPU tensor: a resident torch tensor is returned as-is (no
    copy), numpy is transferred once."""
    import torch
    if torch.is_tensor(factors):
        return factors
    return torch.as_tensor(np.ascontiguousarray(factors, dtype=np.float32), device=device)


def gather_rows(factors, idx, device):
    """Rows `idx` (numpy int) of `factors` as a GPU tensor: a device-side gather if `factors` is a
    resident GPU tensor (no host copy of the big matrix), else a transfer of just those rows."""
    import torch
    if torch.is_tensor(factors):
        return factors[torch.as_tensor(idx, device=device)]
    return torch.as_tensor(np.ascontiguousarray(factors[idx], dtype=np.float32), device=device)


@runtime_checkable
class ItemBlockScores(Protocol):
    """Scores for arbitrary users against one fixed, pre-selected block of items."""

    n_items: int

    def user_scores(self, user_ids: np.ndarray):
        """(len(user_ids), n_items) float32 GPU tensor, safe for the caller to mutate."""
        ...


@runtime_checkable
class UserBlockScores(Protocol):
    """Scores for a contiguous slice of items against the whole user population (Mode B)."""

    n_users: int
    n_items: int

    def item_scores(self, lo: int, hi: int):
        """(hi - lo, n_users) float32 GPU tensor, safe for the caller to mutate."""
        ...


# ---------------------------------------------------------------------------
# Bilinear sources -- ALS, and the rank-1 Popularity/Activity duals (eval._rank1)
# ---------------------------------------------------------------------------

class FactorItemBlock:
    """score(u, i) = user_factors[u] . item_factors[i], for i in `item_ids`."""

    def __init__(self, user_factors, item_factors, item_ids, device):
        self._user_factors = user_factors
        self._device = device
        self._V = gather_rows(item_factors, np.asarray(item_ids), device)   # hoisted once per block
        self.n_items = int(self._V.shape[0])

    @classmethod
    def from_block(cls, user_factors, item_factor_rows, device):
        """When the caller already holds exactly the block's factor rows (e.g. `cold_factors` from a
        fold-in) rather than the full item_factors matrix -- skips a redundant gather."""
        self = cls.__new__(cls)
        self._user_factors = user_factors
        self._device = device
        self._V = as_gpu(item_factor_rows, device)
        self.n_items = int(self._V.shape[0])
        return self

    def user_scores(self, user_ids):
        uf = gather_rows(self._user_factors, np.asarray(user_ids), self._device)
        return uf @ self._V.T


class FactorUserBlock:
    """Mode B dual: score(i, u) for every user, sliced by item."""

    def __init__(self, user_factors, item_factors_rows, device):
        self._Uf = as_gpu(user_factors, device)
        self._Cf = as_gpu(item_factors_rows, device)
        self.n_users = int(self._Uf.shape[0])
        self.n_items = int(self._Cf.shape[0])

    def item_scores(self, lo, hi):
        # Items as ROWS so the caller's top-K runs along the contiguous (user) axis -- top-K along a
        # strided axis is several times slower on the GPU.
        return self._Cf[lo:hi] @ self._Uf.T


# ---------------------------------------------------------------------------
# Precomputed sources -- for scores that are not a product of two factor matrices
# ---------------------------------------------------------------------------

class DenseItemBlock:
    """A score block that was computed ahead of time and is just indexed here.

    This is what makes the exact (un-factorized) content score practical. Freezing the CBHCF user
    profile on `ref_train` makes the content score matrix independent of BOTH the reveal level k and
    the ALS seed, so it is computed once per run and replayed for every (seed, k) instead of being
    recomputed 210 times.

    `block` is (n_rows, n_items); `user_row_of` maps a global user id to its row (-1 for users not
    covered, which must not be queried). Half precision halves residency (12,382 eval users x
    246,687 warm items is 6.1 GB at fp16, 12.2 GB at fp32); reads are promoted to float32 so the
    downstream arithmetic matches the factor path.

    The block may live on the GPU (fastest, if it fits alongside the model's factors) or stay in
    host memory, in which case only the rows for the current chunk cross the bus. `device` is always
    where the RETURNED tensor lands.

    `col_idx` selects a sub-block of columns, gathered per chunk. That is what lets ONE canonical
    block (users x the full candidate pool) serve every sub-pool the sweeps rank over -- warm, cold,
    the degree-matched AUC pool -- instead of storing a separate copy of each. The gather is over a
    chunk of rows, so it is cheap; storing four overlapping copies of a 6 GB block would not be.
    """

    def __init__(self, block, user_row_of, device, col_idx=None):
        import torch
        self._device = device
        self._block = block if torch.is_tensor(block) else torch.as_tensor(np.ascontiguousarray(block))
        self._row_of = np.asarray(user_row_of, dtype=np.int64)
        self._col_idx = None if col_idx is None else torch.as_tensor(
            np.asarray(col_idx, dtype=np.int64), device=self._block.device)
        self.n_items = int(self._block.shape[1] if col_idx is None else len(col_idx))

    def user_scores(self, user_ids):
        import torch
        rows = self._row_of[np.asarray(user_ids)]
        if (rows < 0).any():
            missing = np.asarray(user_ids)[rows < 0]
            raise KeyError(
                f"DenseItemBlock has no precomputed row for {len(missing):,} of the "
                f"{len(rows):,} requested users (e.g. {missing[:5].tolist()}). The block covers only "
                f"the users it was built for -- for CBHCF that is the eval-user set passed to "
                f"build_content_cache, not the whole population. Either query users inside that set "
                f"or rebuild the cache over a superset.")
        out = self._block[torch.as_tensor(rows, device=self._block.device)]
        if self._col_idx is not None:
            out = out[:, self._col_idx]
        out = out.to(device=self._device, dtype=torch.float32)
        # .to() copies whenever it changes device or dtype; when it did neither we must clone, since
        # the caller WILL pin already-liked entries to -inf in place and would corrupt the cache.
        return out if out.data_ptr() != self._block.data_ptr() else out.clone()


class DenseUserBlock:
    """Mode B dual of DenseItemBlock: a precomputed (n_items, n_users) block, sliced by item."""

    def __init__(self, block, device):
        import torch
        self._device = device
        self._block = block if torch.is_tensor(block) else torch.as_tensor(np.ascontiguousarray(block))
        self.n_items, self.n_users = (int(self._block.shape[0]), int(self._block.shape[1]))

    def item_scores(self, lo, hi):
        import torch
        out = self._block[lo:hi].to(device=self._device, dtype=torch.float32)
        return out if out.data_ptr() != self._block.data_ptr() else out.clone()


# ---------------------------------------------------------------------------
# Blending -- the CBHCF score, and any future "two signals, per-item mix"
# ---------------------------------------------------------------------------

class _Additive:
    """Shared arithmetic for the two additive sources.

        score(u, i) = coef_a * a[u, i] + coef_b * b[u, i]

    This is the hybrid form the literature actually uses -- content is a permanent additive term
    whose influence is never scheduled away (CTR/CDL's `v_j = theta_j + eps_j`, DropoutNet's
    permanent content pathway, and every feature-based production hybrid). It replaces the earlier
    per-item convex blend, which drove content's weight to zero once an item passed a fixed
    interaction count -- an assumption no cited method makes.

    **Scaling is multiplicative only, never an offset.** The two operands have different units, so a
    scale is needed, but subtracting a minimum would destroy the property that carries this design:
    at reveal level 0 a cold item's ALS factor is the exact zero vector, so its collaborative score
    is exactly 0 and the item is ranked on content alone. Under a min-max normalization that 0 would
    map to some positive constant and every cold item would collect a spurious collaborative bonus.
    With scale-only, zero stays zero and the content-to-collaborative handoff happens on its own as
    the collaborative term grows -- no schedule, no threshold.

    The coefficients are computed once on the base model (see `scale_over_users`) and held fixed
    across reveal levels; recomputing per level would rescale the score at every k and the warm-up
    curve would mix real signal with normalization drift.
    """

    def __init__(self, a, b, coef_a, coef_b):
        self._a, self._b = a, b
        self._ca, self._cb = float(coef_a), float(coef_b)


class AdditiveItemBlock(_Additive):
    """Sum of two ItemBlockScores over the SAME item block."""

    def __init__(self, a, b, coef_a, coef_b):
        if a.n_items != b.n_items:
            raise ValueError(f"additive operands disagree on the item block: "
                             f"{a.n_items} vs {b.n_items}")
        super().__init__(a, b, coef_a, coef_b)
        self.n_items = a.n_items

    def user_scores(self, user_ids):
        out = self._a.user_scores(user_ids)
        out *= self._ca                                   # in place: user_scores already returns
        out += self._b.user_scores(user_ids) * self._cb   # a tensor the caller may mutate
        return out


class AdditiveUserBlock(_Additive):
    """Mode B dual: sum of two UserBlockScores."""

    def __init__(self, a, b, coef_a, coef_b):
        if a.n_items != b.n_items or a.n_users != b.n_users:
            raise ValueError("additive operands disagree on shape")
        super().__init__(a, b, coef_a, coef_b)
        self.n_items, self.n_users = a.n_items, a.n_users

    def item_scores(self, lo, hi):
        out = self._a.item_scores(lo, hi)
        out *= self._ca
        out += self._b.item_scores(lo, hi) * self._cb
        return out


def scale_over_users(source, user_ids, chunk=1024, kind="std"):
    """A single positive scale summarizing `source` over `user_ids` x its item block, streamed in
    chunks so the full score matrix is never materialized.

    "std" (default) makes the two terms contribute comparable *spread*, which is what matters for a
    ranking, and is robust to the fact that content cosines are non-negative and mass-concentrated
    near zero while collaborative scores straddle zero. "range" reproduces min-max's denominator
    without its offset. Either way the result is used as a divisor only -- see _Additive.
    """
    user_ids = np.asarray(user_ids)
    if kind == "range":
        lo, hi = float("inf"), float("-inf")
        for start in range(0, len(user_ids), chunk):
            s = source.user_scores(user_ids[start:start + chunk])
            lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
        return (hi - lo) or 1.0
    # Streaming mean/variance over every scored cell (Welford is unnecessary here -- the counts are
    # exact and the values are O(1), so the naive sums are well conditioned).
    total = sq = 0.0
    n = 0
    for start in range(0, len(user_ids), chunk):
        s = source.user_scores(user_ids[start:start + chunk]).double()
        total += float(s.sum())
        sq += float((s * s).sum())
        n += s.numel()
    var = max(sq / n - (total / n) ** 2, 0.0)
    return float(np.sqrt(var)) or 1.0
