# This file was created with the assistance of Generative AI.
"""Provider-side equity and exposure metrics: whose items actually get recommended, not just
whether a cold item becomes retrievable. Companion to eval.py's warm-up curve (Mode A/B), this
module asks whether exposure concentrates on already-well-represented providers ("rich get
richer") or whether the providers of cold items get a share proportional to what they contribute.

Plan of record and the reasoning behind every choice below: design_documents/06-provider-equity.md.
The five things that define the measurement, all of which used to be implicit:

  1. PROVIDER IDENTITY is `content.canonical_creator` -- the SAME partition the content models
     see. Building a separate one from raw `author_name` would group the catalogue differently
     from the way the model groups it, and the fairness number would describe a partition the
     model never had access to.
  2. PROVIDER UNIVERSE is every provider with >= 1 item in the CANDIDATE POOL, zeros included.
     Providers who received nothing are the observations a Gini exists to count; providers whose
     items were never candidates never had a chance and are excluded.
  3. EXPOSURE WEIGHTING is a position discount, several reported side by side. Uniform-over-top-K
     is not the absence of a user model, it is the claim that rank 100 is worth as much as rank 1.
  4. THE FAIRNESS BASELINE is reported three ways -- catalog share (all items), catalog share
     (cold items only), and merit share from held-out relevance.
  5. AGGREGATION is six statistics, not one mean over providers, which on a heavy tail is
     dominated by the thousands of one-book authors.

WHAT CHANGED FROM THE FIRST DRAFT OF THIS MODULE (all four were silent, none raised):
  D1  ids were built with a `book_`/`movie_` prefix that `load.Dataset` never uses, so every item
      mapped to UNKNOWN, Gini came out 0.0 and the equity ratio 1.0 -- "perfect equality".
  D2  `value_counts()` dropped zero-exposure providers, so Gini was computed only over providers
      who received something. Reported inequality came out drastically too EQUAL.
  D3  a 38.4M-element object array hashed per (seed, k) made the full user population look
      infeasible. Integer codes turn that into a bincount; it is the population that matters.
  D4  the equity ratio divided an all-recommendations exposure share by a cold-items-only catalog
      share. Numerator and denominator now always cover the same population.

RENAME PENDING: `k` means interaction count here, matching eval.py, not list length (`K`). Hold
off renaming to `n` until eval.py's `k` gets renamed too.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from recsys.protocol import RetrievalModel

UNATTRIBUTED = "UNKNOWN"

#: Bump on ANY change to what these metrics mean -- a new discount in DEFAULT_DISCOUNTS, a change
#: to a reference share, a change to what `ratio_stats` reports. Stamped into persisted results and
#: intended as a component of any cache key built over this module's output. Data fingerprints
#: cannot see a code change, which is the failure mode that produced D2 and D4 in the first place
#: and that nearly resurfaced stale content blocks and a stale tuning grid elsewhere in this repo.
EQUITY_METRICS_VERSION = 1


# ---------------------------------------------------------------------------
# Item -> provider mapping. Integer codes, not an object array -- see D3.
# ---------------------------------------------------------------------------

@dataclass
class ProviderMap:
    """`codes[item_index]` -> provider code in [0, n_providers), or -1 for an item outside the
    candidate pool. `names[code]` -> the canonical creator string.

    The pool restriction is the provider-universe decision (2) made explicit: an item the model
    can never retrieve must not contribute a zero to the Gini, because its provider never had a
    chance. Everything downstream sizes its arrays with `n_providers`, so the universe is fixed
    once here rather than implied differently by each metric.
    """
    codes: np.ndarray          # int32, length n_items
    names: np.ndarray          # object, length n_providers
    n_providers: int
    n_pool_items: int
    unattributed_code: int     # code of UNATTRIBUTED, or -1 if no pooled item is unattributed

    def catalog_counts(self, item_mask=None) -> np.ndarray:
        """Items per provider, length n_providers. `item_mask` (boolean over items) narrows to a
        sub-population -- e.g. the cold items -- for the cold-vs-cold reference share."""
        codes = self.codes if item_mask is None else self.codes[item_mask]
        codes = codes[codes >= 0]
        return np.bincount(codes, minlength=self.n_providers).astype(np.float64)


def build_provider_map(creator_of, pool_items=None, n_items=None) -> ProviderMap:
    """`creator_of` is `content.canonical_creator`'s array (item_index -> creator string).
    `pool_items` is the candidate pool (`model._candidate_ids`); None means every item.

    Providers are exactly the distinct creators appearing among `pool_items`. An item outside the
    pool gets code -1 and is invisible to every metric here.
    """
    creator_of = np.asarray(creator_of, dtype=object)
    n_items = len(creator_of) if n_items is None else n_items
    pool = np.arange(n_items) if pool_items is None else np.asarray(pool_items, dtype=np.int64)

    names, pooled_codes = np.unique(creator_of[pool].astype(str), return_inverse=True)
    codes = np.full(n_items, -1, dtype=np.int32)
    codes[pool] = pooled_codes.astype(np.int32)
    unattributed = int(np.flatnonzero(names == UNATTRIBUTED)[0]) if UNATTRIBUTED in set(names) else -1
    return ProviderMap(codes=codes, names=names, n_providers=len(names),
                       n_pool_items=len(pool), unattributed_code=unattributed)


def load_provider_map(meta_path, dataset, pool_items=None):
    """Convenience: metadata parquet -> ProviderMap, going through `content.canonical_creator` so
    the partition is the model's. Returns (ProviderMap, stats) -- gate on
    `stats["unattributed_frac"]`, which is ~0.5% when the fallback chain is working and jumps to
    ~1.0 if the ids ever stop matching `dataset.index_to_item` (D1)."""
    import pandas as _pd
    from recsys import content

    meta = _pd.read_parquet(meta_path, columns=["parent_asin", "author_name", "store"])
    creator_of, stats = content.canonical_creator(meta, dataset.n_items, dataset.index_to_item,
                                                  unattributed=UNATTRIBUTED)
    return build_provider_map(creator_of, pool_items, dataset.n_items), stats


# ---------------------------------------------------------------------------
# Position discounts. See design doc section 3 for the comparison table; the short version is that
# log is the headline because NDCG uses exactly this discount, so the accuracy curve and the equity
# curve describe the same hypothetical user.
# ---------------------------------------------------------------------------

#: Reported side by side. RBP's parameter is the interpretable one -- expected examination depth is
#: 1/(1-p) -- so 0.90 is "users look at about 10 items", with 0.80/0.95 as the sensitivity band.
DEFAULT_DISCOUNTS = ("uniform@10", "uniform@20", "uniform@100", "log", "rbp0.90", "rbp0.80", "rbp0.95")
HEADLINE_DISCOUNT = "log"


def discount_weights(name: str, K: int) -> np.ndarray:
    """Length-K weight vector, weight[r-1] = attention paid to rank r. Not normalized: every
    consumer here takes shares or a scale-invariant Gini, so a constant factor cannot matter."""
    r = np.arange(1, K + 1, dtype=np.float64)
    if name.startswith("uniform@"):
        return (r <= int(name.split("@")[1])).astype(np.float64)
    if name == "log":
        return 1.0 / np.log2(1.0 + r)                      # exactly NDCG's discount
    if name.startswith("rbp"):
        p = float(name[3:])
        if not 0.0 < p < 1.0:
            raise ValueError(f"RBP persistence must be in (0, 1), got {p}")
        return p ** (r - 1.0)
    if name.startswith("zipf"):
        return 1.0 / r ** (float(name[4:]) if len(name) > 4 else 1.0)
    raise ValueError(f"unknown discount {name!r}")


# ---------------------------------------------------------------------------
# Metric primitives -- same level as eval.py's mode_a_metrics_at_k, operating on already-computed
# recommendations rather than on a model.
# ---------------------------------------------------------------------------

def gini(values) -> float:
    """Gini over a NON-NEGATIVE vector covering the WHOLE provider universe, zeros included (D2).
    0 = every provider equal, 1 = one provider holds everything.

    Passing only the providers who received exposure is the D2 bug and reports a catalogue as far
    more equal than it is; callers should be handing in a length-n_providers array."""
    v = np.sort(np.asarray(values, dtype=np.float64))       # ascending: the weighting below favours the top
    total = v.sum()
    if total <= 0:
        return 0.0
    n = len(v)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * v) / (n * total))


def rank_exposure_counts(rec_ids, pmap: ProviderMap) -> np.ndarray:
    """(K, n_providers) -- how much raw exposure each provider takes at each RANK.

    Every discount is then a matrix-vector product against this, so all seven cost one pass over
    `rec_ids` instead of seven. `rec_ids` is (n_users, K) of global item indices, exactly what
    `model.recommend` / `recommend_cached` return.
    """
    rec_ids = np.asarray(rec_ids)
    if rec_ids.ndim == 1:
        rec_ids = rec_ids[None, :]
    K = rec_ids.shape[1]
    codes = pmap.codes[rec_ids]                             # (n_users, K), -1 outside the pool
    out = np.zeros((K, pmap.n_providers), dtype=np.float64)
    for r in range(K):                                      # per rank: bounded memory, no big tile
        col = codes[:, r]
        col = col[col >= 0]
        out[r] = np.bincount(col, minlength=pmap.n_providers)
    return out


def exposure_from_ranks(rank_counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Discounted exposure per provider: weights (K,) applied to rank_counts (K, n_providers)."""
    return weights @ rank_counts


def _shares(counts):
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)


def merit_shares(dataset, pmap: ProviderMap, weights=None, cold_only=True) -> np.ndarray:
    """Merit-proportional reference share: what each provider "deserves" if exposure should track
    RELEVANCE rather than catalogue footprint (Singh & Joachims 2018; Diaz et al. 2020).

    Relevance is the held-out interactions in `dataset.test_matrix` -- binary, and the only
    relevance signal that is not the model's own opinion of itself (scoring merit with the model
    being audited is circular).

    Discount-matched, which is the point: a position-discounted numerator over an undiscounted
    denominator would be D4 in a new outfit. The target is the exposure the relevant items would
    receive under an IDEAL ranking, so a user with R relevant items has them occupying ranks
    1..R. With binary relevance every ordering of those R is equally ideal, so each takes the
    mean of the top-R weights rather than an arbitrary one of them.

    `cold_only=True` restricts to cold items' held-out interactions -- the population the
    interventions actually target.
    """
    test = dataset.test_matrix.tocsr()
    if cold_only:
        keep = np.zeros(dataset.n_items, dtype=bool)
        keep[dataset.cold_item_ids] = True
    else:
        keep = np.ones(dataset.n_items, dtype=bool)

    merit = np.zeros(pmap.n_providers, dtype=np.float64)
    indptr, indices = test.indptr, test.indices
    per_user_n = np.diff(indptr)
    # Precompute mean(weights[:R]) for every R actually present, so the loop below is a lookup.
    if weights is None:
        ideal_value = None
    else:
        max_r = int(per_user_n.max()) if len(per_user_n) else 0
        csum = np.concatenate([[0.0], np.cumsum(weights)])
        # A user with more relevant items than K cannot have them all shown; the surplus gets zero.
        ideal_value = np.array([csum[min(R, len(weights))] / R if R > 0 else 0.0
                                for R in range(max_r + 1)])

    for u in np.flatnonzero(per_user_n):
        items = indices[indptr[u]:indptr[u + 1]]
        items = items[keep[items]]
        if not len(items):
            continue
        codes = pmap.codes[items]
        codes = codes[codes >= 0]
        if not len(codes):
            continue
        # R is the user's FULL held-out count: an ideal ranking must place all of them, and the
        # per-item value falls as R grows. Filtering to cold items afterwards attributes only the
        # cold share of that ideal exposure, which is what cold_only is asking for.
        val = 1.0 if ideal_value is None else ideal_value[per_user_n[u]]
        np.add.at(merit, codes, val)
    return _shares(merit)


def ratio_stats(ratio: np.ndarray, exposure_share: np.ndarray) -> dict:
    """Six summaries of the per-provider ratio distribution, replacing the single unweighted mean.

    Ratios are multiplicative, so 4x and 0.25x average to 2.1x, not 1.0 -- only the
    exposure-weighted mean (what the average IMPRESSION sees) and the geometric mean are
    defensible central tendencies. The percentiles are invariant to that and safe as reported.
    NaN entries (reference share 0 -- the provider is not in the reference population at all) are
    excluded from every statistic rather than counted as anything.
    """
    finite = np.isfinite(ratio)
    r = ratio[finite]
    if not len(r):
        return {"exposure_weighted_mean": float("nan"), "unweighted_mean": float("nan"),
                "geometric_mean": float("nan"), "median": float("nan"), "p10": float("nan"),
                "p25": float("nan"), "p75": float("nan"), "p90": float("nan"),
                "frac_below_1": float("nan"), "n_scored": 0}
    w = exposure_share[finite]
    wsum = w.sum()
    pos = r > 0
    q = np.percentile(r, [10, 25, 50, 75, 90])
    return {
        "exposure_weighted_mean": float((w @ r) / wsum) if wsum > 0 else float("nan"),
        "unweighted_mean": float(np.mean(r)),
        "geometric_mean": float(np.exp(np.mean(np.log(r[pos])))) if pos.any() else float("nan"),
        "median": float(q[2]), "p10": float(q[0]), "p25": float(q[1]),
        "p75": float(q[3]), "p90": float(q[4]),
        "frac_below_1": float(np.mean(r < 1.0)),
        "n_scored": int(len(r)),
    }


def equity_ratio(exposure: np.ndarray, reference_share: np.ndarray) -> tuple:
    """(ratio, exposure_share). ratio = exposure share / reference share, per provider; 1.0 is
    proportional, >1 over-exposed. NaN where the reference share is zero -- undefined, not
    infinite, and `ratio_stats` drops those.

    D4: `exposure` and `reference_share` MUST describe the same population. Cold-item exposure
    goes with the cold catalog share; all-recommendation exposure goes with the full catalog
    share. Mixing them attributes a big warm provider's whole exposure to its one cold title.
    """
    exposure_share = _shares(np.asarray(exposure, dtype=np.float64))
    ref = np.asarray(reference_share, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref > 0, exposure_share / ref, np.nan)
    return ratio, exposure_share


def equity_ratio_table(exposure, reference_share, pmap: ProviderMap) -> pd.DataFrame:
    """The per-provider view, for eyeballing who is over- and under-exposed. Not used by the
    sweeps (they keep scalars); this is the drill-down when a curve looks surprising."""
    ratio, exposure_share = equity_ratio(exposure, reference_share)
    return pd.DataFrame({"provider": pmap.names, "exposure": exposure,
                         "exposure_share": exposure_share,
                         "reference_share": reference_share, "equity_ratio": ratio}
                        ).sort_values("exposure_share", ascending=False)


# ---------------------------------------------------------------------------
# The accumulator: one bundle of metrics per (k, seed, discount), computed as the recommendations
# arrive and reduced to scalars immediately. Holding per-provider vectors for every (k, seed,
# discount) would be ~1.6 TB at Books scale, which is why nothing is retained.
# ---------------------------------------------------------------------------

class ExposureAccumulator:
    """Consumes `rec_ids` per (k index, seed index) and produces the equity curves.

    Designed to be driven by `eval.sweep_mode_a_cached`'s `on_recs` hook, so the equity numbers
    describe the exact recommendation lists the NDCG curve describes, at no extra retrieval cost.
    It works equally well fed from any other source of top-K lists.
    """

    def __init__(self, pmap, dataset, n_k, n_seeds, K,
                 discounts=DEFAULT_DISCOUNTS, variance_at_ki=None, users=None):
        self.pmap, self.K = pmap, K
        self.discounts = tuple(discounts)
        self.weights = {d: discount_weights(d, K) for d in self.discounts}
        self.n_k, self.n_seeds = n_k, n_seeds

        cold_mask = np.zeros(dataset.n_items, dtype=bool)
        cold_mask[dataset.cold_item_ids] = True
        self.cold_mask = cold_mask
        self.cold_codes = set(int(c) for c in pmap.codes[cold_mask] if c >= 0)
        self._is_cold_provider = np.zeros(pmap.n_providers, dtype=bool)
        self._is_cold_provider[list(self.cold_codes)] = True

        # Reference shares are k- and seed-invariant, so they are built once here.
        self.catalog_share_all = _shares(pmap.catalog_counts())
        self.catalog_share_cold = _shares(pmap.catalog_counts(cold_mask))

        # MERIT IS ONLY DEFINED WHEN THE SCORED POPULATION IS THE MERIT POPULATION. Merit comes
        # from `dataset.test_matrix`, i.e. from the users holding a held-out interaction. Scoring a
        # DIFFERENT set of users -- the whole population, say -- and dividing that exposure share
        # by this merit share compares a numerator and a denominator drawn from different
        # populations, which is precisely the D4 defect in a new outfit. Catalog share has no such
        # problem: it counts items, not users.
        #
        # So it is computed only when the two sets match exactly, and the three `merit_ratio.*`
        # stats come back NaN otherwise. Silently returning a plausible number is the one thing
        # this module must not do.
        _test = dataset.test_matrix.tocsr()
        merit_users = np.flatnonzero(np.diff(_test.indptr))
        scored = np.arange(dataset.n_users) if users is None else np.asarray(users)
        self.merit_users, self.n_scored_users = merit_users, len(scored)
        self.merit_available = (len(scored) == len(merit_users)
                                and np.array_equal(np.sort(scored), np.sort(merit_users)))
        self.merit_share_cold = ({d: merit_shares(dataset, pmap, self.weights[d], cold_only=True)
                                  for d in self.discounts} if self.merit_available else None)
        self._nan_stats = ratio_stats(np.array([]), np.array([]))

        self.results = {d: [[None] * n_seeds for _ in range(n_k)] for d in self.discounts}
        # Per-provider shares at ONE k, across seeds: the deterministic-ranking diagnostic. If Gini
        # is stable across seeds while individual providers' shares swing, the concentration is
        # real but who occupies the head is substantially arbitrary (near-tie amplification).
        self.variance_at_ki = (n_k - 1) if variance_at_ki is None else variance_at_ki
        self.per_provider_by_seed = np.full((n_seeds, pmap.n_providers), np.nan, dtype=np.float32)

    def add(self, ki, si, rec_ids):
        """Fold in one (reveal level, seed) worth of recommendations."""
        rec_ids = np.asarray(rec_ids)
        if rec_ids.ndim == 1:
            rec_ids = rec_ids[None, :]
        rank_counts = self._rank_counts_masked(rec_ids, None)
        # Cold-item exposure is counted separately, over the SAME rank positions: the cold-vs-cold
        # ratio's numerator must cover the same population as its denominator (D4). Masking the
        # codes -- not the ids -- is what keeps a filtered-out slot from being attributed to item 0.
        rank_counts_cold = self._rank_counts_masked(rec_ids, self.cold_mask[rec_ids])

        for d in self.discounts:
            w = self.weights[d]
            exposure_all = exposure_from_ranks(rank_counts, w)
            exposure_cold = exposure_from_ranks(rank_counts_cold, w)

            r_cat, s_all = equity_ratio(exposure_all, self.catalog_share_all)
            r_cold, s_cold = equity_ratio(exposure_cold, self.catalog_share_cold)
            if self.merit_available:
                r_merit, _ = equity_ratio(exposure_cold, self.merit_share_cold[d])
                merit_stats = ratio_stats(r_merit, s_cold)
            else:
                merit_stats = self._nan_stats          # scored population != merit population

            self.results[d][ki][si] = {
                "gini": gini(exposure_all),
                "gini_cold": gini(exposure_cold),
                "cold_exposure_share": float(exposure_cold.sum() / exposure_all.sum())
                                       if exposure_all.sum() > 0 else 0.0,
                "top1pct_share": _top_share(s_all, 0.01),
                "catalog_ratio": ratio_stats(r_cat, s_all),
                "cold_ratio": ratio_stats(r_cold, s_cold),
                "merit_ratio": merit_stats,
            }
            if d == HEADLINE_DISCOUNT and ki == self.variance_at_ki:
                self.per_provider_by_seed[si] = s_all.astype(np.float32)

    def _rank_counts_masked(self, rec_ids, keep):
        """(K, n_providers) rank-resolved exposure counts, over only the slots where `keep` is
        True (`keep=None` counts every slot). Masking the CODES rather than the ids is what keeps
        an excluded slot from being silently attributed to item 0."""
        codes = self.pmap.codes[rec_ids]
        if keep is not None:
            codes = np.where(keep, codes, -1)
        out = np.zeros((rec_ids.shape[1], self.pmap.n_providers), dtype=np.float64)
        for r in range(rec_ids.shape[1]):
            col = codes[:, r]
            col = col[col >= 0]
            out[r] = np.bincount(col, minlength=self.pmap.n_providers)
        return out

    # -- reduction ----------------------------------------------------------
    def curves(self):
        """{discount: {metric: {"mean": [...per k], "std": [...]}}} -- the same shape eval.sweep
        returns, so an equity curve plots on the warm-up curve's x-axis."""
        out = {}
        for d in self.discounts:
            metrics = {}
            for name in ("gini", "gini_cold", "cold_exposure_share", "top1pct_share"):
                metrics[name] = self._reduce(d, lambda r, n=name: r[n])
            for table in ("catalog_ratio", "cold_ratio", "merit_ratio"):
                for stat in ("exposure_weighted_mean", "geometric_mean", "median",
                             "p10", "p90", "frac_below_1", "unweighted_mean"):
                    metrics[f"{table}.{stat}"] = self._reduce(
                        d, lambda r, t=table, s=stat: r[t][s])
            out[d] = metrics
        return out

    def _reduce(self, d, get):
        mean, std = [], []
        for ki in range(self.n_k):
            vals = [get(r) for r in self.results[d][ki] if r is not None]
            # An all-NaN slice is legitimate, not an error: `geometric_mean` is undefined when an
            # arm takes zero cold exposure (ALS at k=0, where every cold factor is exactly zero).
            # np.nanmean warns on it, so short-circuit instead of emitting a RuntimeWarning per k.
            finite = [v for v in vals if v == v]
            mean.append(float(np.mean(finite)) if finite else float("nan"))
            std.append(float(np.std(finite)) if finite else float("nan"))
        return {"mean": mean, "std": std}

    def seed_variance(self, top_n=None):
        """Across-seed spread of each provider's exposure share at `variance_at_ki`. Returns a
        DataFrame sorted by mean share -- the near-tie-arbitrariness readout (design doc,
        "Ranking stochasticity"). `top_n` limits it to the head, where it is legible."""
        m = np.nanmean(self.per_provider_by_seed, axis=0)
        s = np.nanstd(self.per_provider_by_seed, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(m > 0, s / m, np.nan)
        df = pd.DataFrame({"provider": self.pmap.names, "mean_share": m, "std_share": s,
                           "cv": cv}).sort_values("mean_share", ascending=False)
        return df.head(top_n) if top_n else df


def _top_share(shares, frac):
    """Share of exposure held by the top `frac` of providers -- the concentration number that is
    legible without knowing what a Gini of 0.87 feels like."""
    n = max(1, int(math.ceil(len(shares) * frac)))
    return float(np.sort(shares)[::-1][:n].sum())


# ---------------------------------------------------------------------------
# Generic path -- protocol-only, no warm cache. Needed for the Popularity floor (which has none)
# and used by bench_29 as the reference the cached path is checked against.
#
# RENAME PENDING: `k` is interaction count, matching eval.py.
# ---------------------------------------------------------------------------

def _check_model(model):
    if not isinstance(model, RetrievalModel):
        raise TypeError(
            f"{model!r} does not satisfy protocol.RetrievalModel (needs fit, recommend, fold_in)")


def evaluate_provider_equity_at_k(model, dataset, pmap, k, K=100, users=None,
                                  discounts=DEFAULT_DISCOUNTS):
    """Fold `model` in at reveal level k (same mechanism as eval.evaluate_at_k) and return one
    accumulator's worth of metrics, keyed by discount.

    `users` defaults to the WHOLE population: exposure is a property of the recommendation
    surface, so restricting it changes the measurement rather than optimizing it (unlike eval.py's
    restriction to users with a held-out item, which is exact). Pass an explicit array to measure
    a subset deliberately.
    """
    _check_model(model)
    train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
    folded = model.fold_in(dataset, k)
    user_ids = np.arange(dataset.n_users) if users is None else np.asarray(users)
    rec_ids, _ = folded.recommend(user_ids, train_k[user_ids], N=K,
                                  filter_already_liked_items=True)
    acc = ExposureAccumulator(pmap, dataset, n_k=1, n_seeds=1, K=K, discounts=discounts,
                              users=user_ids)
    acc.add(0, 0, rec_ids)
    return {d: acc.results[d][0][0] for d in acc.discounts}


def sweep_provider_equity(models, dataset, pmap, k_levels, K=100, users=None,
                          discounts=DEFAULT_DISCOUNTS, accumulator=False):
    """Provider-equity analog of eval.sweep(): the generic, no-cache path across k_levels and
    seeds. Returns the same curve shape as `ExposureAccumulator.curves()`, or the accumulator
    itself with `accumulator=True`.

    Correct for any RetrievalModel -- including the Popularity floor, which has no warm cache --
    and the reference `bench_29` checks the cached path against. At Books scale prefer
    `sweep_provider_equity_full`, which is the same measurement at a fraction of the retrieval.
    """
    user_ids = np.arange(dataset.n_users) if users is None else np.asarray(users)
    _check_user_coverage(models, user_ids)
    acc = ExposureAccumulator(pmap, dataset, n_k=len(k_levels), n_seeds=len(models), K=K,
                              discounts=discounts, users=user_ids)
    for ki, k in enumerate(k_levels):
        train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)
        for si, model in enumerate(models):
            _check_model(model)
            folded = model.fold_in(dataset, k)
            rec_ids, _ = folded.recommend(user_ids, train_k[user_ids], N=K,
                                          filter_already_liked_items=True)
            acc.add(ki, si, rec_ids)
    return acc if accumulator else acc.curves()


# ---------------------------------------------------------------------------
# Cached path -- the one to actually run at Books scale.
# ---------------------------------------------------------------------------

def covered_users(model):
    """The user ids `model` can score, or None if it can score anybody.

    ALS computes every score from factors, so it is unrestricted. CBHCF and Intervention A are
    not: their content half is a PRECOMPUTED user x item block, built once over the eval-user set
    (`cbhcf.build_content_cache`), and `scores.DenseItemBlock` raises for any user outside it.

    That is a hard constraint, not a tuning knob. The Books block is 12,382 users x 246,687 warm
    items = 6.1 GB at fp16; the full 384,339-user population would be ~189 GB per arm. So a
    whole-population exposure measurement is available for ALS and Popularity and is simply not
    computable for the hybrids -- see design_documents/06-provider-equity.md, decision 2.
    """
    cache = getattr(model, "_cache", None)
    if not cache or cache.get("row_of") is None:
        return None
    return np.flatnonzero(np.asarray(cache["row_of"]) >= 0)


def common_covered_users(*model_groups, n_users=None):
    """Intersection of `covered_users` across every model in every group -- the largest user set
    all arms can be scored on, and therefore the only one they can be COMPARED on.

    Returns None if nothing is restricted. Pass this as `users=` so every arm is measured over the
    same population; measuring arms over different populations would make the between-arm
    difference (the whole point of the section) uninterpretable.
    """
    common = None
    for group in model_groups:
        for model in group or []:
            cov = covered_users(model)
            if cov is None:
                continue
            common = cov if common is None else np.intersect1d(common, cov, assume_unique=False)
    return common


def _check_user_coverage(models, user_ids):
    """Fail before doing any work, not 10 minutes into a sweep, and say what to pass instead."""
    for model in models:
        cov = covered_users(model)
        if cov is None:
            continue
        missing = np.setdiff1d(user_ids, cov)
        if len(missing):
            raise ValueError(
                f"{type(model).__name__} can only score the {len(cov):,} users its precomputed "
                f"content block was built for, but {len(missing):,} of the {len(user_ids):,} "
                f"requested users are outside it (e.g. {missing[:5].tolist()}). Rebuilding the "
                f"block over the whole population is not an option -- it is ~6 GB for 12,382 "
                f"users, so ~189 GB for 384,339. Pass "
                f"`users=equity_metrics.common_covered_users(seed_models, cbhcf_models, ia_models)` "
                f"so every arm is measured over the same population.")


def _require_cached_surface(model):
    for attr in ("build_warm_cache", "recommend_cached", "load_factors_gpu", "free_factors_gpu"):
        if not hasattr(model, attr):
            raise TypeError(
                f"{type(model).__name__} has no {attr}: the cached equity pass needs a model "
                "prepared with prepare_gpu_recommend(candidates='warm_cold'). Use "
                "sweep_provider_equity() for models without a warm cache (e.g. PopularityModel).")


def sweep_provider_equity_full(models, dataset, pmap, k_levels, K=100, users=None,
                               discounts=DEFAULT_DISCOUNTS, verbose=True, free_warm_cache=True,
                               accumulator=False):
    """Provider exposure across `k_levels` for every seed, over the WHOLE user population.

    Why the whole population, when eval.py restricts to users holding a held-out item: that
    restriction is exact for a macro-average (a user with no test item cannot move it) and is
    NOT exact here. Exposure is a count of impressions across the surface you serve, so dropping
    users changes the estimand rather than optimizing it. There is also a sample-size argument --
    136,602 providers against 12,382 x K impressions is ~9 per provider, so a Gini over that
    mostly measures small-sample zero-inflation; the full population gives ~281.

    Same warm-cache trick as eval.sweep_mode_a_cached: warm factors are frozen across the reveal
    sweep, so each seed's warm top-N is built once and only the cold block re-ranks per level.
    `fold_in` is memoized on (dataset fingerprint, k), so when this runs in the same session as
    Section 9's sweep the fold-ins are already computed and cost nothing.

    D5: `_warm_cache` is 460 MB per seed at full population (12 MB at eval-user scale, which is
    why it was never freed). Ten live seeds is 4.6 GB of host memory for a cache that is dead the
    moment the seed's k-loop ends, so it is dropped per seed unless `free_warm_cache=False`.
    """
    import time as _time

    user_ids = np.arange(dataset.n_users) if users is None else np.asarray(users)
    whole_population = users is None
    _check_user_coverage(models, user_ids)      # seconds, not 10 minutes into the sweep
    t = {"setup": 0.0, "warm": 0.0, "foldin": 0.0, "recommend": 0.0, "equity": 0.0}

    _t = _time.perf_counter()
    # Only the REVEALED matrices are precomputed -- they are tiny (n_cold * k nnz, ~0.65 MB each).
    # `ref_train` is added inside the loop rather than precomputing the 21 sums: measured at Books
    # scale the add is 17 ms, so 210 of them cost 3.7 s against a ~10-20 min pass, where holding
    # the sums would cost 688 MB resident for the whole run. Not a close call in either direction.
    revealed = [dataset.revealed_matrix_at_k(k) for k in k_levels]
    warm_liked = dataset.ref_train if whole_population else dataset.ref_train[user_ids]
    acc = ExposureAccumulator(pmap, dataset, n_k=len(k_levels), n_seeds=len(models), K=K,
                              discounts=discounts, users=user_ids)
    t["setup"] = _time.perf_counter() - _t

    for si, model in enumerate(models):
        _check_model(model)
        _require_cached_surface(model)
        model.load_factors_gpu()
        _t = _time.perf_counter(); model.build_warm_cache(user_ids, warm_liked, K); t["warm"] += _time.perf_counter() - _t
        for ki, k in enumerate(k_levels):
            _t = _time.perf_counter(); folded = model.fold_in(dataset, k); t["foldin"] += _time.perf_counter() - _t
            train_k = dataset.ref_train + revealed[ki]
            _t = _time.perf_counter()
            rec = folded.recommend_cached(user_ids, train_k if whole_population else train_k[user_ids], K)
            t["recommend"] += _time.perf_counter() - _t
            _t = _time.perf_counter(); acc.add(ki, si, rec); t["equity"] += _time.perf_counter() - _t
        model.free_factors_gpu()
        if free_warm_cache:
            model._warm_cache = None                       # D5 -- see the docstring
    if verbose:
        print(f"[sweep_provider_equity_full] users={len(user_ids):,} "
              f"providers={pmap.n_providers:,} discounts={len(acc.discounts)}  "
              + "  ".join(f"{k}={v:.1f}s" for k, v in t.items())
              + f"  total={sum(t.values()):.1f}s")
    return acc if accumulator else acc.curves()


def ceiling_equity_generic(models, dataset, pmap, K=100, users=None,
                           discounts=DEFAULT_DISCOUNTS, accumulator=False):
    """`ceiling_equity` for models with no warm cache -- the Popularity floor. Same measurement,
    via plain `recommend()`.

    It exists so the floor is not the one arm in the table with a bare `nan` ceiling. An absent
    number that looks like a computed one is the failure mode this whole module is built to avoid,
    and "the code path did not support it" is not a reason a reader can infer from `nan`.
    """
    user_ids = np.arange(dataset.n_users) if users is None else np.asarray(users)
    _check_user_coverage(models, user_ids)
    ceiling_train = dataset.ref_train + dataset.ceiling_matrix()
    acc = ExposureAccumulator(pmap, dataset, n_k=1, n_seeds=len(models), K=K,
                              discounts=discounts, users=user_ids)
    for si, model in enumerate(models):
        _check_model(model)
        folded = model.fold_in_ceiling(dataset)
        rec_ids, _ = folded.recommend(user_ids, ceiling_train[user_ids], N=K,
                                      filter_already_liked_items=True)
        acc.add(0, si, rec_ids)
    return acc if accumulator else acc.curves()


def ceiling_equity(models, dataset, pmap, K=100, users=None, discounts=DEFAULT_DISCOUNTS,
                   verbose=True, free_warm_cache=True, accumulator=False):
    """The equity analog of eval.ceiling_reference: every cold item folded in with ALL its
    pre-test interactions, so this is the warm-up curve's own asymptote rather than a different
    item population. Answers "what does provider exposure look like once the cold items are fully
    warm" -- the reference point every other curve in the notebook has and this one lacked.

    Structurally one reveal level, so it returns curves whose lists are length 1.
    """
    import time as _time

    user_ids = np.arange(dataset.n_users) if users is None else np.asarray(users)
    whole_population = users is None
    _check_user_coverage(models, user_ids)
    ceiling_train = dataset.ref_train + dataset.ceiling_matrix()
    warm_liked = dataset.ref_train if whole_population else dataset.ref_train[user_ids]
    user_items = ceiling_train if whole_population else ceiling_train[user_ids]
    acc = ExposureAccumulator(pmap, dataset, n_k=1, n_seeds=len(models), K=K,
                              discounts=discounts, users=user_ids)

    _t = _time.perf_counter()
    for si, model in enumerate(models):
        _check_model(model)
        _require_cached_surface(model)
        model.load_factors_gpu()
        model.build_warm_cache(user_ids, warm_liked, K)
        folded = model.fold_in_ceiling(dataset)
        acc.add(0, si, folded.recommend_cached(user_ids, user_items, K))
        model.free_factors_gpu()
        if free_warm_cache:
            model._warm_cache = None
    if verbose:
        print(f"[ceiling_equity] users={len(user_ids):,} seeds={len(models)}  "
              f"{_time.perf_counter() - _t:.1f}s")
    return acc if accumulator else acc.curves()
