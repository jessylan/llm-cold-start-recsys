"""Load Amazon Books (the 5-core interaction set, restricted to common users -- see
data_filtering.ipynb's "Build the 5-core enriched dataset" section), build the sparse user x item
interaction matrix, and construct the cold/warm split used throughout the rest of the pipeline.

The only external data dependency in the pipeline: everything downstream (pop.py, cf.py,
eval.py, and future retrieval methods) consumes the `Dataset` returned by `load_dataset()` and
never touches raw IDs, the source parquet, or scaffolding like the raw per-item interaction
counts used only to pick the cold-item population.

Cold-item split: unchanged mechanism from the original MovieLens loader -- a fixed fraction of
items with enough total interactions are selected as structurally cold, and each one's own
chronological history is split into a revealable pool (first n_reveal interactions) and a
permanently reserved test set (everything after).

Warm-item split: leave-last-out (LOO) per user, not a random split -- for a user's chronological
sequence of warm-item interactions (length N), the first N-2 go to ref_train, the (N-1)th to
ref_val, and the Nth to ref_test. This is the standard protocol in cold-start/sequential-rec
literature. A global time-based cutoff was considered and deliberately rejected: this project's
cold-item evaluation needs a genuinely disjoint held-out population per user, not "got unlucky
relative to one global cutoff."

Positive-signal definition: only ratings >= min_rating (default 4) count as a positive
interaction. Unlike MovieLens's implicit-CF convention (every rating event counts, regardless of
value), Amazon ratings are explicit, and a 1-3 star review is not a reliable positive signal.
"""
from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd
import scipy.sparse as sparse


@dataclass
class Dataset:
    """Everything downstream of the raw load/split step actually needs. Deliberately excludes
    pure scaffolding (the raw DataFrame, per-item interaction counts, the warm/cold row splits)
    that only matters while `load_dataset()` is constructing this object."""

    n_users: int
    n_items: int
    ref_train: sparse.csr_matrix       # warm-item training data -- every model's `fit()` input
    ref_test: sparse.csr_matrix        # warm-item held-out set -- the "normal" reference score
    test_matrix: sparse.csr_matrix     # fixed reserved cold-item test set (every k, same rows)
    ref_val: sparse.csr_matrix = None  # warm-item validation set (LOO's (N-1)th interaction per user)
    reveal_pool: dict = field(default_factory=dict)  # item_idx -> chronological user_index array (len <= n_reveal)
    ceiling_pool: dict = field(default_factory=dict)  # item_idx -> ALL pre-test users (len = item_total - test_size)
    cold_item_ids: np.ndarray = None   # sorted(reveal_pool.keys()), precomputed once
    index_to_user: dict = field(default_factory=dict)
    index_to_item: dict = field(default_factory=dict)
    # A SECOND, disjoint cold population reserved for hyperparameter selection -- itself a Dataset,
    # sharing this one's ref_train/ref_val/ref_test and id maps but carrying its own cold items,
    # reveal pools and reserved test set. Because it satisfies the same interface, every function in
    # eval.py works on it unchanged: `ev.sweep_mode_a_cached(models, dataset.cold_val, ...)` selects
    # a hyperparameter without ever touching the reported cold-test population. `None` on the nested
    # instance (no further recursion) and when cold_val_fraction=0.
    cold_val: "Dataset" = None

    # --- pool -> matrix builders -------------------------------------------------------------
    # All four of these used to walk `reveal_pool` / `ceiling_pool` with a nested Python loop,
    # appending one (row, col) pair at a time. `revealed_*_at_k` is called once per reveal level
    # per seed per sweep, so at Books scale (2,727 cold items x up to 20 revealed) that was ~1.1M
    # list appends per 21-level sweep, repeated for every arm.
    #
    # The pools are FIXED for the life of a Dataset, so the loop only has to happen once. The flat
    # form below records, for every (cold item, user) pair, its local item index, its global item
    # index, the user, and its position within that item's chronological pool -- after which
    # "the first k of each item" is the vectorized mask `pos < k` and every builder is one
    # `csr_matrix` construction with no Python-level iteration.
    #
    # Exactly equivalent, not approximately: COO construction sums duplicates and sorts indices, so
    # the emission ORDER never mattered, only the multiset of pairs -- and the mask reproduces that
    # multiset entry for entry. bench_30 asserts identity against the original loops across every
    # k, including k=0, k beyond the pool length, duplicate pairs and empty pools.

    def _flat_pool(self, which: str):
        """(local_idx, item_idx, user_idx, position_within_item) for `reveal_pool` or
        `ceiling_pool`, built once per Dataset and memoized.

        Deliberately NOT a dataclass field: adding one would change the constructor signature, and
        an unpickled Dataset built before this existed must still work (it rebuilds on first use).
        `cache_pickle` writes the Dataset straight out of `load_dataset()`, before any of these are
        called, so this never bloats the cache file either.
        """
        cache = getattr(self, "_flat_cache", None)
        if cache is None:
            cache = self._flat_cache = {}
        if which in cache:
            return cache[which]

        pool = self.reveal_pool if which == "reveal" else self.ceiling_pool
        ids = np.asarray(self.cold_item_ids, dtype=np.int64)
        # `cold_item_ids` is `sorted(reveal_pool.keys())` by construction and `ceiling_pool` is
        # built over the same items. Assert it rather than assume it: if the pools ever diverge,
        # iterating `ids` would silently drop entries the old `pool.items()` loop included.
        if set(pool.keys()) != set(ids.tolist()):
            raise RuntimeError(
                f"{which}_pool keys do not match cold_item_ids "
                f"({len(pool)} vs {len(ids)}); the flat builders assume they are the same items.")

        if len(ids) == 0:
            empty = np.zeros(0, dtype=np.int64)
            cache[which] = (empty, empty, empty, empty)
            return cache[which]

        lengths = np.fromiter((len(pool[i]) for i in ids), dtype=np.int64, count=len(ids))
        users = (np.concatenate([np.asarray(pool[i], dtype=np.int64) for i in ids])
                 if lengths.sum() else np.zeros(0, dtype=np.int64))
        local = np.repeat(np.arange(len(ids), dtype=np.int64), lengths)
        items = np.repeat(ids, lengths)
        starts = np.cumsum(lengths) - lengths                  # first flat index of each item
        pos = np.arange(len(users), dtype=np.int64) - np.repeat(starts, lengths)
        cache[which] = (local, items, users, pos)
        return cache[which]

    def revealed_matrix_at_k(self, k: int) -> sparse.csr_matrix:
        """User x item sparse matrix of cold items' revealed interactions at reveal level k."""
        _, items, users, pos = self._flat_pool("reveal")
        m = pos < k
        return sparse.csr_matrix((np.ones(int(m.sum())), (users[m], items[m])),
                                 shape=(self.n_users, self.n_items))

    def revealed_item_users_at_k(self, k: int) -> tuple[np.ndarray, sparse.csr_matrix]:
        """Item x user sparse matrix (batch-ready for implicit's `recalculate_item`) of cold
        items' revealed interactions at reveal level k. Returns (item_ids, item_users_matrix)."""
        local, _, users, pos = self._flat_pool("reveal")
        m = pos < k
        item_users = sparse.csr_matrix((np.ones(int(m.sum())), (local[m], users[m])),
                                       shape=(len(self.cold_item_ids), self.n_users))
        return self.cold_item_ids, item_users

    def ceiling_matrix(self) -> sparse.csr_matrix:
        """User x item sparse matrix of every cold item's FULL pre-test (ceiling) interactions -- the
        within-item warm reference. Folding a cold item in with all of this is the item's own
        fully-warm state (degree-matched by construction: same item, same last-5 test as the curve)."""
        _, items, users, _ = self._flat_pool("ceiling")
        return sparse.csr_matrix((np.ones(len(users)), (users, items)),
                                 shape=(self.n_users, self.n_items))

    def ceiling_item_users(self) -> tuple[np.ndarray, sparse.csr_matrix]:
        """Item x user sparse matrix (batch-ready for implicit's `recalculate_item`) of every cold
        item's FULL pre-test interactions. Returns (item_ids, item_users_matrix) -- the ceiling
        analog of revealed_item_users_at_k, used for the within-item warm-reference fold-in."""
        local, _, users, _ = self._flat_pool("ceiling")
        item_users = sparse.csr_matrix((np.ones(len(users)), (local, users)),
                                       shape=(len(self.cold_item_ids), self.n_users))
        return self.cold_item_ids, item_users


def dataset_fingerprint(dataset: "Dataset") -> tuple:
    """A cheap, order-insensitive summary of everything downstream results depend on.

    Use it as the `params` of a `cache_pickle` key so a derived artifact (the Popularity floor, the
    Activity curve, CBHCF's content-score block, a results pickle) can never be silently reused
    against a different split. It changes if the user/item universe changes, if `ref_train` gains or
    loses interactions, or if either cold population changes membership -- the id SUM catches a
    swapped population that happens to be the same size.

    Note what this cannot do: it is computed FROM a Dataset, so it cannot key the Dataset's own
    cache (you would have to build the thing to learn its key). That one must be keyed on the
    INPUTS instead -- see `load_params_fingerprint`.
    """
    cold_val_ids = None if dataset.cold_val is None else dataset.cold_val.cold_item_ids
    base = (
        int(dataset.n_users), int(dataset.n_items),
        int(dataset.ref_train.nnz), int(dataset.ref_test.nnz),
        int(0 if dataset.ref_val is None else dataset.ref_val.nnz),
        int(len(dataset.cold_item_ids)), int(np.asarray(dataset.cold_item_ids, dtype=np.int64).sum()),
        int(dataset.test_matrix.nnz),
        int(0 if cold_val_ids is None else len(cold_val_ids)),
        int(0 if cold_val_ids is None else np.asarray(cold_val_ids, dtype=np.int64).sum()),
    )
    # A WRAPPER may declare extra identity. Every field above reads an attribute that a decorator
    # like `coldllm.SyntheticAugmentedDataset` delegates straight through, so without this an
    # augmented dataset fingerprints IDENTICALLY to the plain one -- and `cf.ALSModel.fold_in` keys
    # its cold-factor memo on this fingerprint, so one model folded against both would silently
    # reuse the wrong factors.
    #
    # APPENDED ONLY WHEN PRESENT, never as a fixed extra slot: a plain Dataset must keep returning
    # the exact 10-tuple it always has. Every `data/cache/*` key, and the `dataset_fingerprint`
    # stamped into all four sections of `outputs/hyperparams.json`, is that tuple -- widening it
    # unconditionally would invalidate the lot and make the steel thread refuse its own artifacts.
    salt = getattr(dataset, "_fingerprint_salt", None)
    return base if salt is None else base + (salt,)


def load_params_fingerprint(data_path: str, **load_kwargs) -> tuple:
    """Input-side key for the Dataset's OWN cache: the source file's identity plus every argument
    that shapes the split. Complements `dataset_fingerprint`, which keys everything downstream."""
    try:
        stat = os.stat(data_path)
        source = (os.path.basename(data_path), stat.st_size, int(stat.st_mtime))
    except OSError:
        source = (os.path.basename(data_path), None, None)
    return (source, tuple(sorted(load_kwargs.items())))


def _build_csr(rows_df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (np.ones(len(rows_df)), (rows_df["user_index"], rows_df["item_index"])),
        shape=(n_users, n_items),
    )


def load_dataset(
    data_path: str = "data/filtered/books_5core_common.parquet",
    min_interactions: int = 25,
    cold_item_fraction: float = 0.10,
    cold_val_fraction: float = 0.10,
    n_reveal: int = 20,
    test_size: int = 5,
    min_rating: float = 4.0,
    min_interactions_warm: int = 5,
    min_loo_interactions: int = 3,
    seed: int = 42,
) -> Dataset:
    """Loads the Amazon Books common-user review set, remaps raw IDs to contiguous indices,
    selects a fixed population of structurally "cold" items, and produces the warm-item
    train/val/test split every model fits/tunes/reports on. Prints a diagnostic summary of every
    filtering/splitting decision before returning.

    Cold-item eligibility filter: only items with >= `min_interactions` (post rating-filter,
    post-dedup) total interactions are eligible to be selected as cold, guaranteeing every
    selected item can supply the full k=0..n_reveal sweep without running out of history
    (n_reveal revealable + at least min_interactions - n_reveal reserved for evaluation).

    Two cold populations, not one. The eligible pool is split 80 / 10 / 10 by default: 80% stay warm
    and are trained on, `cold_item_fraction` becomes the reported TEST cold set, and
    `cold_val_fraction` becomes a disjoint VALIDATION cold set exposed as `Dataset.cold_val`.
    Selecting a hyperparameter (CBHCF's content weight, field weights, a future intervention's
    knobs) on the test cold set would bias every reported number, and `ref_val` is the wrong
    instrument for it -- `ref_val` holds out *warm* interactions, so it answers a different question
    than cold-item retrieval. The validation cold items are drawn from the same eligible pool by the
    same rule, so they are exchangeable with the test ones.

    Set `cold_val_fraction=0` to recover the single-population behaviour. The test population is
    unaffected by this parameter (the two draws come off one generator in a fixed order), but
    `ref_train` is NOT: the validation items must leave it, since an item cannot simultaneously be
    trained on and be cold. Any cached Dataset / fitted model / cached result from a run with a
    different `cold_val_fraction` is therefore stale.

    Reveal and reserve, per cold item (leave-last-`test_size`-out): sort that item's interactions
    chronologically. The LAST `test_size` are the permanently reserved test set (fixed size for every
    cold item, identical across every k and every condition). The first `n_reveal` are the revealable
    pool for the warm-up curve (k=0..n_reveal). Everything before the test -- the first
    (item_total - test_size) interactions -- is the `ceiling_pool`: folding a cold item in with ALL of
    it is that item's own fully-warm state, the within-item warm reference (degree-matched to the
    curve by construction, since it's the same item evaluated on the same last-`test_size` test).
    For items with exactly `min_interactions` the ceiling equals the k=n_reveal curve endpoint; for
    higher-degree items it uses the extra interactions between n_reveal and (item_total - test_size),
    so the reference sits above the curve endpoint (the data-abundance headroom).

    Warm-item leave-last-out split: for a user's chronological sequence of warm-item
    interactions (length N >= min_loo_interactions), the first N-2 go to ref_train, the (N-1)th
    to ref_val, the Nth to ref_test. Users with N < min_loo_interactions contribute everything to
    ref_train -- there's no way to hold a slot out without leaving that user with zero training
    signal, and N-2/N-1/N only makes sense for N >= 3.

    Item-starvation safety net: a `min_interactions_warm` pre-filter drops very sparse warm items
    before the split (an item with very few total interactions has a real chance of losing all
    of them to LOO holdouts by chance, becoming "accidentally cold"). That's a risk-reduction
    lever, not a guarantee -- so after the split, any warm item that still ended up with zero
    ref_train occurrences has ALL of its val/test rows rerouted back to ref_train. This can't
    cascade (rerouting only adds train rows for the affected item, never removes one for another
    item), so one pass is sufficient.
    """
    if min_interactions - test_size < n_reveal:
        raise ValueError(
            f"min_interactions ({min_interactions}) - test_size ({test_size}) = "
            f"{min_interactions - test_size} < n_reveal ({n_reveal}): the first-n_reveal reveal pool "
            "would overlap the last-test_size test set. Raise min_interactions or lower n_reveal/test_size."
        )

    df = pd.read_parquet(data_path, columns=["user_id", "parent_asin", "rating", "timestamp"])

    user_cat = df["user_id"].astype("category")
    item_cat = df["parent_asin"].astype("category")
    df["user_index"] = user_cat.cat.codes
    df["item_index"] = item_cat.cat.codes
    index_to_user = dict(enumerate(user_cat.cat.categories))
    index_to_item = dict(enumerate(item_cat.cat.categories))
    n_users = df["user_index"].nunique()
    n_items = df["item_index"].nunique()

    # Dedupe BEFORE the rating filter: "this user's current opinion of this item, counted only
    # if positive" -- filtering first could let a stale positive review outlive a more recent
    # negative one for the same (user, item) pair.
    df = df.sort_values("timestamp", kind="stable")
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["user_index", "item_index"], keep="last")
    n_dup_collapsed = n_before_dedup - len(df)

    n_before_rating_filter = len(df)
    df = df.loc[df["rating"] >= min_rating].copy()
    n_dropped_by_rating = n_before_rating_filter - len(df)

    # --- Cold-item selection: TWO disjoint populations, test and validation -------------------
    item_total_count = df.groupby("item_index").size().reindex(range(n_items), fill_value=0)
    eligible_items = item_total_count.index[item_total_count >= min_interactions].to_numpy()

    # Drawn SEQUENTIALLY from one generator so that adding a validation population does not disturb
    # the test population: the first draw consumes exactly the randomness it did before
    # cold_val_fraction existed, so `cold_items` is bit-identical to the single-population split.
    # Only ref_train changes (it loses the validation items), which is unavoidable -- an item cannot
    # be both trained on and cold.
    selection_rng = np.random.default_rng(seed)
    n_cold = round(len(eligible_items) * cold_item_fraction)
    cold_items = set(selection_rng.choice(eligible_items, size=n_cold, replace=False).tolist())
    n_cold_val = round(len(eligible_items) * cold_val_fraction)
    if n_cold_val:
        remaining_eligible = np.setdiff1d(eligible_items, np.fromiter(cold_items, dtype=np.int64))
        cold_val_items = set(selection_rng.choice(remaining_eligible, size=n_cold_val,
                                                  replace=False).tolist())
    else:
        cold_val_items = set()

    def _cold_pools(item_set):
        """Reveal / ceiling / reserved-test split for one cold population. Leave-last-`test_size`-out:
        eligibility (>= min_interactions, with min_interactions - test_size >= n_reveal) guarantees
        the three slices are disjoint -- first n_reveal (reveal) and all-but-last test_size (ceiling)
        never reach into the last test_size (test)."""
        rows = df.loc[df["item_index"].isin(item_set)].sort_values(["item_index", "timestamp"],
                                                                   kind="stable")
        reveal, ceiling, reserved = {}, {}, []
        for item_idx, group in rows.groupby("item_index", sort=False):
            users_sorted = group["user_index"].to_numpy()
            reveal[item_idx] = users_sorted[:n_reveal]        # first n_reveal -> warm-up curve
            ceiling[item_idx] = users_sorted[:-test_size]     # all but last m -> within-item ceiling
            reserved.append(group.iloc[-test_size:])          # last m -> fixed reserved test
        matrix = _build_csr(pd.concat(reserved), n_users, n_items)
        return reveal, ceiling, matrix, np.array(sorted(reveal.keys()))

    reveal_pool, ceiling_pool, test_matrix, cold_item_ids = _cold_pools(cold_items)
    val_pools = _cold_pools(cold_val_items) if cold_val_items else None

    # --- Warm-item sparsity pre-filter -------------------------------------------------------
    # BOTH cold populations leave the warm pool: a validation cold item that stayed in ref_train
    # would not be cold at all.
    held_out_items = cold_items | cold_val_items
    warm_rows = df.loc[~df["item_index"].isin(held_out_items)]
    warm_item_count = warm_rows.groupby("item_index").size()
    eligible_warm_items = set(warm_item_count.index[warm_item_count >= min_interactions_warm])
    n_items_dropped_sparse = warm_rows["item_index"].nunique() - len(eligible_warm_items)
    warm_eligible_df = warm_rows.loc[warm_rows["item_index"].isin(eligible_warm_items)].copy()
    n_interactions_dropped_sparse = len(warm_rows) - len(warm_eligible_df)

    # --- Per-user leave-last-out split, vectorized (no Python loop over millions of users) ---
    warm_eligible_df = warm_eligible_df.sort_values(
        ["user_index", "timestamp", "item_index"], kind="stable"  # item_index: deterministic tiebreak
    ).reset_index(drop=True)

    user_groups = warm_eligible_df.groupby("user_index")
    rank_from_end = user_groups.cumcount(ascending=False).to_numpy()
    n_u = user_groups["item_index"].transform("size").to_numpy()

    split = np.full(len(warm_eligible_df), "train", dtype=object)
    can_split = n_u >= min_loo_interactions
    split[can_split & (rank_from_end == 0)] = "test"
    split[can_split & (rank_from_end == 1)] = "val"
    warm_eligible_df["split"] = split

    user_interaction_counts = warm_eligible_df.groupby("user_index").size()
    n_train_only_users = int((user_interaction_counts < min_loo_interactions).sum())
    n_users_with_warm = len(user_interaction_counts)

    # --- Item-starvation safety net: deterministic, not probabilistic ------------------------
    train_items = set(warm_eligible_df.loc[warm_eligible_df["split"] == "train", "item_index"])
    starved_items = eligible_warm_items - train_items
    starved_mask = warm_eligible_df["item_index"].isin(starved_items)
    n_interactions_rerouted = int(starved_mask.sum())
    warm_eligible_df.loc[starved_mask, "split"] = "train"
    n_items_starved = len(starved_items)

    ref_train = _build_csr(warm_eligible_df.loc[warm_eligible_df["split"] == "train"], n_users, n_items)
    ref_val = _build_csr(warm_eligible_df.loc[warm_eligible_df["split"] == "val"], n_users, n_items)
    ref_test = _build_csr(warm_eligible_df.loc[warm_eligible_df["split"] == "test"], n_users, n_items)

    print(f"users: {n_users:,}   items: {n_items:,}")
    print(f"duplicate (user,item) pairs collapsed: {n_dup_collapsed:,}")
    print(f"rows dropped by rating < {min_rating} filter: {n_dropped_by_rating:,} "
          f"({n_dropped_by_rating / n_before_rating_filter:.1%})")
    print(f"cold items (TEST): {len(cold_item_ids):,}  (eligible pool >= {min_interactions}: "
          f"{len(eligible_items):,}, fraction={cold_item_fraction})")
    if val_pools is not None:
        print(f"cold items (VALIDATION): {len(val_pools[3]):,}  (fraction={cold_val_fraction}) "
              f"-- for hyperparameter selection only; disjoint from the test population")
    print(f"warm items dropped by sparsity pre-filter (< {min_interactions_warm} interactions): "
          f"{n_items_dropped_sparse:,}  ({n_interactions_dropped_sparse:,} interactions discarded)")
    print(f"warm items retained: {len(eligible_warm_items):,}")
    print(f"users below min_loo_interactions={min_loo_interactions} (train-only, no val/test): "
          f"{n_train_only_users:,} of {n_users_with_warm:,} ({n_train_only_users / n_users_with_warm:.1%})")
    print(f"items rescued by starvation safety net: {n_items_starved:,}  "
          f"({n_interactions_rerouted:,} val/test interactions rerouted back to train)")
    print(f"ref_train nnz: {ref_train.nnz:,}   ref_val nnz: {ref_val.nnz:,}   ref_test nnz: {ref_test.nnz:,}")
    ceiling_sizes = np.array([len(v) for v in ceiling_pool.values()])
    print(f"reserved cold-item test interactions: {test_matrix.nnz:,}  "
          f"(leave-last-{test_size}-out: {test_size} per cold item x {len(cold_item_ids):,} items)")
    print(f"within-item ceiling pool per cold item: min={ceiling_sizes.min()} "
          f"median={int(np.median(ceiling_sizes))} max={ceiling_sizes.max()}  "
          f"(= item_total - {test_size}; {n_reveal} for exactly-{min_interactions} items)")

    shared = dict(n_users=n_users, n_items=n_items, ref_train=ref_train, ref_test=ref_test,
                  ref_val=ref_val, index_to_user=index_to_user, index_to_item=index_to_item)
    # The validation view shares ref_train/ref_val/ref_test and the id maps by reference (they are
    # the same objects, not copies) and differs only in its cold population -- so it satisfies the
    # Dataset interface and eval.py operates on it with no changes. cold_val stays None on it, so
    # there is no recursion.
    cold_val = None if val_pools is None else Dataset(
        test_matrix=val_pools[2], reveal_pool=val_pools[0], ceiling_pool=val_pools[1],
        cold_item_ids=val_pools[3], **shared)
    return Dataset(
        test_matrix=test_matrix,
        reveal_pool=reveal_pool,
        ceiling_pool=ceiling_pool,
        cold_item_ids=cold_item_ids,
        cold_val=cold_val,
        **shared,
    )


def load_titles(meta_path: str = "data/filtered/books_meta_5core_common.parquet") -> dict[str, str]:
    """Raw parent_asin -> title, for display purposes only (e.g. eval.top_n_recommendations).
    Not a Dataset field since it's read on demand, not a modeling input."""
    meta = pd.read_parquet(meta_path, columns=["parent_asin", "title"])
    return meta.drop_duplicates(subset="parent_asin", keep="last").set_index("parent_asin")["title"].to_dict()
