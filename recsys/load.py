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

    def revealed_matrix_at_k(self, k: int) -> sparse.csr_matrix:
        """User x item sparse matrix of cold items' revealed interactions at reveal level k."""
        rows, cols = [], []
        for item_idx, users in self.reveal_pool.items():
            for u in users[:k]:
                rows.append(u)
                cols.append(item_idx)
        data = np.ones(len(rows))
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def revealed_item_users_at_k(self, k: int) -> tuple[np.ndarray, sparse.csr_matrix]:
        """Item x user sparse matrix (batch-ready for implicit's `recalculate_item`) of cold
        items' revealed interactions at reveal level k. Returns (item_ids, item_users_matrix)."""
        rows, cols = [], []
        for local_idx, item_idx in enumerate(self.cold_item_ids):
            for u in self.reveal_pool[item_idx][:k]:
                rows.append(local_idx)
                cols.append(u)
        data = np.ones(len(rows))
        item_users = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.cold_item_ids), self.n_users))
        return self.cold_item_ids, item_users

    def ceiling_matrix(self) -> sparse.csr_matrix:
        """User x item sparse matrix of every cold item's FULL pre-test (ceiling) interactions -- the
        within-item warm reference. Folding a cold item in with all of this is the item's own
        fully-warm state (degree-matched by construction: same item, same last-5 test as the curve)."""
        rows, cols = [], []
        for item_idx, users in self.ceiling_pool.items():
            for u in users:
                rows.append(u)
                cols.append(item_idx)
        data = np.ones(len(rows))
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def ceiling_item_users(self) -> tuple[np.ndarray, sparse.csr_matrix]:
        """Item x user sparse matrix (batch-ready for implicit's `recalculate_item`) of every cold
        item's FULL pre-test interactions. Returns (item_ids, item_users_matrix) -- the ceiling
        analog of revealed_item_users_at_k, used for the within-item warm-reference fold-in."""
        rows, cols = [], []
        for local_idx, item_idx in enumerate(self.cold_item_ids):
            for u in self.ceiling_pool[item_idx]:
                rows.append(local_idx)
                cols.append(u)
        data = np.ones(len(rows))
        item_users = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.cold_item_ids), self.n_users))
        return self.cold_item_ids, item_users


def _build_csr(rows_df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (np.ones(len(rows_df)), (rows_df["user_index"], rows_df["item_index"])),
        shape=(n_users, n_items),
    )


def load_dataset(
    data_path: str = "data/filtered/books_5core_common.parquet",
    min_interactions: int = 25,
    cold_item_fraction: float = 0.10,
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

    # --- Cold-item selection: unchanged mechanism, operating on the post-filter population ---
    item_total_count = df.groupby("item_index").size().reindex(range(n_items), fill_value=0)
    eligible_items = item_total_count.index[item_total_count >= min_interactions].to_numpy()

    selection_rng = np.random.default_rng(seed)
    n_cold = round(len(eligible_items) * cold_item_fraction)
    cold_items = set(selection_rng.choice(eligible_items, size=n_cold, replace=False).tolist())

    cold_rows = df.loc[df["item_index"].isin(cold_items)].sort_values(["item_index", "timestamp"], kind="stable")

    reveal_pool = {}
    ceiling_pool = {}
    reserved_test_parts = []
    for item_idx, group in cold_rows.groupby("item_index", sort=False):
        users_sorted = group["user_index"].to_numpy()
        # Leave-last-`test_size`-out. Eligibility (>= min_interactions) with min_interactions - test_size
        # >= n_reveal guarantees the three slices below are disjoint: first n_reveal (reveal) and
        # all-but-last test_size (ceiling) never reach into the last test_size (test).
        reveal_pool[item_idx] = users_sorted[:n_reveal]        # first n_reveal -> warm-up curve (k=0..n_reveal)
        ceiling_pool[item_idx] = users_sorted[:-test_size]     # all but last m -> within-item warm ceiling
        reserved_test_parts.append(group.iloc[-test_size:])    # last m -> fixed reserved test

    reserved_test_df = pd.concat(reserved_test_parts)
    test_matrix = _build_csr(reserved_test_df, n_users, n_items)
    cold_item_ids = np.array(sorted(reveal_pool.keys()))

    # --- Warm-item sparsity pre-filter -------------------------------------------------------
    warm_rows = df.loc[~df["item_index"].isin(cold_items)]
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
    print(f"cold items: {len(cold_item_ids):,}  (eligible pool >= {min_interactions}: "
          f"{len(eligible_items):,}, fraction={cold_item_fraction})")
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

    return Dataset(
        n_users=n_users,
        n_items=n_items,
        ref_train=ref_train,
        ref_test=ref_test,
        test_matrix=test_matrix,
        ref_val=ref_val,
        reveal_pool=reveal_pool,
        ceiling_pool=ceiling_pool,
        cold_item_ids=cold_item_ids,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )


def load_titles(meta_path: str = "data/filtered/books_meta_5core_common.parquet") -> dict[str, str]:
    """Raw parent_asin -> title, for display purposes only (e.g. eval.top_n_recommendations).
    Not a Dataset field since it's read on demand, not a modeling input."""
    meta = pd.read_parquet(meta_path, columns=["parent_asin", "title"])
    return meta.drop_duplicates(subset="parent_asin", keep="last").set_index("parent_asin")["title"].to_dict()
