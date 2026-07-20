"""Load MovieLens 100k, build the sparse user x item interaction matrix, and construct the
cold/warm split used throughout the rest of the pipeline.

The only external data dependency in the pipeline: everything downstream (pop.py, cf.py,
eval.py, and future retrieval methods) consumes the `Dataset` returned by `load_dataset()` and
never touches raw IDs, the source CSV, or scaffolding like the raw per-item interaction counts
used only to pick the cold-item population.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # must be set before importing implicit,
# anywhere in the process -- implicit does its own multithreading, and letting the underlying
# BLAS also spawn threads causes oversubscription. load.py imports implicit.evaluation, so this
# guard has to live here too, not only in cf.py -- whichever module is imported first is the one
# that has to set it.

import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.evaluation import train_test_split

ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


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
    reveal_pool: dict = field(default_factory=dict)  # item_idx -> chronological user_index array (len <= n_reveal)
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


def load_dataset(
    data_dir: str = "ml-100k",
    min_interactions: int = 25,
    cold_item_fraction: float = 0.10,
    n_reveal: int = 20,
    seed: int = 42,
) -> Dataset:
    """Downloads/loads MovieLens 100k, remaps raw IDs to contiguous indices, selects a fixed
    population of structurally "cold" items, and produces the warm-item reference train/test
    split every model fits on.

    Eligibility filter: only items with >= `min_interactions` total interactions are eligible to
    be selected as cold, guaranteeing every selected item can supply the full k=0..n_reveal sweep
    without running out of history (n_reveal revealable + at least min_interactions - n_reveal
    reserved for evaluation).

    Reveal and reserve, per cold item: sort that item's interactions chronologically. The first
    `n_reveal` are the revealable pool; everything after is permanently reserved for evaluation,
    unchanged at every k.
    """
    if not os.path.exists(data_dir):
        # extractall() lands the zip's own top-level "ml-100k/" folder inside this parent dir,
        # so data_dir's basename must match that folder name (true for the "ml-100k" default).
        zip_path = f"{data_dir}.zip"
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(data_dir) or ".")

    columns = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(f"{data_dir}/u.data", sep="\t", names=columns)

    user_cat = df["user_id"].astype("category")
    item_cat = df["item_id"].astype("category")
    df["user_index"] = user_cat.cat.codes
    df["item_index"] = item_cat.cat.codes
    index_to_user = dict(enumerate(user_cat.cat.categories))
    index_to_item = dict(enumerate(item_cat.cat.categories))
    n_users = df["user_index"].nunique()
    n_items = df["item_index"].nunique()

    df["interaction"] = 1

    item_total_count = df.groupby("item_index").size().reindex(range(n_items), fill_value=0)
    eligible_items = item_total_count.index[item_total_count >= min_interactions].to_numpy()

    selection_rng = np.random.default_rng(seed)
    n_cold = round(len(eligible_items) * cold_item_fraction)
    cold_items = set(selection_rng.choice(eligible_items, size=n_cold, replace=False).tolist())

    cold_rows = df.loc[df["item_index"].isin(cold_items)].sort_values(["item_index", "timestamp"], kind="stable")

    reveal_pool = {}
    reserved_test_parts = []
    for item_idx, group in cold_rows.groupby("item_index", sort=False):
        users_sorted = group["user_index"].to_numpy()
        reveal_pool[item_idx] = users_sorted[:n_reveal]
        reserved_test_parts.append(group.iloc[n_reveal:])

    reserved_test_df = pd.concat(reserved_test_parts)
    test_matrix = sparse.csr_matrix(
        (reserved_test_df["interaction"], (reserved_test_df["user_index"], reserved_test_df["item_index"])),
        shape=(n_users, n_items),
    )

    warm_rows = df.loc[~df["item_index"].isin(cold_items)]
    warm_matrix = sparse.csr_matrix(
        (warm_rows["interaction"], (warm_rows["user_index"], warm_rows["item_index"])),
        shape=(n_users, n_items),
    )
    ref_train, ref_test = train_test_split(warm_matrix, train_percentage=0.8, random_state=seed)

    cold_item_ids = np.array(sorted(reveal_pool.keys()))

    return Dataset(
        n_users=n_users,
        n_items=n_items,
        ref_train=ref_train,
        ref_test=ref_test,
        test_matrix=test_matrix,
        reveal_pool=reveal_pool,
        cold_item_ids=cold_item_ids,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )


def load_titles(data_dir: str = "ml-100k") -> dict[int, str]:
    """Raw item_id -> title, for display purposes only (e.g. eval.top_n_recommendations). Not
    a Dataset field since it's read on demand, not a modeling input."""
    return (
        pd.read_csv(
            f"{data_dir}/u.item", sep="|", header=None, encoding="latin-1", usecols=[0, 1],
            names=["item_id", "title"],
        )
        .set_index("item_id")["title"]
        .to_dict()
    )
