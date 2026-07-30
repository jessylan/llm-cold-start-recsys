"""Provider-side equity and exposure metrics: whose items actually get recommended, not just
whether a cold item becomes retrievable. Companion to eval.py's warm-up curve (Mode A/B), this
module looks at whether exposure concentrates on already-popular providers ("rich get richer")
or newer/smaller providers get a fair share relative to their catalog footprint.

Every function here takes a `model` (or list of `model`s, for averaging across seeds)
satisfying protocol.RetrievalModel and a `load.Dataset`, same convention as eval.py, so this
plugs into the pipeline the same way the item-level warm-up curve does, not as a separate
special case.

Provider field is author_name for books and store for movies, pulled from
books_meta_common.parquet and movies_meta_common.parquet (built by data_filtering.ipynb on the
raw, unfiltered 0-core Amazon Reviews 2023 .jsonl.gz files). See build_item_provider_map for
why these two fields, not main_category/categories.
"""
import numpy as np
import pandas as pd

from recsys.protocol import RetrievalModel

# ---------------------------------------------------------------------------
# Item -> provider mapping
# ---------------------------------------------------------------------------

def load_provider_metadata(books_meta_path, movies_meta_path):
    """Load books_meta_common.parquet and movies_meta_common.parquet and return a single
    Series: item_id -> provider_id, using the movie_ / book_ prefixed item_id convention from
    data_cleaning.ipynb's unified item table. <-- NOTE - deprecated. To be updated in a future commit. 

    Books use author_name; movies use store -- main_category/categories describe what an item
    *is* (genre/type), not who *made* it, so they'd conflate content-type equity with provider
    equity. store and author_name are the actual "who put this out" fields, though store is
    Amazon's storefront/brand field and isn't a perfect analog to author for movies -- worth a
    caveat in the methodology writeup, not treated as equivalent in meaning.

    Missing values become "UNKNOWN" rather than being dropped, so every item still gets a row.
    """
    books_meta = pd.read_parquet(books_meta_path, columns=["parent_asin", "author_name"])
    movies_meta = pd.read_parquet(movies_meta_path, columns=["parent_asin", "store"])

    books_provider = pd.Series(
        books_meta["author_name"].fillna("UNKNOWN").values,
        index="book_" + books_meta["parent_asin"].astype(str),
        name="provider",
    )
    movies_provider = pd.Series(
        movies_meta["store"].fillna("UNKNOWN").values,
        index="movie_" + movies_meta["parent_asin"].astype(str),
        name="provider",
    )
    return pd.concat([books_provider, movies_provider])


def build_item_provider_map(provider_series, dataset):
    """Build an array provider_of[item_index] -> provider_id, aligned to dataset.index_to_item
    (load.Dataset), the same zero-based item_index used everywhere else in the pipeline.
    """
    lookup = provider_series.to_dict()
    provider_of = np.empty(dataset.n_items, dtype=object)
    for item_idx, item_id in dataset.index_to_item.items():
        provider_of[item_idx] = lookup.get(item_id, "UNKNOWN")
    return provider_of


# ---------------------------------------------------------------------------
# Metric primitives -- operate directly on (rec_ids, provider_of), same level as eval.py's
# recall_and_hit_rate_at_k operating on (model, train_matrix, test_matrix)
# ---------------------------------------------------------------------------

def gini_coefficient(values):
    """Gini coefficient of an array of non-negative values (e.g. per-provider exposure counts).
    0 = perfect equality, 1 = maximal inequality (one entity has everything)."""
    values = np.asarray(values, dtype=float)
    if values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)  # ascending order, so the weighting below favors the top end
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_vals)) / (n * np.sum(sorted_vals)))


def provider_exposure_counts(rec_ids, provider_of):
    """rec_ids: (n_users, K) array of recommended item indices, as returned by
    model.recommend(). provider_of: array, item_index -> provider_id.
    Returns a pandas Series: provider_id -> total exposure count across all users/slots."""
    flat_items = rec_ids.ravel()  # every recommendation shown, across every user, in one list
    flat_providers = provider_of[flat_items]
    return pd.Series(flat_providers).value_counts()


def catalog_share(provider_of, eligible_item_mask=None):
    """Each provider's share of the catalog (or of the eligible/cold-item population if
    eligible_item_mask is passed, e.g. dataset.cold_item_ids as a boolean mask).
    Returns a pandas Series: provider_id -> fraction of catalog."""
    providers = provider_of if eligible_item_mask is None else provider_of[eligible_item_mask]  # mask narrows to just cold items
    counts = pd.Series(providers).value_counts()
    return counts / counts.sum()


def equity_ratio(rec_ids, provider_of, eligible_item_mask=None):
    """Exposure share / catalog share, per provider. 1.0 = proportional exposure,
    <1.0 = under-exposed relative to catalog footprint, >1.0 = over-exposed. The core
    'rich get richer' check: providers already well represented in the catalog getting
    disproportionately MORE exposure than their catalog share would predict."""
    exposure_counts = provider_exposure_counts(rec_ids, provider_of)
    exposure_share = exposure_counts / exposure_counts.sum()
    cat_share = catalog_share(provider_of, eligible_item_mask)

    ratio = pd.DataFrame({"exposure_share": exposure_share, "catalog_share": cat_share}).fillna(0.0)
    ratio["equity_ratio"] = ratio["exposure_share"] / ratio["catalog_share"].replace(0, np.nan)  # avoid divide by zero
    return ratio.sort_values("exposure_share", ascending=False)


# ---------------------------------------------------------------------------
# Provider equity warm-up sweep -- mirrors eval.py's evaluate_at_k / sweep exactly: same
# model.fold_in(dataset, k) + model.recommend(...) calls, so this is called from the same
# notebook cell, on the same seed_models list, as eval.sweep().
# ---------------------------------------------------------------------------

def _check_model(model):
    if not isinstance(model, RetrievalModel):
        raise TypeError(
            f"{model!r} does not satisfy protocol.RetrievalModel "
            "(needs fit, recommend, score_matrix, and fold_in)"
        )


def evaluate_provider_equity_at_k(model, dataset, provider_of, k, K=100):
    """Folds `model` in at reveal level k (same mechanism as eval.evaluate_at_k) and computes:
      - gini: Gini coefficient of exposure across ALL providers
      - cold_equity_ratio_mean: mean equity_ratio, restricted to cold items' providers
        specifically (are cold-item providers, specifically, under- or over-exposed at this k?)

    RENAME PENDING: k means interaction count here (matches eval.py), not list length. Hold
    off renaming to n until eval.py's k gets renamed too.
    """
    _check_model(model)
    train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)  # RENAME PENDING: k -> n
    folded = model.fold_in(dataset, k)  # RENAME PENDING: k -> n

    user_ids = np.arange(dataset.n_users)
    rec_ids, _ = folded.recommend(user_ids, train_k[user_ids], N=K, filter_already_liked_items=True)

    gini = gini_coefficient(provider_exposure_counts(rec_ids, provider_of).values)

    # narrow to cold items' providers specifically, that's the actual equity question
    cold_item_mask = np.zeros(dataset.n_items, dtype=bool)
    cold_item_mask[dataset.cold_item_ids] = True
    ratio_df = equity_ratio(rec_ids, provider_of, eligible_item_mask=cold_item_mask)
    cold_providers = set(provider_of[cold_item_mask])
    cold_ratio_mean = ratio_df.loc[ratio_df.index.isin(cold_providers), "equity_ratio"].mean()

    return {"gini": gini, "cold_equity_ratio_mean": float(cold_ratio_mean)}


def sweep_provider_equity(models, dataset, provider_of, k_levels, K=100):
    """Provider-equity analog of eval.sweep(): runs evaluate_provider_equity_at_k across
    k_levels for `models` (independently-fit instances of one method), averaging across them
    at each k. Returns curve = {"gini": {"mean": [...], "std": [...]},
    "cold_equity_ratio_mean": {"mean": [...], "std": [...]}}, one entry per k in k_levels --
    same shape as eval.sweep()'s curve, so it can be plotted on the same x-axis.

    RENAME PENDING: k / k_levels mean interaction count here, same as eval.py. Hold off
    renaming to n until eval.py's own k gets renamed too.
    """
    metrics = ["gini", "cold_equity_ratio_mean"]
    curve = {m: {"mean": [], "std": []} for m in metrics}
    for k in k_levels:  # RENAME PENDING: k -> n
        per_seed = {m: [] for m in metrics}
        for model in models:
            result = evaluate_provider_equity_at_k(model, dataset, provider_of, k, K=K)
            for m in metrics:
                per_seed[m].append(result[m])
        for m in metrics:
            curve[m]["mean"].append(float(np.mean(per_seed[m])))
            curve[m]["std"].append(float(np.std(per_seed[m])))
    return curve
