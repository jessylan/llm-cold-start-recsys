"""Item content representation, shared by every content-aware retrieval method.

This module owns exactly one job: turn a dataset's raw item metadata into a sparse item x term
matrix `T` whose rows are L2-normalized, so that `cosine_similarity(T) == T @ T.T`. That identity
is load-bearing. It is what lets a content score be computed as

    content_scores = (user_history @ T) @ T.T          # (n_users x n_terms) then (n_users x n_items)

instead of materializing the n_items x n_items similarity matrix, which at 487,790 Books items
would be ~950 GB. Every consumer (cbhcf.py today, later content-embedding methods) depends on it,
so any change here must preserve row normalization.

Two design decisions, both measured rather than assumed (see citations.md section 4):

**Roles, not column names.** The Amazon Books and Movies metadata share one physical schema but
NOT one semantic mapping. Books `features` holds the plot/marketing blurb (98.0% populated, ~174
words) while Books `description` holds editorial reviews; on Movies `features` is 0.01% populated
and `description` is the plot synopsis (91.7%, ~141 words). A shared column list would map Books'
editorial reviews onto Movies' synopses and drop Movies' only real text field. So each dataset
declares a FIELD MAP from role -> column, and downstream code refers only to roles.

**Per-field weighted blocks, not one bag of words.** Concatenating every field into one document
has three failure modes: a 3,000-word description drowns an 8-word title under per-document L2
normalization; the author surname "Price" collides with the noun `price` and the genre "Fantasy"
with the word `fantasy` in a blurb; and there is no lever to say a field matters more. Instead each
role is vectorized separately, L2-normalized, scaled by a per-field weight, and the blocks are
concatenated and row-renormalized. This is BM25F (Robertson, Zaragoza & Taylor, CIKM 2004) carried
into the vector-space model, and it makes the resulting cosine a weighted average of per-field
cosines:

    cos(i, j) = sum_f w_f^2 * cos_f(i, j) / (||i|| ||j||)

An item missing a field contributes a zero block, and the row renormalization automatically
redistributes that field's weight onto the fields it does have -- which is what keeps a Movies item
with no creator field from being silently penalized against a Books item that has one.

**Entity fields are not word-tokenized.** `creator` and `taxonomy` carry entities, not prose, so
they use one atomic token per entity (following MARec's split of TF-IDF for text / multi-hot for
categoricals). Word-splitting "Linda Francis Lee" into `linda`/`francis`/`lee` manufactures matches
against ordinary vocabulary.

**min_df=2 on every block.** A term occurring in exactly one item cannot contribute to ANY
item-item similarity -- similarity through a term requires two items to share it -- yet under
per-field L2 normalization it still consumes the block's norm, stealing weight from fields that can
actually match. Dropping singletons is a correctness fix, not a size optimization. (57.0% of
distinct Books `author_name` values are singletons.)

**The vectorizer is fit inductively.** `fit_content_space(..., fit_rows=warm_item_ids)` fits
vocabulary and IDF on WARM items only, then transforms cold items with that frozen vocabulary --
the faithful simulation of a new item arriving at a deployed system. Measured cost on Books at
min_df=2: cold items lose 0.56% of their tokens and retain 99.4% of their TF-IDF norm, with zero
cold items left empty. The same frozen space transforms a whole new domain (Books -> Movies), which
is the point: cold-starting Movies must not consult Movies metadata when choosing the vocabulary.
"""
from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROLES = ("title", "creator", "taxonomy", "blurb", "reviews")

# Amazon `store` on Books is the author with a role suffix -- "Charles Platt (Author)",
# "Brandon Graham (Author, Artist)". Strip it so the atomic token is the person, not the credit.
_ROLE_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")

# `store` is mostly an author page (91.9% of values map to a single author) but is contaminated by
# publisher storefronts, which would make every DK Publishing title "by the same creator". These
# are dropped rather than tokenized. Extend as more are found; keep it visible, not buried.
DEFAULT_CREATOR_BLOCKLIST = frozenset({
    "", "nan", "none", "various", "various authors", "aa", "anonymous", "unknown",
    "dk", "dk publishing", "perseus", "disney rh", "lonely planet",
})

# Per-role default weights. Title and creator are short and high-precision; taxonomy is coarse
# (only 1,268 distinct category documents across 487,790 Books items -- a genre prior, not a
# discriminator); reviews are long and noisy. Tune on `dataset.ref_val`, never on ref_test.
DEFAULT_WEIGHTS = {"title": 1.0, "creator": 1.0, "taxonomy": 0.5, "blurb": 1.0, "reviews": 0.5}

# Which metadata column fills each role, per dataset. `creator` takes a list: the first non-empty
# column wins (Books `author_name` is cleaner but only 87.9% populated; `store` covers the rest,
# reaching ~99.5%. On Movies `author_name` is 0.0% populated and `store` holds the cast list, so
# the fallback carries the whole role there).
BOOKS_FIELD_MAP = {
    "title": ["title"],
    "creator": ["author_name", "store"],
    "taxonomy": ["categories"],
    "blurb": ["features"],
    "reviews": ["description"],
}
MOVIES_FIELD_MAP = {
    "title": ["title"],
    "creator": ["store"],
    "taxonomy": ["categories"],
    "blurb": ["description"],
    "reviews": [],            # Books' editorial-reviews field has no Movies counterpart
}

# Roles whose values are entities (one atomic token each), not prose to be word-tokenized.
ENTITY_ROLES = frozenset({"creator", "taxonomy"})
_ENTITY_SEP = "|"


def _clean_entity(value: str, blocklist=DEFAULT_CREATOR_BLOCKLIST) -> str:
    """Normalize one entity string to a single atomic token, or '' to drop it."""
    v = _ROLE_SUFFIX.sub("", str(value or "")).strip().lower()
    v = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", v)).strip()
    return "" if v in blocklist else v


def _entity_analyzer(doc: str) -> list[str]:
    """Split an entity field on the separator and emit one token per entity. Tokens keep their
    internal spaces ('kalayna price' is ONE term), which is the whole point -- see module docstring."""
    return [t for t in (_clean_entity(p) for p in str(doc).split(_ENTITY_SEP)) if t]


def _as_text(value) -> str:
    """Amazon metadata stores `features`/`description`/`categories` as lists of strings and the
    rest as scalars. Flatten either to one string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value if v is not None).strip()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _as_entities(value) -> str:
    """Flatten to a separator-joined entity string, preserving entity boundaries (a category PATH
    or a cast LIST must not be collapsed into prose)."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return _ENTITY_SEP.join(str(v) for v in value if v is not None)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    # A flat cast string -- "Clint Eastwood  (Actor),     Bernadette Peters  (Actor)" -- is
    # comma-separated; a single author name has no comma and survives as one entity.
    return _ENTITY_SEP.join(str(value).split(","))


def build_documents(meta: pd.DataFrame, field_map: dict, n_items: int,
                    index_to_item: dict) -> dict[str, np.ndarray]:
    """Row-align raw metadata to item_index order and flatten it to one string per (role, item).

    `meta` is the raw metadata table (one row per parent_asin); `index_to_item` maps item_index ->
    parent_asin, so the returned arrays are indexed by item_index exactly like every matrix in
    load.Dataset. Items with no metadata row become empty strings in every role -- a zero block,
    which the block renormalization handles.
    """
    asin_of_index = pd.Series(index_to_item).reindex(range(n_items))
    aligned = (meta.drop_duplicates(subset="parent_asin", keep="last")
                   .set_index("parent_asin")
                   .reindex(asin_of_index.to_numpy()))

    docs = {}
    for role in ROLES:
        columns = [c for c in field_map.get(role, []) if c in aligned.columns]
        flatten = _as_entities if role in ENTITY_ROLES else _as_text
        if not columns:
            docs[role] = np.full(n_items, "", dtype=object)
            continue
        # Fallback chain: first non-empty column wins, per item (Books creator = author_name, else
        # store). Not concatenation -- two spellings of one author would become two entities.
        out = np.array([flatten(v) for v in aligned[columns[0]].to_numpy()], dtype=object)
        for extra in columns[1:]:
            alt = np.array([flatten(v) for v in aligned[extra].to_numpy()], dtype=object)
            empty = np.array([len(s) == 0 for s in out])
            out[empty] = alt[empty]
        docs[role] = out
    return docs


@dataclass
class ContentSpace:
    """A fitted content vector space: one vectorizer per role, plus the per-role weights and the
    column slice each role occupies in the concatenated matrix.

    Fitted on one item population (warm items of one dataset) and reusable, unchanged, to transform
    any other -- cold items, or a whole new domain. `transform` never refits, which is what makes
    the protocol inductive.
    """

    vectorizers: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    slices: dict = field(default_factory=dict)
    n_features: int = 0

    def transform(self, docs: dict[str, np.ndarray]) -> sparse.csr_matrix:
        """Vectorize `docs` (role -> array of strings, item_index order) into the fitted space.
        Returns a row-L2-normalized CSR item x term matrix, so `X @ X.T` is exactly cosine
        similarity. Terms unseen at fit time are dropped (see module docstring)."""
        n_rows = len(next(iter(docs.values())))
        blocks = []
        for role in ROLES:
            vec = self.vectorizers.get(role)
            if vec is None:
                continue
            column = docs.get(role, np.full(n_rows, "", dtype=object))
            blocks.append(vec.transform(column) * self.weights[role])
        if not blocks:
            return sparse.csr_matrix((n_rows, 0), dtype=np.float32)
        # Each block is already row-unit-norm (TfidfVectorizer norm='l2') times its weight, so the
        # concatenation's dot product is sum_f w_f^2 cos_f. Renormalizing rows turns that into the
        # weighted AVERAGE, and lets an item missing a field spend its full norm on the rest.
        stacked = sparse.hstack(blocks, format="csr", dtype=np.float32)
        return normalize(stacked, norm="l2", axis=1, copy=False)

    def block_of(self, role: str) -> slice:
        """Column slice this role occupies -- lets a caller inspect what one field contributed to a
        similarity. (This is the interpretability that a dense factorization of the space discards.)"""
        return self.slices[role]


def fit_content_space(docs: dict[str, np.ndarray], fit_rows: np.ndarray, weights: dict = None,
                      min_df: int = 2, verbose: bool = True) -> ContentSpace:
    """Fit one TF-IDF block per role on `fit_rows` ONLY.

    `fit_rows` must be the warm item ids -- vocabulary and IDF are chosen without ever seeing a cold
    item, so the cold items being evaluated are genuinely out-of-sample for the content model too,
    not just for the collaborative one. Roles whose fit population yields no usable vocabulary
    (every term a singleton, or the field absent for this dataset) are dropped from the space
    rather than contributing an all-zero block.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    space = ContentSpace(weights={})
    offset = 0
    if verbose:
        print(f"{'role':<10}{'kind':>8}{'vocab':>10}{'fit nnz':>12}{'nonempty(fit)':>15}{'weight':>8}")
    for role in ROLES:
        column = docs.get(role)
        if column is None:
            continue
        fit_docs = column[fit_rows]
        if not any(len(s) for s in fit_docs):
            if verbose:
                print(f"{role:<10}{'-':>8}{'(absent)':>10}")
            continue
        is_entity = role in ENTITY_ROLES
        vec = TfidfVectorizer(
            analyzer=_entity_analyzer if is_entity else "word",
            lowercase=not is_entity,           # the entity analyzer lowercases internally
            stop_words=None if is_entity else "english",
            sublinear_tf=True,                 # 1 + log(tf): stop a 3,000-word blurb from
            min_df=min_df,                     #   outweighing a title by raw repetition
            dtype=np.float32,
        )
        try:
            fitted = vec.fit_transform(fit_docs)
        except ValueError:                     # sklearn raises on an empty vocabulary
            if verbose:
                print(f"{role:<10}{'entity' if is_entity else 'text':>8}{'(no vocab)':>10}")
            continue
        width = fitted.shape[1]
        space.vectorizers[role] = vec
        space.weights[role] = weights[role]
        space.slices[role] = slice(offset, offset + width)
        offset += width
        if verbose:
            nonempty = float((np.diff(fitted.indptr) > 0).mean())
            print(f"{role:<10}{'entity' if is_entity else 'text':>8}{width:>10,}"
                  f"{fitted.nnz:>12,}{nonempty * 100:>14.2f}%{weights[role]:>8.2f}")
    space.n_features = offset
    if verbose:
        print(f"{'TOTAL':<10}{'':>8}{offset:>10,}")
    return space


def load_item_documents(meta_path: str, dataset, field_map: dict = None,
                        columns: list = None) -> dict[str, np.ndarray]:
    """Read a metadata parquet and return role -> per-item-index document arrays for `dataset`.

    Kept here rather than in load.py so the Dataset contract stays about interactions only, and so
    a content method can be added without touching the module every other model imports.
    """
    field_map = field_map or BOOKS_FIELD_MAP
    wanted = {c for cols in field_map.values() for c in cols} | {"parent_asin"}
    meta = pd.read_parquet(meta_path, columns=sorted(columns or wanted))
    return build_documents(meta, field_map, dataset.n_items, dataset.index_to_item)
