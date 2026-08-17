# This file was created with the assistance of Generative AI.
"""Intervention A -- sentence embeddings in place of TF-IDF for the item's own descriptive prose.

CBHCF's content term asks one question: which items are like which other items? The baseline answers
it with TF-IDF, which can only see shared vocabulary. Two books whose blurbs share no words are
exactly orthogonal to it, however obviously alike they are to a reader. Intervention A replaces that
one block with a pretrained sentence encoder, and changes nothing else -- same additive hybrid, same
frozen user profile, same lambda mechanism, same evaluation.

**Only the prose block is replaced.** The five roles are not interchangeable, and handing all of
them to an encoder would swap several things at once (measured on the 487,790-item Books catalogue):

    role       column          treatment              why
    title      title       }
    blurb      features    }-> ENCODER (dense)        243 tokens mean, p99 813 -- prose that fits a
                                                      window whole, read in context, which is the
                                                      capability being bought
    creator    author_name     TF-IDF entity          136,602 distinct authors. Exact match IS the
                               (unchanged)            correct semantics; an encoder would pull
                                                      "Kalayna Price" toward other names
    taxonomy   categories      TF-IDF entity          a controlled vocabulary of 1,269 category
                               (unchanged)            paths; an encoder would blur "Fantasy" into
                                                      "Science Fiction"
    reviews    description     TF-IDF word            847 words mean, 45% over 350. Heterogeneous
                               (unchanged)            stitched editorial sections ("About the
                                                      Author", "Praise for...", back-cover copy) --
                                                      the worst case for a single vector. And it is
                                                      dense with generic praise ("brilliant",
                                                      "page-turner") that IDF discounts automatically
                                                      and an encoder would treat as signal, making
                                                      every thriller resemble every other thriller

**The text block carries weight sqrt(2), and the exponent is the whole point.** It absorbs two roles
that each carried 1.0 in `content.DEFAULT_WEIGHTS`, and weights enter this space SQUARED -- the
identity `content.py` is built on is

    cos(i, j) = sum_f w_f^2 cos_f(i, j) / (||i|| ||j||)

so a block's share of an item's squared norm is `w_f^2`, not `w_f`. The baseline's shares are
title 1, creator 1, taxonomy 0.25, blurb 1, reviews 0.25, totalling 3.5, of which prose
(title + blurb) is 2/3.5 = 57.1%. Reproducing that here needs `w^2 = 2`, i.e. `w = sqrt(2)`:
2/(2 + 1 + 0.25 + 0.25) = 57.1%. Which is also the geometric statement of what merging the two roles
does -- two orthogonal unit blocks combine to norm sqrt(1^2 + 1^2).

Setting it to 2.0 by intuition ("1 + 1") would instead hand the dense block 4/5.5 = 72.7% of the
norm, over-weighting prose by a quarter against the baseline, so part of any measured gain would be
that reweighting rather than the representation. `bench_21` gates the DEFAULT against the baseline's
share for exactly that reason.

**That default is a parity choice, not a claim that sqrt(2) is optimal -- and the shipped
configuration is different.** `notebooks/intervention_a_weight_sweep.ipynb` later searched the prose
and long-text weights for Intervention A AND for the TF-IDF baseline over one identical grid, so
neither arm got a tuning budget the other lacked. It selected `text_weight = 2.0` with `description`
embedded at 1.0 (and, for the baseline, `reviews` 0.5 -> 1.0), recorded in
`outputs/hyperparams.json["steel_thread_config"]`, which is what `steel_thread.ipynb` actually runs.

Both numbers are correct for what they are, and the distinction matters when reading results:

    sqrt(2)  DEFAULT_TEXT_WEIGHT below -- identical prose share to the baseline, so an UNTUNED
             comparison isolates the representation and nothing else
    2.0      selected on dataset.cold_val under the matched-budget sweep; what the reported
             steel-thread numbers use

Anything constructing a space without passing `text_weight` gets the parity default.

**Embeddings are keyed by `parent_asin`, not `item_index`.** The TF-IDF cache is keyed on the
dataset fingerprint and goes stale whenever the split changes, which is correct for it -- its
vocabulary and IDF are FIT on the warm items of one particular split. An encoder fits nothing, so
its output depends only on the text. Keying by ASIN makes the cache split-independent: every row of
the metadata file is encoded once, and any dataset, any future split and any cold population just
looks its ASINs up. Nothing is ever re-encoded.

**Nothing is fit, which makes this MORE inductive than the baseline, not less.** `content.py` goes
to some trouble to fit vocabulary and IDF on warm items only, so cold items are genuinely
out-of-sample. A pretrained encoder has no fitting step at all -- there is no vocabulary to freeze
and no statistic estimated from this catalogue -- so a cold item's vector is computed exactly as a
warm item's is. The usual leakage question does not arise. (What DOES arise: the encoder saw a web
crawl during pretraining, so it may have encountered these books. That is a property of every
pretrained model and is not specific to the cold items.)

**bf16, and why.** Every model is encoded at bfloat16 with `max_seq_length=1024`. The cap truncates
0.26% of items and leaves top-10 item neighbour lists 100% identical to an 8192-token reference, so
it is free; it is set for cross-model comparability (windows in the slate range from 8k to 40k) and
for memory predictability. bf16 is NOT free -- it agrees with fp32 to a cosine of 0.99993 per item
but still reshuffles ~4% of top-10 neighbours. It is used anyway because `Qwen3-Embedding-4B` cannot
be loaded in fp32 on a 24 GB card (16.1 GB of weights before activations), and because a uniform
dtype means that perturbation applies identically to all six models and so cannot favour one.
"""
import os
import re

import numpy as np
import pandas as pd

from recsys import content
from recsys import item_space

# Roles the encoder takes, joined into one document per item, in this order. `blurb` resolves to
# Books `features` and Movies `description` -- addressing ROLES rather than columns is what lets one
# code path serve both catalogues (see content.py's field maps).
TEXT_ROLES = ("title", "blurb")
# Roles that keep `content.py`'s TF-IDF treatment. TEXT_ROLES and SPARSE_ROLES must partition the
# roles actually in use: a role represented twice would be double-counted in the item's norm.
SPARSE_ROLES = ("creator", "taxonomy", "reviews")

# sqrt(w_title^2 + w_blurb^2) = sqrt(2), NOT w_title + w_blurb = 2 -- weights are squared in this
# space, so sqrt(2) is what reproduces the baseline's 57.1% prose share. This is the PARITY default,
# used when no weight is passed; the tuned steel-thread configuration uses 2.0 from
# hyperparams.json["steel_thread_config"]. See the module docstring.
DEFAULT_TEXT_WEIGHT = float(np.sqrt(2.0))
MAX_SEQ_LEN = 1024
TORCH_DTYPE = "bfloat16"

# Post-hoc transforms of the text block, all FIT ON WARM ITEMS ONLY. See fit_text_transform.
TEXT_TRANSFORMS = ("none", "center", "abtt", "whiten")

# --- the `description` field's internal structure ------------------------------------------------
# Books `description` is not one document: it is Amazon's editorial sections concatenated. Measured
# over all 487,790 items, 99.5% of populated descriptions carry at least one of these headings, and
# among the items long enough to be truncated at 1024 tokens, 99.6% do -- so the structure exists
# exactly where it is needed. Splitting here rather than on fixed token windows means every chunk is
# a coherent unit (one complete bio, one complete review) instead of an arbitrary slice.
#
# Groups, not raw headings: a block that is present for 1% of the catalogue (Kirkus) costs 1,024
# dense columns and buys almost nothing, so the fourteen headings collapse to three blocks with
# 92% / ~70% / ~25% coverage.
DESCRIPTION_GROUPS = {
    # The author, on themselves. `From the Author` is the author's own note; `About the Author` is a
    # third-person bio. Both are about the person, which exact-match `creator` cannot compare
    # semantically -- "similar KIND of author" is signal only an encoder can see.
    "author_bio": ["About the Author", "From the Author"],
    # Third-party editorial opinion, from wherever it was syndicated.
    "editorial_review": ["Review", "From Publishers Weekly", "From Booklist", "From Library Journal",
                         "From School Library Journal", "From Kirkus Reviews", "From AudioFile",
                         "Amazon.com Review", "From The New Yorker", "Editorial Reviews"],
    # Publisher copy about the book, plus -- deliberately -- the catch-all. Any text before the
    # first heading, any heading not listed here, and `Unknown` placeholders land here rather than
    # being dropped. Measured residual after the groups above: ~0.2-0.3% of all description text.
    # NOTE: `Excerpt.` carries copyright/ISBN front matter that IDF would have discounted and an
    # encoder will not; if this block underperforms, that is the first thing to suspect.
    "jacket_copy": ["From the Back Cover", "From the Inside Flap", "From the Publisher",
                    "Book Description", "Product Description", "Excerpt.", "Unknown"],
}
DESCRIPTION_RESIDUAL_GROUP = "jacket_copy"

# Longest-first so the alternation prefers "Amazon.com Review" over the substring "Review".
_ALL_MARKERS = sorted({mk for v in DESCRIPTION_GROUPS.values() for mk in v}, key=len, reverse=True)
_MARKER_OF = {mk: g for g, v in DESCRIPTION_GROUPS.items() for mk in v}
_SECTION_RE = re.compile("(" + "|".join(re.escape(mk) for mk in _ALL_MARKERS) + ")", re.IGNORECASE)

WINDOW_WORDS = 750          # ~1,000 tokens, inside every slate model's 1,024 cap
WINDOW_OVERLAP = 100        # only matters for sections with no finer structure of their own
MAX_WINDOWS = 4             # ~85% of all description text; the tail is deep and cheap to lose


def split_description(text: str) -> dict:
    """One `description` string -> {group: text}, losing nothing.

    Text before the first recognised heading, and any heading not in DESCRIPTION_GROUPS, go to
    DESCRIPTION_RESIDUAL_GROUP. Absent groups are simply missing from the returned dict; the caller
    turns those into zero blocks, whose weight `BlockItemSpace.from_blocks` redistributes across the
    groups the item does have -- the same treatment `content.py` gives a missing field.
    """
    out = {}
    if not text:
        return out
    parts = _SECTION_RE.split(text)
    head = parts[0].strip()
    if head:
        out[DESCRIPTION_RESIDUAL_GROUP] = [head]
    for marker, body in zip(parts[1::2], parts[2::2]):
        group = _MARKER_OF.get(marker.strip(), DESCRIPTION_RESIDUAL_GROUP)
        # Canonical spelling of the marker is kept: it tells the encoder what kind of passage this
        # is, which is context the text alone does not always carry.
        out.setdefault(group, []).append(f"{marker.strip()}. {body.strip()}")
    return {g: " ".join(v).strip() for g, v in out.items() if " ".join(v).strip()}


def _windows(text: str, size: int = WINDOW_WORDS, overlap: int = WINDOW_OVERLAP,
             cap: int = MAX_WINDOWS) -> list:
    """Word windows for a section that is itself longer than the model's context.

    Sections are already coherent, so this only fires for the ~18% that have no finer structure.
    Overlap keeps a thought that straddles a boundary present in both halves."""
    w = text.split()
    if len(w) <= size:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(w[i:i + size]) for i in range(0, len(w), step)][:cap]


class Encoder:
    """One slate entry: what to load, how to prompt it, and how big a batch it survives.

    `prefix` is prepended to every document. Only nomic needs one, and which one is a judgement
    call worth recording: its recommended prefixes are task-specific, and CBHCF's content score is
    `cosine(user profile, candidate item)` where the profile is a weighted mean of ITEM vectors --
    both sides are items, so this is symmetric item-item similarity, not asymmetric query-document
    retrieval. `clustering:` is the prefix nomic documents for that, so it is what we use rather
    than the more commonly cited `search_document:`.

    `batch` is measured, not guessed -- see the benchmarks in the session notes. Bigger is not
    better: Qwen3-0.6B is FASTER at 32 than at 64 (219 vs 202 docs/s) because a longer batch is
    padded to its longest member.
    """

    def __init__(self, key, repo, remote_code=False, prefix="", batch=64, dim=None):
        self.key, self.repo, self.remote_code = key, repo, remote_code
        self.prefix, self.batch, self.dim = prefix, batch, dim


ENCODERS = {e.key: e for e in [
    Encoder("gte-modernbert-base", "Alibaba-NLP/gte-modernbert-base", batch=128, dim=768),
    Encoder("nomic-v1.5", "nomic-ai/nomic-embed-text-v1.5", remote_code=True,
            prefix="clustering: ", batch=128, dim=768),
    Encoder("bge-m3", "BAAI/bge-m3", batch=64, dim=1024),
    Encoder("arctic-l-v2.0", "Snowflake/snowflake-arctic-embed-l-v2.0", batch=64, dim=1024),
    Encoder("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", batch=32, dim=1024),
    Encoder("qwen3-4b", "Qwen/Qwen3-Embedding-4B", batch=16, dim=2560),
]}

# The catalogues to encode. Both are done in one pass so a later experiment on Movies -- or a
# Books -> Movies transfer run, which is the point of content.py's role indirection -- never needs
# the GPU again.
CATALOGUES = {
    "books": ("data/filtered/books_meta_5core_common.parquet", content.BOOKS_FIELD_MAP),
    "movies": ("data/filtered/movies_meta_5core_common.parquet", content.MOVIES_FIELD_MAP),
}


def build_text_documents(meta: pd.DataFrame, field_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """(parent_asin, document) over EVERY row of a metadata table, deduped by ASIN.

    Deliberately NOT a function of a `Dataset`: the whole point of keying by ASIN is that the
    encoding pass is independent of any split. `content.build_documents` is the split-aligned
    counterpart, used later to place these vectors into item_index order.
    """
    meta = meta.drop_duplicates(subset="parent_asin", keep="last")
    parts = []
    for role in TEXT_ROLES:
        columns = [c for c in field_map.get(role, []) if c in meta.columns]
        if not columns:
            parts.append(np.full(len(meta), "", dtype=object))
            continue
        out = np.array([content._as_text(v) for v in meta[columns[0]].to_numpy()], dtype=object)
        for extra in columns[1:]:                     # first non-empty column wins, as in content.py
            alt = np.array([content._as_text(v) for v in meta[extra].to_numpy()], dtype=object)
            empty = np.array([len(s) == 0 for s in out])
            out[empty] = alt[empty]
        parts.append(out)
    # ". " between roles, and no stray separator when a role is empty -- a document that begins
    # ". " would spend tokens on punctuation for the ~2% of items with no blurb.
    docs = np.array([". ".join(p for p in row if p) for row in zip(*parts)], dtype=object)
    return meta["parent_asin"].to_numpy(), docs


def embedding_path(model_key: str, catalogue: str, embed_dir: str = "../data/embeddings") -> str:
    return os.path.join(embed_dir, f"{catalogue}_{model_key}.npz")


def encode_catalogue(model_key: str, catalogue: str, *, embed_dir: str = "../data/embeddings",
                     data_root: str = "..", device: str = "cuda:1", overwrite: bool = False,
                     batch: int = None, progress: bool = True, verbose: bool = True) -> str:
    """Encode every row of one catalogue with one model and save it. Idempotent.

    Returns the path. Stored as float16: a normalized vector's components sit around 1/sqrt(d) ~ 0.02,
    comfortably inside float16's range, and it halves a file that is 1-2.5 GB per model. Scoring
    promotes to float32, matching how `cbhcf` already caches its content block.
    """
    path = embedding_path(model_key, catalogue, embed_dir)
    if os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[intervention_a] {catalogue}/{model_key}: already encoded -> {path}")
        return path

    import torch
    from sentence_transformers import SentenceTransformer

    spec = ENCODERS[model_key]
    meta_path, field_map = CATALOGUES[catalogue]
    wanted = {c for role in TEXT_ROLES for c in field_map.get(role, [])} | {"parent_asin"}
    meta = pd.read_parquet(os.path.join(data_root, meta_path), columns=sorted(wanted))
    asins, docs = build_text_documents(meta, field_map)
    if spec.prefix:
        docs = np.array([spec.prefix + d for d in docs], dtype=object)

    # The RESOLVED torch dtype, not the string. transformers normally accepts "bfloat16" and
    # resolves it, but nomic-bert-2048's custom from_pretrained forwards it straight to
    # `model.to(dtype=...)`, which rejects a str: "to() received an invalid combination of
    # arguments - got (dtype=str,)". Passing the real dtype object works for every model.
    st = SentenceTransformer(spec.repo, trust_remote_code=spec.remote_code, device=device,
                             model_kwargs={"torch_dtype": getattr(torch, TORCH_DTYPE)})
    st.max_seq_length = min(MAX_SEQ_LEN, st.max_seq_length)
    seq_len = int(st.max_seq_length)      # read before the model is released, for the provenance stamp
    batch = batch or spec.batch
    if verbose:
        print(f"[intervention_a] {catalogue}/{model_key}: {len(docs):,} documents, "
              f"max_seq_length={seq_len}, batch={batch}, {TORCH_DTYPE}, {device}", flush=True)

    vectors = st.encode(list(docs), batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=progress, convert_to_numpy=True)
    del st
    torch.cuda.empty_cache()

    os.makedirs(embed_dir, exist_ok=True)
    np.savez(path, parent_asin=asins.astype(str), vectors=vectors.astype(np.float16),
             model=spec.repo, max_seq_length=seq_len, dtype=TORCH_DTYPE)
    if verbose:
        print(f"[intervention_a] saved {vectors.shape} -> {path} "
              f"({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return path


def description_path(model_key: str, catalogue: str, embed_dir: str = "../data/embeddings") -> str:
    return os.path.join(embed_dir, f"{catalogue}_{model_key}_desc.npz")


def encode_description_sections(model_key: str, catalogue: str = "books", *,
                                embed_dir: str = "../data/embeddings", data_root: str = "..",
                                device: str = "cuda:1", item_batch: int = 20_000,
                                overwrite: bool = False, verbose: bool = True) -> str:
    """Encode `description` as one vector PER SECTION GROUP per item.

    Storing per group rather than per item is what makes both variants reachable from one encode:
    the caller can keep the groups as separate weighted blocks (each item compared bio-to-bio and
    review-to-review) or pool them into a single block, without touching the GPU again.

    Within a group, text longer than the context window is windowed and pooled LENGTH-WEIGHTED --
    a token-level average, so a 1,780-word review does not count the same as a 165-word one. That
    pooling is defensible because it averages homogeneous text; pooling ACROSS groups would average
    an author bio with a plot review and represent neither.

    Absent groups are stored as exact zero rows, which is the signal `BlockItemSpace.from_blocks`
    uses to redistribute that block's weight onto the groups the item does have.
    """
    path = description_path(model_key, catalogue, embed_dir)
    if os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[intervention_a] {catalogue}/{model_key} description: already encoded -> {path}")
        return path

    import torch
    from sentence_transformers import SentenceTransformer

    spec = ENCODERS[model_key]
    meta_path, field_map = CATALOGUES[catalogue]
    # `reviews` is the role holding the long editorial text (Books `description`).
    columns = [c for c in field_map.get("reviews", []) if c]
    if not columns:
        raise ValueError(f"catalogue {catalogue!r} has no `reviews` role to section-split")
    meta = pd.read_parquet(os.path.join(data_root, meta_path),
                           columns=sorted({"parent_asin", *columns}))
    meta = meta.drop_duplicates(subset="parent_asin", keep="last")
    asins = meta["parent_asin"].to_numpy()
    raw = [content._as_text(v) for v in meta[columns[0]].to_numpy()]

    groups = list(DESCRIPTION_GROUPS)
    sections = [split_description(t) for t in raw]
    if verbose:
        for g in groups:
            n = sum(1 for s in sections if g in s)
            print(f"[intervention_a] {g:<18} present for {n:>7,} / {len(sections):,} items "
                  f"({n / len(sections):.1%})")

    st = SentenceTransformer(spec.repo, trust_remote_code=spec.remote_code, device=device,
                             model_kwargs={"torch_dtype": getattr(torch, TORCH_DTYPE)})
    st.max_seq_length = min(MAX_SEQ_LEN, st.max_seq_length)
    dim = st.get_sentence_embedding_dimension()
    out = {g: np.zeros((len(asins), dim), dtype=np.float16) for g in groups}

    for g in groups:
        rows = [i for i, s in enumerate(sections) if g in s]
        if verbose:
            print(f"[intervention_a] encoding {g}: {len(rows):,} items", flush=True)
        for start in range(0, len(rows), item_batch):
            batch_rows = rows[start:start + item_batch]
            texts, owner, weights = [], [], []
            for i in batch_rows:
                wins = _windows(sections[i][g])
                texts.extend(wins)
                owner.extend([i] * len(wins))
                weights.extend(len(w.split()) for w in wins)   # length weights, per window
            vecs = st.encode(texts, batch_size=spec.batch, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            # Accumulate against BATCH-local rows: a catalogue-sized accumulator would be 2 GB per
            # iteration for rows that are almost all zero.
            local_of = {row: j for j, row in enumerate(batch_rows)}
            owner_local = np.fromiter((local_of[i] for i in owner), dtype=np.int64, count=len(owner))
            weights = np.asarray(weights, dtype=np.float32)
            acc = np.zeros((len(batch_rows), dim), dtype=np.float32)
            # Length-weighted sum per item, then renormalize -> a direction, not a magnitude.
            np.add.at(acc, owner_local, vecs * weights[:, None])
            n = np.sqrt(np.einsum("ij,ij->i", acc, acc))
            out[g][np.asarray(batch_rows)] = (acc / np.maximum(n, 1e-12)[:, None]).astype(np.float16)
            if verbose:
                print(f"\r[intervention_a]   {g}: {min(start + item_batch, len(rows)):,}/{len(rows):,}",
                      end="", flush=True)
        if verbose:
            print(flush=True)
    del st
    torch.cuda.empty_cache()

    os.makedirs(embed_dir, exist_ok=True)
    np.savez(path, parent_asin=asins.astype(str), groups=np.array(groups),
             model=spec.repo, max_seq_length=MAX_SEQ_LEN, dtype=TORCH_DTYPE,
             **{f"vec_{g}": out[g] for g in groups})
    if verbose:
        print(f"[intervention_a] saved {len(groups)} x {out[groups[0]].shape} -> {path} "
              f"({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return path


def chunked_path(model_key: str, catalogue: str, embed_dir: str = "../data/embeddings") -> str:
    return os.path.join(embed_dir, f"{catalogue}_{model_key}_descchunk.npz")


def encode_description_chunked(model_key: str, catalogue: str = "books", *,
                               embed_dir: str = "../data/embeddings", data_root: str = "..",
                               device: str = "cuda:1", item_batch: int = 20_000,
                               overwrite: bool = False, verbose: bool = True) -> str:
    """Encode `description` as ONE vector per item, chunking on fixed word windows.

    The control for `encode_description_sections`. Both produce a single description block of the
    same width at the same weight; the ONLY difference is where the cuts fall -- arbitrary
    750-word windows here, section boundaries there. So comparing them isolates exactly one
    question: does it matter that a chunk is a coherent passage?

    This is also the naive default most pipelines use, which makes it the honest baseline for
    whether any of the structural work earned its complexity.
    """
    path = chunked_path(model_key, catalogue, embed_dir)
    if os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[intervention_a] {catalogue}/{model_key} chunked description: exists -> {path}")
        return path

    import torch
    from sentence_transformers import SentenceTransformer

    spec = ENCODERS[model_key]
    meta_path, field_map = CATALOGUES[catalogue]
    columns = [c for c in field_map.get("reviews", []) if c]
    if not columns:
        raise ValueError(f"catalogue {catalogue!r} has no `reviews` role")
    meta = pd.read_parquet(os.path.join(data_root, meta_path),
                           columns=sorted({"parent_asin", *columns}))
    meta = meta.drop_duplicates(subset="parent_asin", keep="last")
    asins = meta["parent_asin"].to_numpy()
    raw = [content._as_text(v) for v in meta[columns[0]].to_numpy()]
    rows = [i for i, t in enumerate(raw) if t.strip()]

    st = SentenceTransformer(spec.repo, trust_remote_code=spec.remote_code, device=device,
                             model_kwargs={"torch_dtype": getattr(torch, TORCH_DTYPE)})
    st.max_seq_length = min(MAX_SEQ_LEN, st.max_seq_length)
    dim = st.get_sentence_embedding_dimension()
    out = np.zeros((len(asins), dim), dtype=np.float16)
    if verbose:
        print(f"[intervention_a] chunked description: {len(rows):,}/{len(asins):,} items, "
              f"windows of {WINDOW_WORDS}w overlap {WINDOW_OVERLAP} cap {MAX_WINDOWS}", flush=True)

    for start in range(0, len(rows), item_batch):
        batch_rows = rows[start:start + item_batch]
        texts, owner, weights = [], [], []
        for j, i in enumerate(batch_rows):
            wins = _windows(raw[i])               # identical windowing to the section path
            texts.extend(wins)
            owner.extend([j] * len(wins))
            weights.extend(len(w.split()) for w in wins)
        vecs = st.encode(texts, batch_size=spec.batch, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        acc = np.zeros((len(batch_rows), dim), dtype=np.float32)
        np.add.at(acc, np.asarray(owner), vecs * np.asarray(weights, dtype=np.float32)[:, None])
        n = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        out[np.asarray(batch_rows)] = (acc / np.maximum(n, 1e-12)[:, None]).astype(np.float16)
        if verbose:
            print(f"\r[intervention_a]   chunked: {min(start + item_batch, len(rows)):,}/{len(rows):,}",
                  end="", flush=True)
    if verbose:
        print(flush=True)
    del st
    torch.cuda.empty_cache()

    os.makedirs(embed_dir, exist_ok=True)
    np.savez(path, parent_asin=asins.astype(str), vectors=out, model=spec.repo,
             window_words=WINDOW_WORDS, overlap=WINDOW_OVERLAP, max_windows=MAX_WINDOWS)
    if verbose:
        print(f"[intervention_a] saved {out.shape} -> {path} "
              f"({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return path


def chunked_description_embeddings(dataset, model_key: str, catalogue: str = "books",
                                   embed_dir: str = "../data/embeddings",
                                   verbose: bool = True) -> np.ndarray:
    """The naive-chunked description block in item_index order; zero rows where absent."""
    path = chunked_path(model_key, catalogue, embed_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no chunked description embeddings at {path}. Run "
            f"intervention_a.encode_description_chunked('{model_key}', '{catalogue}').")
    blob = np.load(path, allow_pickle=False)
    row_of = pd.Series(np.arange(len(blob["parent_asin"])), index=blob["parent_asin"])
    asin_of_index = pd.Series(dataset.index_to_item).reindex(range(dataset.n_items)).astype(str)
    r = row_of.reindex(asin_of_index.to_numpy()).to_numpy()
    found = ~pd.isna(r)
    out = np.zeros((dataset.n_items, blob["vectors"].shape[1]), dtype=np.float32)
    out[found] = blob["vectors"][r[found].astype(np.int64)].astype(np.float32)
    if verbose:
        nz = int((np.einsum("ij,ij->i", out, out) > 0).sum())
        print(f"[intervention_a] chunked description: {nz:,}/{dataset.n_items:,} items "
              f"({nz / dataset.n_items:.1%})")
    return out


def description_embeddings(dataset, model_key: str, catalogue: str = "books",
                           embed_dir: str = "../data/embeddings",
                           verbose: bool = True) -> dict:
    """{group: (n_items x d) float32} in item_index order. Absent groups are zero rows."""
    path = description_path(model_key, catalogue, embed_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no description embeddings at {path}. Run "
            f"intervention_a.encode_description_sections('{model_key}', '{catalogue}').")
    blob = np.load(path, allow_pickle=False)
    row_of = pd.Series(np.arange(len(blob["parent_asin"])), index=blob["parent_asin"])
    asin_of_index = pd.Series(dataset.index_to_item).reindex(range(dataset.n_items)).astype(str)
    rows = row_of.reindex(asin_of_index.to_numpy()).to_numpy()
    found = ~pd.isna(rows)
    idx = rows[found].astype(np.int64)

    out = {}
    for g in [str(x) for x in blob["groups"]]:
        src = blob[f"vec_{g}"]
        dest = np.zeros((dataset.n_items, src.shape[1]), dtype=np.float32)
        dest[found] = src[idx].astype(np.float32)
        out[g] = dest
        if verbose:
            nz = int((np.einsum("ij,ij->i", dest, dest) > 0).sum())
            print(f"[intervention_a] {g:<18} {nz:,}/{dataset.n_items:,} items ({nz / dataset.n_items:.1%})")
    return out


def item_embeddings(dataset, model_key: str, catalogue: str = "books",
                    embed_dir: str = "../data/embeddings", verbose: bool = True) -> np.ndarray:
    """The saved vectors placed into `item_index` order for `dataset`, as (n_items x d) float32.

    An item with no metadata row gets an all-zero vector, which `BlockItemSpace.from_blocks` then
    lets the other blocks' weight absorb -- the same treatment `content.py` gives an item missing a
    field. Zero rows are reported rather than silently tolerated: a large count means the metadata
    file and the interaction file disagree about the catalogue, which is a data bug, not a modelling
    choice.
    """
    path = embedding_path(model_key, catalogue, embed_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no embeddings for {catalogue}/{model_key} at {path}. Run "
            f"intervention_a.encode_catalogue('{model_key}', '{catalogue}') on the GPU box first.")
    blob = np.load(path, allow_pickle=False)
    row_of = pd.Series(np.arange(len(blob["parent_asin"])), index=blob["parent_asin"])

    asin_of_index = pd.Series(dataset.index_to_item).reindex(range(dataset.n_items)).astype(str)
    rows = row_of.reindex(asin_of_index.to_numpy()).to_numpy()
    found = ~pd.isna(rows)

    vectors = blob["vectors"]
    out = np.zeros((dataset.n_items, vectors.shape[1]), dtype=np.float32)
    out[found] = vectors[rows[found].astype(np.int64)].astype(np.float32)
    if verbose:
        missing = int((~found).sum())
        print(f"[intervention_a] {model_key}: {dataset.n_items - missing:,}/{dataset.n_items:,} "
              f"items embedded (dim {vectors.shape[1]})"
              + (f"   MISSING METADATA: {missing:,}" if missing else ""))
    return out


def fit_text_transform(dense: np.ndarray, fit_rows: np.ndarray, kind: str = "none",
                       n_components: int = None, eps: float = 1e-6,
                       verbose: bool = True) -> dict:
    """Fit a corpus-level transform of the text block on WARM items only.

    **Why this exists at all.** The TF-IDF blocks carry a corpus-statistics stage -- IDF -- fit on
    the warm items, which discounts terms that appear everywhere and keeps the discriminative ones.
    The embedding block has no equivalent: it is whatever the encoder emits. That is not a neutral
    difference. Sentence embeddings are strongly ANISOTROPIC (Ethayarajh 2019; Mu & Viswanath 2018):
    every vector lies in a narrow cone, so cosines bunch into a high, narrow band and the usable
    dynamic range is small. That shape produces decent AVERAGE rank while separating the top of the
    ranking poorly -- which is exactly the AUC-good / NDCG@100-bad split measured against TF-IDF on
    cold_val. So this is a fairness correction, not an extra tuning knob: it gives the text block
    the same kind of warm-fit corpus adaptation the sparse blocks already have.

    **The inductive contract is preserved exactly.** Statistics come from `fit_rows` (the warm items)
    and are then applied UNCHANGED to every item, cold included -- the same protocol
    `content.fit_content_space` uses for vocabulary and IDF. A cold item's vector is still computed
    without any cold item having influenced the transform.

    kinds:
      none    -- identity (the default; what the six-encoder selection ran).
      center  -- subtract the warm mean. The cheapest anisotropy fix: the dominant component of an
                 embedding cloud is usually just its offset from the origin, which carries no
                 discriminative information but dominates every cosine.
      abtt    -- "all-but-the-top" (Mu & Viswanath, ICLR 2018): centre, then remove the projection
                 onto the top `n_components` principal directions (default d/100). Removes the few
                 directions shared by all items while leaving the rest of the space untouched.
      whiten  -- centre, then rescale each principal direction to unit variance (Su et al. 2021).
                 The most aggressive: it equalizes ALL directions, which maximizes dynamic range but
                 also amplifies low-variance directions that may be noise -- hence `eps`.

    Returns a dict carrying everything `apply_text_transform` needs, so the identical transform can
    be replayed on another catalogue (Movies) without refitting.
    """
    if kind not in TEXT_TRANSFORMS:
        raise ValueError(f"unknown text transform {kind!r}; expected one of {TEXT_TRANSFORMS}")
    if kind == "none":
        return {"kind": "none"}

    warm = np.ascontiguousarray(dense[np.asarray(fit_rows)], dtype=np.float32)
    mu = warm.mean(axis=0)
    warm -= mu
    out = {"kind": kind, "mean": mu}

    if kind != "center":
        # Covariance of the warm block only. d x d, so tiny even at 2,560 dims.
        cov = (warm.T @ warm) / max(len(warm) - 1, 1)
        evals, evecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(evals)[::-1]                      # eigh returns ascending
        evals, evecs = evals[order], evecs[:, order]
        if kind == "abtt":
            d = max(1, (n_components if n_components is not None else dense.shape[1] // 100))
            out["components"] = np.ascontiguousarray(evecs[:, :d].astype(np.float32))
            if verbose:
                var = float(evals[:d].sum() / evals.sum())
                print(f"[intervention_a] abtt: removing top {d} of {dense.shape[1]} directions "
                      f"({var:.1%} of warm variance)")
        else:                                                 # whiten
            keep = n_components or dense.shape[1]
            lam = np.maximum(evals[:keep], eps)               # floor: tiny eigenvalues are noise
            out["matrix"] = np.ascontiguousarray(
                (evecs[:, :keep] / np.sqrt(lam)).astype(np.float32))
            if verbose:
                print(f"[intervention_a] whiten: {dense.shape[1]} -> {keep} dims, "
                      f"condition number before {evals[0] / max(evals[-1], eps):.3g}")
    del warm
    return out


def apply_text_transform(dense: np.ndarray, transform: dict, chunk: int = 50_000,
                         verbose: bool = True) -> np.ndarray:
    """Apply a fitted transform to every item and re-L2-normalize.

    Chunked because the intermediate is the size of the block itself -- 5 GB at 2,560 dims -- and
    this step sits next to the other high-water marks. Re-normalization is not optional: the whole
    pipeline rests on rows being unit norm so a dot product IS a cosine (see item_space)."""
    kind = transform["kind"]
    if kind == "none":
        return dense

    mu = transform["mean"]
    n, d = dense.shape
    width = d if kind != "whiten" else transform["matrix"].shape[1]
    out = dense if (kind != "whiten" or width == d) else np.empty((n, width), dtype=np.float32)

    for s in range(0, n, chunk):
        block = dense[s:s + chunk] - mu
        if kind == "abtt":
            V = transform["components"]
            block -= (block @ V) @ V.T
        elif kind == "whiten":
            block = block @ transform["matrix"]
        norms = np.sqrt(np.einsum("ij,ij->i", block, block))
        np.divide(block, np.maximum(norms, 1e-12)[:, None], out=block)
        out[s:s + chunk] = block
    if verbose:
        print(f"[intervention_a] text transform '{kind}' applied: {dense.shape} -> {out.shape}")
    return out


def _sparse_field_map(field_map: dict, sparse_roles) -> dict:
    """Restrict a field map to the roles that will actually be FITTED.

    `content.load_item_documents` reads every column any role mentions and flattens a document for
    each of `ROLES`. Once `description` moves to the dense side that is pure waste, and expensive
    waste: it re-reads the 847-word-mean `description` column out of a ~1 GB parquet and runs
    Python-level string flattening over 487,790 rows, to build documents that `fit_content_space`
    then ignores. Measured effect during the description sweep: the kernel sat at 99.8% of one core
    with RSS at 40.6 GB of 46 GB -- close enough to the ceiling to matter for the widest variant.
    """
    return {role: cols for role, cols in field_map.items() if role in set(sparse_roles)}


def build_space(dataset, model_key: str, *, catalogue: str = "books", fit_rows=None,
                field_map: dict = None, weights: dict = None, text_weight: float = DEFAULT_TEXT_WEIGHT,
                min_df: int = 2, embed_dir: str = "../data/embeddings", data_root: str = "..",
                text_transform: str = "none", n_components: int = None,
                description_mode: str = "tfidf",
                verbose: bool = True) -> item_space.BlockItemSpace:
    """Intervention A's item representation for one encoder: dense text block + TF-IDF entity/reviews
    blocks, jointly row-normalized.

    `fit_rows` must be the WARM item ids, exactly as for `content.fit_content_space` -- the sparse
    half still fits its vocabulary and IDF without ever seeing a cold item. The dense half has
    nothing to fit, so it is untouched by this argument.

    `description_mode` decides what happens to the long editorial field. The four modes are designed
    so that consecutive pairs differ in exactly ONE thing:
      "tfidf"    -- leave it as a TF-IDF block (the six-encoder selection ran this way).
      "chunked"  -- ONE embedding block, cut on fixed 750-word windows. The naive default.
      "pooled"   -- ONE embedding block, cut on SECTION boundaries then averaged.
                    chunked -> pooled isolates: must the cuts be semantically coherent?
      "sections" -- THREE embedding blocks, one per section group, kept separate.
                    pooled -> sections isolates: does keeping the sections apart matter?

    In every mode the field weights are unchanged from `content.DEFAULT_WEIGHTS`: the section blocks
    split `reviews`' 0.5 between them, so total squared weight stays 3.5 and prose stays 64.3% of an
    item's norm. Nothing here introduces a tuned parameter.
    """
    if description_mode not in ("tfidf", "sections", "pooled", "chunked"):
        raise ValueError(f"unknown description_mode {description_mode!r}")
    meta_path, default_map = CATALOGUES[catalogue]
    field_map = field_map or default_map
    if fit_rows is None:
        fit_rows = np.unique(dataset.ref_train.nonzero()[1])

    dense = item_embeddings(dataset, model_key, catalogue, embed_dir, verbose=verbose)

    # Fit on warm items, apply to all -- the same inductive rule the TF-IDF blocks follow. Done
    # BEFORE the blocks are combined, since from_blocks assumes unit-norm rows.
    if text_transform != "none":
        tf = fit_text_transform(dense, fit_rows, kind=text_transform, n_components=n_components,
                                verbose=verbose)
        dense = apply_text_transform(dense, tf, verbose=verbose)

    sparse_roles = list(SPARSE_ROLES)
    if description_mode != "tfidf":
        # `reviews` moves from the sparse side to the dense side, so it must leave the TF-IDF fit or
        # the field would be counted twice in the item's norm.
        sparse_roles = [r for r in sparse_roles if r != "reviews"]
        w_review0 = (weights or content.DEFAULT_WEIGHTS).get(
            "reviews", content.DEFAULT_WEIGHTS["reviews"])
        if description_mode == "chunked":
            # The control: same single block at the same weight as "pooled", cut on fixed windows
            # instead of section boundaries. The pair isolates cut placement and nothing else.
            chunk_block = chunked_description_embeddings(dataset, model_key, catalogue, embed_dir,
                                                         verbose=verbose)
            dense = np.hstack([dense * np.float32(text_weight),
                               chunk_block * np.float32(w_review0)]).astype(np.float32)
            text_weight = 1.0
            if verbose:
                print(f"[intervention_a] description as ONE naive-chunked block at w={w_review0}")
            del chunk_block
            docs = content.load_item_documents(os.path.join(data_root, meta_path), dataset,
                                               field_map=_sparse_field_map(field_map, sparse_roles))
            space = content.fit_content_space(docs, fit_rows, weights=weights, min_df=min_df,
                                              verbose=verbose, roles=sparse_roles)
            sparse_block = space.transform(docs, normalize_rows=False)
            out = item_space.BlockItemSpace.from_blocks(dense, sparse_block, dense_weight=1.0,
                                                        own_dense=True)
            if verbose:
                print(f"[intervention_a] space: dense {out.D.shape} + sparse {sparse_block.shape}"
                      f"  -> {out.n_features:,} features")
            return out
        desc = description_embeddings(dataset, model_key, catalogue, embed_dir, verbose=verbose)
        groups = list(desc)
        # `reviews` carried w=0.5, i.e. w^2 = 0.25 of the item's squared norm. Splitting that across
        # the section blocks keeps the field's total share EXACTLY as the baseline had it, so no new
        # weight is being tuned -- only redistributed within the field it already owned.
        w_review = (weights or content.DEFAULT_WEIGHTS).get("reviews", content.DEFAULT_WEIGHTS["reviews"])
        if description_mode == "sections":
            per = float(w_review / np.sqrt(len(groups)))
            blocks = [(desc[g] * np.float32(per)) for g in groups]
            if verbose:
                print(f"[intervention_a] description as {len(groups)} section blocks "
                      f"({', '.join(groups)}) at w={per:.4f} each "
                      f"(sum w^2 = {len(groups) * per ** 2:.4f} = reviews w^2)")
        else:                                   # "pooled": one block, groups averaged per item
            stacked = np.zeros_like(desc[groups[0]])
            for g in groups:
                stacked += desc[g]
            nrm = np.sqrt(np.einsum("ij,ij->i", stacked, stacked))
            stacked /= np.maximum(nrm, 1e-12)[:, None]
            blocks = [stacked * np.float32(w_review)]
            if verbose:
                print(f"[intervention_a] description pooled across {len(groups)} groups at "
                      f"w={w_review}")
        # Text block first (already unit-norm), then the description blocks, each pre-weighted.
        dense = np.hstack([dense * np.float32(text_weight)] + blocks).astype(np.float32)
        text_weight = 1.0        # weights are already baked into the concatenated block
        del desc, blocks

    docs = content.load_item_documents(os.path.join(data_root, meta_path), dataset,
                                       field_map=_sparse_field_map(field_map, sparse_roles))
    space = content.fit_content_space(docs, fit_rows, weights=weights, min_df=min_df,
                                      verbose=verbose, roles=sparse_roles)
    # normalize_rows=False is load-bearing: these blocks are only PART of the space, and the single
    # joint norm has to span them and the dense block together (see item_space.BlockItemSpace).
    sparse_block = space.transform(docs, normalize_rows=False)

    # own_dense=True: `dense` was built by this function and is not used again, so the weighting can
    # be done in place. Without it the peak is doubled (5.0 -> 10.0 GB for a 2,560-dim encoder).
    out = item_space.BlockItemSpace.from_blocks(dense, sparse_block, dense_weight=text_weight,
                                                own_dense=True)
    del dense
    if verbose:
        # einsum, not np.abs(...).sum(axis=1): the latter materializes a whole extra copy of the
        # dense block (5 GB at 2,560 dims) purely to print a diagnostic count.
        empty = int((np.einsum("ij,ij->i", out.D, out.D) == 0).sum())
        print(f"[intervention_a] space: dense {out.D.shape} (weight {text_weight}) + sparse "
              f"{sparse_block.shape}  -> {out.n_features:,} features   "
              f"items with no text vector: {empty:,}")
    return out
