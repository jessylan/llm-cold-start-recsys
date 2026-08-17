# This file was created with the assistance of Generative AI.
"""How an item's content vector is stored, and how a content score is computed from it.

`content.py` produces ONE shape of item representation: a sparse item x term matrix with
L2-normalized rows. `cbhcf.py` was written directly against that shape -- it called
`self._content[ids].T.tocsr()`, built a `cupyx` CSR from it, and assumed every product was sparse.
That was fine while TF-IDF was the only content model.

Intervention A breaks the assumption. A sentence-embedding block is DENSE (487,790 x 1,024), and
storing it as sparse would cost 500M explicit nonzeros -- ~6 GB for a matrix with no zeros in it --
while replacing a GEMM with a far slower SpMM. Worse, Intervention A is not purely dense either: it
keeps `content.py`'s TF-IDF blocks for `creator` and `taxonomy` (atomic entity tokens, where exact
match is the correct semantics and an encoder would blur "Fantasy" into "Science Fiction") and for
`reviews` (847 words mean, where IDF's automatic discounting of generic praise language is doing
real work). So the representation is genuinely MIXED: one dense block beside several sparse ones.

This module is the seam. It answers exactly two questions for any representation --

    hoist(item_ids)                 what setup can be done once per item block?
    scores(history_rows, hoisted)   what is the score for these users against that block?

-- and nothing else. `cbhcf.py` talks only to this interface, so it is agnostic to whether the
content model behind it is sparse, dense, or both.

**The load-bearing invariant survives unchanged.** Every implementation guarantees row-L2-normalized
item vectors, so `cosine(i, j) == x_i . x_j` and the content score still factors as

    cb[users, items] = (history[users] @ X) @ X[items].T

which is what keeps this tractable at 487,790 items -- the ~950 GB item-item similarity matrix never
exists. For the mixed case that factorization distributes over the blocks,

    (H @ [D | S]) @ [D | S][items].T  ==  (H @ D) @ D[items].T  +  (H @ S) @ S[items].T

so the dense half runs as a GEMM and the sparse half as an SpMM, and they are summed. This is an
identity, not an approximation: `bench_21` gates it against the explicit dense cosine.

**Normalization must span every block at once.** A mixed space CANNOT be built by normalizing the
dense part, normalizing the sparse part, and concatenating -- that would give each half its own unit
norm and silently discard the per-field weighting between them. `BlockItemSpace.from_blocks` takes
UNNORMALIZED weighted blocks and divides both by the single joint norm
`sqrt(||d_i||^2 + ||s_i||^2)`. This is why `content.ContentSpace.transform` grew a
`normalize_rows=False` option: its blocks are no longer always the whole space.
"""
import numpy as np
import scipy.sparse as sparse


def as_item_space(item_content) -> "ItemSpace":
    """Accept either a raw matrix or an already-built space.

    Existing callers (the steel thread, `hyperparameter_tuning`) pass `content.ContentSpace.transform`'s
    CSR matrix straight to `CBHCFModel.fit`, and must keep working with byte-identical results --
    `bench_21` is the gate on that. So a plain matrix is wrapped in `SparseItemSpace`, whose methods
    are literally the lines `cbhcf.py` used to run inline.
    """
    if isinstance(item_content, (SparseItemSpace, BlockItemSpace)):
        return item_content
    return SparseItemSpace(item_content)


class SparseItemSpace:
    """A sparse item x term space -- `content.py`'s output, and the pre-Intervention-A behaviour.

    Every method here is the code `cbhcf.py` previously ran inline, moved behind the interface
    without a numerical change of any kind.
    """

    def __init__(self, matrix):
        self.T = sparse.csr_matrix(matrix).astype(np.float32)
        self.n_items, self.n_features = self.T.shape

    # --- CPU ---------------------------------------------------------------------------------

    def hoist(self, item_ids):
        """`T[items].T` as CSR, hoisted out of the caller's user-chunk loop.

        Worth 12.6x measured: re-slicing and transposing a 136M-nonzero matrix once per chunk
        dominated everything else in the block build.
        """
        return self.T[np.asarray(item_ids)].T.tocsr()

    def scores(self, history_rows, hoisted) -> np.ndarray:
        profile = history_rows @ self.T                      # (b x n_terms), still sparse
        return np.asarray((profile @ hoisted).todense(), dtype=np.float32)

    # --- GPU (cuSPARSE SpMM) -----------------------------------------------------------------

    def gpu_hoist(self, item_ids):
        """Resident `T[items]` on the GPU. Note the orientation: the product is computed as
        `T[items] @ profile.T` rather than `profile @ T[items].T`, because the output is ~99% dense
        -- SpGEMM (sparse x sparse -> sparse) is the wrong algorithm for it and in fact exhausts
        cuSPARSE's workspace, while SpMM (sparse x dense -> dense) is the right shape."""
        import cupyx.scipy.sparse as cusp
        return cusp.csr_matrix(self.T[np.asarray(item_ids)].tocsr().astype(np.float32))

    def gpu_scores(self, hoisted, history_rows):
        """(b x n_block) cupy array. The caller owns freeing the pool."""
        import cupy as cp
        profile = (history_rows @ self.T).astype(np.float32)
        return (hoisted @ cp.asarray(np.asfortranarray(profile.toarray().T))).T


class BlockItemSpace:
    """A dense block beside a sparse block, jointly row-normalized -- Intervention A's space.

    `dense` is (n_items x d) float32 with the per-field weight already applied; `sparse_block` is
    (n_items x n_terms) CSR, likewise pre-weighted. Both must come from `from_blocks`, which is what
    applies the single joint normalization that makes `x_i . x_j` a true cosine across the whole
    space rather than within each half.

    `sparse_block` may be None (a purely dense space), which is what an ablation that drops the
    entity/reviews fields would use.
    """

    def __init__(self, dense, sparse_block):
        self.D = np.ascontiguousarray(dense, dtype=np.float32)
        self.S = None if sparse_block is None else sparse.csr_matrix(sparse_block).astype(np.float32)
        self.n_items = self.D.shape[0]
        self.n_features = self.D.shape[1] + (0 if self.S is None else self.S.shape[1])
        if self.S is not None and self.S.shape[0] != self.n_items:
            raise ValueError(f"block row counts disagree: dense {self.D.shape[0]:,} vs "
                             f"sparse {self.S.shape[0]:,}")

    @classmethod
    def from_blocks(cls, dense, sparse_block, dense_weight=1.0,
                    own_dense: bool = False) -> "BlockItemSpace":
        """Weight, then jointly normalize. THE constructor -- do not build one any other way.

        `dense` arrives row-unit-norm from the encoder (every model is called with
        `normalize_embeddings=True`) and `sparse_block` arrives as `content.ContentSpace.transform(
        ..., normalize_rows=False)`: per-role blocks each unit-norm, each scaled by its own weight,
        concatenated but NOT renormalized. Scaling the dense block by `dense_weight` and dividing
        everything by the joint norm makes the resulting dot product the weighted AVERAGE of the
        per-field cosines, exactly as in `content.py`:

            cos(i, j) = sum_f w_f^2 cos_f(i, j) / (||i|| ||j||)

        and -- the property that matters for a catalogue where fields are missing -- an item with no
        blurb, or no author, contributes a zero block there and the renormalization automatically
        redistributes that field's weight across the fields it does have.

        An item with NO content at all in any block would divide by zero; its row is left at exactly
        zero instead, which scores 0 against everything. That is the honest answer for an item we
        know nothing about, and it is what the sparse path already does.

        `own_dense=True` says the caller will not use `dense` again, letting the weighting happen
        IN PLACE. This is a memory fix, not a micro-optimization: `ascontiguousarray` returns the
        same object when the input is already contiguous float32, so `arr * weight` allocates a
        second full-size array while the first is still referenced. At Qwen3-4B's 2,560 dimensions
        that is 5.0 GB doubled to 10.0 GB, on a box where this step is already the high-water mark.
        Default False because a caller may well reuse its input -- `bench_21` does exactly that.
        """
        D = np.ascontiguousarray(dense, dtype=np.float32)
        if not own_dense and D is dense:
            D = D.copy()                       # caller keeps its array; we need our own to scale
        D *= np.float32(dense_weight)
        S = None if sparse_block is None else sparse.csr_matrix(sparse_block).astype(np.float32)

        sq = np.einsum("ij,ij->i", D, D)
        if S is not None:
            sq = sq + np.asarray(S.multiply(S).sum(axis=1)).ravel()
        norm = np.sqrt(sq, dtype=np.float32)
        inv = np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)

        D *= inv[:, None]
        if S is not None:
            S = sparse.diags(inv).astype(np.float32) @ S
        return cls(D, S)

    # --- CPU ---------------------------------------------------------------------------------

    def hoist(self, item_ids):
        ids = np.asarray(item_ids)
        # The dense half is hoisted as (d x n_block) so the per-chunk op is one contiguous GEMM.
        return (np.ascontiguousarray(self.D[ids].T),
                None if self.S is None else self.S[ids].T.tocsr())

    def scores(self, history_rows, hoisted) -> np.ndarray:
        dense_T, sparse_T = hoisted
        # (b x n_items) sparse @ (n_items x d) dense -> (b x d) dense, then one GEMM to (b x n_block).
        out = np.asarray(history_rows @ self.D, dtype=np.float32) @ dense_T
        if sparse_T is not None:
            profile = history_rows @ self.S
            out += np.asarray((profile @ sparse_T).todense(), dtype=np.float32)
        return out

    # --- GPU -------------------------------------------------------------------------------------

    def gpu_hoist(self, item_ids):
        import cupy as cp
        import cupyx.scipy.sparse as cusp
        ids = np.asarray(item_ids)
        return (cp.asarray(np.ascontiguousarray(self.D[ids].T)),
                None if self.S is None else
                cusp.csr_matrix(self.S[ids].tocsr().astype(np.float32)))

    def gpu_scores(self, hoisted, history_rows):
        import cupy as cp
        dense_T, sparse_T = hoisted
        profile = np.asarray(history_rows @ self.D, dtype=np.float32)
        out = cp.asarray(profile) @ dense_T
        if sparse_T is not None:
            prof_s = (history_rows @ self.S).astype(np.float32)
            out += (sparse_T @ cp.asarray(np.asfortranarray(prof_s.toarray().T))).T
        return out
