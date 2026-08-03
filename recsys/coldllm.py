"""Intervention B: ColdLLM-style synthetic interaction generation for strict cold-start items
(arXiv:2402.09176). A two-stage funnel -- Filtering Simulation narrows every cold item's
candidate pool from all users down to the top-K most content-similar, then Refining Simulation
asks an LLM a yes/no for each surviving (item, user) pair. Positive predictions become a
synthetic interactions matrix.

This is NOT a RetrievalModel -- its output is data, not a scorer. The synthetic matrix returned
by refine_candidates() is meant to be added to a training matrix before fitting an existing
model (cf.ALSModel, CBHCFModel), the same slot Dataset.revealed_matrix_at_k(k) already
occupies. It must NEVER be folded into whatever matrix eval.py passes as the "already
interacted" exclusion set for recommend()/ndcg_at_k/recall_and_hit_rate_at_k/auc_at_full --
unlike real revealed history, synthetic interactions aren't guaranteed disjoint from
dataset.test_matrix, so masking with them would silently corrupt evaluation. Keep it in its own
matrix, added only to whatever gets passed to fit()/fold_in(), never to an exclusion-set
argument.

Two Refining-stage prompting strategies share this module -- direct yes/no and reason-then-answer
-- both exercised in a single run of notebooks/intervention_b_coldllm.ipynb; see
_build_prompt/yes_probability's `reasoning` flag.
"""
import re

import numpy as np
import scipy.sparse as sparse
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

_ANSWER_RE = re.compile(r"Answer:\s*(yes|no)", re.IGNORECASE)


class VLLMColdLLMSimulator:
    """Runs Refining Simulation's yes/no judgment as an in-process vLLM `task="generate"`
    engine. (Filtering Simulation doesn't use an LLM -- see filter_candidates.)

    `**engine_kwargs` passes straight through to `vllm.LLM(...)` (e.g.
    `gpu_memory_utilization`, `dtype`, `max_model_len`) for tuning to the actual hardware."""

    def __init__(self, model: str, **engine_kwargs):
        self.generate_llm = LLM(model=model, task="generate", **engine_kwargs)

    def yes_probability(self, prompts: list, batch_size: int = 256, reasoning: bool = False) -> np.ndarray:
        """Hard 0.0/1.0 per prompt -- the paper fine-tunes its Refining model via LoRA, so it gets
        a smooth confidence score; this is zero-shot, so `threshold` in refine_candidates() only
        ever sees a 0.0 or a 1.0 either way. Two mutually exclusive decoding strategies, matching
        `reasoning`'s prompt from _build_prompt():

        `reasoning=False` (Intervention B1): guided decoding constrains the ENTIRE completion to
        the literal choices "yes"/"no" -- the output IS one of the two choices, no parsing needed.

        `reasoning=True` (Intervention B2): the model is asked to justify itself first, so the
        completion can't be constrained to just two tokens. Guided decoding instead constrains it
        to a regex grammar -- free text of up to ~600 chars followed by a mandatory final
        "Answer: yes"/"Answer: no" line -- which still guarantees a parseable answer exists
        (just after however much reasoning text the model produces), then `_ANSWER_RE` pulls the
        last such line out. `max_tokens` is raised to 200 to leave room for that reasoning text
        before the answer.

        Either way, vLLM batches the whole prompt list internally within one `.chat()` call (the
        point of running in-process rather than one request at a time), so `batch_size` only
        bounds how many prompts are queued per call, not concurrency."""
        if reasoning:
            sampling_params = SamplingParams(
                temperature=0, max_tokens=200,
                guided_decoding=GuidedDecodingParams(regex=r"[\s\S]{1,600}\nAnswer: (yes|no)"),
            )
        else:
            sampling_params = SamplingParams(
                temperature=0, max_tokens=5,
                guided_decoding=GuidedDecodingParams(choice=["yes", "no"]),
            )
        answers = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            conversations = [[{"role": "user", "content": p}] for p in batch]
            outputs = self.generate_llm.chat(conversations, sampling_params)
            texts = [o.outputs[0].text.strip() for o in outputs]
            if reasoning:
                answers.extend(_parse_final_answer(t) for t in texts)
            else:
                answers.extend(t.lower() for t in texts)
        return np.array([1.0 if a == "yes" else 0.0 for a in answers], dtype=np.float32)


def _row_scaled(matrix) -> sparse.csr_matrix:
    """Divide each row by its own sum, leaving all-zero rows at zero (a user with no training
    history has no content profile and must score 0, not NaN)."""
    m = sparse.csr_matrix(matrix)
    counts = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, counts, out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    return (sparse.diags(inv) @ m).astype(np.float32)


def user_content_profile(train_matrix: sparse.csr_matrix, item_content: sparse.csr_matrix) -> sparse.csr_matrix:
    """A user's TF-IDF content profile for Filtering Simulation: the row-scaled average of the
    content vectors of the items in their training history."""
    return _row_scaled(train_matrix) @ sparse.csr_matrix(item_content)


def filter_candidates(
    item_content: sparse.csr_matrix,
    user_profiles: sparse.csr_matrix,
    cold_item_ids: np.ndarray,
    top_k: int = 50,
) -> dict:
    """Stage 1 -- Filtering Simulation, via TF-IDF cosine similarity -- `item_content` is
    the same (n_items x n_terms) row-L2-normalized matrix CBHCF scores with
    (content.ContentSpace.transform), so an inner product against a cold item's row IS its
    cosine similarity to every user's profile. For each cold item, ranks every user by that
    similarity and keeps the top_k. Returns {item_idx: user_idx array}. Plain numpy
    argpartition over a sparse-dense inner product."""
    candidates = {}
    for item_idx in cold_item_ids:
        item_vec = item_content[int(item_idx)].T
        scores = (user_profiles @ item_vec).toarray().ravel()
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        candidates[int(item_idx)] = top[np.argsort(-scores[top])]
    return candidates


def _build_prompt(history_texts: list, item_text: str, max_history: int = 10, reasoning: bool = False) -> str:
    """Simple, fixed prompt template. `reasoning=False` (Intervention B1) leaves the prompt as a
    bare yes/no question -- paired with yes_probability()'s choice-constrained decoding, the
    model jumps straight to the answer token. `reasoning=True` (Intervention B2) appends an
    instruction to justify the answer first, on the hypothesis that a brief chain-of-thought
    surfaces content-relevance signal (genre/author/theme overlap with the user's history) that a
    forced-immediate answer skips past -- paired with yes_probability()'s regex-constrained
    decoding, which still guarantees a parseable final line."""
    history = "\n".join(f"- {t}" for t in history_texts[:max_history]) or "(no prior history)"
    prompt = (
        "A user has previously shown interest in the following items:\n"
        f"{history}\n\n"
        "Would this user be interested in the following new item?\n"
        f"- {item_text}"
    )
    if reasoning:
        prompt += (
            "\n\nFirst give a one-sentence reason, then respond on a new final line in the "
            "exact format \"Answer: yes\" or \"Answer: no\"."
        )
    return prompt


def _parse_final_answer(text: str) -> str:
    """Pulls the yes/no out of a reasoning completion's mandatory final "Answer: yes/no" line
    (see yes_probability's `reasoning=True` grammar). Takes the LAST match in case the model's
    reasoning text itself echoes the word "Answer:" earlier on; falls back to "no" only if the
    guided-decoding grammar somehow didn't produce one (shouldn't happen, but refine_candidates()
    should never crash on a single malformed completion out of thousands)."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].lower() if matches else "no"


def refine_candidates(
    simulator: VLLMColdLLMSimulator,
    candidates: dict,
    item_metadata: list,
    train_matrix: sparse.csr_matrix,
    threshold: float = 0.5,
    batch_size: int = 16,
    reasoning: bool = False,
) -> sparse.csr_matrix:
    """Stage 2 -- Refining Simulation. For every (item, candidate_user) pair surviving Stage 1,
    prompts the LLM for P(Yes) and keeps pairs at or above `threshold` as synthetic interactions.
    Returns a sparse (n_users, n_items) matrix -- see module docstring for why this must be kept
    separate from whatever matrix is used for recommend()'s "already interacted" exclusion.

    `reasoning` selects the prompting strategy for every pair (see _build_prompt/yes_probability):
    False (Intervention B1) is a bare yes/no question with choice-constrained decoding; True
    (Intervention B2) asks for a one-sentence justification before the answer, with regex-
    constrained decoding. Passed straight through to both -- the two must agree, since B2's
    prompt only makes sense paired with B2's parsing and vice versa."""
    n_users, n_items = train_matrix.shape
    train_csr = sparse.csr_matrix(train_matrix)  # indptr/indices below are row (user) sliced

    prompts, pair_users, pair_items = [], [], []
    for item_idx, user_ids in candidates.items():
        item_text = item_metadata[item_idx]
        for user_idx in user_ids:
            start, end = train_csr.indptr[user_idx], train_csr.indptr[user_idx + 1]
            history_item_ids = train_csr.indices[start:end]
            history_texts = [item_metadata[i] for i in history_item_ids]
            prompts.append(_build_prompt(history_texts, item_text, reasoning=reasoning))
            pair_users.append(user_idx)
            pair_items.append(item_idx)

    if not prompts:
        return sparse.csr_matrix((n_users, n_items))

    probs = simulator.yes_probability(prompts, batch_size=batch_size, reasoning=reasoning)
    keep = probs >= threshold
    rows = np.array(pair_users)[keep]
    cols = np.array(pair_items)[keep]
    data = np.ones(len(rows))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


class SyntheticAugmentedDataset:
    """Wraps a real load.Dataset so fold_in() sees synthetic interactions at every reveal level
    k, not just k=0 -- lets the existing eval.sweep()/sweep_mode_a_cached() machinery produce a
    full warm-up curve for a ColdLLM-augmented model, unmodified.

    Overrides ONLY revealed_item_users_at_k -- what ALSModel.fold_in() uses to recalculate
    cold-item factors -- combining it with `synthetic_matrix`. Deliberately does NOT override
    revealed_matrix_at_k, which eval.py separately uses to build the "already interacted"
    exclusion matrix (train_k = dataset.ref_train + dataset.revealed_matrix_at_k(k)). Synthetic
    interactions have no guarantee of being disjoint from dataset.test_matrix (unlike real
    revealed history, guaranteed disjoint by load.py's reveal/reserve split), so augmenting the
    exclusion set too would risk silently masking out the very test items being measured -- see
    this module's top-level docstring.

    `restrict_to_item_ids`, if given, also narrows `test_matrix` to just those cold items --
    without it, a curve run against a subsample of cold items (see
    notebooks/intervention_b_coldllm.ipynb) would be diluted by every other cold item, which
    received no synthetic augmentation at all and is therefore identical to the unaugmented
    baseline. Restricting both curves being compared to the SAME item subset keeps the
    comparison fair.

    Everything else (ref_train, cold_item_ids, index_to_item, ...) delegates straight through.
    """

    def __init__(self, dataset, synthetic_matrix, restrict_to_item_ids=None):
        self._dataset = dataset
        self._synthetic = sparse.csr_matrix(synthetic_matrix)
        if restrict_to_item_ids is not None:
            test_coo = dataset.test_matrix.tocoo()
            keep = np.isin(test_coo.col, restrict_to_item_ids)
            self.test_matrix = sparse.csr_matrix(
                (test_coo.data[keep], (test_coo.row[keep], test_coo.col[keep])),
                shape=dataset.test_matrix.shape,
            )

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def revealed_item_users_at_k(self, k):
        item_ids, item_users = self._dataset.revealed_item_users_at_k(k)
        synthetic_item_users = self._synthetic[:, item_ids].T.tocsr()
        combined = (item_users + synthetic_item_users).tocsr()
        combined.data = np.minimum(combined.data, 1.0)  # avoid double-weighting a rare overlap
        return item_ids, combined
