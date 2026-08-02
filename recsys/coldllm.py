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
    """Runs two in-process vLLM engines, both loading the SAME model -- one configured for
    `task="generate"` (Refining Simulation's yes/no), one for `task="embed"` with mean pooling
    (Filtering Simulation's embeddings). This matches ColdLLM's own design (arXiv:2402.09176,
    Sec 4.2.1): item/user embeddings for Filtering come from mean-pooling the SAME LLM's token
    embeddings (Eq. 7: E_llm = (1/n) sum E_token), not a separate embedding model.

    vLLM fixes one task per engine instance, so the two stages need two instances even though
    they're the same model. Both are built LAZILY -- only the first time embed_texts() or
    yes_probability() is actually called, not at construction -- so a simulator used for only
    one stage (as this module's Filtering/Refining cells each do, via their own separate
    instance) never pays for the other engine's memory at all. Calling both methods on the SAME
    instance still ends up with both engines resident at once by the time the second one
    builds, since nothing here tears the first one down -- prefer two separate instances, one
    per stage, if that matters. When both ARE resident together, that's the model's weights
    loaded TWICE (once per engine), i.e. double the GPU memory of a single copy; split
    `gpu_memory_utilization` between them (e.g. ~0.45 each) if they share one GPU, or run them
    on separate GPUs via `CUDA_VISIBLE_DEVICES`-scoped processes if that's not workable in one
    process.

    `**engine_kwargs` passes straight through to both `vllm.LLM(...)` constructors (e.g.
    `gpu_memory_utilization`, `dtype`, `max_model_len`) for tuning to the actual hardware."""

    def __init__(self, model: str, **engine_kwargs):
        self.model = model
        self._engine_kwargs = engine_kwargs
        self._generate_llm = None
        self._embed_llm = None

    @property
    def generate_llm(self):
        """Built on first access, not at construction -- see class docstring."""
        if self._generate_llm is None:
            self._generate_llm = LLM(model=self.model, task="generate", **self._engine_kwargs)
        return self._generate_llm

    @property
    def embed_llm(self):
        """Built on first access, not at construction -- see class docstring."""
        if self._embed_llm is None:
            self._embed_llm = LLM(
                model=self.model, task="embed",
                override_pooler_config={"pooling_type": "MEAN", "normalize": False},
                **self._engine_kwargs,
            )
        return self._embed_llm

    def embed_texts(self, texts: list, batch_size: int = 256) -> np.ndarray:
        """Mean-pooled embedding per text via vLLM's embed-task engine -- ColdLLM's own
        Filtering-stage representation (arXiv:2402.09176 Eq. 7: mean pooling over token
        embeddings), computed directly by vLLM's pooling engine rather than by hand. Unlike the
        paper, there's no learned MLP projection on top -- these raw mean-pooled vectors feed
        straight into filter_candidates()'s inner product. `batch_size` bounds how many texts
        are queued to the engine per call; vLLM schedules/batches internally either way, so
        this is a memory safeguard, not a concurrency knob (unlike a per-request HTTP client,
        there's no serial-request penalty here to work around)."""
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            outputs = self.embed_llm.embed(batch)
            embeddings.extend(o.outputs.embedding for o in outputs)
        return np.asarray(embeddings, dtype=np.float32)

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


def user_preference_embeddings(train_matrix: sparse.csr_matrix, item_embeddings: np.ndarray) -> np.ndarray:
    """Mean of a user's training-history items' embeddings -- the embedding-space analog of
    cbhcf.CBHCFModel._content_score_matrix's train_matrix @ similarity, but producing one dense
    per-user vector instead of a per-(user, item) score."""
    history_counts = np.asarray(train_matrix.sum(axis=1)).ravel()
    summed = train_matrix @ item_embeddings
    safe_counts = np.where(history_counts > 0, history_counts, 1)
    pooled = summed / safe_counts[:, None]
    pooled[history_counts == 0] = 0.0
    return pooled


def filter_candidates(
    item_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    cold_item_ids: np.ndarray,
    top_k: int = 50,
) -> dict:
    """Stage 1 -- Filtering Simulation. For each cold item, ranks every user by dot-product
    similarity to that item's embedding and keeps the top_k. Returns {item_idx: user_idx array}.
    Plain numpy argpartition over a brute-force inner product -- this is what the paper itself
    does too (Eq. 17: an inner product against every user), not an approximation chosen for
    lack of FAISS; revisit only if this becomes the bottleneck at real dataset size."""
    candidates = {}
    for item_idx in cold_item_ids:
        scores = user_embeddings @ item_embeddings[item_idx]
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
