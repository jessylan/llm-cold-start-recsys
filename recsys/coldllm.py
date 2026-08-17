# This file was created with the assistance of Generative AI.
"""Intervention B: ColdLLM-style synthetic interaction generation for strict cold-start items
(arXiv:2402.09176). A two-stage funnel -- Filtering Simulation narrows every cold item's
candidate pool from all users down to the top-K most content-similar, then Refining Simulation
asks an LLM about each surviving (item, user) pair. The survivors become a synthetic interactions
matrix.

SCORES, THEN SELECTION -- and the separation is the main design decision here. The paper's
refiner is LoRA-tuned, so its yes/no verdict carries a calibrated confidence. Run zero-shot, the
same question is answered "yes" for ~98% of the candidates Filtering has already selected: the
verdict is nearly constant and a threshold has nothing to sort. What IS informative is the
ordering within those yes-es, so Refining stores a graded log-odds per pair (`yes_logodds`) and
every decision about which pairs become interactions happens afterwards, in memory:

    select_top_n        the n best-ranked users per item          (the intervention)
    random_n_matrix     n users drawn at random from the pool     (control: is there a signal?)
    popularity_n_matrix the n most active users in the pool       (control: is it just activity?)

Keeping selection out of the expensive pass means N, calibration, and any ablation are array
operations over saved scores rather than another run of the model. `assemble_user_priors` adds the
one calibration that can reorder a per-item ranking -- each user's own baseline agreeableness,
estimated over shared probe items and subtracted PMI-style.

This is NOT a RetrievalModel -- its output is data, not a scorer. The synthetic matrix is meant to
be added to a training matrix before fitting an existing model (cf.ALSModel, CBHCFModel), the same
slot Dataset.revealed_matrix_at_k(k) already occupies. It must NEVER be folded into whatever
matrix eval.py passes as the "already interacted" exclusion set for
recommend()/ndcg_at_k/recall_and_hit_rate_at_k/auc_at_full -- unlike real revealed history,
synthetic interactions aren't guaranteed disjoint from dataset.test_matrix, so masking with them
would silently corrupt evaluation. Keep it in its own matrix, added only to whatever gets passed
to fit()/fold_in(), never to an exclusion-set argument.

Two Refining-stage prompting strategies share this module -- direct yes/no and reason-then-answer
-- both exercised in a single run of notebooks/intervention_b_coldllm.ipynb; see
_build_prompt/yes_logodds's `reasoning` flag. Their log-odds LEVELS are not comparable to each
other (see _answer_position); their rankings are.
"""
import json
import os
import re
import time

import numpy as np
import scipy.sparse as sparse

_ANSWER_RE = re.compile(r"Answer:\s*(yes|no)", re.IGNORECASE)

# Surface forms of the two answers. A tokenizer treats "yes", " yes", "Yes" and "YES" as DIFFERENT
# tokens, so scoring only the bare lowercase form silently throws away most of the probability mass
# and is the classic bug in LM-as-classifier code. Mass is summed over all of them instead.
_YES_FORMS = ("yes", " yes", "Yes", " Yes", "YES", " YES")
_NO_FORMS = ("no", " no", "No", " No", "NO", " NO")

# Real items each user is scored against to estimate their own baseline.
#
# The prompt count looks alarming -- ~79k distinct candidate users against 136k pairs means this
# pass issues several times as many PROMPTS as the thing it corrects -- but prompts are the wrong
# unit. All of a user's probes share that user's entire history prefix and differ only in the
# trailing item, so with prefix caching a user costs one full prefill (~1,500 tokens) plus ~100
# tokens per extra probe. Going 4 -> 8 is roughly +20% of tokens, not +100%.
#
# The binding consideration is statistical, and it points the other way from cost. The prior is a
# SAMPLE MEAN, carrying error sigma_within/sqrt(n), and that error is subtracted from every score
# -- so an under-estimated prior injects more noise into the ranking than the bias it removes. The
# correction only pays when
#
#     var_between_users  >  var_within_user / n_probe
#
# which `assemble_user_priors` reports the components of, so `n_probe` can be checked against real
# numbers instead of chosen by feel.
DEFAULT_N_PROBE = 8

# How many top logprobs to ask for at the answer position. Wider than it looks like it needs to be
# on purpose: vLLM returns the UNMASKED distribution, so an emphatic model pushes the losing answer
# out of a narrow window and its score has to be bounded instead of read (see `_logodds_at`). The
# values are computed regardless, so a wider window costs almost nothing. NOTE this must also be
# raised on the ENGINE -- vLLM refuses any request above its own `max_logprobs`, which defaults
# to 20 -- which VLLMColdLLMSimulator does for its callers.
DEFAULT_N_LOGPROBS = 64


def _logsumexp(values) -> float:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum())) if np.isfinite(m) else float(m)


def _logodds_at(position):
    """log P(yes) - log P(no) from ONE generated position's logprob dict.

    Returns `(logodds, imputed)`; `imputed` flags that one side had to be bounded rather than read.

    Why log-odds rather than P(yes): a difference of logprobs is invariant to whatever constant
    the distribution was normalised by, so it is well defined without depending on that. Log space
    also preserves resolution the probability domain destroys -- the direct strategy's scores sit
    between P(yes)=0.99992 and 0.9999998, which are indistinguishable as float32 probabilities but
    six log-odds units apart.

    THE MISSING-SIDE CASE, measured rather than assumed. vLLM reports the top-k of the UNMASKED
    vocabulary distribution even under structured decoding -- the returned candidates include
    tokens the grammar forbids ("_no", ".no"), which is the proof. So the constraint does not
    guarantee both answers appear, and when the model is emphatic (having just written "a
    different genre", say) the losing answer falls outside the top-k entirely. Those pairs are the
    most confidently negative ones -- precisely the worst to drop, since dropping them would bias
    selection toward the pairs the model was unsure about.

    The absent side is therefore BOUNDED instead of discarded: everything outside the returned
    top-k is, by construction, no more probable than the smallest logprob in it, so that value is a
    valid ceiling. The resulting score understates the magnitude but keeps the pair at the correct
    extreme of the ranking, which is all top-N needs.
    """
    if not position:
        return float("nan"), False
    by_text = {}
    for lp in position.values():
        text = getattr(lp, "decoded_token", None)
        if text is not None:
            by_text.setdefault(text, lp.logprob)
    yes = [v for t, v in by_text.items() if t in _YES_FORMS]
    no = [v for t, v in by_text.items() if t in _NO_FORMS]
    if not yes and not no:
        return float("nan"), False          # not an answer position at all
    floor = min(by_text.values())
    imputed = not yes or not no
    return _logsumexp(yes or [floor]) - _logsumexp(no or [floor]), imputed


def _answer_position(output):
    """The generated position holding the yes/no answer.

    Direct prompts answer at position 0. Reasoning prompts answer LAST, after their own
    justification, so the scan runs from the end -- and the score there is conditioned on the text
    the model just produced, which is why reasoning log-odds are comparable within that strategy
    but not against the direct strategy's levels.
    """
    logprobs = getattr(output, "logprobs", None)
    if not logprobs:
        return None
    # EITHER side identifies the answer position, not both: the losing answer is often outside the
    # returned top-k when the model is emphatic (see _logodds_at). Requiring both would skip the
    # real answer position and then fail on some earlier, meaningless one.
    for position in reversed(logprobs):
        if not position:
            continue
        texts = {getattr(lp, "decoded_token", None) for lp in position.values()}
        if texts & set(_YES_FORMS) or texts & set(_NO_FORMS):
            return position
    return None


def _structured_outputs(**kwargs):
    """Build vLLM's structured-decoding params across the 0.8 -> 0.26 rename.

    Up to ~0.9 this was `GuidedDecodingParams`, passed as `SamplingParams(guided_decoding=...)`.
    Current vLLM calls it `StructuredOutputsParams` / `structured_outputs=`. Both spellings are
    tried so the module runs on either, because the vLLM version is dictated by whatever torch the
    rest of the environment can tolerate -- and that is not a decision this module should force.

    Returns `(kwarg_name, params_object)` for splatting into SamplingParams.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return "structured_outputs", StructuredOutputsParams(**kwargs)
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return "guided_decoding", GuidedDecodingParams(**kwargs)


_WSL_PIN_MEMORY_MIN_KERNEL = (4, 19, 121)


def _enable_wsl_pin_memory():
    """Opt in to pinned memory under WSL2, without which vLLM 0.26 will not start here at all.

    vLLM disables pinned memory whenever it detects WSL, because support depends on the driver.
    In 0.26 that is no longer merely a slow path: `is_uva_available()` is defined as
    `is_pin_memory_available()`, and the GPU worker raises `RuntimeError: UVA is not available`
    during engine init. So on WSL the conservative default is not a performance hint, it is a hard
    failure -- and vLLM's own escape hatch is this environment variable.

    Only set when the kernel clears vLLM's own floor, and never over an explicit setting: someone
    who exported VLLM_WSL2_ENABLE_PIN_MEMORY=0 has a reason. Set before `import vllm`, since
    vllm.envs reads the environment at import time.
    """
    import platform
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") is not None:
        return
    release = platform.uname().release
    if "microsoft" not in release.lower():
        return                                    # not WSL; the default is already correct
    try:
        version = tuple(int(p) for p in re.match(r"(\d+(?:\.\d+)*)", release).group(1).split("."))
    except (AttributeError, ValueError):
        return                                    # unparseable: leave vLLM's default alone
    if version >= _WSL_PIN_MEMORY_MIN_KERNEL:
        os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"


def _add_to_path(var, directory, first=False):
    parts = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    if directory in parts:
        return
    os.environ[var] = os.pathsep.join([directory] + parts if first else parts + [directory])


def _prepare_build_toolchain():
    """Make vLLM's runtime kernel compilation find the RIGHT tools, without installing anything.

    Building the engine JIT-compiles CUDA kernels (flashinfer's sampler, inductor's fused ops).
    Two things have to be on the environment for that to work, and neither is guaranteed by simply
    having the right packages installed:

    `ninja` -- torch shells out to it. The ninja wheel puts that executable in the environment's
    `bin/`, the same directory as the running interpreter, so it exists whenever vLLM does but is
    only VISIBLE when the environment was activated. A Jupyter kernel launched straight from an
    interpreter path, or `/path/to/env/bin/python script.py`, inherits a PATH without it and dies
    with `FileNotFoundError: 'ninja'` minutes into engine init.

    `nvcc` -- and this one silently picks the WRONG compiler rather than failing to find one. A
    distro CUDA toolkit at /usr/bin/nvcc shadows the `nvidia-cuda-nvcc` wheel that torch's own
    dependencies install, so a cu130 torch ends up compiling cu13 sources with, on this box, a
    CUDA 12.0 compiler -- which fails deep inside a ninja build log rather than saying so. The
    matching toolchain ships in `nvidia/cu<major>/`, selected here against `torch.version.cuda` so
    it tracks whatever torch was built for instead of a hardcoded version, and put FIRST on PATH
    so it wins over the distro one. CUDA_HOME is set alongside it because torch's extension
    builder consults that before falling back to a PATH search.

    `libnvrtc` -- the RUNTIME half of the same story, and the reason LD_LIBRARY_PATH is set too.
    flashinfer dlopens it by bare name; with nothing on the library path the lookup fails and it
    falls back, logging `ImportError: libnvrtc.so.13: cannot open shared object file` even though
    the library is installed two directories away. Harmless but wasteful. This works only because
    vLLM starts its engine in a SPAWNED SUBPROCESS: the dynamic linker reads LD_LIBRARY_PATH at
    process start, so setting it here cannot help this process, but every engine launched after
    it inherits the corrected environment.

    Anything the caller set explicitly is left alone -- an operator pointing at a deliberate
    toolchain should not be overridden by a default.
    """
    import sys
    _add_to_path("PATH", os.path.dirname(os.path.abspath(sys.executable)))

    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    try:
        import torch
        major = (torch.version.cuda or "").split(".")[0]
        import nvidia
        roots = [os.path.join(p, f"cu{major}") for p in nvidia.__path__]
    except Exception:
        return                      # no bundled toolchain to point at; leave the environment be
    for root in roots:
        if os.path.exists(os.path.join(root, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = root
            _add_to_path("PATH", os.path.join(root, "bin"), first=True)
            # Appended, not prepended: torch resolves its own CUDA libraries via RPATH and should
            # keep winning. This only has to make the library findable when nothing else offers it.
            libdir = os.path.join(root, "lib")
            if os.path.isdir(libdir):
                _add_to_path("LD_LIBRARY_PATH", libdir)
            return


def _engine_role(role: str) -> dict:
    """Tell vLLM this is a generation engine, across the 0.8 -> 0.26 rename.

    `LLM(task="generate")` up to ~0.10; current vLLM splits that into `runner` ("generate" vs
    "pooling") and `convert`, and passing the old name is a hard TypeError from EngineArgs rather
    than a warning. Asked of the dataclass rather than guessed from a version string, since the
    field is what actually decides.
    """
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs
    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    if "runner" in fields:
        return {"runner": role}
    if "task" in fields:
        return {"task": role}
    return {}          # neither: a version that infers the role from the model itself


class VLLMColdLLMSimulator:
    """Runs Refining Simulation's yes/no judgment as an in-process vLLM `task="generate"`
    engine. (Filtering Simulation doesn't use an LLM -- see filter_candidates.)

    `**engine_kwargs` passes straight through to `vllm.LLM(...)` (e.g.
    `gpu_memory_utilization`, `dtype`, `max_model_len`) for tuning to the actual hardware.

    vLLM is imported HERE, not at module scope, so that everything else in this module --
    Filtering Simulation, the work queue, assembly, `SyntheticAugmentedDataset` -- can be used in
    an environment without vLLM installed. Only the Refining stage genuinely needs it.
    """

    def __init__(self, model: str, **engine_kwargs):
        _enable_wsl_pin_memory()
        _prepare_build_toolchain()
        # The engine enforces its own ceiling on requested logprobs (default 20) and rejects
        # anything above it, so scoring's requirement is declared here rather than left for every
        # caller to remember. setdefault, so an explicit choice still wins.
        engine_kwargs.setdefault("max_logprobs", DEFAULT_N_LOGPROBS)
        from vllm import LLM
        self.generate_llm = LLM(model=model, **_engine_role("generate"), **engine_kwargs)
        self.last_n_imputed = 0

    def yes_logodds(self, prompts: list, batch_size: int = 256, reasoning: bool = False,
                    progress_cb=None, n_logprobs: int = DEFAULT_N_LOGPROBS) -> np.ndarray:
        """A GRADED score per prompt: log P(yes) - log P(no) at the answer position.

        This replaces an earlier hard 0/1 verdict, and the reason is worth stating. The paper's
        refiner is LoRA-tuned, so its yes/no carries a calibrated confidence; zero-shot, the same
        question is answered "yes" for ~98% of the candidates Filtering already selected. A binary
        verdict throws away the ordering WITHIN those yes-es -- which is the only part of this
        judge's output that is actually informative -- and leaves a threshold with nothing to sort.
        Reading the logprobs recovers that ordering, and costs nothing extra: the values are
        already computed for the token being generated.

        What this score is NOT is a probability of interaction. It is a statement about text, and
        an uncalibrated one; use it to RANK candidates (see select_top_n), not as a likelihood.
        Levels are additionally not comparable between the two strategies -- see _answer_position.

        Structured decoding is kept for parseability -- it does NOT guarantee both answers appear
        among the returned logprobs, because vLLM reports the unmasked distribution (see
        `_logodds_at`). `n_logprobs` is therefore generous: a wider window makes the losing answer
        far more likely to be present, and the values are computed anyway so asking for more of
        them is nearly free.

        Returns float32, NaN only where the answer position could not be located at all. The count
        of BOUNDED scores (one side outside the window) is recorded on `self.last_n_imputed`, since
        a systematic rise there would mean the window is too narrow to rank the confident tail.
        """
        from vllm import SamplingParams
        if reasoning:
            # The reasoning budget and max_tokens must not collide. At 600 chars the grammar can
            # consume essentially the whole 200-token allowance before reaching "Answer:", and a
            # completion truncated before the answer has NO scorable position -- measured at 1.7%
            # of pairs, which would silently drop those candidates from selection. 320 chars is
            # still several times the one sentence the prompt asks for, and leaves ample room.
            key, params = _structured_outputs(regex=r"[\s\S]{1,320}\nAnswer: (yes|no)")
            sampling_params = SamplingParams(temperature=0, max_tokens=200, logprobs=n_logprobs,
                                             **{key: params})
        else:
            key, params = _structured_outputs(choice=["yes", "no"])
            sampling_params = SamplingParams(temperature=0, max_tokens=5, logprobs=n_logprobs,
                                             **{key: params})
        scores, n_imputed = [], 0
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            conversations = [[{"role": "user", "content": p}] for p in batch]
            # vLLM's own tqdm bar is suppressed. It redraws one stderr line with carriage returns,
            # while `progress_cb` prints newline-terminated lines to stdout -- in a notebook the
            # two streams are flushed as separate messages, so the overwrite lands nowhere and the
            # output fills with orphaned bars and blank space. run_worker already reports per-batch
            # rate, chunk totals and an ETA, which is the information the bar was providing.
            outputs = self.generate_llm.chat(conversations, sampling_params, use_tqdm=False)
            for o in outputs:
                value, imputed = _logodds_at(_answer_position(o.outputs[0]))
                scores.append(value)
                n_imputed += bool(imputed)
            if progress_cb is not None:
                progress_cb(len(scores), len(prompts))
        self.last_n_imputed = n_imputed
        return np.asarray(scores, dtype=np.float32)


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
    block_size: int = 128,
    verbose: bool = False,
) -> dict:
    """Stage 1 -- Filtering Simulation, via TF-IDF cosine similarity. `item_content` is the same
    (n_items x n_terms) row-L2-normalized matrix CBHCF scores with, so an inner product against a
    cold item's row IS its cosine similarity to every user's profile. For each cold item, ranks
    every user by that similarity and keeps the top_k. Returns {item_idx: user_idx array}.

    Scored a BLOCK of items at a time, which is the difference between minutes and hours. scipy's
    sparse product iterates over the LEFT operand's nonzeros, so `user_profiles @ one_item_column`
    costs a full pass over all ~463M nonzeros of `user_profiles` no matter how few terms the item
    has -- almost every lookup lands on an empty row. Doing that once per item made Filtering
    O(n_cold * nnz(user_profiles)) ~ 1.3e12 operations single-threaded. Blocking amortises each
    pass across `block_size` items and cuts the number of passes by that factor, for identical
    output.

    `block_size` trades memory for passes: the dense score block is n_users x block_size (~400 MB
    at 128 users-columns of float64 for this dataset), so raise it if there is headroom.
    """
    cold = np.asarray(cold_item_ids, dtype=np.int64)
    candidates = {}
    for start in range(0, len(cold), block_size):
        blk = cold[start:start + block_size]
        # (n_users x n_terms) @ (n_terms x block) -> one pass over user_profiles for the whole block
        scores = (user_profiles @ item_content[blk].T).toarray()
        for col, item_idx in enumerate(blk):
            row = scores[:, col]
            k = min(top_k, row.size)
            top = np.argpartition(-row, k - 1)[:k]
            candidates[int(item_idx)] = top[np.argsort(-row[top])]
        if verbose:
            print(f"\r[filtering] {min(start + block_size, len(cold)):,}/{len(cold):,} items",
                  end="", flush=True)
    if verbose:
        print(f"\r[filtering] {len(cold):,}/{len(cold):,} items", flush=True)
    return candidates


def _build_prompt(history_texts: list, item_text: str, max_history: int = 10, reasoning: bool = False) -> str:
    """Simple, fixed prompt template. `reasoning=False` (Intervention B1) leaves the prompt as a
    bare yes/no question -- paired with yes_logodds()'s choice-constrained decoding, the
    model jumps straight to the answer token. `reasoning=True` (Intervention B2) appends an
    instruction to justify the answer first, on the hypothesis that a brief chain-of-thought
    surfaces content-relevance signal (genre/author/theme overlap with the user's history) that a
    forced-immediate answer skips past -- paired with yes_logodds()'s regex-constrained
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


def answer_from_text(text: str) -> str:
    """The yes/no a reasoning completion ended on, for DIAGNOSTICS only.

    The score that drives selection comes from `yes_logodds`, not from parsing text. This remains
    because reading a handful of actual completions is the only way to tell whether B2's
    justifications are reasoning or decoration, and that is worth being able to check."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].lower() if matches else ""


# =====================================================================================
# The work queue -- how Refining Simulation survives being interrupted, and shares GPUs
# =====================================================================================
# Refining is ~136,000 LLM calls per prompting strategy and runs for hours, on a box where the
# second GPU is not always available. Three requirements follow, and one design satisfies all of
# them: split the cold items into CHUNKS and let each worker CLAIM one at a time.
#
#   1. Resumable  -- a chunk's result is written to disk the moment it is done, so an interrupted
#                    run resumes at chunk granularity instead of starting over.
#   2. Elastic    -- workers discover work rather than being assigned it, so ONE worker does the
#                    whole job, a second joining halfway just starts taking chunks, and either can
#                    be killed at any moment.
#   3. No broker  -- claiming is an atomic O_CREAT|O_EXCL file create. The filesystem is the lock;
#                    there is no coordinator process to run, crash, or wait for.
#
# A worker that dies mid-chunk leaves a claim with no result. Claims carry a heartbeat (the file's
# mtime, refreshed while working), so any other worker may steal a claim whose heartbeat has gone
# stale -- that is what stops a killed worker from stranding its chunk forever.
#
# `run_key` covers everything that changes the SCORES -- fingerprint, model, top_k, strategy,
# prompt construction -- so changing any of them starts a clean run rather than blending results
# computed under different rules. It deliberately does NOT cover the selection rule (N, calibrated
# or raw): those consume the scores offline, and folding them into the key would throw away hours
# of GPU work every time a sensitivity curve was plotted.

_CLAIM_STALE_SECONDS = 1800      # 30 min with no heartbeat -> the claim may be stolen
_HEARTBEAT_SECONDS = 60


def run_key(dataset_fp, model: str, top_k: int, strategy: str, prompt_tag: str = "",
            pass_name: str = "pairs") -> str:
    """Short stable id for one scoring configuration.

    `pass_name` separates the main pair-scoring pass from the per-user calibration pass, which
    scores different prompts and must never share a work directory with it."""
    import hashlib
    payload = repr((tuple(dataset_fp), model, int(top_k), strategy, prompt_tag, pass_name))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def plan_work(candidates: dict, work_dir: str, chunk_size: int = 64, meta: dict = None) -> dict:
    """Split `candidates` into chunks and write the manifest. Idempotent: an existing manifest for
    this directory is returned untouched, so re-running the planning cell never renumbers chunks
    out from under a worker that is mid-flight."""
    os.makedirs(os.path.join(work_dir, "claims"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "results"), exist_ok=True)
    manifest_path = os.path.join(work_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)

    item_ids = sorted(int(i) for i in candidates)
    chunks = [item_ids[s:s + chunk_size] for s in range(0, len(item_ids), chunk_size)]
    manifest = {"chunks": {f"{i:05d}": c for i, c in enumerate(chunks)},
                "n_items": len(item_ids), "chunk_size": chunk_size,
                "n_pairs": int(sum(len(candidates[i]) for i in item_ids)),
                "meta": meta or {}}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest


def _claim_path(work_dir, cid):
    return os.path.join(work_dir, "claims", f"chunk_{cid}.json")


def _result_path(work_dir, cid):
    return os.path.join(work_dir, "results", f"chunk_{cid}.npz")


def _write_result(work_dir, cid, rows, cols, scores, n_prompts):
    """Publish a chunk's result atomically: write a temp file, then rename it into place.

    STORES SCORES, NOT DECISIONS -- every pair scored, with its log-odds, rather than only the
    pairs that survived some rule. Generating these scores is the expensive, irreversible part of
    the whole intervention (hours of GPU); turning them into a synthetic matrix is microseconds.
    Persisting the decision instead would mean that every later question -- a different N, a
    calibrated ranking, a sensitivity curve, an ablation nobody thought of yet -- costs another
    full run. Persisting the scores makes all of those offline array operations.

    The rename is what makes it atomic. A half-written .npz left behind by a kill would otherwise
    look like a finished chunk to `claim_chunk`/`assemble_scores` and be skipped forever --
    silently dropping those pairs.

    numpy is handed an open FILE OBJECT rather than a path on purpose: `np.savez(path, ...)`
    appends ".npz" unless the name already ends in it, so a temp path like `chunk_0.npz.tmp1234`
    is quietly written as `chunk_0.npz.tmp1234.npz` and the rename then fails on a missing file.
    Passing a handle takes that renaming behaviour out of the picture entirely.
    """
    final = _result_path(work_dir, cid)
    tmp = f"{final}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, rows=np.asarray(rows, dtype=np.int64),
                 cols=np.asarray(cols, dtype=np.int64),
                 scores=np.asarray(scores, dtype=np.float32), n_prompts=np.int64(n_prompts))
    os.replace(tmp, final)
    return final


def claim_chunk(work_dir: str, manifest: dict, worker_id: str,
                stale_seconds: int = _CLAIM_STALE_SECONDS):
    """Atomically take the next available chunk id, or None when everything is done or in flight.

    `os.open(..., O_CREAT | O_EXCL)` either creates the claim or raises FileExistsError -- the
    whole mutual exclusion, with no lock server and no race window. A claim whose heartbeat has
    gone stale and that produced no result is deleted and retried, which is how a killed worker's
    chunk gets picked up by somebody else."""
    for cid in manifest["chunks"]:
        if os.path.exists(_result_path(work_dir, cid)):
            continue
        claim = _claim_path(work_dir, cid)
        if os.path.exists(claim):
            age = time.time() - os.path.getmtime(claim)
            if age < stale_seconds:
                continue
            try:                              # stale: reclaim it
                os.remove(claim)
            except FileNotFoundError:
                continue                      # somebody else got there first
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue                          # lost the race; try the next chunk
        with os.fdopen(fd, "w") as f:
            json.dump({"worker": worker_id, "started": time.time()}, f)
        return cid
    return None


def progress(work_dir: str, manifest: dict) -> dict:
    """Counts for a status cell: done / in flight / remaining."""
    done, claimed = 0, 0
    for cid in manifest["chunks"]:
        if os.path.exists(_result_path(work_dir, cid)):
            done += 1
        elif os.path.exists(_claim_path(work_dir, cid)):
            claimed += 1
    total = len(manifest["chunks"])
    return {"done": done, "in_flight": claimed, "todo": total - done - claimed, "total": total}


def run_worker(work_dir: str, manifest: dict, candidates: dict, item_metadata: list,
               train_matrix, model: str, *, reasoning: bool = False, unit: str = "item",
               batch_size: int = 256, max_history: int = 10, worker_id: str = None,
               engine_kwargs: dict = None, verbose: bool = True, max_chunks: int = None) -> list:
    """Claim and process chunks until none remain.

    Returns one record per chunk completed BY THIS WORKER: `{"chunk", "n_prompts", "n_kept",
    "seconds"}`. Per-chunk timings rather than a total, because a total is dominated by engine
    build and first-call kernel compilation -- fine for a long run, wildly misleading for a short
    one, and it is the steady-state rate that a remaining-time estimate needs.

    ONE vLLM engine is built for the whole call and reused across every chunk -- engine startup is
    a minute or two, so building one per chunk would dominate the run. That is also why a worker is
    a long-lived process rather than a per-chunk invocation.

    `max_chunks` stops after that many chunks are completed HERE, leaving the rest claimable. It is
    what lets an expensive pass be sampled before it is committed to -- run a few chunks, look at
    what they say, then decide whether to finish. Unclaimed chunks are untouched, so "finishing"
    is just calling this again without the limit.

    The engine is released before returning, so the next pass (or the evaluation half of the
    notebook) gets the card back rather than waiting on garbage collection.

    Safe to run in two notebooks at once (one per GPU), to start the second one hours late, or to
    interrupt either: the claim protocol makes all three equivalent to "some worker will get to it".
    """
    worker_id = worker_id or f"pid{os.getpid()}-{int(time.time())}"
    if progress(work_dir, manifest)["todo"] == 0 and progress(work_dir, manifest)["in_flight"] == 0:
        if verbose:
            print(f"[{worker_id}] nothing to do -- all {len(manifest['chunks'])} chunks complete")
        return []

    simulator = VLLMColdLLMSimulator(model=model, **(engine_kwargs or {}))
    train_csr = sparse.csr_matrix(train_matrix)
    completed = []
    consecutive_failures, max_consecutive_failures = 0, 3
    while True:
        cid = claim_chunk(work_dir, manifest, worker_id)
        if cid is None:
            break
        items = manifest["chunks"][cid]
        t0 = time.perf_counter()
        # Build pairs USER-MAJOR, not item-major. The history block is ~10/11 of a prompt's tokens
        # and depends only on the user, so consecutive prompts for the same user share a long
        # identical prefix -- which vLLM's automatic prefix caching then serves from the KV cache
        # instead of re-prefilling. A user recurs across many cold items' candidate lists, so
        # item-major ordering scatters those repeats and throws the saving away. Order does not
        # otherwise matter: every pair carries its own (user, item) and the kept ones go into a
        # sparse matrix, which is order-independent.
        # `unit` decides what a chunk element IS. For the main pass it is a cold ITEM and
        # candidates[item] are the users to judge it against. For the calibration pass it is a
        # USER and candidates[user] are the shared probe items -- the same machinery, claims and
        # resumability, scoring E_i[score(u, i)] instead of score(u, i).
        if unit == "user":
            pairs = [(int(u), int(i)) for u in items for i in candidates[u]]
        else:
            pairs = [(int(u), int(i)) for i in items for u in candidates[i]]
        pairs.sort()
        prompts, pair_users, pair_items = [], [], []
        for user_idx, item_idx in pairs:
            lo, hi = train_csr.indptr[user_idx], train_csr.indptr[user_idx + 1]
            history = [item_metadata[j] for j in train_csr.indices[lo:hi]]
            prompts.append(_build_prompt(history, item_metadata[item_idx],
                                         max_history=max_history, reasoning=reasoning))
            pair_users.append(user_idx)
            pair_items.append(item_idx)

        # Refresh the claim while working. Without a beat from inside the batch loop a chunk that
        # runs longer than the staleness window looks abandoned, and another worker redoes it.
        def _beat(done, total, _cid=cid, _t0=t0):
            os.utime(_claim_path(work_dir, _cid), None)
            if verbose and total:
                el = time.perf_counter() - _t0
                print(f"[{worker_id}] chunk {_cid}: {done:,}/{total:,} prompts "
                      f"({el:.0f}s, {done / max(el, 1e-9):.1f}/s)", flush=True)

        # One bad chunk must not end a multi-hour run. The claim is deliberately LEFT IN PLACE on
        # failure: this worker moves on, and the chunk becomes retryable by anyone once its
        # heartbeat goes stale -- which avoids both losing it and spinning on it immediately. No
        # partial result is written, because a chunk missing some of its pairs would look complete
        # to `assemble_scores` and silently drop them.
        try:
            scores = (simulator.yes_logodds(prompts, batch_size=batch_size, reasoning=reasoning,
                                            progress_cb=_beat)
                      if prompts else np.zeros(0, dtype=np.float32))
        except Exception as exc:
            consecutive_failures += 1
            print(f"[{worker_id}] chunk {cid} FAILED ({type(exc).__name__}: {exc}); claim left for "
                  f"retry, moving on ({consecutive_failures} consecutive)", flush=True)
            if consecutive_failures >= max_consecutive_failures:
                print(f"[{worker_id}] {consecutive_failures} chunks failed in a row -- stopping "
                      f"rather than burning the queue", flush=True)
                break
            continue
        consecutive_failures = 0
        # EVERY scored pair is written, not just the winners -- selection happens offline.
        _write_result(work_dir, cid, pair_users, pair_items, scores, len(prompts))
        os.utime(_claim_path(work_dir, cid), None)
        seconds = time.perf_counter() - t0
        n_bad = int(np.isnan(scores).sum())
        completed.append({"chunk": cid, "n_prompts": len(prompts), "n_unscored": n_bad,
                          "seconds": seconds})
        if verbose:
            p = progress(work_dir, manifest)
            eta = p["todo"] * seconds / 3600.0        # this worker alone; a second one halves it
            finite = scores[np.isfinite(scores)]
            spread = (f"log-odds p10 {np.percentile(finite, 10):+.2f} "
                      f"p50 {np.percentile(finite, 50):+.2f} "
                      f"p90 {np.percentile(finite, 90):+.2f}" if finite.size else "no finite scores")
            print(f"[{worker_id}] chunk {cid}: {len(prompts):,} prompts "
                  f"({seconds:.0f}s, {len(prompts) / max(seconds, 1e-9):.1f}/s)   {spread}"
                  + (f"   {n_bad} UNSCORED" if n_bad else "") +
                  f"   {p['done']}/{p['total']} done, {p['in_flight']} in flight, "
                  f"~{eta:.1f} h left at this rate", flush=True)
        if max_chunks is not None and len(completed) >= max_chunks:
            if verbose:
                left = progress(work_dir, manifest)
                print(f"[{worker_id}] stopping at max_chunks={max_chunks}; {left['todo']} chunk(s) "
                      f"still unclaimed and resumable", flush=True)
            break

    # Hand the card back explicitly. vLLM holds ~90% of it, and the next pass builds its own engine
    # -- waiting on garbage collection to notice would turn that into an OOM.
    del simulator
    try:
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    if verbose:
        n = sum(c["n_prompts"] for c in completed)
        print(f"[{worker_id}] finished -- {len(completed)} chunk(s), {n:,} prompts by this worker",
              flush=True)
    return completed


def assemble_scores(work_dir: str, manifest: dict, strict: bool = True):
    """Concatenate every chunk result into flat (users, items, scores) arrays.

    Returns the SCORED PAIRS, not a matrix -- the matrix depends on a selection rule (how many per
    item, calibrated or raw), and those are cheap decisions that should not require re-running the
    LLM. See `select_top_n`.

    `strict` refuses to assemble a partial run -- silently returning one that is missing chunks
    would understate the intervention and read as a weak result rather than an incomplete one."""
    missing = [c for c in manifest["chunks"] if not os.path.exists(_result_path(work_dir, c))]
    if missing and strict:
        raise RuntimeError(f"{len(missing)} of {len(manifest['chunks'])} chunks have no result "
                           f"(e.g. {missing[:5]}). Run a worker until progress() reports todo=0, "
                           f"or pass strict=False to assemble what exists.")
    rows, cols, scores, n_prompts = [], [], [], 0
    for cid in manifest["chunks"]:
        path = _result_path(work_dir, cid)
        if not os.path.exists(path):
            continue
        blob = np.load(path)
        rows.append(blob["rows"])
        cols.append(blob["cols"])
        scores.append(blob["scores"])
        n_prompts += int(blob["n_prompts"])
    u = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    i = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    s = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    finite = s[np.isfinite(s)]
    stats = {"n_prompts": n_prompts, "n_scored": int(s.size), "missing_chunks": len(missing),
             "n_unscored": int((~np.isfinite(s)).sum()),
             "logodds_p10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
             "logodds_p50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
             "logodds_p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
             "frac_yes": float((finite > 0).mean()) if finite.size else float("nan")}
    return u, i, s, stats


def assemble_user_priors(work_dir: str, manifest: dict, strict: bool = True):
    """Mean log-odds per user over the shared probe items -> {user_idx: prior}.

    This is the marginal that `select_top_n(priors=...)` divides out. Averaging over several real
    items (see `probe_items`) rather than one placeholder makes it an honest Monte Carlo estimate
    of E_i[score(u, i)] taken IN DISTRIBUTION -- a content-free probe would measure the model's
    reaction to a degenerate prompt instead of the user's baseline."""
    u, i, s, stats = assemble_scores(work_dir, manifest, strict=strict)
    priors, order = {}, np.argsort(u, kind="stable")
    u, s = u[order], s[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    within, n_used = [], []
    for chunk_u, chunk_s in zip(np.split(u, bounds), np.split(s, bounds)):
        finite = chunk_s[np.isfinite(chunk_s)]
        if chunk_u.size and finite.size:
            priors[int(chunk_u[0])] = float(finite.mean())
            if finite.size > 1:
                within.append(float(finite.var(ddof=1)))
                n_used.append(finite.size)

    # The components that decide whether subtracting this prior helps at all. var(prior_hat)
    # observed across users is inflated by the sampling error of each mean, so the between-user
    # variance has to be recovered as var(prior_hat) - var_within/n rather than read off directly.
    means = np.array(list(priors.values())) if priors else np.zeros(0)
    var_within = float(np.mean(within)) if within else float("nan")
    n_eff = float(np.mean(n_used)) if n_used else float("nan")
    var_of_means = float(means.var(ddof=1)) if means.size > 1 else float("nan")
    var_between = var_of_means - var_within / n_eff if np.isfinite(var_within) else float("nan")
    stats.update({
        "n_users_with_prior": len(priors),
        "var_within_user": var_within,          # noise in one probe
        "var_of_prior_means": var_of_means,     # what you naively see across users
        "var_between_users": var_between,       # signal the correction is meant to remove
        "n_probe_effective": n_eff,
        # n_probe needed for the removed bias to exceed the injected noise.
        "n_probe_required": (var_within / var_between
                             if np.isfinite(var_between) and var_between > 0 else float("inf")),
    })
    return priors, stats


def select_top_n(users, items, scores, n_per_item, n_users, n_items, priors=None):
    """Keep the `n_per_item` highest-scoring users for each item -> binary csr (n_users x n_items).

    Ranking, not thresholding, because the ordering is the trustworthy part of an uncalibrated
    judge: any monotone recalibration -- a different prompt, a more agreeable model, a temperature
    change -- leaves a top-N selection unchanged while moving a probability cutoff arbitrarily.

    Fixing the COUNT also keeps the arms comparable. If one prompting strategy accepted 49 users
    per item and another 25, a difference between their curves would confound how many synthetic
    interactions each injected with how good they were; ALS's fold-in is directly sensitive to the
    number of rows behind a cold item. Holding N equal asks the sharper question.

    `priors` maps user -> baseline log-odds; when given, each score has its user's own baseline
    subtracted. This is the one calibration that can change a per-item ranking: a GLOBAL or
    PER-ITEM correction is constant within one item's 50 candidates and cannot reorder them, but
    users vary within that ranking -- a user with a long, generic history is agreeable about
    everything -- and subtracting their own mean converts "which users are agreeable" into "which
    users are agreeable ABOUT THIS ITEM". Structurally this is PMI: cancel a marginal to recover an
    association, in logit space rather than log space because these probabilities saturate near 1.

    Ties break on the pair's own (score, user) so a rerun selects the same set.
    """
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    if priors:
        base = np.array([priors.get(int(u), 0.0) for u in users], dtype=np.float64)
        adjusted -= base
    # NaN (unscorable) must lose to every real score rather than sorting arbitrarily.
    adjusted = np.where(np.isfinite(adjusted), adjusted, -np.inf)

    # Sort by item, then descending score, then user id for a deterministic tie-break.
    order = np.lexsort((users, -adjusted, items))
    items_s, adj_s, users_s = items[order], adjusted[order], users[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(items_s)) + 1))
    rank = np.arange(len(items_s)) - np.repeat(starts, np.diff(np.append(starts, len(items_s))))
    keep = (rank < n_per_item) & np.isfinite(adj_s)
    r, c = users_s[keep], items_s[keep]
    return sparse.csr_matrix((np.ones(len(r), dtype=np.float64), (r, c)), shape=(n_users, n_items))


def select_top_n_floored(users, items, scores, n_per_item, n_users, n_items, priors=None,
                         floor: float = 0.0):
    """top-N, but only over pairs the model actually endorsed (score > `floor`).

    The abstention variant. `select_top_n` ignores the SIGN of the score, so an item whose whole
    candidate pool was rejected still receives its N least-rejected users -- and that is not
    harmless: with no synthetic data, fold-in returns the zero vector at k=0 and CBHCF ranks the
    item on content alone, whereas N wrong users make the collaborative half contribute noise and
    can push it below that. Here such an item receives nothing instead, which is exactly what the
    unaugmented baseline does for it.

    This is also the more faithful reading of ColdLLM: the paper's Refining stage is a FILTER whose
    positive predictions become interactions, not a fixed quota per item. The cost is that items no
    longer receive equal counts, so an arm built this way must be compared against a control
    matched to ITS per-item counts (`random_n_matrix` accepts a dict for exactly this), never
    against a constant-N one.
    """
    keep = np.isfinite(np.asarray(scores, dtype=np.float64))
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    if priors:
        adjusted -= np.array([priors.get(int(u), 0.0) for u in np.asarray(users)])
    keep &= adjusted > floor
    return select_top_n(np.asarray(users)[keep], np.asarray(items)[keep], adjusted[keep],
                        n_per_item, n_users, n_items)


def per_item_counts(matrix) -> dict:
    """{item_idx: how many interactions it received} -- to match a control to a variable-N arm."""
    csc = sparse.csc_matrix(matrix)
    counts = np.diff(csc.indptr)
    return {int(i): int(counts[i]) for i in np.flatnonzero(counts)}


def random_n_matrix(candidates: dict, n_per_item, n_users: int, n_items: int, seed: int = 0):
    """CONTROL ARM: n users drawn uniformly from each item's Filtering candidates -- no LLM.

    `n_per_item` may be an int (constant N) or a {item: count} mapping. The mapping form exists so
    a variable-N arm -- `select_top_n_floored`, where items receive different numbers -- can still
    be compared against a control that injects exactly the same volume item by item. Without that,
    a curve gap would confound which users were chosen with how many.

    This is the control that decides whether Intervention B did anything. Top-N always produces
    exactly n interactions per item, so the synthetic matrix looks healthy even if the LLM's
    ordering is pure noise -- the very symptom a 98%-yes rate advertised under a hard yes/no rule
    is now hidden by construction. Random-N puts it back in view: it has everything the LLM arm
    has (the same candidate pool, the same count, the same downstream fit) except the ordering. If
    the LLM arm does not beat it, content similarity did the work and the LLM is decorative.

    Draw a DIFFERENT sample per seed (pass the model seed) so the band across seeds reflects the
    draw as well as ALS initialisation, rather than reporting one lucky draw's spread as if it
    were seed noise.
    """
    rng = np.random.default_rng(seed)
    per_item = (dict(n_per_item) if isinstance(n_per_item, dict)
                else {int(i): int(n_per_item) for i in candidates})
    rows, cols = [], []
    for item_idx in sorted(candidates):          # sorted so a seed reproduces exactly
        user_ids = np.asarray(candidates[item_idx])
        k = min(per_item.get(int(item_idx), 0), len(user_ids))
        if k:
            rows.append(rng.choice(user_ids, size=k, replace=False))
            cols.append(np.full(k, int(item_idx), dtype=np.int64))
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n_users, n_items))


def popularity_n_matrix(candidates: dict, n_per_item: int, train_matrix, n_users: int, n_items: int):
    """CONTROL ARM: the n most ACTIVE users among each item's candidates -- no LLM.

    Catches the specific failure where the LLM is a popularity proxy in disguise. A user with a
    long history gives the model more text to agree with, so `select_top_n` could be recovering
    "this user reads a lot" rather than "this user would read THIS". If the LLM arm merely matches
    this, that is what it is doing. Deterministic, so it needs no per-seed variation.
    """
    degree = np.diff(sparse.csr_matrix(train_matrix).indptr)
    rows, cols = [], []
    for item_idx, user_ids in candidates.items():
        user_ids = np.asarray(user_ids)
        k = min(n_per_item, len(user_ids))
        if k:
            top = user_ids[np.lexsort((user_ids, -degree[user_ids]))[:k]]
            rows.append(top)
            cols.append(np.full(k, int(item_idx), dtype=np.int64))
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n_users, n_items))


def probe_items(item_metadata, warm_item_ids, n_probe: int = DEFAULT_N_PROBE, seed: int = 0):
    """A fixed random sample of REAL items, used to measure every user's baseline agreeableness.

    The SAME probe set is used for every user on purpose. The prior only has to be accurate up to a
    constant, because what it must do is reorder users WITHIN one item's candidate list -- so a
    paired design, where every user is measured against identical items, removes probe-set variance
    from exactly the comparison that matters. An independent draw per user would inject noise into
    the quantity being corrected.

    Drawn from warm items because they are guaranteed to have real metadata; the train/test split
    is random over items, so warm and cold text are drawn from the same distribution.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray([i for i in np.asarray(warm_item_ids)
                       if item_metadata[int(i)] and item_metadata[int(i)].strip()])
    return np.sort(rng.choice(pool, size=min(n_probe, len(pool)), replace=False))


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

    def __init__(self, dataset, synthetic_matrix, restrict_to_item_ids=None, tag: str = ""):
        self._dataset = dataset
        self._synthetic = sparse.csr_matrix(synthetic_matrix)
        # Makes this wrapper fingerprint DIFFERENTLY from the dataset it wraps. Every field
        # `load.dataset_fingerprint` reads is delegated by __getattr__ below, so without this an
        # augmented dataset is indistinguishable from the plain one -- and `cf.ALSModel.fold_in`
        # memoizes cold-item factors keyed on that fingerprint. A model folded against both would
        # silently reuse the first one's factors for the second. `load.dataset_fingerprint` appends
        # this only when present, so plain Datasets keep the exact tuple every cache key and every
        # artifact in outputs/hyperparams.json already stores.
        self._fingerprint_salt = ("coldllm", tag, int(self._synthetic.nnz))
        if restrict_to_item_ids is not None:
            test_coo = dataset.test_matrix.tocoo()
            keep = np.isin(test_coo.col, restrict_to_item_ids)
            self.test_matrix = sparse.csr_matrix(
                (test_coo.data[keep], (test_coo.row[keep], test_coo.col[keep])),
                shape=dataset.test_matrix.shape,
            )

    def __getattr__(self, name):
        # Guard the delegation target itself. __getattr__ fires for ANY attribute missing from the
        # instance, so before __init__ has run (or during unpickling) `self._dataset` is itself
        # missing -- and looking it up here would re-enter __getattr__ forever. Raising
        # AttributeError is what the protocol expects for a genuinely absent attribute.
        if name.startswith("_") and name in ("_dataset", "_synthetic", "_fingerprint_salt"):
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def revealed_item_users_at_k(self, k):
        item_ids, item_users = self._dataset.revealed_item_users_at_k(k)
        synthetic_item_users = self._synthetic[:, item_ids].T.tocsr()
        combined = (item_users + synthetic_item_users).tocsr()
        combined.data = np.minimum(combined.data, 1.0)  # avoid double-weighting a rare overlap
        return item_ids, combined
