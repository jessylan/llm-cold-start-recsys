"""The one definition of `outputs/hyperparams.json["steel_thread_config"]` -- both directions.

That key is what `steel_thread.ipynb` and `intervention_b_coldllm.ipynb` actually run on: the
CBHCF field weights and lambda, and Intervention A's encoder, weights and lambda, all selected on
`dataset.cold_val` by `notebooks/intervention_a_weight_sweep.ipynb`. It used to be **transcribed by
hand** from that notebook's printed table, which meant a re-tune silently changed nothing -- the
sweep rewrote its own section, the steel thread kept reading the stale config, and the split
fingerprint could not object because re-tuning does not change the split.

`build()` closes that gap. `load()` is here for the matching reason: the reader carries two guards
(refuse a config tuned on a different split; refuse a config older than the sweep it derives from)
and it was previously copy-pasted into every notebook that needed it, so the guards could -- and
did -- exist in some copies and not others. One module, one contract, two consumers.

    weight sweep -> hyperparams.json["intervention_a_weight_sweep"]["verified"]
                 -> steel_config.build()
                 -> hyperparams.json["steel_thread_config"]   <- what the notebooks read via load()

THE MAPPING. The sweep verifies `N_VERIFY` grid points per arm on the full objective and records
each as `{arm, p, r, lambda, objective, warm}`. The winner of each arm becomes one half of the
config:

  - `B_tfidf`  -> `cbhcf`.          Arm B splits its prose weight across two TF-IDF blocks as
                                    `p/sqrt(2)` each, so the block carries `w^2 = p^2` -- the same
                                    share of the item norm arm A's single dense block carries at
                                    weight `p`. So `title = blurb = p/sqrt(2)`, `reviews = r`, and
                                    `creator`/`taxonomy` are the fixed 1.0/0.5 both arms hold
                                    constant.
  - `A_arctic` -> `intervention_a`. `text_weight = p`, `description_weight = r`. The sweep pools
                                    the description groups into one renormalized block, so
                                    `description_mode` is always "pooled" here.

Verified against the committed artifacts: `python -m recsys.steel_config --check` re-derives the
stored config from the stored `verified` block and diffs the two.
"""
import json
import math
import os
import time

from recsys import intervention_a as ia

#: Held fixed by BOTH arms of the sweep (see its `space_arm_a` / `space_arm_b`): the entity blocks
#: are not part of the search, so they are not read out of `verified` -- they are the constants the
#: search was conditioned on. If the sweep ever varies them, this must read them instead.
FIXED_ENTITY_WEIGHTS = {"creator": 1.0, "taxonomy": 0.5}
ARM_TO_SECTION = {"B_tfidf": "cbhcf", "A_arctic": "intervention_a"}
DEFAULT_PATH = "outputs/hyperparams.json"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def best_per_arm(verified):
    """The highest-objective verified point for each arm. Raises rather than guessing if an arm is
    missing -- a half-written config is the failure mode this module exists to prevent."""
    best = {}
    for point in verified.values():
        arm = point["arm"]
        if arm not in ARM_TO_SECTION:
            raise ValueError(f"unknown arm {arm!r} in verified points; expected {list(ARM_TO_SECTION)}")
        if arm not in best or point["objective"] > best[arm]["objective"]:
            best[arm] = point
    missing = set(ARM_TO_SECTION) - set(best)
    if missing:
        raise ValueError(f"no verified points for {sorted(missing)} -- the sweep's verification "
                         "section did not finish; refusing to write a half-tuned config")
    return best


def build(hp):
    """Assemble the `steel_thread_config` value from a loaded `hyperparams.json`. Pure -- does no
    I/O, so `--check` can compare it against what is already stored."""
    sweep = hp.get("intervention_a_weight_sweep")
    if sweep is None:
        raise KeyError("hyperparams.json has no `intervention_a_weight_sweep` -- run "
                       "notebooks/intervention_a_weight_sweep.ipynb first")
    best = best_per_arm(sweep["verified"])
    b, a = best["B_tfidf"], best["A_arctic"]
    model_key = sweep["model"]
    spec = ia.ENCODERS[model_key]

    per_block = b["p"] / math.sqrt(2.0)          # see THE MAPPING in the module docstring
    field_weights = {"title": per_block, "creator": FIXED_ENTITY_WEIGHTS["creator"],
                     "taxonomy": FIXED_ENTITY_WEIGHTS["taxonomy"], "blurb": per_block,
                     "reviews": b["r"]}

    return {
        "selected_on": (
            f"cold_val, {sweep['seeds']} seeds, mean NDCG@100 over k={sweep['k_full']} + "
            "within-item ceiling, subject to the warm-NDCG guard; field weights and lambda searched "
            "on the SAME grid for both arms so neither has a tuning advantage "
            "(notebooks/intervention_a_weight_sweep.ipynb, assembled by recsys.steel_config)"),
        "cbhcf": {
            "field_weights": field_weights,
            "content_weight": float(b["lambda"]),
            "coldval_objective": float(b["objective"]),
            "coldval_warm_ndcg": float(b["warm"]),
            "note": (f"arm B_tfidf at prose weight p={b['p']:.4f} (title=blurb=p/sqrt(2)="
                     f"{per_block:.4f}) and long-text weight r={b['r']:g}; creator/taxonomy held "
                     "at 1.0/0.5 by both arms of the sweep."),
        },
        "intervention_a": {
            "model_key": model_key,
            "model_repo": spec.repo,
            "text_weight": float(a["p"]),
            "description_mode": "pooled",
            "description_weight": float(a["r"]),
            "content_weight": float(a["lambda"]),
            "coldval_objective": float(a["objective"]),
            "coldval_warm_ndcg": float(a["warm"]),
            "max_seq_length": ia.MAX_SEQ_LEN,
            "encode_dtype": ia.TORCH_DTYPE,
        },
        # The pre-sweep numbers the sweep itself printed as its baseline, so the config records what
        # the tuning bought. CAVEAT: these come from `hp["cbhcf"]` and `hp["intervention_a"]`, written
        # by earlier notebooks -- if those were not re-run after a change that moves the content
        # space, this mixes vintages and is not a clean delta. Absent if they never ran.
        "untuned_reference": {
            "cbhcf_objective": hp.get("cbhcf", {}).get("coldval_at_selected_lambda", {}).get("objective"),
            "intervention_a_objective": (hp.get("intervention_a", {}).get("per_model", {})
                                         .get(model_key, {}).get("best_objective")),
        },
        "dataset_fingerprint": list(sweep["dataset_fingerprint"]),
    }


def write(path=DEFAULT_PATH, backup=True):
    """Derive and store `steel_thread_config`, stamping it with the time it was derived (which is
    what `load`'s staleness guard compares against). Touches no other key."""
    with open(path) as f:
        hp = json.load(f)
    built = build(hp)
    if backup:
        with open(path + ".bak", "w") as f:
            json.dump(hp, f, indent=2)
    hp["steel_thread_config"] = {**built, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(path, "w") as f:
        json.dump(hp, f, indent=2)
    return hp["steel_thread_config"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load(path, fingerprint, verbose=True):
    """Return `steel_thread_config`, or None if there is nothing to read (caller falls back to
    untuned defaults). Raises rather than returning a config that would silently misreport:

      - tuned on a DIFFERENT SPLIT -- `dataset_fingerprint` mismatch;
      - STALE -- derived before the weight sweep that should have produced it. Re-tuning does not
        change the split, so the fingerprint check above cannot see this; it is the failure this
        module was written to close, and it is only detectable because both sections carry an ISO
        timestamp (sortable as a plain string).
    """
    if not os.path.exists(path):
        if verbose:
            print(f"No {path}; falling back to UNTUNED defaults -- run "
                  f"intervention_a_weight_sweep.ipynb for selected values.")
        return None
    with open(path) as f:
        hp = json.load(f)
    rec = hp.get("steel_thread_config")
    if rec is None:
        if verbose:
            print(f"{path} has no steel_thread_config; run intervention_a_weight_sweep.ipynb "
                  f"(its last cell derives it), or `python -m recsys.steel_config`.")
        return None

    fingerprint = tuple(fingerprint)
    if tuple(rec["dataset_fingerprint"]) != fingerprint:
        raise RuntimeError(
            f"{path} steel_thread_config was tuned on a different split -- refusing to use it. "
            f"artifact fingerprint {tuple(rec['dataset_fingerprint'])}, current {fingerprint}. "
            f"Re-run intervention_a_weight_sweep.ipynb, or delete the section to fall back "
            f"to untuned defaults.")

    sweep_ts = hp.get("intervention_a_weight_sweep", {}).get("timestamp")
    cfg_ts = rec.get("timestamp")
    if sweep_ts and (cfg_ts is None or cfg_ts < sweep_ts):
        when = cfg_ts or "UNDATED (hand-written, before this was automated)"
        raise RuntimeError(
            f"{path}: steel_thread_config is older than the intervention_a_weight_sweep section it "
            f"is derived from (config {when}, sweep {sweep_ts}). Every lambda and field weight "
            f"downstream would be the PREVIOUS tuning's. Run `python -m recsys.steel_config` "
            f"to re-derive it.")
    return rec


# ---------------------------------------------------------------------------
# CLI:  python -m recsys.steel_config [--check] [--hyperparams PATH]
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Derive hyperparams.json['steel_thread_config'].")
    ap.add_argument("--hyperparams", default=DEFAULT_PATH,
                    help=f"path to hyperparams.json (default: {DEFAULT_PATH})")
    ap.add_argument("--check", action="store_true",
                    help="derive and diff against the stored config; write nothing, exit 1 on drift")
    args = ap.parse_args(argv)

    with open(args.hyperparams) as f:
        hp = json.load(f)
    built = build(hp)
    stored = hp.get("steel_thread_config")

    if args.check:
        if stored is None:
            print("CHECK FAIL: no steel_thread_config stored to compare against")
            return 1
        # `selected_on` and `note` are prose regenerated from the sweep's own parameters; the
        # numbers are the contract. Compare those exactly.
        drift = []
        for section in ("cbhcf", "intervention_a", "untuned_reference"):
            for k, v in built[section].items():
                if k == "note":
                    continue
                sv = stored.get(section, {}).get(k)
                if isinstance(v, dict):
                    same = sv == v
                elif isinstance(v, float) and isinstance(sv, (int, float)):
                    same = math.isclose(float(sv), v, rel_tol=1e-12, abs_tol=0.0)
                else:
                    same = sv == v
                if not same:
                    drift.append(f"  {section}.{k}: stored {sv!r}  derived {v!r}")
        if list(stored.get("dataset_fingerprint", [])) != built["dataset_fingerprint"]:
            drift.append(f"  dataset_fingerprint: stored {stored.get('dataset_fingerprint')}  "
                         f"derived {built['dataset_fingerprint']}")
        if drift:
            print("CHECK FAIL -- derived config differs from stored:")
            print("\n".join(drift))
            return 1
        print("CHECK OK -- the derived config reproduces the stored steel_thread_config exactly.")
        return 0

    rec = write(args.hyperparams)
    c, a = rec["cbhcf"], rec["intervention_a"]
    print(f"backed up   {args.hyperparams}.bak")
    print(f"wrote       {args.hyperparams} -> steel_thread_config")
    print(f"  cbhcf           lambda={c['content_weight']:g}  field_weights="
          + ", ".join(f"{k}={v:g}" for k, v in c["field_weights"].items())
          + f"  coldval={c['coldval_objective']:.5f}")
    print(f"  intervention_a  lambda={a['content_weight']:g}  text_weight={a['text_weight']:g}  "
          f"description_weight={a['description_weight']:g}  coldval={a['coldval_objective']:.5f}")
    print(f"  fingerprint     {rec['dataset_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
