#!/usr/bin/env python3
"""Generate `solution_architecture_generated.html` -- the CF retrieval-stage steel thread's
reference architecture diagram, built from structured Python data via diagram_lib rather
than hand-edited HTML.

Run:  python build_solution_architecture.py   ->   solution_architecture_generated.html

To change stage text or tags: edit the Step(...)/Sequence(...)/Fork(...) data below and
re-run. To change stage NUMBERING (insert/remove/reorder a stage): also edit the tree
below -- numbers are assigned automatically (diagram_lib.assign_labels).

Cross-references are name-based: every stage that gets referenced elsewhere has a
key=... , and prose refers to it as {{that_key}} instead of typing its number. One final
resolve_refs() pass (at the bottom of this file) swaps every {{key}} for its real,
current label -- so renumbering never leaves a stale "stage 9" behind again.
"""
from pathlib import Path

from diagram_lib import (
    Step, Sequence, Fork, Metric, Code,
    render_pipeline, render_legend, render_metrics_table,
    render_deferred_card, render_deferred_section,
    render_header, render_footer, render_page, resolve_refs,
)

registry = {}  # key -> assigned label, populated by render_pipeline() below

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

pipeline = Sequence([

    Step(
        key="ingestion",
        title="Raw Data Ingestion",
        tag="generic",
        desc=(
            "Loads the raw interaction records - <code>user_id, item_id, rating, timestamp</code> "
            "from MovieLens's <code>u.data</code>. The system's only external data dependency; "
            "everything downstream assumes this exists. A future intervention (content embeddings, "
            "LLM-simulated interactions) gains a sibling step here, not a replacement, since CF still "
            "needs raw interactions."
        ),
        meta=[Code("u.data"), "100,000 interactions", "943 users", "1,682 items"],
    ),

    Step(
        key="id_norm",
        title="Identifier Normalization",
        tag="generic",
        desc=(
            "Maps raw user/item IDs to contiguous zero-based integers, plus a reverse lookup for "
            "readable output. Required because sparse-matrix math needs dense integer addressing. "
            "This is also the exact join key a content-feature table would align to under a future "
            "content-based intervention - the seam where the two approaches connect."
        ),
        meta=[Code("user_index"), Code("item_index"), Code("index_to_user / index_to_item")],
    ),

    Step(
        key="sparse_matrix",
        title="Sparse Interaction Matrix Construction",
        tag="generic",
        desc=(
            "Assembles (user, item) pairs into a <b>binary</b> CSR sparse user&times;item matrix - "
            "any rating counts as an interaction, no weighting. Matches exactly what {{als}} reads "
            "downstream. Confidence-weighted interactions are a deliberately deferred extension, not "
            "used here."
        ),
        meta=[Code("user_item"), "6.30% density", Metric("100,000 non-zeros")],
    ),

    Sequence(
        key="split_group",
        boxed=True,
        box_label="Step {{split_group}} - split, in two independent parts",
        children=[
            Step(
                key="cold_item_split",
                title="Cold-Item Split",
                tag="generic",
                desc=(
                    "A fixed population of items plays the role of \"cold\" throughout this notebook. "
                    "<b>Eligibility:</b> only items with at least 25 total interactions qualify - this "
                    "guarantees every candidate can supply a full <code>k=0..20</code> reveal sweep "
                    "without running out of history partway through. <b>Selection:</b> 10% of the "
                    "eligible population is sampled <i>uniformly at random</i> (no stratification by "
                    "popularity tier). <b>Reveal and reserve, per cold item:</b> sort its interactions "
                    "chronologically; the first 20 are the revealable pool (exactly <code>k</code> of "
                    "these are treated as known history at reveal level <code>k</code>); everything "
                    "after position 20 is permanently reserved for evaluation, unchanged at every "
                    "<code>k</code>. This keeps the evaluated population perfectly stable across the "
                    "whole sweep - only the amount of revealed history changes. This same reveal "
                    "mechanism is called repeatedly by stages {{warmup_curve_mode_a}} and "
                    "{{warmup_curve_mode_b}}. The plain-random draw turned out to make no promise about "
                    "how homogeneous the resulting population's difficulty would be - reserved-test-user "
                    "counts range from 5 to 487 across these 87 items, which stage "
                    "{{warmup_curve_mode_b}} has to account for directly."
                ),
                meta=[Code("MIN_INTERACTIONS=25"), "872 / 1,682 items eligible", "87 cold items (10%, plain random)",
                      Metric("7,372 reserved test interactions (5-487 per item, mean 84.7)")],
            ),
            Step(
                key="reference_set",
                title="Reference Set",
                tag="generic",
                desc=(
                    "<b>Independent, not a fold-in.</b> A plain <b>random</b> 80/20 split on the warm "
                    "items only (everything {{cold_item_split}} did not select as cold) - no reveal "
                    "levels, no fold-in, no sweep. It's a reference, not a comparison: it exists purely "
                    "to answer \"what does normal performance look like\" for stages "
                    "{{warmup_reference_score}}/{{cold_eval}}/{{warmup_curve_mode_a}} to be read against, "
                    "so plotting a warm-up curve for an already-warm item would answer no question worth "
                    "asking here."
                ),
                meta=[Code("train_test_split()"), "1,595 warm items only", Metric("72,833 train / 18,055 test")],
            ),
        ],
    ),

    Fork(
        key="model_fork",
        fork_note=None,  # auto: "stage 5 forks - order between 5a / 5b is arbitrary"
        merge_note="merges - both expose the same <code>model.recommend()</code> interface",
        branches=[
            Step(
                key="popularity",
                title="Popularity Baseline",
                tag="generic",
                desc=(
                    "Ranks items by raw training-interaction count - the same global list for every "
                    "user, personalized only by excluding each user's already-seen items. No factors, no "
                    "training loop, no hyperparameters, no notion of individual user preference at all - "
                    "exactly why it's the right floor for this comparison. Built identically regardless "
                    "of which algorithm occupies {{als}}; exists to answer \"is {{als}} earning its "
                    "complexity.\" Also the conceptual template for stage {{warmup_curve_mode_b}}'s "
                    "user-activity floor - \"rank by observed volume, no personalization\" applied to the "
                    "opposite axis."
                ),
                meta=[Code("PopularityModel"), "no training", Metric("top item: 482 interactions")],
            ),
            Step(
                key="als",
                title="Collaborative Filtering - ALS",
                tag="algo",
                desc=(
                    "Learns latent user/item factor vectors by alternately solving, in closed form, for "
                    "all user factors with item factors fixed, then all item factors with user factors "
                    "fixed. <b>The reason ALS - not BPR - is the algorithm here:</b> that alternating "
                    "structure gives <code>recalculate_item(itemid, item_users)</code> natively - an "
                    "exact closed-form solve for <i>one item's</i> factor, holding every user's factor "
                    "completely untouched. BPR's joint stochastic optimization has no equivalently clean "
                    "\"hold one side fixed\" operation, so a full retrain would let user factors drift "
                    "along with the item's, confounding the exact thing this experiment isolates. "
                    "<b>The primary swap point</b> in this diagram for future interventions (content "
                    "embeddings, a content-blended model, LLM-simulated interactions)."
                ),
                meta=[Code("AlternatingLeastSquares"), "factors=50", "regularization=0.01", "iterations=15",
                      Code("recalculate_item()")],
            ),
        ],
    ),

    Step(
        key="warmup_reference_score",
        title="Warm-Item Reference Score",
        tag="diag",
        desc=(
            "Computed <b>ahead of</b> stage {{cold_eval}}'s cold-start evaluation, on purpose: reading "
            "stage {{cold_eval}}'s near-zero numbers with no frame of reference tells you little. Both "
            "models are evaluated on {{reference_set}}'s held-out split, using each model's own "
            "natively-fit representations - no fold-in involved, that mechanism exists specifically for "
            "the cold-item sweeps. Fit here as <code>N_SEEDS=10</code> independent ALS fits (factor "
            "initialization is random), reporting mean &plusmn; spread; these exact same 10 fits are "
            "reused, unchanged, by stages {{cold_eval}}, {{warmup_curve_mode_a}}, and "
            "{{warmup_curve_mode_b}}, so every downstream stage rests on the same underlying models "
            "rather than independently-drawn samples. This is also where the shared evaluation helpers "
            "get defined, including <b>AUC</b> - a full-catalog pairwise ranking metric (Mann-Whitney "
            "rank-sum form) that needs no <code>K</code> at all, included specifically because every "
            "other metric here needed a debatable choice of <code>K</code>."
        ),
        meta=[
            Metric("NDCG@100 - Pop 0.2975 / ALS 0.4635 ± 0.0018"),
            Metric("Recall@100 - Pop 0.4503 / ALS 0.6524 ± 0.0021"),
            Metric("HitRate@100 - Pop 0.9576 / ALS 0.9897 ± 0.0013"),
            Metric("AUC (no K) - Pop 0.8717 / ALS 0.8840 ± 0.0010"),
        ],
    ),

    Step(
        key="cold_eval",
        title="Cold Start Retrieval Evaluation",
        tag="diag",
        desc=(
            "Every reserved test interaction (stage {{cold_item_split}}) belongs to a cold item, "
            "evaluated at reveal level <code>k=0</code> - no history revealed at all. For {{als}}, this "
            "has an exact, provable answer, verified computationally rather than merely observed to be "
            "small: with zero observed interactions, <code>recalculate_item</code>'s regularized "
            "closed-form solve collapses to the <i>exact zero vector</i>, every time, for every seed - a "
            "zero vector's dot product with any user factor is exactly zero, so the item cannot outrank "
            "anything on NDCG/Recall/HitRate@100, all exactly 0.0000. Popularity's <code>k=0</code> "
            "failure is a separate, structural guarantee: a cold item's training-interaction count is "
            "exactly zero, so it can never appear in a popularity ordering regardless of training "
            "dynamics. <b>AUC is the exception</b> - a zero-factor item still lands <i>somewhere</i> in "
            "the full-catalog ranking, just slightly below chance (0.4213, not 0.5000), since the median "
            "real item score sits just above zero. Read alongside stage {{warmup_reference_score}}'s "
            "reference score, computed just before this."
        ),
        meta=[
            Metric("NDCG@100 - Pop 0.0000 / ALS 0.0000 ± 0.0000"),
            Metric("Recall@100 - Pop 0.0000 / ALS 0.0000 ± 0.0000"),
            Metric("HitRate@100 - Pop 0.0000 / ALS 0.0000 ± 0.0000"),
            Metric("AUC - Pop 0.0328 / ALS 0.4213 ± 0.0009"),
        ],
    ),

    Step(
        key="warmup_curve_mode_a",
        title="Warm-Up Curve - User-to-Item (Mode A)",
        tag="diag",
        desc=(
            "The standard consumer-facing question: for a given user, does a cold item surface in "
            "<i>their</i> top-K as it accumulates its own history? Built by <b>fold-in, not "
            "retraining.</b> For each <code>k</code> in the <code>0..20</code> sweep, computes {{als}}'s "
            "fold-in factor for every cold item using exactly the first <code>k</code> of its own "
            "revealed interactions via <code>recalculate_item</code>, evaluates against the same fixed "
            "reserved test set throughout, and repeats this across the 10 models fit in stage "
            "{{warmup_reference_score}} - it does not refit or re-invoke stage {{model_fork}} at any "
            "point on the curve. <code>K=100</code>, not a stricter cutoff: a cold item competing "
            "against ~1,600 fully-trained warm items for one of only 10 slots never wins that "
            "competition anywhere in the sweep, for any user - the resulting curve is a flat, "
            "uninformative zero end to end at <code>K=10</code>. At <code>K=100</code> a real, growing "
            "signal appears on NDCG/Recall/HitRate. AUC (no <code>K</code>) adds a subtler read: ALS "
            "stays ahead of Popularity throughout, but the <i>gap</i> shrinks from 0.39 at k=0 to 0.12 "
            "at k=20, not widens - Popularity's climb from an extreme structural disadvantage closes "
            "distance faster in relative terms than ALS's climb, which is real but flattening."
        ),
        meta=[
            Metric("k=20 HitRate@100 - Pop 0.0000 / ALS 0.1071 (mean of 10)"),
            Metric("k=20 AUC - Pop 0.5339 / ALS 0.6495 (gap 0.12, down from 0.39 at k=0)"),
        ],
    ),

    Step(
        key="warmup_curve_mode_b",
        title="Warm-Up Curve - Item-to-User (Mode B)",
        tag="diag",
        desc=(
            "The targeted-marketing question, the transposed dual of stage {{warmup_curve_mode_a}}: for "
            "a single cold item, rank <i>users</i> by predicted affinity, and check whether the users "
            "known (from the reserved test set) to actually want it land near the top of <i>its</i> "
            "top-K. Same fold-in factors as {{warmup_curve_mode_a}}, same frozen-user-factor principle, "
            "just the ranking axis transposed - no new fitting. {{popularity}} has no meaningful version "
            "of this (no per-user signal to rank users by, only a tie), so it's compared against its "
            "direct dual instead: a frozen <b>global-user-activity</b> ranking (each user's total "
            "interaction count in {{reference_set}}'s train split). Uses its own <code>K_MODE_B=10</code>, "
            "not stage {{warmup_reference_score}}'s <code>K=100</code>, since ~940 candidate users is a "
            "much smaller pool than ~1,600 items and a large K there saturates HitRate for both models "
            "alike. <b>Precision@K</b> is reported alongside Recall@K since the two are capped in "
            "opposite directions by each item's reserved-test-user count <code>r_i</code> (5 to 487) - a "
            "real cross-item comparability problem NDCG@K and AUC don't have. The <b>Downsampled Test "
            "Set</b> (a fixed, random 5-user sample per item - every eligible item is guaranteed at "
            "least 5 by construction, <code>MIN_INTERACTIONS(25) - N_REVEAL(20) = 5</code>) fixes that "
            "comparability problem for NDCG/Precision/Recall/HitRate; AUC keeps the full reserved test "
            "set throughout, since its ceiling is trivially 1.0 regardless of <code>r_i</code>. The two "
            "populations tell genuinely different stories: on the fairness-controlled downsampled "
            "metrics, {{als}} <b>overtakes</b> the activity floor around <code>k&asymp;11-15</code>; on "
            "the full-population AUC, {{als}} still trails at <code>k=20</code> - localizing the "
            "remaining gap to the high-<code>r_i</code>, near-popular items specifically, not the "
            "cold-item population as a whole."
        ),
        meta=[
            Metric("k=20 HitRate@10 (n=5/item) - Activity 0.2989 / ALS 0.3839"),
            Metric("k=20 AUC (full set) - Activity 0.7990 / ALS 0.7493"),
            Metric("ceilings @K=10 - NDCG/Recall/HitRate/AUC 1.0000, Precision 0.5000"),
        ],
    ),
])

pipeline_html = render_pipeline(pipeline, registry)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

legend_html = render_legend([
    ("background: var(--tag-generic);", "Shared / Generic",
     "Identical under any CF or hybrid variant - reused as-is by future interventions.", False),
    ("background: var(--accent);", "Algorithm-Specific",
     "The thing that actually changes when swapping in a content-based or other intervention.", False),
    ("background: var(--tag-diag);", "Diagnostic / Benchmark",
     "Not production - the comparison harness. Must stay identical across variants.", False),
    ("background: var(--card); border: 1.5px dashed var(--tag-future);", "Future / Deferred",
     "Not built. An identified extension point on one or more numbered stages below.", True),
])

# ---------------------------------------------------------------------------
# Metrics tables
# ---------------------------------------------------------------------------

metrics_sections = [
    render_metrics_table(
        caption="Cold-item population (stage {{cold_item_split}}) - 25+ interaction eligibility, plain random 10%",
        headers=["Metric", "Value"],
        rows=[
            ["Eligible items (≥25 interactions)", Metric("872 / 1,682 (51.8%)")],
            ["Cold items selected", Metric("87 (10.0% of eligible)")],
            ["Reserved test interactions", Metric("7,372 total")],
            ["Reserved per item - min / median / mean / max", Metric("5 / 57 / 84.7 / 487")],
        ],
        note=(
            "That min/median/mean/max spread is exactly what motivated stage "
            "{{warmup_curve_mode_b}}'s Downsampled Test Set - a plain random 10% draw makes no promise "
            "about how homogeneous the resulting population's difficulty will be."
        ),
    ),
    render_metrics_table(
        caption="Warm-item reference (stage {{warmup_reference_score}}) - random 80/20 split, 1,595 warm items, "
                "72,833 train / 18,055 test, 943 eval users",
        headers=["Metric", "Popularity", "ALS (mean of 10 seeds)"],
        rows=[
            ["NDCG@100", Metric("0.2975"), Metric("0.4635 ± 0.0018")],
            ["Recall@100", Metric("0.4503"), Metric("0.6524 ± 0.0021")],
            ["HitRate@100", Metric("0.9576"), Metric("0.9897 ± 0.0013")],
            ["AUC (no K)", Metric("0.8717"), Metric("0.8840 ± 0.0010")],
        ],
        note=(
            "This is the ceiling stage {{warmup_curve_mode_a}}'s curve is climbing toward, not a value "
            "it should ever exceed - computed on warm items only, cold items held at strict k=0."
        ),
        margin_top=22,
    ),
    render_metrics_table(
        caption="Warm-up curve, user-to-item (stage {{warmup_curve_mode_a}}) - K=100, 860 eval users",
        headers=["k", "NDCG@100 (Pop / ALS)", "Recall@100 (Pop / ALS)", "HitRate@100 (Pop / ALS)", "AUC (Pop / ALS)"],
        rows=[
            [Metric("0"), Metric("0.0000 / 0.0000"), Metric("0.0000 / 0.0000"), Metric("0.0000 / 0.0000"), Metric("0.0328 / 0.4213")],
            [Metric("5"), Metric("0.0000 / 0.0000"), Metric("0.0000 / 0.0000"), Metric("0.0000 / 0.0000"), Metric("0.2855 / 0.4989")],
            [Metric("10"), Metric("0.0000 / 0.0015"), Metric("0.0000 / 0.0068"), Metric("0.0000 / 0.0134"), Metric("0.4044 / 0.5574")],
            [Metric("15"), Metric("0.0000 / 0.0037"), Metric("0.0000 / 0.0156"), Metric("0.0000 / 0.0447"), Metric("0.4817 / 0.6068")],
            [Metric("20"), Metric("0.0000 / 0.0089"), Metric("0.0000 / 0.0359"), Metric("0.0000 / 0.1071"), Metric("0.5339 / 0.6495")],
        ],
        note=(
            "Popularity never recovers on NDCG/Recall/HitRate across the full tested range - cracking a "
            "global top-100 needs far more than 20 interactions ({{popularity}}'s top item: 482). AUC "
            "tells a subtler story: ALS stays ahead of Popularity throughout, but the <i>gap</i> shrinks "
            "from 0.39 at k=0 to 0.12 at k=20, not widens - Popularity's climb from an extreme structural "
            "disadvantage (zero count) closes distance faster in relative terms than ALS's climb, which "
            "is real but flattening (diminishing marginal gain per revealed interaction, consistent with "
            "a regularized least-squares solve converging asymptotically)."
        ),
        margin_top=22,
    ),
    render_metrics_table(
        caption="Warm-up curve, item-to-user (stage {{warmup_curve_mode_b}}) - K_MODE_B=10, "
                "Downsampled Test Set (n=5/item) for NDCG/Precision/Recall/HitRate; full test set for AUC",
        headers=["k", "NDCG (ALS / Activity)", "Precision (ALS / Activity)", "Recall (ALS / Activity)",
                  "HitRate (ALS / Activity)", "AUC, full set (ALS / Activity)"],
        rows=[
            [Metric("0"), Metric("0.0082 / 0.0681"), Metric("0.0046 / 0.0379"), Metric("0.0092 / 0.0759"),
             Metric("0.0460 / 0.2989"), Metric("0.5000 / 0.7926")],
            [Metric("10"), Metric("0.0528 / 0.0684"), Metric("0.0308 / 0.0379"), Metric("0.0616 / 0.0759"),
             Metric("0.2862 / 0.2989"), Metric("0.6836 / 0.7959")],
            [Metric("15"), Metric("0.0627 / 0.0687"), Metric("0.0383 / 0.0379"), Metric("0.0766 / 0.0759"),
             Metric("0.3471 / 0.2989"), Metric("0.7146 / 0.7974")],
            [Metric("20"), Metric("0.0722 / 0.0707"), Metric("0.0428 / 0.0379"), Metric("0.0855 / 0.0759"),
             Metric("0.3839 / 0.2989"), Metric("0.7493 / 0.7990")],
            [Metric("ceiling"), Metric("1.0000"), Metric("0.5000"), Metric("1.0000"), Metric("1.0000"), Metric("1.0000")],
        ],
        note=(
            "ALS crosses over and overtakes the activity floor on all four downsampled metrics between "
            "k&asymp;11 and k&asymp;15, and keeps widening the lead through k=20. AUC (full test set, "
            "every r_i from 5 to 487 included) still shows ALS trailing at k=20 - the remaining gap is "
            "concentrated in the high-r_i, near-popular items specifically, where \"target the generally "
            "active users\" is close to unbeatable by construction, not spread evenly across the cold-"
            "item population."
        ),
        margin_top=22,
    ),
]

# ---------------------------------------------------------------------------
# Deferred / extension points
# ---------------------------------------------------------------------------

deferred_html = render_deferred_section([
    render_deferred_card(
        "Confidence-Weighted Interactions", "{{sparse_matrix}}",
        "<code>confidence = 1 + &alpha;&middot;rating</code> (Hu et al. 2008). ALS reads interaction "
        "magnitude natively, so this would be a direct extension of the current interaction matrix - "
        "not used here, every interaction counts equally regardless of rating.",
    ),
    render_deferred_card(
        "Positive-Only Filtering", "{{sparse_matrix}}",
        "Exclude low ratings (e.g. &lt;3 stars) from the interaction set entirely, rather than "
        "including them at low weight. Addresses \"1-star &ne; dislike.\"",
    ),
    render_deferred_card(
        "Time-Based Split", "{{cold_item_split}}",
        "Not the same thing as stage {{cold_item_split}}'s chronological <i>reveal</i> - that orders "
        "each cold item's own interactions internally, but every warm item still trains on 100% of its "
        "history regardless of date. A true time-based split would draw one global cutoff T across the "
        "whole dataset (train on everything before T, test after), so \"cold\" means \"didn't exist in "
        "the catalog yet,\" not \"deliberately withheld.\" A realism upgrade, not a correctness fix - "
        "stage {{cold_item_split}} already guarantees structural coldness without it.",
    ),
    render_deferred_card(
        "Content Embeddings + LLM-Simulated Interactions", "{{ingestion}}, {{als}}",
        "How content embeddings, improved content embeddings, and LLM-simulated interactions enter this "
        "same fold-in mechanism is an open design question - not yet designed. {{popularity}} stays "
        "as-is: the floor doesn't change just because the item representation being measured against it "
        "does.",
    ),
    render_deferred_card(
        "Cold-Item Selection Ceiling", "{{cold_item_split}}",
        "The r_i skew (5 to 487 reserved users/item) that motivated stage {{warmup_curve_mode_b}}'s "
        "Downsampled Test Set was discovered via diagnostics, not designed around at selection time. A "
        "symmetric <code>MAX_INTERACTIONS</code> ceiling alongside the existing floor - narrowing the "
        "eligible population itself rather than fixing it downstream - was considered and not yet "
        "implemented; the Downsampled Test Set is a complementary fix for the K-based metrics, not a "
        "substitute for addressing the underlying population choice.",
    ),
    render_deferred_card(
        "Cold-User Split", "{{cold_item_split}}, {{warmup_curve_mode_a}}",
        "The row-based mirror of stages {{cold_item_split}} and {{warmup_curve_mode_a}} together - the "
        "same eligibility-filter-then-random-sample selection over users instead of items, entire users "
        "held out, then the same chronological-reveal-plus-fold-in treatment to build a true user "
        "warm-up curve. A related but distinct cold-start problem; the machinery to build it now already "
        "exists, just needs pointing at the other axis.",
    ),
])

# ---------------------------------------------------------------------------
# Header, footer, assembly
# ---------------------------------------------------------------------------

header_html = render_header(
    eyebrow="Solution Architecture - Reference Diagram",
    title="Collaborative-Filtering Retrieval Pipeline",
    dek=(
        "The current steel thread: a pure-CF retrieval stage (ALS, benchmarked against a popularity "
        "floor) evaluated on MovieLens 100k under a structural cold-item split, isolating the effect an "
        "item's own accumulating history has on retrieval quality by fold-in (recalculate_item) rather "
        "than retraining - user preferences are fit once and held completely constant throughout. Two "
        "warm-up curves are built from this same fold-in mechanism: one from the standard user-to-item "
        "perspective, one from the transposed item-to-user (targeted-marketing) perspective. Built for "
        "comparing against interventions later on."
    ),
    meta_items=[
        ("Models", "Popularity ({{popularity}}) + Alternating Least Squares ({{als}})"),
        ("Split", "{{cold_item_split}} cold-item (25+ interactions eligible, 10% plain random) &rarr; "
                  "{{reference_set}} reference (independent, not a fold-in)"),
        ("Reference-before-evaluation", "stage {{warmup_reference_score}} runs ahead of stage "
                                          "{{cold_eval}}, on purpose"),
        ("Warm-up curves", "fold-in via recalculate_item, not retrained, 10 seeds, k=0..20; "
                            "user-to-item ({{warmup_curve_mode_a}}) and item-to-user "
                            "({{warmup_curve_mode_b}})"),
        ("Dataset", "MovieLens 100k (proxy for Amazon Books)"),
        ("Scope", "retrieval / candidate generation only"),
        ("Source", "baseline_steel_thread_cf.py"),
    ],
)

footer_html = render_footer(["baseline_steel_thread_cf.py", "implicit 0.7.3", "MovieLens 100k", "699 Capstone"])

page_html = render_page(
    page_title="Solution Architecture - CF Retrieval Pipeline",
    header_html=header_html,
    legend_html=legend_html,
    pipeline_html=pipeline_html,
    metrics_sections_html=metrics_sections,
    deferred_html=deferred_html,
    footer_html=footer_html,
)

final_html = resolve_refs(page_html, registry)  # {{key}} -> real label, one pass, whole page

out_path = Path(__file__).parent / "solution_architecture_generated.html"
out_path.write_text(final_html, encoding="utf-8")
print(f"wrote {out_path}")
print(f"registry: {registry}")
