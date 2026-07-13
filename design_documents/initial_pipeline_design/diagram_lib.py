"""
Reusable architecture-diagram builder: numbered pipeline stages that can be a plain
sequence, a boxed/grouped sub-sequence (dependent sub-steps, e.g. "4.1"/"4.2"), or a
fork (parallel, order-arbitrary branches, e.g. "5a"/"5b") -- and any of those can nest
inside a Fork's branches, since a branch is itself a Step, Sequence, or Fork.

Cross-references are name-based, not number-based: give a node `key="some_name"`, then
write "{{some_name}}" anywhere in ANY text (another step's desc, a deferred card, a table
caption) instead of hand-typing its number. assign_labels() builds a key->label registry
while numbering the tree; resolve_refs() does one final substitution pass over the fully
assembled page so every reference updates automatically when stages are added, removed,
or reordered -- no grep-and-fix required.

Visual design (palette, type, card/fork/box CSS) is content-agnostic and lives entirely
in PAGE_CSS below; a new diagram just supplies new Step/Sequence/Fork data plus the
legend/metrics/deferred/footer content via the render_page() helpers.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

Node = Union["Step", "Sequence", "Fork"]

TAG_LABELS = {
    "generic": "Shared / Generic",
    "algo": "Algorithm-Specific",
    "diag": "Diagnostic / Benchmark",
}


class Metric(str):
    """Meta item rendered with the accent 'metric' styling (a computed result)."""


class Code(str):
    """Meta item rendered inside a <code> tag (an identifier, not a result)."""


def _meta_item(item):
    if isinstance(item, Code):
        return f"<code>{item}</code>"
    if isinstance(item, Metric):
        return f'<span class="metric">{item}</span>'
    return f"<span>{item}</span>"


@dataclass
class Step:
    """A single leaf stage: one numbered box (in a Sequence) or one lettered card (in a Fork)."""
    title: str
    tag: str  # "generic" | "algo" | "diag"
    desc: str  # may include inline <b>/<i>/<code> and {{key}} cross-references
    meta: List[object] = field(default_factory=list)  # plain str, Metric(...), or Code(...)
    key: Optional[str] = None  # stable name for {{key}} cross-references elsewhere
    label: str = ""  # assigned by assign_labels(), not set by hand


@dataclass
class Sequence:
    """Ordered, top-to-bottom, connected by a line. children may be Step, Sequence, or Fork.
    boxed=True wraps them in the dashed group box -- use for a related dependent sub-part
    of a bigger numbered step (like 4.1/4.2), not for the top-level flow itself."""
    children: List[Node]
    boxed: bool = False
    box_label: Optional[str] = None  # may contain {{key}} cross-references
    key: Optional[str] = None
    label: str = ""


@dataclass
class Fork:
    """Parallel, order-arbitrary branches. Each branch may itself be a Step, Sequence, or
    Fork (a branch can fork again, or contain its own multi-step sequence)."""
    branches: List[Node]
    fork_note: Optional[str] = None   # defaults to an auto-generated "stage N forks..." note
    merge_note: Optional[str] = None  # defaults to a generic "merges" note
    key: Optional[str] = None
    label: str = ""


def assign_labels(node, registry, prefix=None):
    """
    Walks the tree, sets `.label` on every Step/Sequence/Fork, and records key->label in
    `registry` for every node that has a `key`. Rules: a Sequence's children get plain
    integers at the top level (1, 2, 3...) or decimals when nested (4.1, 4.2...). A Fork's
    branches get letters off its own label (5a, 5b...), recursing with that letter as the
    new prefix if a branch nests further.
    """
    if isinstance(node, Sequence):
        for i, child in enumerate(node.children, start=1):
            child.label = str(i) if prefix is None else f"{prefix}.{i}"
            if child.key:
                registry[child.key] = child.label
            if isinstance(child, (Sequence, Fork)):
                assign_labels(child, registry, prefix=child.label)
    elif isinstance(node, Fork):
        for i, branch in enumerate(node.branches):
            letter = chr(ord("a") + i)
            branch.label = f"{prefix}{letter}" if prefix else letter
            if branch.key:
                registry[branch.key] = branch.label
            if isinstance(branch, (Sequence, Fork)):
                assign_labels(branch, registry, prefix=branch.label)


_REF_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def resolve_refs(text, registry):
    """Replace every {{key}} in `text` with registry[key] -- the key's current numeric
    label. Call once, over the fully assembled page, after assign_labels() has run.
    Raises KeyError with the offending key if it was never assigned (typo, or the
    referenced node's `key=` was removed) -- fails loudly rather than shipping a
    literal "{{typo}}" into the diagram."""
    def repl(m):
        key = m.group(1)
        if key not in registry:
            raise KeyError(
                f"Unknown cross-reference {{{{{key}}}}} -- no Step/Sequence/Fork has key={key!r}. "
                f"Known keys: {sorted(registry)}"
            )
        return registry[key]
    return _REF_PATTERN.sub(repl, text)


def _render_step_sequential(step):
    decimal_cls = " decimal" if "." in step.label else ""
    meta_html = "".join(_meta_item(m) for m in step.meta)
    return f"""
    <!-- Stage {step.label} -->
    <div class="stage">
      <div class="stage-spine"><div class="stage-num{decimal_cls}">{step.label}</div><div class="stage-line"></div></div>
      <div class="stage-body">
        <div class="stage-card">
          <div class="stage-head">
            <span class="stage-title">{step.title}</span>
            <span class="tag {step.tag}">{TAG_LABELS[step.tag]}</span>
          </div>
          <p class="stage-desc">{step.desc}</p>
          <div class="stage-meta">{meta_html}</div>
        </div>
      </div>
    </div>"""


def _render_step_fork_branch(step):
    meta_html = "".join(_meta_item(m) for m in step.meta)
    return f"""
      <div class="fork-card">
        <div class="fork-badge">{step.label}</div>
        <div class="stage-head">
          <span class="stage-title">{step.title}</span>
          <span class="tag {step.tag}">{TAG_LABELS[step.tag]}</span>
        </div>
        <p class="stage-desc">{step.desc}</p>
        <div class="stage-meta">{meta_html}</div>
      </div>"""


def _render_sequence(seq):
    inner = "\n".join(_render_node(c) for c in seq.children)
    if not seq.boxed:
        return inner
    return f"""
    <div class="step-group">
      <div class="step-group-label">{seq.box_label}</div>
      {inner}
    </div>"""


def _render_fork(fork):
    branch_labels = " / ".join(b.label for b in fork.branches)
    fork_note = fork.fork_note or f"stage {fork.label} forks - order between {branch_labels} is arbitrary"
    merge_note = fork.merge_note or "merges - shared interface"

    branch_html = []
    for b in fork.branches:
        if isinstance(b, Step):
            branch_html.append(_render_step_fork_branch(b))
        else:
            # Nested Sequence/Fork as a branch: not visually tuned beyond this diagram's
            # needs yet -- falls back to the same node renderer, stacked in the column.
            branch_html.append(_render_node(b))

    return f"""
    <div class="split-divider"><span>&#9660; {fork_note}</span></div>
    <div class="fork-row">{"".join(branch_html)}
    </div>
    <div class="split-divider"><span>&#9650; {merge_note}</span></div>"""


def _render_node(node):
    if isinstance(node, Step):
        return _render_step_sequential(node)
    if isinstance(node, Sequence):
        return _render_sequence(node)
    if isinstance(node, Fork):
        return _render_fork(node)
    raise TypeError(f"Unknown node type: {type(node)}")


def render_pipeline(root_sequence, registry, heading="Pipeline - execution order"):
    """root_sequence: a top-level (unboxed) Sequence. Assigns labels into `registry`
    (mutated in place -- reuse the same dict for resolve_refs() on the rest of the page),
    then renders. Note: the returned HTML still contains unresolved {{key}} placeholders;
    call resolve_refs() over the fully assembled page before writing it out."""
    assign_labels(root_sequence, registry, prefix=None)
    body = "\n".join(_render_node(c) for c in root_sequence.children)
    return f"""
  <section class="pipeline">
    <h2 class="section-heading">{heading}</h2>
{body}
  </section>"""


# ---------------------------------------------------------------------------
# Page furniture: legend, metrics tables, deferred/extension cards, skeleton.
# ---------------------------------------------------------------------------

def render_legend(items):
    """items: list of (swatch_style_css, title, body_html, is_future: bool)"""
    cards = []
    for swatch_css, title, body, is_future in items:
        future_cls = " is-future" if is_future else ""
        cards.append(f"""
    <div class="legend-item{future_cls}">
      <div class="swatch" style="{swatch_css}"></div>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>""")
    return f"""
  <section class="legend" aria-label="Tag taxonomy">{"".join(cards)}
  </section>"""


def render_metrics_table(caption, headers, rows, note=None, margin_top=None):
    """rows: list of lists of cells. Wrap numeric/tabular cells in Metric(...) to get the
    'num' (tabular-nums) styling -- explicit, not guessed from the string's contents."""
    style = f' style="margin-top:{margin_top}px;"' if margin_top else ""
    head_html = "".join(f"<th>{h}</th>" for h in headers)
    row_html = ""
    for row in rows:
        cells = "".join(
            f'<td class="num">{c}</td>' if isinstance(c, Metric) else f"<td>{c}</td>"
            for c in row
        )
        row_html += f"<tr>{cells}</tr>\n"
    note_html = f'\n    <p style="font-size:12.5px;color:var(--ink-faint);margin-top:8px;">\n      {note}\n    </p>' if note else ""
    return f"""
    <table class="metrics"{style}>
      <caption>{caption}</caption>
      <thead>
        <tr>{head_html}</tr>
      </thead>
      <tbody>
        {row_html.strip()}
      </tbody>
    </table>{note_html}"""


def render_deferred_card(title, extends, body):
    return f"""
      <div class="future-card">
        <div class="future-head">
          <span class="future-title">{title}</span>
          <span class="extends">extends &rarr; {extends}</span>
        </div>
        <p>{body}</p>
      </div>"""


def render_deferred_section(cards, heading="Extension points - deferred, not built"):
    return f"""
  <section>
    <h2 class="section-heading">{heading}</h2>
    <div class="future-grid">{"".join(cards)}
    </div>
  </section>"""


def render_header(eyebrow, title, dek, meta_items):
    meta_html = "".join(f"<span><b>{k}:</b> {v}</span>" for k, v in meta_items)
    return f"""
  <header>
    <div class="eyebrow">{eyebrow}</div>
    <h1>{title}</h1>
    <p class="dek">
      {dek}
    </p>
    <div class="meta-row">{meta_html}
    </div>
  </header>"""


def render_footer(items):
    return f"""
  <footer>{"".join(f"<span>{i}</span>" for i in items)}
  </footer>"""


def render_page(page_title, header_html, legend_html, pipeline_html, metrics_sections_html, deferred_html, footer_html):
    metrics_block = "\n".join(metrics_sections_html) if isinstance(metrics_sections_html, list) else metrics_sections_html
    return f"""<title>{page_title}</title>
<style>
{PAGE_CSS}
</style>

<div class="page">
{header_html}
{legend_html}
{pipeline_html}

  <section class="metrics-wrap">
{metrics_block}
  </section>
{deferred_html}
{footer_html}

</div>
"""


PAGE_CSS = """
  :root {
    --paper: #EEF1F0;
    --card: #FFFFFF;
    --ink: #12181F;
    --ink-soft: #4A5560;
    --ink-faint: #7C8790;
    --line: #C7CED3;
    --accent: #C4622D;
    --accent-soft: #F3E2D6;
    --tag-generic: #3B6E8F;
    --tag-generic-soft: #E1EBF1;
    --tag-diag: #6B7A3F;
    --tag-diag-soft: #E9EDDE;
    --tag-future: #8A8478;
    --tag-future-soft: #F2F0EC;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Georgia", serif;
    --sans: ui-sans-serif, "Avenir Next", "Segoe UI", system-ui, sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }

  * { box-sizing: border-box; }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  a { color: var(--accent); }

  .page {
    max-width: 880px;
    margin: 0 auto;
    padding: 56px 28px 96px;
    display: flex;
    flex-direction: column;
    gap: 48px;
  }

  header {
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-bottom: 1px solid var(--line);
    padding-bottom: 28px;
  }

  .eyebrow {
    font-family: var(--mono);
    font-size: 12.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent);
  }

  h1 {
    font-family: var(--serif);
    font-weight: 500;
    font-size: 34px;
    line-height: 1.15;
    margin: 0;
    text-wrap: balance;
    color: var(--ink);
  }

  .dek {
    font-size: 15.5px;
    color: var(--ink-soft);
    max-width: 62ch;
  }

  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
    font-family: var(--mono);
    font-size: 12.5px;
    color: var(--ink-faint);
    margin-top: 4px;
  }

  .meta-row b { color: var(--ink-soft); font-weight: 600; }

  .legend {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }

  .legend-item {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 14px 14px 15px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .legend-item.is-future { border-style: dashed; }

  .swatch {
    width: 20px;
    height: 20px;
    border-radius: 4px;
  }

  .legend-item h3 {
    font-family: var(--sans);
    font-size: 13.5px;
    font-weight: 700;
    margin: 0;
    color: var(--ink);
  }

  .legend-item p {
    font-size: 12.5px;
    color: var(--ink-soft);
    margin: 0;
    line-height: 1.4;
  }

  section.pipeline {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .section-heading {
    font-family: var(--serif);
    font-size: 21px;
    font-weight: 500;
    margin: 0 0 20px;
    color: var(--ink);
  }

  .stage {
    position: relative;
    display: grid;
    grid-template-columns: 56px 1fr;
    gap: 0 20px;
  }

  .stage-spine {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stage-num {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--card);
    border: 2px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-weight: 700;
    font-size: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1;
    flex-shrink: 0;
  }

  .stage-num.decimal {
    width: 48px;
    height: 48px;
    font-size: 12.5px;
    letter-spacing: -0.02em;
  }

  .step-group {
    position: relative;
    border: 1.5px dashed var(--tag-generic);
    border-radius: 10px;
    padding: 20px 18px 2px;
    margin: 4px 0 10px;
  }

  .step-group-label {
    position: absolute;
    top: -11px;
    left: 18px;
    background: var(--paper);
    padding: 0 8px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--tag-generic);
  }

  .step-group .stage:last-of-type .stage-line {
    display: none;
  }

  .stage-line {
    width: 2px;
    flex: 1;
    background: var(--line);
    margin: 2px 0;
  }

  .stage:last-of-type .stage-line { display: none; }

  .stage-body {
    padding-bottom: 28px;
  }

  .stage-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 18px 20px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .stage-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .stage-title {
    font-family: var(--serif);
    font-size: 18px;
    font-weight: 500;
    color: var(--ink);
  }

  .tag {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 100px;
    white-space: nowrap;
  }

  .tag.generic { background: var(--tag-generic-soft); color: var(--tag-generic); }
  .tag.algo    { background: var(--accent-soft); color: var(--accent); }
  .tag.diag    { background: var(--tag-diag-soft); color: var(--tag-diag); }

  .split-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 4px 0;
  }

  .split-divider::before, .split-divider::after {
    content: "";
    flex: 1;
    height: 2px;
    background: var(--accent);
  }

  .split-divider span {
    font-family: var(--mono);
    font-size: 11.5px;
    letter-spacing: 0.03em;
    color: var(--accent);
    white-space: nowrap;
  }

  .split-divider span code {
    background: var(--accent-soft);
    border-radius: 3px;
    padding: 1px 5px;
  }

  .fork-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .fork-card {
    background: var(--card);
    border: 2px solid var(--accent);
    border-radius: 6px;
    padding: 16px 18px 18px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    position: relative;
  }

  .fork-badge {
    position: absolute;
    top: -13px;
    left: 16px;
    background: var(--accent);
    color: #fff;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 12px;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .stage-desc {
    font-size: 14.5px;
    color: var(--ink-soft);
    max-width: 66ch;
  }

  .stage-desc b { color: var(--ink); font-weight: 600; }

  .stage-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink-faint);
    padding-top: 6px;
    border-top: 1px dashed var(--line);
  }

  .stage-meta code {
    background: var(--paper);
    border-radius: 3px;
    padding: 1px 5px;
    color: var(--ink-soft);
  }

  .stage-meta .metric { color: var(--accent); font-weight: 600; }

  .future-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
  }

  .future-card {
    background: var(--tag-future-soft);
    border: 1.5px dashed var(--tag-future);
    border-radius: 6px;
    padding: 15px 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .future-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
  }

  .future-title {
    font-family: var(--serif);
    font-size: 15.5px;
    font-weight: 500;
    color: var(--ink);
  }

  .extends {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--tag-future);
    white-space: nowrap;
  }

  .future-card p {
    font-size: 13.5px;
    color: var(--ink-soft);
    margin: 0;
  }

  .metrics-wrap { overflow-x: auto; }

  table.metrics {
    border-collapse: collapse;
    width: 100%;
    font-size: 13.5px;
    min-width: 560px;
  }

  table.metrics caption {
    text-align: left;
    font-family: var(--serif);
    font-size: 16px;
    color: var(--ink);
    margin-bottom: 10px;
    caption-side: top;
  }

  table.metrics th, table.metrics td {
    text-align: left;
    padding: 8px 14px 8px 0;
    border-bottom: 1px solid var(--line);
  }

  table.metrics th {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-faint);
    font-weight: 600;
  }

  table.metrics td.num {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    color: var(--ink);
  }

  footer {
    border-top: 1px solid var(--line);
    padding-top: 20px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink-faint);
    display: flex;
    flex-wrap: wrap;
    gap: 6px 24px;
  }

  @media (max-width: 640px) {
    .legend { grid-template-columns: repeat(2, 1fr); }
    .future-grid { grid-template-columns: 1fr; }
    .stage { grid-template-columns: 40px 1fr; }
  }

  @media (prefers-reduced-motion: no-preference) {
    .stage-card, .future-card { transition: border-color 0.15s ease; }
  }
"""
