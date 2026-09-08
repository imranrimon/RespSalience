"""Static consistency checks for the manuscript sources.

Runs without the datasets, so it is safe in CI. It catches the class of defect that repeatedly slipped
through during revision: references to labels that no longer exist, citations with no bibliography
entry, floats nobody points at, figures placed at a fraction of their authored width (which is what
makes tick labels collide), and tables too wide for a single IEEE column.

    python tools/check_manuscript.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
B = chr(92)
TEX = ["prism_breath.tex", "prism_breath_supplementary.tex", "appendix_additions.tex",
       "appendix_evals.tex", "appendix_reviewer_analyses.tex", "appendix_selection.tex",
       "tripod_checklist.tex"]
COLUMN_IN = 3.5          # IEEEtran single column
TEXT_IN = 7.16           # IEEEtran full width


def main() -> int:
    present = [f for f in TEX if (MS / f).exists()]
    if not present:
        print("no manuscript sources found; nothing to check")
        return 0
    text = "".join((MS / f).read_text(encoding="utf-8") for f in present)
    problems = []

    labels = set(re.findall(B + B + r"label\{([^}]+)\}", text))
    refs = set(re.findall(B + B + r"ref\{([^}]+)\}", text))
    for r in sorted(refs - labels):
        problems.append(f"dangling reference: {r}")

    bib = MS / "prism_breath.bib"
    if bib.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
        cites = {x.strip() for c in re.findall(B + B + r"cite\{([^}]+)\}", text) for x in c.split(",")}
        for c in sorted(cites - keys):
            problems.append(f"undefined citation: {c}")

    for lab in sorted(l for l in labels if l.startswith(("tab:", "fig:"))):
        if lab not in refs:
            problems.append(f"float never referenced: {lab}")

    # A figure placed far below its authored width has its text scaled down with it.
    try:
        import fitz  # PyMuPDF, optional
    except ImportError:
        fitz = None
    for m in re.finditer(B + B + r"begin\{(figure\*?)\}(.*?)" + B + B + r"end\{\1\}", text, re.S):
        env, body = m.group(1), m.group(2)
        inc = re.search(B + B + r"includegraphics\[width=([^\]]+)\]\{([^}]+)\}", body)
        if not inc:
            continue
        placed = float(re.sub(r"[^0-9.]", "", inc.group(1)) or 0)
        limit = TEXT_IN if env.endswith("*") else COLUMN_IN
        if placed > limit + 0.05:
            problems.append(f"figure wider than its float: {inc.group(2)} at {placed}in in {env}")
        if fitz:
            for cand in (MS / inc.group(2), MS / "fig" / inc.group(2),
                         MS / (inc.group(2) + ".pdf"), MS / "fig" / (inc.group(2) + ".pdf")):
                if cand.exists():
                    with fitz.open(cand) as d:
                        native = d[0].rect.width / 72
                    if placed and native and placed / native < 0.65:
                        problems.append(
                            f"figure down-scaled to {placed/native:.0%} of authored width "
                            f"(labels will collide): {inc.group(2)} native {native:.2f}in -> {placed}in")
                    break

    # A figure present BOTH in the graphicspath and in the document directory is a trap: the
    # graphicspath copy wins, so a regenerated figure left in the document directory is silently
    # ignored while a stale copy is typeset. This actually happened -- four figures compiled from
    # copies a month older than the numbers they illustrated.
    import hashlib
    gp = re.search(B + B + r"graphicspath\{\{([^}]*)\}\}", text)
    if gp:
        gdir = MS / gp.group(1).strip("/")
        seen = set()
        for m in re.finditer(B + B + r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            a, b = MS / name, gdir / name
            if a.exists() and b.exists():
                ha = hashlib.md5(a.read_bytes()).hexdigest()
                hb = hashlib.md5(b.read_bytes()).hexdigest()
                if ha != hb:
                    newer = "document dir" if a.stat().st_mtime > b.stat().st_mtime else "graphicspath"
                    problems.append(
                        f"figure shadowed: {name} differs between manuscript/ and "
                        f"{gp.group(1)} -- LaTeX will use the {gp.group(1)} copy; the newer file is in "
                        f"the {newer}")
            elif not b.exists() and not a.exists():
                problems.append(f"figure not found in either manuscript/ or {gp.group(1)}: {name}")

    # A wide table in a single-column float overflows into the gutter.
    for m in re.finditer(B + B + r"begin\{(table\*?)\}(.*?)" + B + B + r"end\{\1\}", text, re.S):
        env, body = m.group(1), m.group(2)
        if env.endswith("*"):
            continue
        lab = re.search(B + B + r"label\{([^}]+)\}", body)
        rows = [ln for ln in body.splitlines() if "&" in ln]
        width = max((len(ln) for ln in rows), default=0)
        ncol = max((ln.count("&") for ln in rows), default=0) + 1
        # Column count alone is a poor predictor -- six narrow columns fit comfortably. Row width
        # is what actually overflows \columnwidth.
        if width > 95:
            problems.append(f"table may overflow a single column, use table*: "
                            f"{lab.group(1) if lab else '?'} ({ncol} cols, {width} chars)")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print(f"manuscript checks passed ({len(present)} sources, {len(labels)} labels, {len(refs)} refs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
