"""Find quantitative claims in the manuscript that nothing re-derives.

verify_manuscript_claims.py proves that the numbers it knows about are right. It says nothing about
the numbers it has never heard of, and that gap is exactly how this paper's errors happened: the
escalation section carried four mutually inconsistent number sets, and the pretraining ablation shipped
a false positive, because neither was in the harness.

This walks the LaTeX, pulls out every number that reads like a result, and reports which are covered.
Coverage is by VALUE (a claim is covered if the harness checks that value in that neighbourhood of the
paper), which is deliberately loose -- it is a triage tool for finding blind spots, not a proof.

    python tools/audit_claims.py                 # summary by section
    python tools/audit_claims.py --uncovered     # list the blind spots
    python tools/audit_claims.py --dupes         # same quantity stated with different values
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
VERIFIER = ROOT / "scripts" / "verify_manuscript_claims.py"

SOURCES = ["prism_breath.tex", "appendix_additions.tex", "appendix_evals.tex",
           "appendix_reviewer_analyses.tex", "appendix_selection.tex",
           "appendix_reproducibility.tex", "prism_breath_supplementary.tex"]

# A "result" number: 3-decimal metrics, p-values, percentages, and 2-decimal rates. Deliberately
# excludes years, reference numbers, section numbers and table geometry.
NUM = re.compile(r"(?<![\w.])(\d\.\d{2,3}|\d{1,3}\.\d\\?%|\d{1,3}\\%|\d+\.\d+\{?\\?times)")
PVAL = re.compile(r"p\s*[=<]\s*\$?([\d.]+(?:\{?\\times\}?10\^\{?-?\d+\}?)?)")


def strip_comments(t):
    return "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("%"))


def sections(text):
    """Map each line to the most recent \\section/\\subsection label, for grouping."""
    out, cur = [], "(preamble)"
    for line in text.splitlines():
        m = re.search(r"\\(?:sub)?section\*?\{([^}]*)\}", line)
        if m:
            cur = m.group(1)[:48]
        out.append(cur)
    return out


def verifier_values():
    """Every literal the harness checks, as rounded strings."""
    t = VERIFIER.read_text(encoding="utf-8")
    vals = set()
    for m in re.finditer(r"ck\.add\(\s*[^,]+,\s*([-\d.e+]+)", t):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        vals.add(f"{v:.3f}")
        vals.add(f"{v:.2f}")
        vals.add(f"{v:.1f}")
    # Tuples of claimed values, e.g. ("<2mo", 0.650, 0.939)
    for m in re.finditer(r"\(\s*\"[^\"]+\",\s*([\d.]+),\s*([\d.]+)\s*\)", t):
        for g in m.groups():
            v = float(g)
            vals.update({f"{v:.3f}", f"{v:.2f}", f"{v:.1f}"})
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncovered", action="store_true")
    ap.add_argument("--dupes", action="store_true")
    a = ap.parse_args()

    covered = verifier_values()
    per_section = defaultdict(lambda: {"n": 0, "cov": 0, "miss": []})
    value_sites = defaultdict(list)

    for src in SOURCES:
        p = MS / src
        if not p.exists():
            continue
        text = strip_comments(p.read_text(encoding="utf-8", errors="ignore"))
        secs = sections(text)
        for i, line in enumerate(text.splitlines()):
            if line.lstrip().startswith("\\label") or "includegraphics" in line:
                continue
            for m in NUM.finditer(line):
                raw = m.group(1)
                try:
                    v = float(re.sub(r"[\\%a-z{}]", "", raw))
                except ValueError:
                    continue
                if v > 1000 or (v == int(v) and v < 10):      # counts and small integers: not metrics
                    continue
                key = f"{v:.3f}"
                sec = secs[i]
                rec = per_section[f"{src}:{sec}"]
                rec["n"] += 1
                hit = any(f"{v:.{d}f}" in covered for d in (3, 2, 1))
                if hit:
                    rec["cov"] += 1
                else:
                    rec["miss"].append((i + 1, raw, line.strip()[:90]))
                value_sites[key].append((src, i + 1, sec))

    tot = sum(r["n"] for r in per_section.values())
    cov = sum(r["cov"] for r in per_section.values())
    print(f"{'section':58s} {'claims':>7s} {'backed':>7s} {'gap':>5s}")
    print("-" * 82)
    for k in sorted(per_section, key=lambda x: -(per_section[x]["n"] - per_section[x]["cov"])):
        r = per_section[k]
        gap = r["n"] - r["cov"]
        if r["n"]:
            print(f"{k[:58]:58s} {r['n']:>7d} {r['cov']:>7d} {gap:>5d}")
    print("-" * 82)
    print(f"{'TOTAL':58s} {tot:>7d} {cov:>7d} {tot-cov:>5d}   "
          f"({100*cov/max(1,tot):.0f}% of result-like numbers re-derived)")

    if a.uncovered:
        print("\nUNBACKED (no harness check re-derives this value):")
        for k in sorted(per_section, key=lambda x: -(per_section[x]["n"] - per_section[x]["cov"])):
            r = per_section[k]
            if not r["miss"]:
                continue
            print(f"\n  {k}")
            for ln, raw, ctx in r["miss"][:14]:
                print(f"    L{ln:<5d} {raw:>8s}  {ctx}")
            if len(r["miss"]) > 14:
                print(f"    ... and {len(r['miss'])-14} more")

    if a.dupes:
        print("\nSAME VALUE IN MANY PLACES (check they mean the same thing):")
        for v, sites in sorted(value_sites.items(), key=lambda x: -len(x[1]))[:12]:
            if len(sites) < 3:
                continue
            where = ", ".join(f"{s}:{ln}" for s, ln, _ in sites[:6])
            print(f"  {v}  x{len(sites):<3d} {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
