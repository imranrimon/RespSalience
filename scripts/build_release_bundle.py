"""Assemble the checkpoint release for archiving under a DOI.

What goes in, and why:

  weights/published/     the 30 checkpoints of the model the paper reports (SSL-free, deterministic,
                         six seeds x five folds). These are what a reader needs to reproduce the
                         headline numbers without a GPU.
  weights/ssl_variant/   the 30 checkpoints of the self-supervised variant the paper tested and
                         REJECTED. Included deliberately: the null ablation is a reported result, and
                         a reader cannot check a null without both arms.
  predictions/           per-fold prediction CSVs for both, plus the materialised ensemble. Tiny, and
                         they let every reported metric be recomputed with no model loading at all.
  MANIFEST.csv           one row per file with size and SHA-256, so the archive can be verified.
  README.md              the format, how to load a checkpoint, and what the numbers should come to.

Deliberately NOT included: any dataset, or anything derived from one. The signals are not ours to
redistribute -- Pediatric-RR is public under its own terms, NCHSDB is credentialed access.

Run:  python scripts/build_release_bundle.py --out dist/respsalience-weights-v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "outputs", "paper_runs")

PUBLISHED = [f"pedrr_kfold_salience_nossl332det_seed{i}" for i in range(6)]
SSL_VARIANT = [f"pedrr_kfold_salience_ssl332det_seed{i}" for i in range(6)]
ENSEMBLE = "pedrr_kfold_salience_published"

README = """# RespSalience — trained model weights

Reference weights for *Beyond Respiratory-Rate Regression: Decision-Aware Modeling for
Contactless Pediatric Tachypnea Screening*.

**This is research software and not a medical device.** It has not been cleared by any regulatory
authority and must not inform care for an individual patient. Tachypnea is a screening sign, not a
diagnosis. See the repository's `NOTICE` for the full statement.

## Contents

| Path | What it is |
|---|---|
| `weights/published/` | The model the paper reports: SSL-free, deterministic, 6 seeds x 5 folds = 30 checkpoints. |
| `weights/ssl_variant/` | The self-supervised variant the paper tested and **rejected** (no effect on either metric). Included so the null ablation can be checked — a null cannot be verified from one arm. See `MANIFEST.csv` for exactly which seeds are present. |
| `predictions/` | Per-fold prediction CSVs, plus the seed-ensembled run. Every reported metric can be recomputed from these without loading a model. |
| `MANIFEST.csv` | Size and SHA-256 for every file. |

## Loading a checkpoint

```python
import torch
from train_resp_salience_ssl import Encoder, SalHead     # from the code repository

ck = torch.load("weights/published/seed0_fold0.pt", map_location="cpu", weights_only=False)
enc, head = Encoder(), SalHead()
enc.load_state_dict(ck["encoder_state_dict"])
head.load_state_dict(ck["head_state_dict"])
enc.eval(); head.eval()

cfg = ck["config"]        # bins_bpm, fs_hz, window_samples, channel, bandpass_hz, seed, fold
```

The head emits a salience score per rate bin. Normalise it, then read the tachypnea probability as
the mass at or above the WHO age threshold — the whole point of the paper:

```python
import numpy as np
s = head(enc(signal)).detach().numpy().ravel()
s = s / s.sum()
bins = np.array(cfg["bins_bpm"])
tachy_prob = s[bins >= tau_age].sum()     # tau_age: 60 (<2mo), 50 (2-12mo), 40 (12-60mo)
```

`cfg["bins_bpm"]` is essential: without the rate grid the head's output cannot be interpreted, which
is why it is stored in the file rather than in documentation that can drift from it.

## Preprocessing

Inputs are 1440-sample optical-flow windows at 25 Hz (≈57.6 s), band-passed 0.1–1.7 Hz, detrended and
z-scored — exactly `prep()` in `scripts/train_resp_salience_ssl.py`. Feeding a differently prepared
signal will produce a confident and wrong answer.

## What the weights should reproduce

Six-seed ensemble on the 332-subject Pediatric-RR cohort, subject-disjoint five-fold CV:

| Metric | Value |
|---|---|
| Tachypnea AUROC | 0.927 |
| RR-MAE | 5.00 breaths/min |
| AUROC, <2 months | 0.939 |

Training is deterministic: retraining at a fixed seed reproduces these predictions bit-identically.
Run `python scripts/verify_manuscript_claims.py` in the code repository to re-derive every reported
number (110 checks).

## Datasets

None are included, and none may be redistributed from here. Pediatric-RR-332 is public via Mendeley
Data (`10.17632/72dd3rkttf`); the NCH Sleep DataBank requires PhysioNet credentialed access;
AIR-125/400 comes from its original authors. Each remains under its provider's terms.

## Citation

See `CITATION.cff` in the code repository: https://github.com/imranrimon/RespSalience
"""


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(runs, dest, rows, kind):
    os.makedirs(dest, exist_ok=True)
    n = 0
    for r in runs:
        seed = r.rsplit("_seed", 1)[-1]
        for k in range(5):
            src = os.path.join(RUNS, r, f"fold{k}", "train", "resp_salience_best.pt")
            if not os.path.exists(src):
                print(f"  [warn] missing {src}")
                continue
            out = os.path.join(dest, f"seed{seed}_fold{k}.pt")
            shutil.copy2(src, out)
            rows.append([os.path.relpath(out, os.path.dirname(dest)).replace("\\", "/"),
                         kind, os.path.getsize(out), sha256(out)])
            n += 1
    return n


def collect_preds(runs, dest, rows):
    n = 0
    for r in runs:
        for k in range(5):
            src = os.path.join(RUNS, r, f"fold{k}", "test_eval", "prism_breath_predictions.csv")
            if not os.path.exists(src):
                continue
            d = os.path.join(dest, r)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f"fold{k}.csv")
            shutil.copy2(src, out)
            rows.append([os.path.relpath(out, os.path.dirname(dest)).replace("\\", "/"),
                         "predictions", os.path.getsize(out), sha256(out)])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "respsalience-weights-v1"))
    ap.add_argument("--zip", action="store_true", help="also write a .zip beside the directory")
    a = ap.parse_args()

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)

    rows = []
    n1 = collect(PUBLISHED, os.path.join(a.out, "weights", "published"), rows, "published")
    n2 = collect(SSL_VARIANT, os.path.join(a.out, "weights", "ssl_variant"), rows, "ssl_variant")
    n3 = collect_preds(PUBLISHED + [ENSEMBLE], os.path.join(a.out, "predictions"), rows)

    counts = ("\n## Contents as built\n\n"
              "- published weights: %d checkpoint(s)\n"
              "- ssl_variant weights: %d checkpoint(s)\n"
              "- prediction files: %d\n" % (n1, n2, n3))
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README + counts)
    for extra in ("LICENSE", "NOTICE", "CITATION.cff"):
        src = os.path.join(ROOT, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, extra))

    with open(os.path.join(a.out, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "kind", "bytes", "sha256"])
        w.writerows(sorted(rows))

    total = sum(r[2] for r in rows)
    print(f"published weights : {n1}")
    print(f"ssl variant       : {n2}")
    print(f"prediction files  : {n3}")
    print(f"total             : {total/1e6:.1f} MB in {a.out}")

    if a.zip:
        z = shutil.make_archive(a.out, "zip", a.out)
        print(f"archive           : {z} ({os.path.getsize(z)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
