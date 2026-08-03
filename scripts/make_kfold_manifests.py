"""Generate subject-disjoint K-fold manifests for Pediatric-RR cross-validation.

Every subject appears in exactly ONE test fold, so pooling the K test folds yields
one prediction per subject over the WHOLE cohort (~5x the effective test-n of the
single 50-clip split) and removes the selection-on-test leakage. Per fold:
  test  = fold k subjects
  val   = deterministic val-frac slice of the remaining subjects (for conformal)
  train = the rest
A shared label-free SSL pretrain manifest (all subjects) is also written; SSL is
unlabeled so pretraining on all clips does not leak labels into any test fold.

Usage:
  python scripts/make_kfold_manifests.py --k 5 --seed 100 \
      --signal-dir data/pedrr_signals_learned --out-dir data/organized/kfold_learned
"""
from __future__ import annotations
import argparse, json, random, collections
from pathlib import Path


def _signal_name(rec: dict) -> str:
    # matches the {dataset}_{subject}_{video}.npz provenance contract
    ds = str(rec.get("dataset", "Pediatric-RR"))
    return f"{ds}_{rec['subject_id']}_{rec['video_id']}.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=[
        "data/organized/pedrr_indist_train.json",
        "data/organized/pedrr_indist_val.json",
        "data/organized/pedrr_indist_test.json"])
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out-dir", default="data/organized/kfold_learned")
    args = ap.parse_args()

    sig_dir = Path(args.signal_dir)
    recs = []
    for m in args.manifests:
        recs += json.load(open(m))
    # keep only clips whose signal actually exists (robust to missing extractions)
    kept = [r for r in recs if (sig_dir / _signal_name(r)).exists()]
    dropped = len(recs) - len(kept)

    by_subj = collections.OrderedDict()
    for r in kept:
        by_subj.setdefault(r["subject_id"], []).append(r)
    subjects = sorted(by_subj)
    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    folds = [subjects[i::args.k] for i in range(args.k)]  # round-robin -> balanced folds

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, subjs) -> int:
        rr = [r for s in subjs for r in by_subj[s]]
        json.dump(rr, open(path, "w"))
        return len(rr)

    # shared label-free SSL pretrain manifest (all subjects)
    n_pre_val = max(5, int(0.1 * len(subjects)))
    dump(out / "pretrain_train.json", subjects)
    dump(out / "pretrain_val.json", subjects[:n_pre_val])

    print(f"K={args.k} seed={args.seed} subjects={len(subjects)} (dropped {dropped} clips w/o signal)")
    total_test = 0
    for k in range(args.k):
        test_subj = folds[k]
        test_set = set(test_subj)
        rest = [s for s in subjects if s not in test_set]
        n_val = max(3, int(args.val_frac * len(rest)))
        val_subj, train_subj = rest[:n_val], rest[n_val:]
        assert not (set(train_subj) & set(val_subj)) and not (set(train_subj) & test_set) \
            and not (set(val_subj) & test_set), "folds must be subject-disjoint"
        fd = out / f"fold{k}"; fd.mkdir(exist_ok=True)
        nt = dump(fd / "train.json", train_subj)
        nv = dump(fd / "val.json", val_subj)
        nte = dump(fd / "test.json", test_subj)
        total_test += nte
        print(f"  fold{k}: train={nt} val={nv} test={nte}")
    print(f"  pooled test coverage = {total_test} clips (== full cohort, each subject tested once)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
