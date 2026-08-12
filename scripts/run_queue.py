# RUN: C:\Users\rimon\anaconda3\envs\resp_env\python.exe scripts\run_queue.py --concurrency 2
#      (MASTER launcher: waits for the two in-flight agefix retrains to finish, then drains the #18
#       data-efficiency jobs + the salience/ablation retrains through a 2-wide GPU pool, then summarizes.)
#      Custom job list:   ... scripts\run_queue.py --jobs-file myjobs.jsonl
#      Preview the plan:  ... scripts\run_queue.py --dry-run
"""
Generic N-WIDE (default 2) job queue for GPU retrains on the single RTX 8000.

WHY 2-WIDE: the box is COMPUTE-bound, not memory-bound (46 GB free). One PRISM 5-fold run is ~2.5-3 h;
two concurrent trainings already saturate the GPU (~93% util). So the right pattern is to keep EXACTLY
2 single-process trainings running and feed the rest as slots free -- which is what this queue does.

FLOW
----
  1. WAIT for the two in-flight Task-Scheduler retrains to finish, by polling for the literal marker
     'DONE_EXITCODE' in BOTH outputs/agefix_run.log AND outputs/agefix_clean_run.log. This guarantees
     we never oversubscribe the GPU (2 agefix + queue jobs). (--no-wait skips; --wait-timeout bounds it.)
  2. BUILD the job list. Default: concatenate the JSONL emitted by
       scripts/data_efficiency.py --emit-jobs           (the #18 configs)
       scripts/retrain_salience_ablations.py --emit-jobs (the salience + ablation variants)
     (toggle either off with --no-data-efficiency / --no-variants; or pass --jobs-file to override).
  3. DRAIN the list keeping at most --concurrency jobs alive. Each job is ONE subprocess (a
     single-process launcher that itself runs pretrain->train->eval sequentially, num_workers=0), logged
     to outputs/queue/<name>.log. On success a outputs/queue/<name>.done marker is written; on failure a
     <name>.failed marker. RE-RUNNING resumes: jobs whose .done exists are skipped, and each underlying
     script is independently resume-safe, so partially-finished variants continue where they left off.
  4. After the pool empties, run the final summary job (scripts/data_efficiency.py --summarize) unless
     --no-summary.

JOBS FILE FORMAT (--jobs-file): one job per line. Either a JSON object {"name","argv":[...],"cwd"?}
(preferred; no shell quoting headaches on Windows paths with spaces) or a plain shell command string
(run with shell=True). Blank lines and lines starting with '#' are ignored.

TIME (2-wide): #18 data-efficiency ~12-24 h; salience+ablation ~25-40 h; the full master queue is the
two sharing the pool, ~40-60 h wall-clock. THIS SANDBOX CANNOT TRAIN (CUDA spawn / WinError 5) -- run
this on the real GPU box.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIR = ROOT / "outputs" / "queue"
DEFAULT_MARKERS = [ROOT / "outputs" / "agefix_run.log", ROOT / "outputs" / "agefix_clean_run.log"]
DONE_TOKEN = "DONE_EXITCODE"


# --------------------------------------------------------------------------- #
def child_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")  # survive Windows console-close events
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


def marker_present(path: Path) -> bool:
    try:
        return DONE_TOKEN in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def wait_for_markers(markers: List[Path], timeout: float, poll: float) -> bool:
    start = time.time()
    while True:
        pending = [m for m in markers if not marker_present(m)]
        if not pending:
            print(f"[queue] all in-flight retrains done ({DONE_TOKEN} seen in every marker).", flush=True)
            return True
        waited = time.time() - start
        if timeout and waited > timeout:
            print(f"[queue] WAIT TIMEOUT after {waited/3600:.1f} h; still pending: "
                  f"{[str(p) for p in pending]}", flush=True)
            return False
        print(f"[queue] waiting for {DONE_TOKEN} in: {[p.name for p in pending]} "
              f"(elapsed {waited/60:.1f} min)", flush=True)
        time.sleep(poll)


# --------------------------------------------------------------------------- #
def build_jobs_from_emitter(python: str, script: str, extra: List[str]) -> List[Dict]:
    cmd = [python, script, "--emit-jobs", "--python", python] + extra
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"[queue] emitter failed: {script} (rc={proc.returncode})")
    jobs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        jobs.append(json.loads(line))
    return jobs


def parse_jobs_file(path: Path) -> List[Dict]:
    jobs = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            jobs.append(json.loads(line))
        else:
            jobs.append({"name": f"job{i:04d}", "shell": line, "cwd": str(ROOT)})
    return jobs


# --------------------------------------------------------------------------- #
class RunningJob:
    def __init__(self, job: Dict, proc: subprocess.Popen, log_fh):
        self.job = job
        self.proc = proc
        self.log_fh = log_fh
        self.start = time.time()


def launch(job: Dict) -> RunningJob:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    name = job["name"]
    log_path = QUEUE_DIR / f"{name}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    cwd = job.get("cwd", str(ROOT))
    header = f"# {name}\n# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if "argv" in job:
        header += "# argv: " + " ".join(str(c) for c in job["argv"]) + "\n"
        log_fh.write(header); log_fh.flush()
        proc = subprocess.Popen([str(c) for c in job["argv"]], cwd=cwd, env=child_env(),
                                stdout=log_fh, stderr=subprocess.STDOUT)
    else:
        header += "# shell: " + job["shell"] + "\n"
        log_fh.write(header); log_fh.flush()
        proc = subprocess.Popen(job["shell"], cwd=cwd, env=child_env(), shell=True,
                                stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"[queue] START {name}  -> {log_path.name}", flush=True)
    return RunningJob(job, proc, log_fh)


def drain(jobs: List[Dict], concurrency: int, poll: float, overwrite: bool) -> Dict[str, int]:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: List[RunningJob] = []
    done, failed, skipped = [], [], []

    def is_done(name: str) -> bool:
        return (QUEUE_DIR / f"{name}.done").exists()

    while pending or running:
        # fill free slots
        while pending and len(running) < concurrency:
            job = pending.pop(0)
            name = job["name"]
            if is_done(name) and not overwrite:
                print(f"[queue] SKIP {name} (.done exists)", flush=True)
                skipped.append(name)
                continue
            # clear any stale failure marker before a retry
            (QUEUE_DIR / f"{name}.failed").unlink(missing_ok=True)
            running.append(launch(job))
        # poll running
        still: List[RunningJob] = []
        for rj in running:
            rc = rj.proc.poll()
            if rc is None:
                still.append(rj)
                continue
            rj.log_fh.close()
            name = rj.job["name"]
            mins = (time.time() - rj.start) / 60.0
            if rc == 0:
                (QUEUE_DIR / f"{name}.done").write_text("ok\n", encoding="utf-8")
                print(f"[queue] DONE  {name}  ({mins:.1f} min)", flush=True)
                done.append(name)
            else:
                (QUEUE_DIR / f"{name}.failed").write_text(f"rc={rc}\n", encoding="utf-8")
                print(f"[queue] FAIL  {name}  (rc={rc}, {mins:.1f} min) -> see {name}.log", flush=True)
                failed.append(name)
        running = still
        if running or pending:
            time.sleep(poll)

    return {"done": done, "failed": failed, "skipped": skipped}


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, default=2, help="max concurrent jobs (GPU is compute-bound: keep 2)")
    ap.add_argument("--jobs-file", default="", help="JSONL / shell job list (overrides the default builders)")
    ap.add_argument("--no-data-efficiency", action="store_true", help="exclude the #18 jobs")
    ap.add_argument("--no-variants", action="store_true", help="exclude the salience/ablation jobs")
    ap.add_argument("--variants", default="all", help="passed to retrain_salience_ablations --variants")
    ap.add_argument("--include-deprecated", action="store_true", help="also emit the deprecated v3 variant")
    ap.add_argument("--no-summary", action="store_true", help="skip the final data_efficiency --summarize")
    ap.add_argument("--no-wait", action="store_true", help="do NOT wait for the agefix markers first")
    ap.add_argument("--markers", default="", help="comma list of marker files (default the two agefix logs)")
    ap.add_argument("--wait-timeout", type=float, default=0.0, help="seconds to wait for markers (0=forever)")
    ap.add_argument("--wait-poll", type=float, default=30.0)
    ap.add_argument("--poll", type=float, default=5.0, help="job-completion poll interval (s)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--overwrite", action="store_true", help="ignore queue .done markers and relaunch")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # ---- assemble jobs ----------------------------------------------------- #
    if args.jobs_file:
        jobs = parse_jobs_file(Path(args.jobs_file))
    else:
        jobs = []
        if not args.no_data_efficiency:
            jobs += build_jobs_from_emitter(args.python, "scripts/data_efficiency.py", [])
        if not args.no_variants:
            extra = ["--variants", args.variants]
            if args.include_deprecated:
                extra += ["--include-deprecated"]
            jobs += build_jobs_from_emitter(args.python, "scripts/retrain_salience_ablations.py", extra)

    if not jobs:
        print("[queue] no jobs to run.", flush=True)
        return 0

    print(f"[queue] {len(jobs)} jobs, concurrency={args.concurrency}", flush=True)
    if args.dry_run:
        for j in jobs:
            cmd = " ".join(str(c) for c in j["argv"]) if "argv" in j else j.get("shell", "")
            print(f"  {j['name']}: {cmd}")
        if not args.no_summary:
            print(f"  (final) de_summary: {args.python} scripts/data_efficiency.py --summarize")
        return 0

    # ---- wait for the in-flight retrains ----------------------------------- #
    if not args.no_wait:
        markers = ([Path(m.strip()) for m in args.markers.split(",") if m.strip()]
                   if args.markers else DEFAULT_MARKERS)
        ok = wait_for_markers(markers, args.wait_timeout, args.wait_poll)
        if not ok:
            print("[queue] aborting: markers not satisfied (pass --no-wait to override).", flush=True)
            return 2

    # ---- drain the pool ---------------------------------------------------- #
    result = drain(jobs, args.concurrency, args.poll, args.overwrite)

    # ---- final summary ----------------------------------------------------- #
    if not args.no_summary and not args.no_data_efficiency:
        print("[queue] running final data-efficiency summary", flush=True)
        subprocess.run([args.python, "scripts/data_efficiency.py", "--summarize"],
                       cwd=str(ROOT), env=child_env())

    print(f"\n[queue] FINISHED  done={len(result['done'])} "
          f"failed={len(result['failed'])} skipped={len(result['skipped'])}", flush=True)
    if result["failed"]:
        print("[queue] FAILURES (inspect outputs/queue/<name>.log):", flush=True)
        for n in result["failed"]:
            print(f"   - {n}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
