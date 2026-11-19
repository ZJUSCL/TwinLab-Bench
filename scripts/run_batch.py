"""Evaluate one actor on many cases: run_case.py per case (up to --jobs at once), then the environment check of
every run, then one table.

python scripts/run_batch.py --cases cases/02*/case.yaml --n-runs 1 \
    --actor-agent codex --actor-model deepseek-chat \
    --env-model claude-sonnet-5-5 --judge-model claude-opus-5-5 \
    --actor-timeout-s 10800 --checkpoints case --jobs 2 --out runs --env-check-scratch /tmp/twinlab_env_check

Output: the usual run dirs under <out>/<case_id>/, plus <out>/batches/<timestamp>/ with one log per case and
attempt and batch_summary.json. Invalid runs are rerun up to MAX_ATTEMPTS times; see run_one.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # run from a checkout without installing
from twinlab.actors import AGENTS                 # noqa: E402
from twinlab.case import CHECKPOINT_MODES, Case   # noqa: E402
from twinlab.env_check import check_env           # noqa: E402
from twinlab.scale import round1                  # noqa: E402

ACTOR_EFFORTS = ["low", "medium", "high", "xhigh", "max"]   # same list as run_case.py (claude --effort levels)
ROOT_LINE = "[run_case] "   # run_case.py prints its output dir on this line (after the [case] line)
MAX_ATTEMPTS = 3            # run_case.py calls per case while runs are judged invalid; then the missing runs score 0


def run_attempt(case: Case, a, n_runs: int, log: Path) -> dict:
    """One run_case.py call with n_runs runs, then the environment check of each run."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts/run_case.py"), "--case", str(case.path),
           "--n-runs", str(n_runs), "--actor-agent", a.actor_agent, "--actor-model", a.actor_model,
           "--env-model", a.env_model, "--judge-model", a.judge_model, "--actor-timeout-s", str(a.actor_timeout_s),
           "--checkpoints", a.checkpoints, "--out", str(a.out),
           *(["--actor-effort", a.actor_effort] if a.actor_effort is not None else [])]
    print(f"[batch] start {case.id} (log {log})", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    roots = [line[len(ROOT_LINE):] for line in log.read_text().splitlines() if line.startswith(ROOT_LINE)]
    assert len(roots) <= 1, f"{log}: several output dirs {roots}"
    root = Path(roots[0]) if roots else None
    att = {"rc": rc, "root": str(root), "log": str(log)}
    if root is None:
        print(f"[batch] {case.id} FAILED before any run (rc={rc}); see {log}", flush=True)
        return att
    checks = {}
    for run_dir in sorted(root.glob("run_*")):
        if (run_dir / "env_session.jsonl").is_file():
            rep = check_env(run_dir, a.env_check_scratch / case.id / root.name / run_dir.name,
                            run_dir / "env_session.jsonl")
            checks[run_dir.name] = {"problems": len(rep["problems"]), "to_read": len(rep["to_read"])}
    summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").is_file() else {}
    att.update(unfinished=summary.get("unfinished"), env_check=checks,
               results=[json.loads(p.read_text()) for p in sorted(root.glob("run_*/result.json"))])
    return att


def run_one(case: Case, a, logs: Path) -> dict:
    """a.n_runs valid runs of a case. Runs judged invalid (environment flag or contamination) are rerun, up to
    MAX_ATTEMPTS run_case.py calls in all; a slot still without a valid run after that scores 0 (derived_100 0,
    pass false) and keeps its place in the mean (user 2026-11-18). A stopped run (rc != 0 or unfinished) ends the
    attempts: it is resumed by hand, never rerun."""
    valid, invalid, attempts = [], [], []
    for k in range(1, MAX_ATTEMPTS + 1):
        missing = a.n_runs - len(valid)
        log = logs / (f"{case.id}.log" if k == 1 else f"{case.id}.attempt{k}.log")
        att = run_attempt(case, a, missing, log)
        attempts.append({x: att.get(x) for x in ("rc", "root", "log", "unfinished", "env_check")})
        for r in att.get("results", []):
            (valid if r["valid"] else invalid).append({"root": att["root"], "run": r["run"], "score": r["score"]})
        if att["rc"] != 0 or att.get("unfinished") or att["root"] == "None":
            break
        if len(valid) == a.n_runs:
            break
        print(f"[batch] {case.id}: attempt {k} left {a.n_runs - len(valid)} of {a.n_runs} runs without a valid "
              f"result" + ("; rerunning" if k < MAX_ATTEMPTS else f"; {MAX_ATTEMPTS} attempts used, scored 0"),
              flush=True)
    last = attempts[-1]
    stopped = last["rc"] != 0 or bool(last["unfinished"]) or last["root"] == "None"
    row = {"case": case.id, "rc": last["rc"], "root": last["root"], "log": last["log"], "unfinished": last["unfinished"],
           "attempts": attempts, "n_runs": a.n_runs, "n_valid": len(valid),
           "n_invalid_runs": len(invalid), "invalid_runs": [f"{x['root']}/{x['run']}" for x in invalid]}
    if stopped:   # not final: the stopped run is resumed by hand
        row.update({"n_scored_0": None, "pass@1": None, "mean_score": None, "mean_cost_used": None})
    else:
        n0 = a.n_runs - len(valid)
        row.update({"n_scored_0": n0,
                    "pass@1": sum(v["score"]["pass"] for v in valid) / a.n_runs,
                    "mean_score": round1(sum(v["score"]["derived_100"] for v in valid) / a.n_runs),
                    "mean_cost_used": (sum(v["score"]["cost_used"] for v in valid) / len(valid)) if valid else None})
    print(f"[batch] done {case.id}: attempts={len(attempts)} pass@1={row['pass@1']} mean_score={row['mean_score']} "
          f"valid={len(valid)}/{a.n_runs} invalid_runs={len(invalid)} scored_0={row['n_scored_0']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, nargs="+", type=Path, help="case.yaml paths (shell globs are fine)")
    ap.add_argument("--n-runs", required=True, type=int)
    ap.add_argument("--actor-agent", required=True, choices=sorted(AGENTS))
    ap.add_argument("--actor-model", required=True)
    ap.add_argument("--actor-effort", choices=ACTOR_EFFORTS,
                    help="omit for the CLI default of this model (machine.yaml actor_default_effort)")
    ap.add_argument("--env-model", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--actor-timeout-s", required=True, type=int)
    ap.add_argument("--checkpoints", required=True, choices=CHECKPOINT_MODES)
    ap.add_argument("--jobs", required=True, type=int, help="cases evaluated at the same time")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--env-check-scratch", required=True, type=Path, help="dir outside runs/ and cases/")
    a = ap.parse_args()
    a.out = a.out.resolve()
    cases = [Case(p, a.checkpoints) for p in a.cases]   # every case must load before anything runs
    batch = a.out / "batches" / time.strftime("%Y%m%d-%H%M%S")
    (batch / "logs").mkdir(parents=True)
    (batch / "batch_config.json").write_text(json.dumps({**vars(a), "cases": [str(c.path) for c in cases]},
                                                        indent=1, default=str))
    print(f"[batch] {len(cases)} cases, {a.jobs} at a time -> {batch}", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(lambda c: run_one(c, a, batch / "logs"), cases))
    (batch / "batch_summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{'case':40s} {'rc':>3s} {'valid':>7s} {'tries':>5s} {'pass@1':>7s} {'score':>7s} {'cost':>7s}")
    for r in rows:
        print(f"{r['case']:40s} {r['rc']:3d} {str(r['n_valid']) + '/' + str(r['n_runs']):>7s} "
              f"{len(r['attempts']):5d} {str(r['pass@1']):>7s} {str(r['mean_score']):>7s} "
              f"{str(r['mean_cost_used']):>7s}")
    done = [r for r in rows if r["mean_score"] is not None]
    if len(done) == len(rows):   # every case final: invalid-after-retries slots are in as 0, the divisor is all cases
        print(f"[batch] mean score over {len(rows)} cases: {round1(sum(r['mean_score'] for r in rows) / len(rows))}")
    print(f"[batch] summary: {batch / 'batch_summary.json'}")
    # a stopped run (gateway outage, harness assert, timeout) is reported and the batch exits non-zero, so the caller
    # sees it (stop on errors with the log, resume by hand; no automatic resume)
    stopped = [r for r in rows if r["rc"] != 0 or r.get("unfinished")]
    if stopped:
        print(f"\n[batch] {len(stopped)} case(s) did not finish:", flush=True)
        for r in stopped:
            print(f"  {r['case']}: rc={r['rc']} unfinished={r.get('unfinished')} log={r['log']} root={r['root']}")
            print(f"    resume: python scripts/resume_run.py --run-dir {r['root']}/<run>")
        sys.exit(1)


if __name__ == "__main__":
    main()
