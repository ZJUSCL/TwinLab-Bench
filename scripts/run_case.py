"""Evaluate one actor on one case, n times.

python scripts/run_case.py --case cases/001-coral-lcm-porites-01/case.yaml --n-runs 1 \
    --actor-agent claude-code --actor-model claude-opus-5-5 \
    --env-model claude-sonnet-5-5 --judge-model claude-opus-5-5 \
    --actor-timeout-s 10800 --checkpoints case --out runs

--actor-agent codex runs Codex with the provider of machine.yaml (e.g. --actor-model deepseek-chat).
--checkpoints: case = the case's own setting; on/off override it (2x2 design).
Output: <out>/<case_id>/<timestamp>/run_<i>/ and summary.json. A run that stops on an error is resumable with
scripts/resume_run.py; the other runs continue.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.actors import AGENTS          # noqa: E402
from twinlab.case import CHECKPOINT_MODES, Case   # noqa: E402
from twinlab.run import one_run, run_config, summarize   # noqa: E402

ACTOR_EFFORTS = ["low", "medium", "high", "xhigh", "max"]   # claude --effort levels (CLI 2.1.296 --help)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--n-runs", required=True, type=int)
    ap.add_argument("--actor-agent", required=True, choices=sorted(AGENTS))
    ap.add_argument("--actor-model", required=True)
    ap.add_argument("--actor-effort", choices=ACTOR_EFFORTS,
                    help="claude-code only: passed as --effort; omitted = the CLI's default effort")
    ap.add_argument("--env-model", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--actor-timeout-s", required=True, type=int)
    ap.add_argument("--checkpoints", required=True, choices=CHECKPOINT_MODES)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    case = Case(a.case, a.checkpoints)
    root = (a.out / case.id / time.strftime("%Y%m%d-%H%M%S")).resolve()
    # the environment and the judge get their own /tmp (twinlab.claude_cli.private_tmp_prefix), so the run directory
    # must not be under the host /tmp
    assert not root.is_relative_to("/tmp"), f"--out {a.out} is under /tmp; put the runs (and the repository) elsewhere"
    print(f"[run_case] {root}", flush=True)
    failed = []
    for i in range(a.n_runs):
        cfg = run_config(case, i, a.actor_agent, a.actor_model, a.env_model, a.judge_model, a.actor_timeout_s,
                         a.checkpoints, a.actor_effort)
        try:
            one_run(case, root / f"run_{i}", cfg)
        except Exception as e:   # keep going with the other runs; lab state is saved after every call -> resumable
            print(f"[run] run_{i} STOPPED: {str(e).splitlines()[0]}", flush=True)
            failed.append(f"run_{i}")
    summarize(root)
    if failed:
        print(f"[run] resumable runs: {failed}\n  python scripts/resume_run.py --run-dir {root}/<run>", flush=True)


if __name__ == "__main__":
    main()
