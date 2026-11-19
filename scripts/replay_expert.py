"""Expert replay: feed the original expert's trajectory (case.expert_reference_trajectory) through the environment
and the judge, without an actor. The expert must pass; if not, the environment or the rubric is broken (never the
trajectory).

python scripts/replay_expert.py --case cases/001-coral-lcm-porites-01/case.yaml --env-model claude-sonnet-5-5 \
    --judge-model claude-opus-5-5 --seed 0 --checkpoints case --out runs

Output: <out>/<case_id>/replay-<timestamp>/replay_result.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.case import CHECKPOINT_MODES, Case    # noqa: E402
from twinlab.env_agent import EnvAgent             # noqa: E402
from twinlab.judge import run_judge, score         # noqa: E402
from twinlab.lab import Lab                        # noqa: E402
from twinlab.outcome import compute as compute_outcome   # noqa: E402

MAX_CHECKPOINT_HOPS = 8        # max resubmissions of deferred steps for one trajectory request
OFFLINE_COMMAND_TIMEOUT_S = 120
os.environ["TWINLAB_ROOT"] = str(Path(__file__).resolve().parents[1])   # offline_commands refer to the checkout
STREAM_FORMAT = "claude-code"  # the offline commands are recorded as a Claude Code stream for the judge


def run_offline_commands(case: Case, run_dir: Path, commands: list):
    """The expert's own analysis of the workspace files, executed for real in a freshly built actor workspace and
    recorded in actor-stream format, so the judge sees exactly what the commands showed."""
    if not commands:
        return
    ws = run_dir / "actor_workspace"
    case.write_actor_workspace(ws)
    append_offline_commands(run_dir, commands, "replay")


def append_offline_commands(run_dir: Path, commands: list, tag: str):
    """Run commands in the replay's actor workspace (which by then may hold files delivered by earlier requests,
    058-02) and append them to the replay stream."""
    ws = run_dir / "actor_workspace"
    stream = run_dir / "actor_stream_replay_0.jsonl"
    with open(stream, "a") as f:
        for i, cmd in enumerate(commands):
            p = subprocess.run(["bash", "-c", cmd], cwd=ws, capture_output=True, text=True,
                               timeout=OFFLINE_COMMAND_TIMEOUT_S)
            assert p.returncode in (0, 1), f"offline command failed rc={p.returncode}: {cmd}\n{p.stderr}"
            out = (p.stdout + p.stderr).strip()
            tid = f"{tag}_{i}"
            f.write(json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": cmd}}]}}) + "\n")
            f.write(json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": tid, "content": out}]}}) + "\n")
            print(f"[offline] $ {cmd}\n{out[:800]}\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--env-model", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--checkpoints", required=True, choices=CHECKPOINT_MODES)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    case = Case(a.case, a.checkpoints)
    run_dir = (a.out / case.id / ("replay-" + time.strftime("%Y%m%d-%H%M%S"))).resolve()
    run_dir.mkdir(parents=True)
    lab = Lab(case, EnvAgent(case, run_dir, a.env_model, resume=False), run_dir, a.seed)
    traj = case.d["expert_reference_trajectory"]
    run_offline_commands(case, run_dir, traj.get("offline_commands", []))
    # optional: the expert's analysis of files that a request delivered, run right after that request (1-based)
    after = {int(k): v for k, v in traj.get("offline_after_requests", {}).items()}
    assert set(after) <= set(range(1, len(traj["requests"]) + 1)), sorted(after)
    for k, req in enumerate(traj["requests"], start=1):
        # with checkpoints enforced, the expert resubmits the deferred steps until the request is complete
        for _ in range(MAX_CHECKPOINT_HOPS):
            res = lab.call("lab", {"request": req})
            assert lab.fatal is None and not res["isError"], f"replay broke at: {req}\n{res}"
            print(res["content"][0]["text"][:600], "\n", flush=True)
            deferred = lab.last_deferred_steps
            if not deferred:
                break
            req = "Continue with the remaining steps of the previous request: " + "; ".join(deferred)
        else:
            raise AssertionError(f"request still had deferred steps after {MAX_CHECKPOINT_HOPS} hops: {req}")
        if k in after:
            append_offline_commands(run_dir, after[k], f"replay_after{k}")
    lab.call("submit_final_report", traj["final_report"])
    if lab.env.session_created:   # an expert path with no request has no env session to archive (as twinlab/run.py)
        lab.env.archive_session(run_dir)
    outcome = compute_outcome(case, run_dir, lab)
    j = run_judge(case, run_dir, lab, a.judge_model, outcome, STREAM_FORMAT)
    s = score(case, j, lab)
    res = {"score": s, "judge": j, "final_outcome": outcome, "checkpoints_enforced": case.checkpoints["enforce"]}
    (run_dir / "replay_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(s, indent=1))
    print("EXPERT REPLAY", "PASS" if s["pass"] and s["env_valid"] else "FAIL", run_dir)


if __name__ == "__main__":
    main()
