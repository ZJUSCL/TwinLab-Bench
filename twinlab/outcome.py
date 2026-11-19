"""Objective outcome of the actor's final answer, for campaign cases.

A campaign case asks the actor to deliver a recipe, not a diagnosis, so the score needs one objective number that
does not depend on the judge reading prose. The case declares:

  final_outcome:
    command: ["python3", "calculators.py", "outcome"]     # run in env_ws, --spec <json> appended
    from_report_fields: [final_pH, final_salt_mM, ...]    # final_report fields, passed through verbatim

The calculator is executed ONCE (see the one-run-representative rule: no averaging of repeats; a noise-free number
comes from the calculator's underlying value, not from replicates). Its stdout must be JSON. Anything unexpected is
an error, never a fallback, because a silently missing outcome would be scored as a zero the actor did not earn.
"""
import json
import subprocess

from .env_agent import CALL_TIMEOUT_S

KEY = "final_outcome"


def has_outcome(case) -> bool:
    return KEY in case.d


def compute(case, run_dir, lab):
    """Run the case's outcome calculator on the submitted report fields. None if the case has no final_outcome."""
    if not has_outcome(case):
        return None
    spec = case.d[KEY]
    if lab.report is None:
        return {"submitted": False, "reason": "no final report"}
    missing = [f for f in spec["from_report_fields"] if f not in lab.report]
    if missing:
        return {"submitted": True, "computed": False, "reason": f"report fields missing: {missing}"}
    args = {f: lab.report[f] for f in spec["from_report_fields"]}
    cmd = [*spec["command"], "--spec", json.dumps(args)]
    ws = run_dir / "env_ws"
    assert ws.is_dir(), ws
    p = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=CALL_TIMEOUT_S)
    assert p.returncode == 0, f"final_outcome calculator failed: {p.returncode}\n{p.stderr[-4000:]}"
    out = json.loads(p.stdout)
    print(f"[outcome] {json.dumps(out, ensure_ascii=False)}", flush=True)
    return {"submitted": True, "computed": True, "fields": args, **out}
