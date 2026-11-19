"""Re-judge a finished run against the CURRENT rubric of a case (no actor, no environment calls).

python scripts/rejudge_run.py --run-dir runs/<case>/<stamp>/run_0 --case cases/<case>/case.yaml \
    --judge-model claude-opus-5-5

The lab's end state is rebuilt from lab_state.json; the original result.json is kept and the new judgment is written
to rejudge-<stamp>.json in the run directory.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.case import Case                              # noqa: E402
from twinlab.judge import audit_actor, run_judge, score    # noqa: E402
from twinlab.lab import Lab                                # noqa: E402
from twinlab.outcome import compute as compute_outcome     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--validity-only", action="store_true",
                    help="take the environment audit from the new judgment but keep rubric points and traps from the "
                         "run's latest judgment (newest rejudge-*.json, else result.json), which must be on the "
                         "current rubric")
    a = ap.parse_args()
    cfg = json.loads((a.run_dir / "run_config.json").read_text())
    case = Case(a.case, cfg["checkpoints"])
    st = json.loads((a.run_dir / "lab_state.json").read_text())
    assert st["fatal"] is None, f"run ended on a fatal error: {st['fatal'][:200]}"
    lab = Lab(case, None, a.run_dir, 0)
    for k in ("cost_used", "consumed", "n_requests", "report", "closed_at"):
        setattr(lab, k, st[k])
    assert lab.report is not None, "no final report in this run"
    audit = audit_actor(case, a.run_dir)
    outcome = compute_outcome(case, a.run_dir, lab)
    j = run_judge(case, a.run_dir, lab, a.judge_model, outcome, cfg["actor_agent"])
    if a.validity_only:
        prev_files = sorted(a.run_dir.glob("rejudge-*.json")) or [a.run_dir / "result.json"]
        prev = json.loads(prev_files[-1].read_text())
        assert prev["score"]["rubric_sha256"] in case.rubric_sha_accepted, \
            f"{prev_files[-1].name} is on rubric {prev['score']['rubric_sha256']}, current {case.rubric_sha256}"
        cur_traps = sorted(t["id"] for t in case.d["traps"])
        assert sorted(prev["judge"]["traps"]) == cur_traps, \
            f"{prev_files[-1].name} traps {sorted(prev['judge']['traps'])} differ from current {cur_traps}"
        print(f"[rejudge] validity only: rubric points and traps kept from {prev_files[-1].name}")
        j = {**j, "rubric": prev["judge"]["rubric"], "traps": prev["judge"]["traps"],
             "validity_only_from": prev_files[-1].name}
    s = score(case, j, lab)
    res = {"run": str(a.run_dir), "case": str(a.case), "score": s, "audit": audit, "judge": j,
           "final_outcome": outcome, "valid": s["env_valid"] and not audit["contaminated"]}
    out = a.run_dir / f"rejudge-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({"out": str(out), "derived_100": s["derived_100"], "total_100": s["total_100"],
                      "points_100": s["points_100"], "total_native": s["total"], "max_native": s["max_native"],
                      "pass": s["pass"], "components": s["components"],
                      "valid": res["valid"], "refusals": j["env_refusals"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
