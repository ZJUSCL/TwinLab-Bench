"""Memorization probes: run before evaluation. If the model reproduces case-specific numbers, the case is burned.

python scripts/probe_memorization.py --case cases/001-coral-lcm-porites-01/case.yaml --model claude-opus-5-5 --out runs
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.case import Case                      # noqa: E402
from twinlab.claude_cli import run_claude_json     # noqa: E402

CALL_TIMEOUT_S = 900

# probes must answer from memory only: no file, shell, web, or agent tools (--tools "" alone still left some)
PROBE_DENY = "Bash Read Glob Grep Edit Write WebSearch WebFetch Task Agent NotebookEdit ToolSearch"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", type=Path)
    src.add_argument("--prompts", type=Path,
                     help="pre-screen a candidate before a case exists: JSON {id, probes: [{prompt, leak_if_contains}]}")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.case:
        case = Case(a.case, "case")   # probes do not use the lab
        cid, probes = case.id, case.d["meta"]["contamination_controls"]["memorization_probes"]
    else:
        spec = json.loads(a.prompts.read_text())
        cid, probes = spec["id"], spec["probes"]
    d = (a.out / cid / ("probe-" + time.strftime("%Y%m%d-%H%M%S"))).resolve()
    d.mkdir(parents=True)
    res = []
    tmp = d / "probe_tmp"
    for n, pr in enumerate(probes):
        # empty working dir inside the call's private /tmp: the model sees /tmp/q<n>, not a path naming the case
        # (Claude Code reports the working directory to the model; 071 probe named its case dir, 2026-10-26)
        cwd = tmp / f"q{n}"
        cwd.mkdir(parents=True)
        out = run_claude_json(["-p", pr["prompt"], "--safe-mode", "--tools", "", "--disallowedTools", PROBE_DENY, "--strict-mcp-config",
                               "--no-session-persistence", "--model", a.model, "--output-format", "json"],
                              cwd=cwd, timeout_s=CALL_TIMEOUT_S, log_path=d / "probe_calls.jsonl", tmp_dir=tmp)
        hits = [s for s in pr["leak_if_contains"] if s in out["result"]]
        res.append({"prompt": pr["prompt"], "answer": out["result"], "leak_hits": hits})
        print(f"[probe] hits={hits} :: {pr['prompt'][:80]}", flush=True)
    summary = {"model": a.model, "burned": any(r["leak_hits"] for r in res), "probes": res}
    (d / "probe_result.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({"burned": summary["burned"], "dir": str(d)}))


if __name__ == "__main__":
    main()
