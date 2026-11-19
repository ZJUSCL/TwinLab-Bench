"""One evaluation run, start to finish. Read this file first: it shows the order in which the parts work together.

  1. EnvAgent (env_agent.py)   the simulated lab's brain: a Claude Code session that knows the hidden truth
  2. Lab (lab.py)              ledger and checks around the environment, served over MCP (mcp_server.py)
  3. actor (actors/)           the evaluated agent, in a sandbox, talking only to the lab
  4. outcome (outcome.py)      campaign cases: the objective result of the submitted recipe
  5. judge (judge.py)          rubric, traps, environment audit, contamination audit, score -> result.json

Files of a run (runs/<case_id>/<timestamp>/run_<i>/): run_config.json, actor_state.json, actor_prompt.md,
actor_stream_*.jsonl, lab_ledger.jsonl, lab_state.json, env_calls.jsonl, env_session.jsonl, judge_prompt.md,
result.json. A run that stops on an error keeps lab_state.json and can be resumed (resume_run).
"""
import json
import time
import uuid
from pathlib import Path

from . import settings
from .actors import make_agent, run_actor
from .actors.base import ACTOR_TIME_LOG
from .case import Case
from .claude_cli import backend_record
from .env_agent import EnvAgent
from .judge import audit_actor, run_judge, score
from .lab import Lab
from .mcp_server import serve
from .outcome import compute as compute_outcome
from .prompts import MESSAGES
from .scale import round1


AGENTS_WITH_EFFORT = ("claude-code", "codex")   # agent CLIs whose reasoning effort is set and recorded


def run_config(case: Case, seed: int, actor_agent: str, actor_model: str, env_model: str, judge_model: str,
               actor_timeout_s, checkpoints: str, actor_effort) -> dict:
    """Everything that defines a run; written to run_config.json and read back by resume and rejudge.
    actor_effort None = the agent CLI's default for this model (machine.yaml actor_default_effort), passed
    explicitly and recorded with actor_effort_source "cli_default"."""
    effort_source = "flag"
    if actor_agent not in AGENTS_WITH_EFFORT:   # a human actor (scripts/human_baseline.py, human_web.py)
        assert actor_effort is None, (actor_agent, actor_effort)
        effort_source = "none"
    elif actor_effort is None:
        assert actor_model in settings.ACTOR_DEFAULT_EFFORT, \
            f"machine.yaml actor_default_effort has no entry for {actor_model}: measure the CLI default first"
        actor_effort, effort_source = settings.ACTOR_DEFAULT_EFFORT[actor_model], "cli_default"
    roles = (("actor", actor_model if actor_agent == "claude-code" else None), ("env", env_model),
             ("judge", judge_model))
    backends = {role: rec for role, m in roles if m is not None for rec in [backend_record(m)] if rec is not None}
    if "actor" in backends:   # the actor runs on its per-run config dir (actor_config), not the backend's
        del backends["actor"]["config_dir"]
    return {"case": str(case.path), "seed": seed, "actor_agent": actor_agent, "actor_model": actor_model,
            "actor_effort": actor_effort, "actor_effort_source": effort_source,
            "env_model": env_model, "judge_model": judge_model, "actor_timeout_s": actor_timeout_s,
            "checkpoints": checkpoints, "checkpoints_enforced": case.checkpoints["enforce"],
            "case_files_sha256": case.files_sha256(),   # checked before every lab request and on resume (lab.py)
            # non-Claude models only (machine.yaml model_backends); Claude-only runs keep the old keys exactly
            **({"model_backends": backends} if backends else {})}


def one_run(case: Case, run_dir: Path, cfg: dict) -> dict:
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=1))
    agent = make_agent(cfg["actor_agent"], run_dir, cfg["actor_effort"])
    (run_dir / "actor_state.json").write_text(json.dumps(
        {"session_id": agent.new_session_id(), "invocations": []}, indent=1))
    env = EnvAgent(case, run_dir, cfg["env_model"], resume=False)
    lab = Lab(case, env, run_dir, cfg["seed"])
    srv, mcp_url = serve(lab)
    print(f"[run] {run_dir.name}: lab MCP on 127.0.0.1:{srv.server_address[1]}, actor {cfg['actor_agent']} {cfg['actor_model']}",
          flush=True)
    agent.prepare(case, cfg["actor_model"])
    prompt = case.actor_prompt()
    (run_dir / "actor_prompt.md").write_text(prompt, encoding="utf-8")
    state = json.loads((run_dir / "actor_state.json").read_text())
    info = run_actor(agent, prompt, cfg["actor_model"], mcp_url, cfg["actor_timeout_s"], lab, state["session_id"],
                     tag="r0", start_with_resume=False)
    srv.shutdown()
    lab.save_state()
    record_actor(run_dir, info)
    return finish_run(case, run_dir, lab, cfg)


def record_actor(run_dir: Path, info: dict):
    """actor_state.json accumulates every actor invocation across resumes."""
    p = run_dir / "actor_state.json"
    st = json.loads(p.read_text())
    st["session_id"] = info["session_id"]   # Codex assigns its session id during the first invocation
    st["invocations"] += info["invocations"]
    p.write_text(json.dumps(st, indent=1))
    return st


def finish_run(case: Case, run_dir: Path, lab: Lab, cfg: dict) -> dict:
    """Common tail of a run and of a resumed run: fail with a resume hint, or audit + judge + score."""
    if lab.fatal is not None:
        (run_dir / "ENV_FATAL.txt").write_text(lab.fatal)
        raise RuntimeError(f"run stopped on an error in {run_dir} (see ENV_FATAL.txt). After fixing the cause:\n"
                           f"  python scripts/resume_run.py --run-dir {run_dir}\n{lab.fatal}")
    actor_st = json.loads((run_dir / "actor_state.json").read_text())
    if lab.env.session_created:   # no lab request at all means no env session to archive
        lab.env.archive_session(run_dir)
    audit = audit_actor(case, run_dir)
    outcome = compute_outcome(case, run_dir, lab)   # campaign cases: objective result of the submitted recipe
    if outcome is not None:
        (run_dir / "final_outcome.json").write_text(json.dumps(outcome, indent=1, ensure_ascii=False))
    j = run_judge(case, run_dir, lab, cfg["judge_model"], outcome, cfg["actor_agent"])
    s = score(case, j, lab)
    res = {"run": run_dir.name, "seed": cfg["seed"], "actor": actor_st, "score": s, "audit": audit, "judge": j,
           "final_outcome": outcome, "valid": s["env_valid"] and not audit["contaminated"]}
    (run_dir / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[run] {run_dir.name}: score_100={s['derived_100']} total_100={s['total_100']} (native {s['total']}"
          f"/{s['max_native']}) pass={s['pass']} valid={res['valid']} "
          f"cost={s['cost_used']} {s['budget_unit']} requests={s['n_requests']}", flush=True)
    return res


def resume_run(run_dir: Path) -> dict:
    """Continue a run that stopped on a harness/environment/actor/judge error, with the current (fixed) code.

    - Lab state (ledger, noise RNG, pending notes, env session) is restored from lab_state.json.
    - env_ws/calc_state is rolled back to its snapshot from before a request that never completed (Lab.restore).
    - The environment session is resumed; it is told that its last (failed) answer was discarded.
    - The actor session is resumed with a notice: its last request was not processed and nothing was charged.
    - If the actor had already submitted its report, only the audit and judge tail is re-run.
    - The case files must match run_config's case_files_sha256 (else the run is invalid: rerun it); the actor gets
      the time limit minus the time its earlier invocations used (actor_time.jsonl).
    """
    cfg = json.loads((run_dir / "run_config.json").read_text())
    case = Case(Path(cfg["case"]), cfg["checkpoints"])
    state_file = run_dir / "lab_state.json"
    # the MCP server writes lab_state.json after each lab tool call: a run whose actor stopped before its first one
    # has none (058-02 mimo 20261105-123443, 018-03 luna); its lab starts fresh
    st = json.loads(state_file.read_text()) if state_file.exists() else None
    if st is None:
        assert not (run_dir / "lab_ledger.jsonl").exists() or not (run_dir / "lab_ledger.jsonl").read_text().strip(), \
            f"{run_dir}: lab requests in the ledger but no lab_state.json"
    actor_st = json.loads((run_dir / "actor_state.json").read_text())
    tag = f"r{sum(1 for _ in run_dir.glob('actor_stream_r*_0.jsonl'))}"   # r0 = original, r1.. = resumes
    fatal_file = run_dir / "ENV_FATAL.txt"
    if fatal_file.exists():
        fatal_file.rename(run_dir / f"ENV_FATAL_before_{tag}.txt")
    if st is None:
        print(f"[resume] {run_dir} as {tag}; no lab_state.json: the actor never called a lab tool, lab starts fresh",
              flush=True)
    else:
        print(f"[resume] {run_dir} as {tag}; previous fatal={st['fatal'] is not None}; "
              f"failed_request={json.dumps(st['failed_request'])[:300]}", flush=True)

    env = EnvAgent(case, run_dir, cfg["env_model"], resume=True)   # its workspace was built at the start
    lab = Lab(case, env, run_dir, cfg["seed"])
    lab.check_case_files()   # a case edited since the start: the run is invalid and rerun, not resumed (2026-11-05)
    if st is not None:
        lab.restore(st)
    else:   # no env session yet: a fresh one, created by the first lab request
        env.restore({"session_id": str(uuid.uuid4()), "n_calls": 0, "session_created": False})
    lab.save_state()
    # the actor time limit covers the whole run: earlier invocations are deducted; runs started
    # before actor_time.jsonl existed have no record and get the full limit, printed
    tlog = run_dir / ACTOR_TIME_LOG
    used = sum(json.loads(x)["seconds"] for x in tlog.read_text().splitlines()) if tlog.exists() else 0.0
    remaining = cfg["actor_timeout_s"] - used
    print(f"[resume] actor time used {used:.0f} s of {cfg['actor_timeout_s']} s "
          f"({'from ' + ACTOR_TIME_LOG if tlog.exists() else 'not recorded: run predates it'}); "
          f"remaining {remaining:.0f} s", flush=True)
    if lab.report is None and remaining <= 0:
        # the actor was cut off at its time limit and the run stopped later (judge timeout, 2026-11-18 003-03):
        # the actor part is over, as in an unbroken run; only audit + judge are redone
        print("[resume] actor time limit used up, no report: the actor part is over; re-running audit + judge only",
              flush=True)
    elif lab.report is None:
        if lab.closed_at is not None:
            lab.closed_at = time.time()          # fresh grace period for the final report
            prompt = MESSAGES["budget_exhausted"]
        elif st is not None and st["fatal"] is not None:
            prompt = MESSAGES["outage_resume"]
        else:
            prompt = MESSAGES["interrupted_resume"]   # actor process crashed (e.g. API 429) or was cut off
        # run_config.json written before 2026-10-19 has no actor_effort; those runs never passed --effort
        agent = make_agent(cfg["actor_agent"], run_dir, cfg.get("actor_effort"))
        agent.check_prepared()
        srv, mcp_url = serve(lab)
        print(f"[resume] lab MCP on 127.0.0.1:{srv.server_address[1]}; actor prompt: {prompt[:120]}", flush=True)
        session_id = actor_st["session_id"]
        if session_id is None:
            # an agent that assigns its own session id (Codex) records it only after its first invocation ends; a
            # run killed during that invocation has none in actor_state.json: read it from the first stream
            # (2026-11-05 GPT pools stopped mid-run)
            session_id = agent.session_id_from_stream(run_dir / "actor_stream_r0_0.jsonl")
            print(f"[resume] session id recovered from actor_stream_r0_0.jsonl: {session_id}", flush=True)
        info = run_actor(agent, prompt, cfg["actor_model"], mcp_url, remaining, lab,
                         session_id, tag=tag, start_with_resume=True)
        srv.shutdown()
        lab.save_state()
        record_actor(run_dir, info)
    else:
        print("[resume] report already submitted; re-running audit + judge only", flush=True)
    return finish_run(case, run_dir, lab, cfg)


def summarize(root: Path) -> dict:
    """summary.json of a set of runs of one case (root/run_*): pass@1, pass^k, means; valid runs only."""
    results = [json.loads(p.read_text()) for p in sorted(root.glob("run_*/result.json"))]
    runs = sorted(p.name for p in root.glob("run_*"))
    valid = [r for r in results if r["valid"]]
    passes = [r["score"]["pass"] for r in valid]
    first = json.loads((root / runs[0] / "run_config.json").read_text())
    summary = {
        "case": first["case"], "actor_agent": first["actor_agent"], "actor_model": first["actor_model"],
        "env_model": first["env_model"], "judge_model": first["judge_model"],
        "checkpoints_enforced": first["checkpoints_enforced"], "n_runs": len(runs), "n_finished": len(results),
        "unfinished": [r for r in runs if not (root / r / "result.json").exists()], "n_valid": len(valid),
        "pass@1": sum(passes) / len(passes) if passes else None,
        f"pass^{len(valid)}": all(passes) if passes else None,
        # 0-100 scale: mean of derived_100 (core missed -> 0)
        "mean_score": round1(sum(r["score"]["derived_100"] for r in valid) / len(valid)) if valid else None,
        "mean_cost_used": sum(r["score"]["cost_used"] for r in valid) / len(valid) if valid else None,
        "per_run": [{"run": r["run"], "valid": r["valid"], **{k: r["score"][k] for k in
                     ("derived_100", "total_100", "total", "pass", "cost_used", "materials_used",
                      "traps_triggered")}}
                    for r in results],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary
