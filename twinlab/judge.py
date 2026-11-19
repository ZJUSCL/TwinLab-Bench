"""Grading a finished run.

  run_judge    one Claude Code call (--safe-mode --restricted, only Read on judge_read/) with prompts/judge_system.md:
               it sees the full case (hidden truth, rubric, traps), every lab request with what the actor was shown,
               the actor's own tool use (cut inline, untruncated in judge_read/actor_activity_full.md), and the final
               report; it scores the rubric and traps and audits the environment
  audit_actor  deterministic scan of the actor's streams for the source record (blocklist) and the case canary
  score        rubric points, trap and overuse penalties, pass rule; a run is valid only if the judge found no
               environment flaw and the audit found no contamination
"""
import base64
import json
import uuid
from pathlib import Path

from . import prompts
from .actors import guard, tool_events
from .case import RUBRIC_COMPONENTS as COMPONENTS
from .case import Case
from .scale import rescale
from .claude_cli import (close_objects, config_dir_for, jsonl_lines, run_claude_json, session_transcripts,
                         structured_answer, structured_answer_instructions)

N_CORE_CHECKS = 2   # independent core checks per judgment; any failing check sets the core item to 0 (2026-11-05)
CALL_TIMEOUT_S = 3600  # 2 of 20 deepseek-flash rejudges (040-03, 008-02) passed 1200 s under 20 parallel sessions (2026-11-04)
# the actor's own tool use shown inline to the judge: per call / per output / in total (whole entries cut from the
# middle). Sizes from 209 Opus runs on current case versions (2026-11-03): 600/400 cut 2465 of 4178 outputs and 1631
# calls; at 4000/1500 the inline log has median 35k, p95 116k, max 222k chars. The untruncated log goes to
# JUDGE_READ_DIR/FULL_LOG_NAME, the only file the judge can Read.
ACTIVITY_CMD_CHARS = 1500
ACTIVITY_OUT_CHARS = 4000
ACTIVITY_MAX_CHARS = 200000
FULL_LOG_WRAP_CHARS = 1000   # Claude Code's Read cuts lines over 2000 chars
JUDGE_KEYS = ["traps", "env_leak", "env_inconsistency", "env_coverage_gap", "env_refusals", "env_minor", "summary",
              "rubric"]   # order of the StructuredOutput object (judge_schema)
JUDGE_READ_DIR = "judge_read"
FULL_LOG_NAME = "actor_activity_full.md"


def judge_schema(case: Case) -> dict:
    rub = case.scoring["rubric"]
    trap_ids = [t["id"] for t in case.d["traps"]]
    flag = {"type": "object", "properties": {"flag": {"type": "boolean"}, "evidence": {"type": "string"}},
            "required": ["flag", "evidence"]}
    # rubric is the LAST key: deepseek-flash judges wrote {"rubric": {...}} and closed the object, 3-5 times in a row
    # (036-01 and 008-02 rejudges 2026-11-04, error_max_structured_output_retries); with the short keys first, the
    # long rubric object ends the answer
    return close_objects({   # every object closed: --json-schema rejects extra keys (claude_cli)
        "type": "object",
        "properties": {
            "traps": {"type": "object", "required": trap_ids,
                      "properties": {t: {"type": "object", "required": ["triggered", "evidence"],
                                         "properties": {"triggered": {"type": "boolean"},
                                                        "evidence": {"type": "string"}}} for t in trap_ids}},
            "env_leak": flag,
            "env_inconsistency": flag,
            "env_coverage_gap": flag,
            "env_refusals": {"type": "array", "items": {
                "type": "object", "required": ["request", "what"],
                "properties": {"request": {"type": "integer"}, "what": {"type": "string"}}}},
            "env_minor": {"type": "array", "items": {
                "type": "object", "required": ["request", "what"],
                "properties": {"request": {"type": "integer"}, "what": {"type": "string"}}}},
            "summary": {"type": "string"},
            "rubric": {"type": "object", "required": list(rub),
                       "properties": {k: {"type": "object", "required": ["points", "justification"],
                                          "properties": {"points": {"type": "number", "minimum": 0,
                                                                    "maximum": v["points"]},
                                                         "justification": {"type": "string"}}}
                                      for k, v in rub.items()}},
        },
        "required": JUDGE_KEYS,
    })


def interaction_log(run_dir: Path) -> str:
    lines = []
    for line in jsonl_lines(run_dir / "lab_ledger.jsonl"):
        r = json.loads(line)
        if r["type"] == "lab":
            e = r["env"]
            shown = r.get("shown", f"(rejected: {r.get('rejected')})")
            lines.append(f"### Request #{r['n']}\nACTOR: {r['request']}\n"
                         f"ENV verdict={e['verdict']} actions={e['action_types']} cost={r.get('cost', 0)} "
                         f"rejected={r.get('rejected')} image files={e['images']}\n"
                         f"SHOWN TO ACTOR:\n{shown}\n"
                         f"ENV grounding (internal): {e['grounding']}\n")
        elif r["type"] == "lab_after_close":
            lines.append(f"### Request after lab closed (refused)\nACTOR: {r['request']}\n")
        elif r["type"] == "lab_zero_budget":
            lines.append(f"### Request in a zero-budget session (refused, no lab)\nACTOR: {r['request']}\n")
    return "\n".join(lines)


def actor_activity_log(run_dir: Path, actor_agent: str, full_path: Path) -> str:
    """The actor's own tool use outside the lab (shell commands, file reads, web), in order, as numbered entries.
    Every entry is written untruncated to full_path (the one file the judge can Read); the returned inline log cuts
    each entry and names the entry's line range in full_path wherever it cut (the 600-char cut hid
    statistics the actor had computed, 007-02 pause pilot and 071-03 rejudge)."""
    entries = []   # (label, text, cut)
    for e in tool_events(run_dir, actor_agent):
        if e["kind"] == "call":
            if not e["name"].startswith("mcp__lab__"):
                entries.append((f"[{e['name']}]", json.dumps(e["input"], ensure_ascii=False), ACTIVITY_CMD_CHARS))
        elif not (e["lab"] or "Lab request #" in e["text"]):   # lab results are already in the interaction log
            entries.append(("->", e["text"], ACTIVITY_OUT_CHARS))
    full = [f"# Actor's own work outside the lab, full text ({len(entries)} entries)",
            f"Same entries as the judge prompt, untruncated. Lines longer than {FULL_LOG_WRAP_CHARS} characters are "
            f"wrapped onto the next lines.", ""]
    inline, lines = [], []   # lines[n-1]: (first, last) line of entry n in full_path
    n_cut = 0
    for n, (label, text, cut) in enumerate(entries, 1):
        first = len(full) + 1
        full.append(f"=== entry {n} {label} ===")
        for line in text.split("\n"):
            full.extend(line[i:i + FULL_LOG_WRAP_CHARS] for i in range(0, max(len(line), 1), FULL_LOG_WRAP_CHARS))
        full.append("")
        lines.append((first, len(full)))
        if len(text) > cut:
            n_cut += 1
            inline.append(f"[e{n}] {label} {text[:cut]}\n  ...[entry {n} truncated at {cut} of {len(text)} chars; "
                          f"full text in {full_path} lines {first}-{len(full)}]")
        else:
            inline.append(f"[e{n}] {label} {text}")
    full_path.parent.mkdir(exist_ok=True)
    full_path.write_text("\n".join(full) + "\n", encoding="utf-8")
    txt = "\n".join(inline)
    n_mid = 0
    if len(txt) > ACTIVITY_MAX_CHARS:   # cut whole entries from the middle
        head, tail = [], []
        i, j = 0, len(inline) - 1
        while i <= j and sum(map(len, head)) + len(inline[i]) <= ACTIVITY_MAX_CHARS // 2:
            head.append(inline[i])
            i += 1
        while j >= i and sum(map(len, tail)) + len(inline[j]) <= ACTIVITY_MAX_CHARS // 2:
            tail.insert(0, inline[j])
            j -= 1
        n_mid = j - i + 1
        assert n_mid > 0, (i, j)
        txt = "\n".join(head + [f"...[entries {i + 1}-{j + 1} omitted here (log over {ACTIVITY_MAX_CHARS} chars); "
                                f"full text in {full_path} lines {lines[i][0]}-{lines[j][1]}]"] + tail)
    print(f"[judge] actor activity: {len(entries)} entries, {n_cut} cut inline, {n_mid} omitted in the middle; "
          f"inline {len(txt)} chars, full log {full_path.stat().st_size} bytes at {full_path}", flush=True)
    return txt


def run_judge(case: Case, run_dir: Path, lab, model: str, outcome, actor_agent: str) -> dict:
    """actor_agent names the format of the run's actor streams (see actors.tool_events)."""
    report = json.dumps(lab.report, indent=1, ensure_ascii=False) if lab.report else "NO FINAL REPORT SUBMITTED"
    oc = (f"\n# Objective outcome of the submitted recipe (computed by the case's outcome calculator, authoritative "
          f"for the outcome rubric items)\n{json.dumps(outcome, indent=1, ensure_ascii=False)}\n" if outcome else "")
    read_dir = (run_dir / JUDGE_READ_DIR).resolve()
    full_path = read_dir / FULL_LOG_NAME
    activity = actor_activity_log(run_dir, actor_agent, full_path)
    assert sorted(p.name for p in read_dir.iterdir()) == [FULL_LOG_NAME], f"{read_dir} must hold only {FULL_LOG_NAME}"
    prompt = (f"# Case specification\n```yaml\n{case.judge_text}\n```\n\n# Interaction log\n{interaction_log(run_dir)}\n\n"
              f"# Actor's own work outside the lab (shell, file reads, web; entries cut inline, full text in "
              f"{full_path})\n{activity}\n\n"
              f"# Final ledger\n{json.dumps(lab.ledger())}\n{oc}\n# Actor final report\n{report}\n")
    (run_dir / "judge_prompt.md").write_text(prompt, encoding="utf-8")
    schema = judge_schema(case)
    # a persisted session (not --no-session-persistence): StructuredOutput rejections are counted from its transcript.
    # Only the Read tool, and only inside read_dir (the cwd): --restricted confines file tools to the working directory
    # and --permission-prompts none denies anything that would prompt (both verified 2026-11-03: Read of run_dir/
    # result.json and /etc/hostname refused).
    sid = str(uuid.uuid4())
    args = ["-p", prompt, "--safe-mode", "--restricted", "--tools", "Read", "--permission-prompts", "none",
            "--strict-mcp-config", "--session-id", sid,
            "--model", model, "--system-prompt", prompts.text("judge_system.md") + structured_answer_instructions()
            + f"Your StructuredOutput object has these {len(JUDGE_KEYS)} top-level keys, all required, in this "
              f"order: {', '.join(JUDGE_KEYS)}. The rubric object comes last and holds one entry per rubric item.\n",
            "--output-format", "json", "--json-schema", json.dumps(schema)]
    log_path = run_dir / "judge_calls.jsonl"
    out = run_claude_json(args, cwd=read_dir, timeout_s=CALL_TIMEOUT_S, log_path=log_path,
                          tmp_dir=run_dir / "judge_tmp")
    j = structured_answer(out, schema, log_path)
    # recorded beside the judgment (not part of the schema): which parts of the full log the judge opened, and any
    # read it attempted outside read_dir (refused by the CLI; printed and kept visible, the judgment still stands)
    j["full_log_reads"] = judge_full_log_reads(sid, full_path, config_dir_for(model))
    j["read_denials"] = out["permission_denials"]
    print(f"[judge] full-log reads: {j['full_log_reads']}", flush=True)
    if j["read_denials"]:
        print(f"[judge] WARNING judge tried to read outside {read_dir}: {j['read_denials']}", flush=True)
    core_check(case, run_dir, j, model, prompt, read_dir, log_path)
    return j


def core_check(case: Case, run_dir: Path, j: dict, model: str, prompt: str, read_dir: Path, log_path: Path):
    """Second, independent call on the core items the judge gave full credit; an item that fails it is set to 0
    (2026-11-05: deepseek-flash judges were lenient on core items in 3 of 10 runs against Opus, all in one direction:
    contradictory report statements read favourably, work log used for statements the report lacks, related
    statements taken as equivalent). The check and its outcome are kept in j["core_check"]."""
    rub = case.scoring["rubric"]
    items = [k for k in case.scoring["pass_rule"]["required_full"] if j["rubric"][k]["points"] == rub[k]["points"]]
    if not items:
        j["core_check"] = {"items": [], "downgraded": []}
        return
    # two lists with one-way booleans: a single "ok" flag on 0-level entries was read as "does not apply" by
    # deepseek-flash (052-01 replay-20261105-081823: every 0-level clause ok=false next to holds=true)
    full_c = {"type": "object", "required": ["condition", "quote", "met"],
              "properties": {"condition": {"type": "string"}, "quote": {"type": "string"}, "met": {"type": "boolean"}}}
    zero_c = {"type": "object", "required": ["clause", "closest_quote", "applies"],
              "properties": {"clause": {"type": "string"}, "closest_quote": {"type": "string"},
                             "applies": {"type": "boolean"}}}
    schema = close_objects({"type": "object", "required": ["items"], "properties": {"items": {
        "type": "object", "required": items,
        "properties": {k: {"type": "object",
                           "required": ["full_conditions", "zero_clauses", "holds", "failing_condition", "evidence"],
                           "properties": {"full_conditions": {"type": "array", "minItems": 1, "items": full_c},
                                          "zero_clauses": {"type": "array", "items": zero_c},
                                          "holds": {"type": "boolean"}, "failing_condition": {"type": "string"},
                                          "evidence": {"type": "string"}}} for k in items}}}})
    first = "\n".join(f"- {k}: {j['rubric'][k]['justification']}" for k in items)
    answers, down = [], []
    # 2026-11-05 (judge reliability, 23 validation runs vs Opus): one check still passed core items whose report met a
    # 0-level clause elsewhere or used a route the criterion does not name, and repeated checks disagreed with each
    # other; N_CORE_CHECKS independent checks, an item fails if any check fails it (holds false or any entry not ok)
    for r in range(N_CORE_CHECKS):
        sid = str(uuid.uuid4())
        args = ["-p", prompt + f"\n\n# Core items to check (first grader gave full credit)\n{first}\n",
                "--safe-mode", "--restricted", "--tools", "Read", "--permission-prompts", "none",
                "--strict-mcp-config", "--session-id", sid, "--model", model,
                "--system-prompt", prompts.text("core_check_system.md") + structured_answer_instructions(),
                "--output-format", "json", "--json-schema", json.dumps(schema)]
        out = run_claude_json(args, cwd=read_dir, timeout_s=CALL_TIMEOUT_S, log_path=log_path,
                              tmp_dir=run_dir / "judge_tmp")
        c = structured_answer(out, schema, log_path)
        answers.append(c["items"])
        for k in items:
            a = c["items"][k]
            bad = ([f"full condition not met: {e['condition']}" for e in a["full_conditions"] if not e["met"]]
                   + [f"0-level clause applies: {e['clause']} ({e['closest_quote'][:300]})"
                      for e in a["zero_clauses"] if e["applies"]])
            if k not in down and (not a["holds"] or bad):
                down.append(k)
                why = a["failing_condition"] if not a["holds"] else bad[0]
                j["rubric"][k]["justification"] = (f"[core check {r}: 0, {why}; {a['evidence']}] "
                                                   + j["rubric"][k]["justification"])
                j["rubric"][k]["points"] = 0
        print(f"[judge] core check {r} of {items}: downgraded so far {down}", flush=True)
    j["core_check"] = {"items": items, "downgraded": down, "answers": answers}


def judge_full_log_reads(sid: str, full_path: Path, config_dir: Path) -> list:
    """The Read calls of judge session sid: offset/limit, or 'whole file'; any other path is named."""
    (tr,) = session_transcripts(sid, config_dir)
    reads = {}   # by tool_use id: the transcript can hold the same tool_use on more than one line
    for line in jsonl_lines(tr):
        msg = json.loads(line).get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            for c in msg["content"]:
                if c.get("type") == "tool_use" and c.get("name") == "Read":
                    r = {k: c["input"][k] for k in ("offset", "limit") if k in c["input"]} or {"whole_file": True}
                    if Path(c["input"]["file_path"]) != full_path:
                        r["file_path"] = c["input"]["file_path"]
                    reads[c["id"]] = r
    return list(reads.values())


# opaque machine ids and signatures: short (numeric) needles match inside them by chance -- thinking 'signature'
# (052, 2026-10-20); a Claude Code event 'uuid' matched needle "35097" (039-01 kimi run 20261105-115701)
OPAQUE_KEYS = {"signature", "uuid", "session_id", "parent_tool_use_id", "tool_use_id", "request_id", "id"}


def _drop_signatures(x):
    """Drop OPAQUE_KEYS values and base64 payloads (images in tool results) before the needle scan: needle "Luh" matched
    inside a lab PNG's data: URI in a Codex function_call_output (sol high 039-01 run 20261106-072635)."""
    if isinstance(x, dict):
        if x.get("type") == "base64" and "data" in x:   # Claude Code image/document source block
            return {k: v for k, v in x.items() if k != "data"}
        return {k: _drop_signatures(v) for k, v in x.items() if k not in OPAQUE_KEYS}
    if isinstance(x, list):
        return [_drop_signatures(v) for v in x]
    if isinstance(x, str) and x.startswith("data:") and ";base64," in x[:100]:
        return x[:x.index(";base64,")]
    return x


# The audit scans what reached the actor from outside and what it sent out: tool calls (inputs) and tool results, never the actor's own
# text or thinking. Recognising the source from memory is measured by the memorization probes; scanning thinking
# also treated models unequally (Codex reasoning is encrypted, open models' thinking is plain text; 10 runs of
# hy4/mimo/qwen were marked contaminated on 2026-11-05 for naming the source in their own thinking only).
# Codex: the exec event stream shows a shell command's raw output even when the PostToolUse guard replaced it, so the
# audit reads the session rollout (what was sent to the model): response items other than reasoning and the model's
# own messages.
OWN_TEXT_BLOCKS = {"text", "thinking", "redacted_thinking"}            # Claude Code assistant content blocks
OLD_GUARD_LIST_PREFIX = 48   # chars of the old base64 blocklist file that identify its content
CODEX_ROLLOUTS = "actor_config/sessions/**/rollout-*.jsonl"
CODEX_REQUEST_ITEMS = {"function_call", "custom_tool_call", "local_shell_call", "tool_search_call"}   # the actor sent these
BLOCKED_ATTEMPTS_INVALID = 3   # stopped requests in one run (guard denials + Codex web-search withdrawals)


def _external_parts(event: dict):
    """The parts of one actor stream/rollout event that came from outside or went out (None: nothing to scan)."""
    t = event.get("type")
    if t == "assistant":   # Claude Code: keep tool calls (and any server-tool blocks), drop own text/thinking
        blocks = [c for c in event["message"]["content"] if c.get("type") not in OWN_TEXT_BLOCKS]
        return blocks or None
    if t == "user":        # Claude Code: the prompt and tool results
        return event["message"]
    if t == "response_item":   # Codex rollout: tool calls and outputs, web_search_call, user/developer messages
        p = event["payload"]
        if p["type"] == "reasoning" or (p["type"] == "message" and p["role"] == "assistant"):
            return None
        return p
    return None            # system/init, final result text, rollout event_msg/turn bookkeeping


def _received_parts(event: dict):
    """The parts of an event that reached the actor from outside; its own tool calls are requests, not content."""
    parts = _external_parts(event)
    if parts is None:
        return None
    if event["type"] == "assistant":
        return [b for b in parts if b.get("type") != "tool_use"] or None
    if event["type"] == "response_item" and parts["type"] in CODEX_REQUEST_ITEMS:
        return None
    return parts


def _denied_results(stream: Path) -> int:
    """Tool results in a Claude Code actor stream that the guard hook's PreToolUse denied."""
    denied = {guard.DENIED_RESULT.format(tool=t) for t in guard.CLAUDE_PRE_TOOLS}
    n = 0
    for raw in stream.read_text().split("\n"):
        if not raw.strip():
            continue
        m = json.loads(raw)
        if m.get("type") != "user" or isinstance(m["message"]["content"], str):
            continue
        for c in m["message"]["content"]:
            if c.get("type") == "tool_result" and c.get("is_error") and c.get("content") in denied:
                n += 1
    return n


def audit_actor(case: Case, run_dir: Path) -> dict:
    """Deterministic scan for the source. A request that the guard stopped is not contamination by
    itself: the actor is told the source is excluded and can go on. The run is contaminated when a blocked string
    reached the actor in received content, or after BLOCKED_ATTEMPTS_INVALID stopped requests. A blocked string in a
    request that nothing had to stop (the actor naming its own local file) does not count."""
    # runs before 2026-11-06 kept the blocklist base64-encoded in the sandbox (hooks/blocklist.b64); mimo-v2.6-pro
    # 058-02 printed and decoded it (title, DOI). Its content reaching a tool result counts as reaching the source;
    # only listing the directory does not (mimo 008-02, 087-02).
    old_list = base64.b64encode(json.dumps(case.blocklist).encode()).decode()[:OLD_GUARD_LIST_PREFIX]
    needles = case.guard_needles + [old_list]
    hits = []
    agent = json.loads((run_dir / "run_config.json").read_text())["actor_agent"]
    if agent == "codex":
        files = sorted(run_dir.glob(CODEX_ROLLOUTS))
        assert files, f"{run_dir}: no Codex session rollout to audit"
    else:
        files = sorted(run_dir.glob("actor_stream_*.jsonl"))
    for f in files:
        for i, raw in enumerate(f.read_text().split("\n")):
            if not raw.strip():
                continue
            parts = _received_parts(json.loads(raw))
            if parts is None:
                continue
            line = json.dumps(_drop_signatures(parts), ensure_ascii=False)
            for n in needles:
                if n in line:
                    hits.append({"file": f.name, "line": i + 1, "needle": n})
    guard = run_dir / "actor_config/hooks/guard.log"
    guard_recs = [json.loads(x) for x in guard.read_text().splitlines()] if guard.exists() else []
    relay = run_dir / "websearch_guard.json"   # Codex web-search relay (actors/websearch_relay.py), since 2026-11-06
    withdrawn = json.loads(relay.read_text())["events"] if relay.exists() else []
    blocked = [g for g in guard_recs if g.get("blocked")]
    n_denied = len(blocked)
    stream_denied = None
    if agent == "claude-code":
        # guard.log is inside the sandbox (the actor can delete it); the stream is written by the harness outside.
        # The larger count is used: runs before 2026-11-11 denied with another message ("This host is not reachable
        # from the lab network."), which the stream count does not match, so guard.log is their record.
        stream_denied = sum(_denied_results(f) for f in files)
        if stream_denied != len(blocked):
            print(f"[audit] WARNING {run_dir}: actor stream has {stream_denied} denied tool calls, guard.log "
                  f"{len(blocked)}; the larger is used (fewer in guard.log: edited by the actor)", flush=True)
        n_denied = max(stream_denied, len(blocked))
    n_blocked = n_denied + len(withdrawn)
    return {"contaminated": bool(hits) or n_blocked >= BLOCKED_ATTEMPTS_INVALID, "hits": hits[:50],
            "n_blocked_attempts": n_blocked,
            "guard_log_blocked": len(blocked), "stream_denied": stream_denied,
            "websearch_withdrawn": [{"attempt": e["attempt"], "hits": e["hits"]} for e in withdrawn],
            "guard_blocked": blocked,
            "guard_redacted": [g for g in guard_recs if g.get("redacted")],   # outputs withheld by PostToolUse
            "n_websearch": sum(g["tool"] == "WebSearch" for g in guard_recs)}


def score(case: Case, j: dict, lab) -> dict:
    rub = case.scoring["rubric"]
    rule = case.scoring["pass_rule"]
    pts = {k: j["rubric"][k]["points"] for k in rub}
    for k, v in pts.items():
        assert 0 <= v <= rub[k]["points"], (k, v)
    n_traps = sum(t["triggered"] for t in j["traps"].values())
    pen = case.scoring["penalties"]
    used = {k: lab.consumed[k] - case.initial_consumed[k] for k in case.consumables}   # used during this run
    overuse_pen = 0.0
    for rule_ in pen["consumable_overuse"]:   # [{prefix, threshold, penalty}]: penalty per unit beyond threshold
        n = sum(v for k, v in used.items() if k.startswith(rule_["prefix"]))
        overuse_pen += max(0, n - rule_["threshold"]) * -rule_["penalty"]
    total = sum(pts.values()) + n_traps * pen["per_trap_triggered"] - overuse_pen
    # pass iff derived score != 0; derived = total if every required_full item is full, else 0.
    # No min_total or percent-of-total threshold (pass_rule.min_total is not read).
    derived = total if all(pts[k] == rub[k]["points"] for k in rule["required_full"]) else 0
    passed = lab.report is not None and derived != 0
    comps = {c: {"points": sum(pts[k] for k in rub if rub[k]["component"] == c),
                 "max": sum(rub[k]["points"] for k in rub if rub[k]["component"] == c)} for c in COMPONENTS}
    # 0-100 reporting scale (twinlab/scale.py); native fields above stay for old tools
    s100 = rescale(pts, total, rub, rule["required_full"])
    return {"points": pts, "components": comps, "rubric_sha256": case.rubric_sha256,
            "traps_triggered": n_traps, "overuse_penalty": overuse_pen, "total": total, **s100,
            "pass": passed, "cost_used": lab.cost_used, "budget_unit": case.budget_unit,
            "materials_used": used, "n_requests": lab.n_requests,
            "report_submitted": lab.report is not None,
            "env_valid": not (j["env_leak"]["flag"] or j["env_inconsistency"]["flag"]
                              or j["env_coverage_gap"]["flag"])}
