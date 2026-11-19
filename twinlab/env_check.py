"""Deterministic after-the-fact check of the environment agent of one run (no model calls).

python scripts/check_env.py --run-dir runs/<case>/<ts>/run_0 --scratch <dir outside runs/ and cases/>
                            [--transcript <env session .jsonl>]   # default: <run-dir>/env_session.jsonl

The environment is a small model; instead of comparing it with a large-model reference we check what it did:
  1. replay: every shell command of the environment that calls calculators.py is re-run, in order, in a fresh copy
     of env_ws (empty calc_state). The replayed output must be what the environment saw, and the final calc_state must
     equal the run's (the calculator is deterministic; a mismatch means edited state or commands run out of order);
  2. billing: per request, env.action_types must equal the calculator's CHARGE lines, and the ledger cost must equal
     their sum by the case cost table (only for cases whose calculator prints CHARGE lines);
  3. numbers: every number in an observation must occur in the calculator output of that request, in the request
     text, or in the cost table. Leftovers are listed with context for a reader (they may be invented, self-computed
     or mis-attributed numbers, or legitimate fixed facts from the case files);
  4. refusals: not_performed_items and calculator REFUSED lines are listed for a reader;
  5. lost output (2026-11-04, output_problems): an observation ending with an ellipsis is a problem; a billed
     calculator output whose decimals mostly do not reach the observation is listed for a reader.
Writes <run-dir>/env_check.json (machine-found problems and the items a reader should look at) and prints a report.
"""
import collections
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from . import calc_guard, settings
from .claude_cli import jsonl_lines

FREE_ACTIONS = {"inspect", "ask_expert_knowledge", "continued_step"}
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_]|\.\d)")
# anchored to the line start (2026-10-30): a printed docstring's mid-line "(CHARGE: quote x1, cost 0)" was billed
# (004-02 20261029-234203 #1). Output the environment joined or prefixed (`| tr '\n' ' '`, `| uniq -c`) is no longer
# read; such commands are flagged by filtered_calculator_commands and billed as max(env listing, CHARGE lines)
# (twinlab/lab.py)
# one CHARGE line may list several keys: "CHARGE: a x12, b x12" (2026-10-19: only the first key was read before)
CHARGE_ITEM = r"\w+ x\d+(?: \[once per request\])?"
CHARGE_RE = re.compile(rf"^[ \t]*CHARGE: ({CHARGE_ITEM}(?:, {CHARGE_ITEM})*)", re.M)
# CHARGE lines the harness copies into env-readable records (session_record.md: what the actor was shown, the env's
# grounding) get this prefix so CHARGE_RE no longer reads them: 047-02 run 20261029-233832 #15 the env ran
# `tail -5 session_record.md` in a calculator command and a recorded 'CHARGE: trial x1' was billed again
RECORD_CHARGE_RE = re.compile(r"^([ \t]*)CHARGE:", re.M)
RECORDED_CHARGE = r"\1recorded CHARGE:"
CALC_INVOKE_RE = re.compile(r"python3?(?:\s+-\w+)*\s+(?:\S*/)?calculators\.py\b"   # also `python3 -I calculators.py`
                            # argv-list form inside a heredoc script, subprocess.run(["python", "-I", "calculators.py",
                            # ...]) (004-02 20261104-152714 #4: the start was neither replayed nor its CHARGE read)
                            r"""|["']python3?["'](?:\s*,\s*["']-\w+["'])*\s*,\s*["'](?:\S*/)?calculators\.py["']""")
LOOP_OUTPUT_FILTER_RE = re.compile(r"\bdone\s*(?:(?<!\|)\|(?!\|)|>)")   # `for ...; do ...; done | grep`
CHARGE_ITEM_RE = re.compile(r"(\w+) x(\d+)( \[once per request\])?")
STATE_TAMPER_RE = re.compile(r"calc_state|sed -i|\brm |\bmv |\bcp |python3? -c|>\s*[\w./]")
STDERR_REDIRECT_RE = re.compile(r"2>\s*(?:&1|/dev/null)")
# stdout to /dev/null (`>/dev/null`, `1>/dev/null`, `&>/dev/null`); not `2>/dev/null`. In guarded runs
# (twinlab/calc_guard.py) such a calculator run either billed nothing or was refused before it counted.
STDOUT_DEVNULL_RE = re.compile(r"(?<![\w>&])[&1]?>>?\s*/dev/null")
SAVED_OUTPUT_RE = re.compile(r"Full output saved to: (\S+)")
DISCARDED_RE = re.compile(r"Your answer to actor request #(\d+) was DISCARDED by the harness")   # twinlab/lab.py Lab.restore
TRUNCATED_END_RE = re.compile(r"(?:\.\.\.|…)\s*$")   # observation cut off at its end (087-02 replay 141338 #1)
PROVENANCE_RE = re.compile(r"^[ \t]*_source\b")   # calculator provenance line (twinlab/lab.py cuts them from observations)
UNREPORTED_MIN_SHARE = 0.5   # a billed output with fewer of its result numbers in the observation is flagged
# a value the calculator hands the environment for its own ledger, not a result for the actor (004 `sink` prints
# "add to sucrose_h: 3.01"; 004-02 20261104-152714 had 9 billed_output_unreported flags from it alone)
LEDGER_VALUE_RE = re.compile(r"\badd to \w+: -?\d+(?:\.\d+)?")
# env_system.md rule 14 (2026-11-04): deepseek-flash envs opened the actor's files through `..` out of the symlinks
# and read or imported the calculator source (env audits a/b)
# (reading the docstring, e.g. `head -60 calculators.py`, is how Sonnet envs learn the usage: allowed)
CALC_IMPORT_RE = re.compile(r"\bimport\s+calculators\b|\bfrom\s+calculators\s+import\b")
# from 2026-11-04 the usage is in env_ws/calculator_usage.txt and rule 14 forbids opening the source: deepseek-flash
# envs read hidden maps and index tables out of it (env audits c/d); any non-run use of calculators.py is recorded
CALC_SOURCE_READ_RE = re.compile(r"\b(?:cat|head|tail|sed|awk|less|more|grep|rg|nl|strings|wc)\b[^|;&\n]*calculators\.py")
PARENT_DIR_RE = re.compile(r"(?:^|[\s'\"=/])\.\./")
ABS_PATH_RE = re.compile(r"(?<![\w.])/(?:home|data|mnt|opt|srv)/[^\s'\"|;&)]+")
TOOL_PATH_RE = re.compile(r'"(?:file_path|path)":\s*"([^"]+)"')


def outside_file_access(turn: dict, ws: Path) -> list:
    """Shell commands and Read/Grep/Glob calls of one request that go outside the env workspace (`..`, an absolute path
    not under ws) or import the calculator (rule 14). Paths under ws itself are allowed."""
    ws_s = str(ws.resolve())
    found = []
    for b in turn["bash"]:
        c = b["command"]
        # Claude Code saves a large tool output under <config>/projects/.../tool-results/ and the env reads it there
        outside = [p for p in ABS_PATH_RE.findall(c) if not p.startswith(ws_s) and "/tool-results/" not in p]
        if CALC_IMPORT_RE.search(c) or CALC_SOURCE_READ_RE.search(c) or PARENT_DIR_RE.search(c) or outside:
            found.append(f"Bash: {c[:200]}")
    for t in turn["other_tools"]:
        if not t.split(" ", 1)[0] in ("Read", "Grep", "Glob"):
            continue
        for p in TOOL_PATH_RE.findall(t):
            if (".." in Path(p).parts or (p.startswith("/") and not p.startswith(ws_s) and "/tool-results/" not in p)
                    or p.endswith("calculators.py")):
                found.append(t[:200])
    return found
CONTEXT_CHARS = 60
REPLAY_TIMEOUT_S = 300
ENV_TMP_REPLAY = "env_tmp_replay"   # sibling of the replay env_ws: the run's env_tmp copy, bound as /tmp
ZSH = str(settings.SHELL_BIN)   # the same shell the environment's Bash tool ran (twinlab.claude_cli)


def load_turns(transcript: Path) -> dict:
    """request number -> {"bash": [{"command", "output"}], "other_tools": [...]} from a Claude Code transcript."""
    turns, cur, pending = {}, None, {}
    for line in jsonl_lines(transcript):
        msg = json.loads(line).get("message")
        if not isinstance(msg, dict):
            continue
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for c in content:
            kind = c["type"]
            if msg["role"] == "user" and kind == "text":
                # a resume after a fatal error tells the environment its answer to request #k was discarded
                # (Lab.restore) and calc_state is rolled back to before #k (Lab.rollback_calc_state): drop the
                # commands of the discarded attempt so only the attempt the ledger accepted is replayed
                disc = DISCARDED_RE.search(c["text"])
                if disc and int(disc.group(1)) in turns:
                    k = int(disc.group(1))
                    print(f"[env_check] request {k}: discarded attempt dropped ({len(turns[k]['bash'])} shell "
                          f"commands, {len(turns[k]['other_tools'])} other tool calls); replaying the resubmitted "
                          f"attempt only", flush=True)
                    # what the discarded attempt's commands printed (real tool output, same calc_state as the
                    # resubmitted attempt after the rollback) stays in the environment's context and may be quoted
                    # in the accepted observation (044, 2026-10-19): kept for the numbers check, not replayed
                    discarded = turns[k]["discarded_seen"] + [b["output"] for b in turns[k]["bash"]]
                    turns[k] = {"bash": [], "other_tools": [], "discarded_seen": discarded}
                    cur = turns[k]
                m = re.search(r"## Actor request #(\d+)", c["text"])
                if m:   # a resumed request re-sends the same number: later commands belong to the same request
                    cur = turns.setdefault(int(m.group(1)), {"bash": [], "other_tools": [], "discarded_seen": []})
            elif kind == "tool_use" and cur is not None:
                if c["id"] in pending:   # the transcript records each assistant tool call twice under the same id
                    continue
                if c["name"] == "Bash":
                    pending[c["id"]] = {"command": c["input"]["command"], "output": ""}
                    cur["bash"].append(pending[c["id"]])
                elif c["name"] != "StructuredOutput":
                    cur["other_tools"].append(f"{c['name']} {json.dumps(c['input'], ensure_ascii=False)[:200]}")
            elif kind == "tool_result" and c["tool_use_id"] in pending:
                out = c["content"]
                if isinstance(out, list):
                    out = "\n".join(x["text"] for x in out if x["type"] == "text")
                # Claude Code replaces a too-large tool output by a preview plus the path of the full text, which the
                # environment then reads with Read (040, 2026-10-19); compare against the full text it could see
                saved = SAVED_OUTPUT_RE.search(out)
                if saved:
                    full = Path(saved.group(1))
                    assert full.is_file(), f"tool output was saved to {full}, which no longer exists"
                    print(f"[env_check] tool output read from saved file {full}", flush=True)
                    out = full.read_text(encoding="utf-8")
                pending[c["tool_use_id"]]["output"] = out
    return turns


def _invocation_output_filtered(cmd: str, start: int) -> bool:
    """Whether the simple command starting at cmd[start] (a calculator invocation) ends in an unquoted pipe or output
    redirection before the next unquoted ; && || or newline."""
    quote, i = None, start
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"':
                i += 1
        elif ch in "'\"":
            quote = ch
        elif ch == "\\":
            i += 1
        elif ch in ";\n" or cmd.startswith("&&", i) or cmd.startswith("||", i):
            return False
        elif ch in "|>":
            return True
        i += 1
    return False


def filtered_calculator_commands(commands: list) -> list:
    """The shell commands among `commands` in which a calculator invocation (python3 calculators.py ...) has its
    output piped or redirected (stderr merges 2>&1, 2>/dev/null are allowed), directly or through the loop it runs in
    (`...; done | grep`). The environment then saw filtered output, so CHARGE lines may be missing (002-02
    20261029-234237 #6: `| grep` dropped read_plate x3) or merged (002-02 20261029-231704 #24: `| uniq -c`).
    Pipes elsewhere in the same command (`tail session_record.md | cut ...; python3 calculators.py ...`, a heredoc
    spec file) do not count."""
    flagged = []
    for c in commands:
        s = STDERR_REDIRECT_RE.sub(" ", c)
        starts = [m.start() for m in CALC_INVOKE_RE.finditer(s)]
        if any(_invocation_output_filtered(s, i) for i in starts) or \
                (starts and LOOP_OUTPUT_FILTER_RE.search(s, starts[0])):
            flagged.append(c)
    return flagged


def strip_stdout_devnull(command: str) -> str:
    """command without its stdout-to-/dev/null redirections, for filtered_calculator_commands in guarded runs."""
    return STDOUT_DEVNULL_RE.sub(" ", command)


def charges_from_outputs(outputs: list, per_item: set, tag: str) -> collections.Counter:
    """Billable action counts of one request from the CHARGE lines in the outputs of its calculator commands.
    xN of a per_item action adds up over lines and commands; a per-request action (no per_item in case_env.yaml,
    or tagged [once per request]) counts once however many commands print it (rule 10; 001-01 two library preps).
    Used by the live harness (twinlab/lab.py, outputs the environment saw) and by check_env (replayed outputs)."""
    charges, once = collections.Counter(), set()
    for out in outputs:
        lines = list(CHARGE_RE.finditer(out))
        items = [x for line in lines for x in line.group(1).split(", ")]
        if any(", " in line.group(1) for line in lines):
            print(f"[{tag}] multi-key CHARGE line(s) read as {items}", flush=True)
        for item in items:
            m = CHARGE_ITEM_RE.fullmatch(item)
            assert m, f"[{tag}] unparsed CHARGE item {item!r}"
            if m.group(1) not in FREE_ACTIONS:
                charges[m.group(1)] += int(m.group(2))
                if m.group(3) or m.group(1) not in per_item:
                    once.add(m.group(1))
    for k in once:
        charges[k] = 1
    return charges


def recorded_charge_lines(text: str) -> str:
    """text with every line-start CHARGE line prefixed 'recorded ', for files the environment can read back."""
    out = RECORD_CHARGE_RE.sub(RECORDED_CHARGE, text)
    assert CHARGE_RE.search(out) is None, f"a CHARGE line survived the record rewrite: {out[:300]!r}"
    return out


def numbers(text: str) -> list:
    return [float(m.group(0)) for m in NUM_RE.finditer(text)]


def output_problems(outputs: list, observation: str, request: str) -> list:
    """Signs that an observation lost calculator results the actor paid for (087-02 replay 20261103-141338 #1: the
    Sonnet env stopped at 'C17-C19 ... ', the billed NMR result never reached the actor). Two independent signs:
    observation_truncated (the observation ends with an ellipsis) and billed_output_unreported (a calculator output with
    a billable CHARGE line whose result numbers are mostly absent from the observation; result numbers = the decimals
    of its lines other than CHARGE and provenance lines, without LEDGER_VALUE_RE values, minus the numbers of the
    request text; integers are left out,
    day and period counters made them fire on every step). Over the 1347 requests of runs/ on 2026-11-04 the first
    fired once (this case), the second on 32 requests, mostly legitimate (fractions reported as percent, source dumps
    printed with a run): check_env lists the first as a problem, the second for a reader. Used by the live harness
    (twinlab/lab.py, flag only, no retry) and by check_env."""
    found = []
    if TRUNCATED_END_RE.search(observation):
        found.append({"kind": "observation_truncated", "what": f"observation ends with {observation.rstrip()[-80:]!r}"})
    obs_nums, req_nums = set(numbers(observation)), set(numbers(request))
    for out in outputs:
        billable = [k for line in CHARGE_RE.finditer(out) for k in CHARGE_ITEM_RE.findall(line.group(1))
                    if k[0] not in FREE_ACTIONS]
        if not billable:
            continue
        body = "\n".join(x for x in out.splitlines() if not CHARGE_RE.match(x) and not PROVENANCE_RE.match(x))
        body = LEDGER_VALUE_RE.sub(" ", body)
        result = {v for v in numbers(body) if v != int(v)} - req_nums
        if not result:
            continue
        absent = sorted(result - obs_nums)
        if len(absent) > (1 - UNREPORTED_MIN_SHARE) * len(result):
            first = next(x for x in out.splitlines() if x.strip())
            found.append({"kind": "billed_output_unreported",
                          "what": f"{len(absent)}/{len(result)} result numbers of a billed calculator output "
                                  f"({first.strip()[:120]!r}) absent from the observation, e.g. {absent[:8]}"})
    return found


def fresh_workspace(env_ws: Path, scratch: Path) -> Path:
    ws = scratch / "env_ws_replay"
    if ws.exists():
        shutil.rmtree(ws)
    # symlinks are dereferenced: the replay re-runs the environment's own shell commands and must not be able to
    # reach the case assets through a link
    shutil.copytree(env_ws, ws, symlinks=False, ignore=shutil.ignore_patterns("calc_state"))
    # the harness appends to session_record.md during the run; the replay starts from the empty record so that a
    # `tail session_record.md` inside a calculator command cannot print more than the environment saw
    record = ws / "session_record.md"
    record.write_text(record.read_text(encoding="utf-8").splitlines(keepends=True)[0], encoding="utf-8")
    # runs since 2026-10-26 gave the environment its own /tmp (<run>/env_tmp, claude_cli.private_tmp_prefix); it
    # writes load specs there with commands that do not call the calculator (002-02 20261030-004819 #13 /tmp/w/spec.json),
    # which the replay skips: the replay gets a copy of the run's final env_tmp as its /tmp (claude internals left out)
    env_tmp = env_ws.parent / "env_tmp"
    if env_tmp.is_dir():
        tmp_copy = scratch / ENV_TMP_REPLAY
        if tmp_copy.exists():
            shutil.rmtree(tmp_copy)
        shutil.copytree(env_tmp, tmp_copy, symlinks=False,
                        ignore=shutil.ignore_patterns("claude-*", "cc-socks", "CLAUDE_ENV_FILE"))
        print(f"[env_check] replay /tmp = copy of {env_tmp}: {sorted(p.name for p in tmp_copy.iterdir())}", flush=True)
    return ws


def replay_command(command: str, env_ws: Path, ws: Path) -> str:
    """Re-run one of the environment's shell commands (loops, pipes and variables included) in the replay
    workspace and return its combined output. The run's own env_ws path is redirected to the replay copy; for runs
    with a private /tmp the command sees the copy made by fresh_workspace as /tmp (bwrap, as in the run)."""
    redirected = command.replace(str(env_ws), str(ws))
    assert str(env_ws.parent) not in redirected, f"command still refers to the run directory: {command[:300]}"
    # zsh, as in the run (twinlab.claude_cli sets SHELL to zsh): unquoted variables are not word-split there
    cmd = [ZSH, "-c", redirected]
    tmp_copy = ws.parent / ENV_TMP_REPLAY
    if (env_ws.parent / "env_tmp").is_dir():
        assert tmp_copy.is_dir(), f"{tmp_copy} missing: make the replay workspace with fresh_workspace"
        # ws is bound again after /tmp because the scratch dir may itself be under the host /tmp
        cmd = [str(settings.BWRAP), "--die-with-parent", "--bind", "/", "/", "--dev", "/dev",
               "--bind", str(tmp_copy), "/tmp", "--bind", str(ws), str(ws), "--chdir", str(ws), *cmd]
    # guarded runs (twinlab/calc_guard.py): python3 goes through the same guard, so a refused calculator run is
    # refused again (state restored, same message) and a passed one runs as it did
    run_dir = env_ws.parent
    guarded = (run_dir / calc_guard.GUARD_BIN).is_dir()
    penv = {**os.environ, **calc_guard.process_env(run_dir, ws.parent / "env_guard_replay.jsonl")} if guarded else None
    p = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=REPLAY_TIMEOUT_S, env=penv)
    # the replay path is removed so that digits in it (tracebacks print it) are not compared as numbers
    return f"{p.stdout.strip()}\n{p.stderr.strip()}".replace(str(ws), "<env_ws>")


def check_env(run: Path, scratch: Path, transcript: Path) -> dict:
    """Check one run; writes and returns <run>/env_check.json. transcript: the env agent's session (normally
    <run>/env_session.jsonl)."""
    run = run.resolve()
    scratch = scratch.resolve()
    assert "runs" not in scratch.parts and "cases" not in scratch.parts, f"scratch must be outside runs/ and cases/: {scratch}"
    assert transcript.is_file(), f"no environment transcript at {transcript} (runs before 2026-10-13 need --transcript)"

    turns = load_turns(transcript)
    requests = [json.loads(line) for line in jsonl_lines(run / "lab_ledger.jsonl")]
    requests = [r for r in requests if r["type"] == "lab"]
    case_env = yaml.safe_load((run / "env_ws" / "case_env.yaml").read_text(encoding="utf-8"))
    cost = {k: v["cost"] for k, v in case_env["actions"].items()}
    per_item = {k for k, v in case_env["actions"].items() if v.get("per_item")}
    table_numbers = {float(v) for v in cost.values()} | {float(case_env["resources"]["budget"]["total"])}
    env_ws = run / "env_ws"
    ws = fresh_workspace(env_ws, scratch)
    print(f"[env_check] {run}: {len(requests)} lab requests, {len(turns)} transcript turns, replay in {ws}", flush=True)

    problems, to_read, shell_notes, billing_fixes, guard_refusals, run_dir_cmds = [], [], [], [], [], []
    guarded = (run / calc_guard.GUARD_BIN).is_dir()
    print(f"[env_check] calculator guard (twinlab/calc_guard.py) {'on' if guarded else 'off'} in this run", flush=True)
    any_charge_line = False
    running_cost = 0.0
    for r in requests:
        n, env = r["n"], r["env"]
        assert n in turns, f"request {n} has no turn in the transcript {transcript}"
        turn = turns[n]
        seen = "\n".join(b["output"] for b in turn["bash"])
        calc_out, calc_outs = "", []
        for b in turn["bash"]:
            if not CALC_INVOKE_RE.search(b["command"]):
                continue   # file reads and greps (also `grep ... calculators.py`) do not change calculator state
            if STATE_TAMPER_RE.search(STDERR_REDIRECT_RE.sub(" ", b["command"])):
                shell_notes.append({"request": n, "command": b["command"][:600]})
            if str(run) in b["command"].replace(str(env_ws), ""):
                # the environment looked at the run directory itself (luna 038-02 20261106-063128: `ls -la <run_0>/;
                # cat session_record.md; python3 calculators.py status ...`): not replayable outside the run, listed
                # for a reader; a skipped command that changed calculator state shows up in the final state check
                run_dir_cmds.append({"request": n, "command": b["command"][:600]})
                print(f"[env_check] request {n}: command refers to the run directory, not replayed: "
                      f"{b['command'][:200]}", flush=True)
                continue
            out = replay_command(b["command"], env_ws, ws)
            calc_out += "\n" + out
            missing = sorted(set(numbers(out)) - set(numbers(b["output"])))
            if missing:
                problems.append({"request": n, "kind": "replay_mismatch",
                                 "what": f"`{b['command'][:160]}`: {len(missing)} replayed numbers absent from what the "
                                         f"environment saw, e.g. {missing[:8]}"})
            calc_outs.append(out)
        charges = charges_from_outputs(calc_outs, per_item, f"env_check request {n}")
        any_charge_line = any_charge_line or bool(charges)
        cmds = [b["command"] for b in turn["bash"]]
        if guarded:   # a /dev/null stdout there billed nothing or was refused before it counted (calc_guard.py)
            cmds = [strip_stdout_devnull(c) for c in cmds]
        filtered = filtered_calculator_commands(cmds)
        guard_ref = [b["command"][:300] for b in turn["bash"] if calc_guard.REFUSAL_MARK in b["output"]]
        if guard_ref:
            guard_refusals.append({"request": n, "commands": guard_ref})
        if filtered:   # the harness billed max(env listing, CHARGE lines) for this request (twinlab/lab.py)
            problems.append({"request": n, "kind": "filtered_calculator_output",
                             "what": f"{len(filtered)} calculator command(s) piped or redirected, CHARGE lines may be "
                                     f"missing: " + " || ".join(c[:200] for c in filtered)})

        if "billing_from_charge_lines" in r:   # the harness replaced the environment's listing (twinlab/lab.py)
            billing_fixes.append({"request": n, **r["billing_from_charge_lines"]})
        billed = collections.Counter(x for x in env["action_types"] if x not in FREE_ACTIONS)
        if charges and billed != charges and not filtered:   # filtered: reported above
            problems.append({"request": n, "kind": "billing",
                             "what": f"action_types {dict(billed)} != CHARGE lines {dict(charges)}"})
        if "rejected" in r:   # the harness rejected the request, nothing billed (lab.py; 002-02 mimo 20261105-112005)
            print(f"[env_check] #{n} rejected by the harness, not billed: {r['rejected']}", flush=True)
            continue
        expected_cost = round(sum(cost[k] * v for k, v in billed.items()), 6)
        if abs(expected_cost - r["cost"]) > 1e-6:
            problems.append({"request": n, "kind": "billing",
                             "what": f"ledger cost {r['cost']} != cost table sum of action_types {expected_cost}"})
        running_cost += r["cost"]
        if abs(running_cost - r["ledger"]["cost_used"]) > 1e-6:
            problems.append({"request": n, "kind": "ledger",
                             "what": f"cost_used {r['ledger']['cost_used']} != running sum {round(running_cost, 6)}"})

        obs = env["observation"]
        signs = output_problems([b["output"] for b in turn["bash"] if CALC_INVOKE_RE.search(b["command"])], obs,
                                r["request"])
        problems += [{"request": n, **p} for p in signs if p["kind"] == "observation_truncated"]
        unreported = [p["what"] for p in signs if p["kind"] == "billed_output_unreported"]
        allowed =set(numbers(calc_out)) | set(numbers(r["request"])) | table_numbers | {float(r["cost"])}
        if turn["discarded_seen"]:
            from_discarded = set(numbers("\n".join(turn["discarded_seen"]))) - allowed
            print(f"[env_check] request {n}: {len(from_discarded)} more numbers allowed from the discarded attempt's "
                  f"shell output", flush=True)
            allowed |= from_discarded
        unexplained = []
        for m in NUM_RE.finditer(obs):
            v = float(m.group(0))
            if v not in allowed and abs(v) not in allowed:
                ctx = obs[max(0, m.start() - CONTEXT_CHARS):m.end() + CONTEXT_CHARS].replace("\n", " / ")
                unexplained.append({"number": m.group(0), "context": ctx})
        refused = [line[:300] for line in seen.splitlines()
                   if "REFUSED" in line and calc_guard.REFUSAL_MARK not in line]
        if unexplained or env["not_performed_items"] or refused or turn["other_tools"] or unreported:
            to_read.append({"request": n, "verdict": env["verdict"], "numbers_not_from_calculator": unexplained,
                            "not_performed_items": env["not_performed_items"], "calculator_refused": refused,
                            "other_tools": turn["other_tools"], "billed_output_unreported": unreported})
        print(f"#{n:2d} {env['verdict']:20s} cost={r['cost']} charges={dict(charges)} numbers={len(numbers(obs))} "
              f"unexplained={len(unexplained)} not_performed={len(env['not_performed_items'])} refused={len(refused)}",
              flush=True)

    run_state_dir, replay_state_dir = run / "env_ws" / "calc_state", ws / "calc_state"
    run_states = sorted(p.name for p in run_state_dir.glob("*.json")) if run_state_dir.is_dir() else []
    replay_states = sorted(p.name for p in replay_state_dir.glob("*.json")) if replay_state_dir.is_dir() else []
    if run_states != replay_states:
        problems.append({"request": "state", "kind": "state", "what": f"state files {run_states} != replay {replay_states}"})
    for name in set(run_states) & set(replay_states):
        if json.loads((run_state_dir / name).read_text()) != json.loads((replay_state_dir / name).read_text()):
            problems.append({"request": "state", "kind": "state", "what": f"final calc_state/{name} differs from the replay"})

    report = {"run_dir": str(run), "transcript": str(transcript), "n_requests": len(requests),
              "billing_checked_against_charge_lines": any_charge_line, "problems": problems, "to_read": to_read,
              "calculator_commands_with_scripts_or_file_writes": shell_notes,
              "billing_from_charge_lines": billing_fixes, "guard_refused_calculator_runs": guard_refusals,
              "commands_referring_to_run_dir_not_replayed": run_dir_cmds}
    (run / "env_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for f in billing_fixes:
        print(f"[env_check] request {f['request']}: harness billed from CHARGE lines {f['charge_lines']}; "
              f"the environment had listed {f['env_listed']}")
    for g in guard_refusals:
        print(f"[env_check] request {g['request']}: guard refused (not run, not billed) {g['commands']}")
    print(f"\n[env_check] machine-found problems: {len(problems)}")
    for p in problems:
        print(f"  request {p['request']} [{p['kind']}] {p['what']}")
    n_unexplained = sum(len(t["numbers_not_from_calculator"]) for t in to_read)
    n_unreported = sum(len(t["billed_output_unreported"]) for t in to_read)
    print(f"[env_check] for a reader: {len(to_read)} requests with {n_unexplained} numbers not from the calculator, "
          f"{n_unreported} billed outputs mostly unreported, refusals or non-calculator tool use; details in "
          f"{run / 'env_check.json'}")
    if not any_charge_line:
        print("[env_check] the calculator printed no CHARGE lines: action_types were checked against the cost table only")
    return report
