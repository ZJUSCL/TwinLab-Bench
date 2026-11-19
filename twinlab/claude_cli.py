"""Launch Claude Code processes: the environment of a process, one-shot JSON calls, and claude.ai usage limits.

The environment agent, the judge, and the memorization probe call `claude -p ... --output-format json` through
run_claude_json(). The Claude Code actor builds its own command (actors/claude_code.py) but uses claude_env().
"""
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

import jsonschema

from . import settings

# Inherited variables with these prefixes are dropped so that a harness started from inside a Claude Code session
# does not leak its child-session state (CLAUDE_CODE_CHILD_SESSION etc.) into the processes it starts.
# CTX_: the user's shell alias also sets CTX_HOOKS_ON (context hooks); harness sessions run without them.
STRIP_ENV_PREFIXES = ("CLAUDE", "ANTHROPIC", "CTX_")
STRIP_ENV_LOWER = ("http_proxy", "https_proxy", "no_proxy", "all_proxy")

LIMIT_RE = re.compile(r"hit your [a-z ]*limit", re.I)
RESET_RE = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(UTC\)", re.I)
LIMIT_MARGIN_S = 90


def claude_env(config_dir: Path) -> dict:
    """os.environ minus inherited Claude/proxy variables, plus the variables of the user's interactive launch
    (config dir, locale, proxy, zsh as SHELL)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(STRIP_ENV_PREFIXES) and k.lower() not in STRIP_ENV_LOWER}
    env.update(
        CLAUDE_CONFIG_DIR=str(config_dir),
        TZ="",
        LANG="en_US.UTF-8",
        LC_ALL="",
        HTTP_PROXY=settings.PROXY,
        HTTPS_PROXY=settings.PROXY,
        no_proxy=settings.NO_PROXY,
        SHELL=str(settings.SHELL_BIN),
    )
    return env


# Claude Code model slots that a non-Claude backend fills with the same model, so that no helper or subagent call
# (haiku helper, sonnet/opus aliases of the Task tool) goes to another model.
BACKEND_MODEL_VARS = ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                      "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


def model_backend(model: str):
    """None for a Claude model (logged-in config, unchanged); the machine.yaml model_backends entry otherwise."""
    if model.startswith("claude-"):
        assert model not in settings.MODEL_BACKENDS, f"{model}: Claude models run on the logged-in config"
        return None
    assert model in settings.MODEL_BACKENDS, f"model {model} has no model_backends entry in machine.yaml"
    return settings.MODEL_BACKENDS[model]


def config_dir_for(model: str) -> Path:
    """Config dir of env and judge sessions of this model (where their transcripts are written)."""
    b = model_backend(model)
    return settings.CLAUDE_CONFIG_DIR if b is None else b["config_dir"]


def claude_env_for(model: str, config_dir: Path) -> dict:
    """claude_env for a call of `model`. A non-Claude model gets its backend's endpoint, token and proxy, and every
    model slot of the CLI set to it."""
    env = claude_env(config_dir)
    b = model_backend(model)
    if b is None:
        return env
    del env["HTTP_PROXY"], env["HTTPS_PROXY"]
    if b["proxy"] is not None:
        env.update(HTTP_PROXY=b["proxy"], HTTPS_PROXY=b["proxy"])
    env.update(ANTHROPIC_BASE_URL=b["base_url"], ANTHROPIC_AUTH_TOKEN=b["auth_token"], ANTHROPIC_API_KEY="",
               NO_PROXY=settings.NO_PROXY, **{k: model for k in BACKEND_MODEL_VARS}, **b["cli_env"])
    return env


def backend_record(model: str):
    """What run_config.json records about a model's backend (never the token); None for a Claude model."""
    b = model_backend(model)
    if b is None:
        return None
    return {"base_url": b["base_url"], "api_format": b["api_format"], "proxy": b["proxy"],
            "config_dir": str(b["config_dir"]), "cli_env": b["cli_env"], "served": served_model(model, b)}


PROBE_MAX_TOKENS = 5


def served_model(model: str, b: dict) -> dict:
    """What the endpoint serves under this name right now, from a 5-token call (record before each run).
    api_format openai: the proxy's OpenAI route, upstream model id and system_fingerprint (deepseek-flash gave
    fingerprint bc19ba97cbe016479903222bb1578db6 on 11-05). anthropic: /v1/messages, the model id it answers with
    (this format has no fingerprint)."""
    import urllib.request
    handlers = [urllib.request.ProxyHandler({"http": b["proxy"], "https": b["proxy"]} if b["proxy"] else {})]
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": PROBE_MAX_TOKENS}
    if b["api_format"] == "openai":
        route = "/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {b['auth_token']}"}
    else:
        assert b["api_format"] == "anthropic", f"{model}: unknown api_format {b['api_format']!r}"
        route = "/v1/messages"
        headers = {"Content-Type": "application/json", "x-api-key": b["auth_token"], "anthropic-version": "2023-06-01"}
    req = urllib.request.Request(b["base_url"].rstrip("/") + route, json.dumps(body).encode(), headers)
    d = json.loads(urllib.request.build_opener(*handlers).open(req, timeout=120).read())
    rec = {"model": d["model"], "system_fingerprint": d.get("system_fingerprint"), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[backend] {model} served as {rec}", flush=True)
    return rec


def private_tmp_prefix(tmp_dir: Path, cwd: Path) -> list:
    """bwrap prefix that keeps the whole filesystem but gives the process its own /tmp (tmp_dir on the host): the
    environment writes scratch files such as /tmp/w.txt, which collided between parallel runs (redundancy audit
    2026-10-26). bwrap does not keep the working directory by itself, hence --chdir; cwd must not be under the host
    /tmp. A cwd inside tmp_dir is entered as /tmp/<rel>, so the process does not see the host path (the memorization
    probe's cwd must not name the case, 071 probe 2026-10-26)."""
    assert not cwd.resolve().is_relative_to("/tmp"), f"cwd {cwd} is under /tmp, which the call gets as its own"
    tmp_dir.mkdir(exist_ok=True)
    inner = cwd.resolve()
    if inner.is_relative_to(tmp_dir.resolve()):
        inner = Path("/tmp") / inner.relative_to(tmp_dir.resolve())
    return [str(settings.BWRAP), "--die-with-parent", "--bind", "/", "/", "--dev", "/dev",
            "--bind", str(tmp_dir), "/tmp", "--chdir", str(inner)]


def run_claude_json(args: list, cwd: Path, timeout_s: int, log_path: Path, tmp_dir: Path) -> dict:
    """Run `claude -p ... --output-format json` on the logged-in config (--safe-mode is the caller's job), with
    tmp_dir as its /tmp. Appends the raw call record to log_path and fails fast on any error."""
    return run_claude_json_env(args, cwd, timeout_s, log_path, tmp_dir, {})


def run_claude_json_env(args: list, cwd: Path, timeout_s: int, log_path: Path, tmp_dir: Path,
                        extra_env: dict) -> dict:
    """run_claude_json with extra process variables on top of claude_env (the env agent's guarded PATH,
    twinlab/calc_guard.py)."""
    # the prompt goes through stdin: judge prompts exceed the per-argument limit of execve (~128 KB)
    i = args.index("-p")
    prompt = args[i + 1]
    cmd = [*private_tmp_prefix(tmp_dir, cwd), str(settings.CLAUDE_BIN), *args[:i + 1], *args[i + 2:]]
    asked = args[args.index("--model") + 1]
    config_dir = config_dir_for(asked)
    structured = "--json-schema" in args
    if structured:   # StructuredOutput rejections are counted from the lines this call appends to the transcript
        sid = session_id_of(args)
        before = session_transcripts(sid, config_dir)
        assert len(before) <= 1, before
        n_before = len(jsonl_lines(before[0])) if before else 0
    while True:   # a claude.ai usage limit is waited out, then the same call is retried
        t0 = time.time()
        p = subprocess.run(cmd, cwd=cwd, env={**claude_env_for(asked, config_dir), **extra_env}, input=prompt,
                           capture_output=True, text=True, timeout=timeout_s)
        if (p.returncode != 0 or '"is_error":true' in p.stdout) and limit_hit(p.stdout + p.stderr):
            wait_for_reset(p.stdout + p.stderr, f"claude call in {cwd.name}")
            continue
        break
    rec = {"cmd": [a if len(a) < 400 else a[:400] + "...<truncated>" for a in cmd], "cwd": str(cwd),
           "rc": p.returncode, "secs": round(time.time() - t0, 1), "stdout": p.stdout, "stderr": p.stderr[-4000:]}
    rejections = None
    if structured and p.returncode == 0:   # a crashed call is reported by the assert below, uncounted
        after = session_transcripts(sid, config_dir)
        assert len(after) == 1, f"session {sid}: expected 1 transcript, found {after}"
        new_lines = jsonl_lines(after[0])[n_before:]
        rejections = structured_output_rejections(new_lines)
        rec["structured_output_rejections"] = len(rejections)
        rec["structured_output_rejection_errors"] = rejections
        rec["compactions"] = sum(COMPACT_MARK in line for line in new_lines)
        # replies cut at the output cap (thinking counts; 2026-11-04 deepseek-flash judges); logged, not fatal
        rec["max_tokens_stops"] = len({json.loads(line)["message"]["id"] for line in new_lines
                                       if MAX_TOKENS_MARK in line and '"type":"assistant"' in line})
        if rec["max_tokens_stops"]:
            print(f"[claude] WARNING {rec['max_tokens_stops']} replies stopped at max_tokens", flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    assert p.returncode == 0, f"claude exited {p.returncode}; see {log_path}\nstderr: {p.stderr[-2000:]}\nstdout: {p.stdout[-2000:]}"
    # an auto-compacted env or judge works from a summary of the case instead of the case (5 of 10 deepseek-flash
    # judges, 2026-11-04, before the backend's CLAUDE_CODE_MAX_CONTEXT_TOKENS); none of 702 Claude env sessions did
    assert not rec.get("compactions"), f"session auto-compacted {rec['compactions']}x; see {log_path}"
    out = json.loads(p.stdout)
    assert not out["is_error"], f"claude returned is_error: {out.get('result')}; see {log_path}"
    # a safety stop silently re-serves the call with another model (kquiz-009 2026-10-19: claude-opus-5 answered
    # with safety_stops 2); haiku is the CLI's own helper model and does not answer the prompt
    served = [m for m in out["modelUsage"] if "haiku" not in m]
    assert served == [asked], f"asked {asked}, served {served} (safety_stops={out.get('safety_stops')}); see {log_path}"
    print(f"[claude] {rec['secs']}s cost=${out.get('total_cost_usd')} turns={out.get('num_turns')}", flush=True)
    if structured:
        print(f"[claude] StructuredOutput rejections={len(rejections)}"
              + "".join(f"\n[claude]   rejected: {r[:300]}" for r in rejections), flush=True)
    return out


# ---- JSON answers through --json-schema (StructuredOutput tool) ----
# The CLI offers the model a StructuredOutput tool whose input is validated against the schema; a rejected input
# comes back as an is_error tool_result and the model corrects it in the same call; the accepted object is in
# out["structured_output"]. The extra turn the CLI logs after it is a <synthetic> message with 0 tokens (study
# removed 10-26 on a wrong cost reason, restored
# 10-30). Every object in the schemas is closed (additionalProperties false): extra keys are rejected, not ignored.

STRUCTURED_ANSWER_MARK = "# ANSWER SUBMISSION"
COMPACT_MARK = '"subtype":"compact_boundary"'   # the transcript line Claude Code writes when it auto-compacts
MAX_TOKENS_MARK = '"stop_reason":"max_tokens"'


def close_objects(schema):
    """Copy of a JSON schema with additionalProperties false on every object that lists properties."""
    if isinstance(schema, list):
        return [close_objects(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: (v if k in ("required", "enum", "properties") else close_objects(v)) for k, v in schema.items()}
    if "properties" in out:
        out["properties"] = {k: close_objects(v) for k, v in schema["properties"].items()}
        out["additionalProperties"] = False
    return out


def structured_answer_instructions() -> str:
    return (f"\n\n{STRUCTURED_ANSWER_MARK}\nEnd every turn by calling the StructuredOutput tool once with your answer; "
            "its input schema lists the keys and what each holds. Text outside that tool call is not read. If the "
            "tool rejects the input, correct what the error names and call it again.\n")


def session_id_of(args: list) -> str:
    for flag in ("--session-id", "--resume"):
        if flag in args:
            return args[args.index(flag) + 1]
    raise AssertionError(f"--json-schema call without --session-id/--resume (rejections are read from its transcript): {args[:3]}")


def jsonl_lines(path: Path) -> list:
    """Records of a jsonl file split on newline only: str.splitlines() also splits on characters JSON leaves raw
    inside strings (form feed, U+2028, ...), e.g. PDF text in a judge's Read result (072-02 mimo 20261105-131105)."""
    return [x for x in path.read_text(encoding="utf-8").split("\n") if x]


def session_transcripts(sid: str, config_dir: Path) -> list:
    return list((config_dir / "projects").glob(f"*/{sid}.jsonl"))


def structured_output_rejections(lines: list) -> list:
    """Error texts of the StructuredOutput calls the CLI rejected, from transcript lines (each tool_use id once)."""
    so_ids, errs = set(), {}
    for line in lines:
        msg = json.loads(line).get("message")
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            continue
        for c in msg["content"]:
            if c.get("type") == "tool_use" and c.get("name") == "StructuredOutput":
                so_ids.add(c["id"])
            elif c.get("type") == "tool_result" and c.get("is_error"):
                errs[c["tool_use_id"]] = c["content"] if isinstance(c["content"], str) else json.dumps(c["content"])
    return [e for tid, e in errs.items() if tid in so_ids]


def structured_answer(out: dict, schema: dict, log_path: Path) -> dict:
    """The object the CLI accepted through StructuredOutput, validated again here (fail fast)."""
    assert out.get("structured_output") is not None, \
        f"no structured_output in the CLI result; see {log_path}\nresult: {str(out.get('result'))[:2000]}"
    obj = out["structured_output"]
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        raise AssertionError(f"structured_output does not match the schema: {e.message} at "
                             f"{list(e.absolute_path)}; see {log_path}")
    return obj


# ---- LEGACY: JSON answers as final text (runs started 2026-10-26..10-30) ----
# From 793eb461 (10-26) to the restore of --json-schema (10-30) the schema went into the system prompt under
# "# ANSWER FORMAT" and the harness parsed the final message. Env sessions of those runs keep this path on resume
# (EnvAgent detects the marker); new runs never use it.

FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(.*)\n```", re.S)
STRAY_TAIL_RE = re.compile(r"[\s}\]]*(?:```[\s}\]]*)?")


def json_answer_instructions(schema: dict) -> str:
    return ("\n\n# ANSWER FORMAT\nEnd every turn with exactly one JSON object that validates against the schema "
            "below and nothing else: no text before or after it, no code fence. Anything meant for the actor (for "
            "example that no figure is attached) goes inside the object, never in front of it.\n```json\n"
            f"{json.dumps(schema, indent=1)}\n```\n")


def parse_json_answer(out: dict, schema: dict, log_path: Path) -> dict:
    text = out["result"].strip()
    # the whole message may be one ```json fenced block (070 replay, 2026-10-26: Sonnet env fenced its first answer
    # despite the instruction); anything besides that one block still fails
    m = FENCED_JSON_RE.fullmatch(text)
    if m:
        print(f"[claude] answer was one fenced JSON block; fence removed", flush=True)
        text = m.group(1)
    # one sentence of prose, then the object (001-01 Opus runs 20261027-041148 #11 and 20261027-043031 #1: Sonnet env
    # wrote "The calculator printed no gel figure ..." before the JSON, twice in a day once --json-schema was gone).
    # Accepted only as plain text without "{" followed by one object that ends the message; the prose is printed.
    # prose, then one fenced block that ends the message (069-02 Opus run 20261029-180354 #1: "Nothing in your request
    # could be done. ..." then ```json ... ```); the prose must not contain "{" and is printed, the fence is removed.
    j = text.find("```")
    if j > 0 and "{" not in text[:j]:
        m = FENCED_JSON_RE.fullmatch(text[j:])
        if m:
            print(f"[claude] PROSE BEFORE FENCED JSON dropped ({j} chars): {text[:j].strip()[:300]!r}", flush=True)
            text = m.group(1)
    i = text.find("{")
    # complete object, then only stray closing braces/brackets or a closing fence (003-03 Opus run 20261029-204001:
    # judge wrote a valid object plus one extra "}"); the tail is cut and printed, any other tail still fails below.
    if i >= 0:
        try:
            _, end = json.JSONDecoder().raw_decode(text, i)
        except json.JSONDecodeError:
            end = len(text)
        if text[end:].strip() and STRAY_TAIL_RE.fullmatch(text[end:]):
            print(f"[claude] STRAY TAIL AFTER JSON dropped: {text[end:]!r}", flush=True)
            text = text[:end]
    if i > 0 and not text.startswith("{"):
        try:
            obj, end = json.JSONDecoder().raw_decode(text, i)
        except json.JSONDecodeError:
            end = -1
        if end == len(text):
            print(f"[claude] PROSE BEFORE JSON dropped ({i} chars): {text[:i].strip()[:300]!r}", flush=True)
            text = text[i:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        # the end of the message goes on the first line: the discard note a resumed env reads is built from this line
        # (lab.Lab.restore); 001-03 run 20261029-202451 ended with '","}' twice and the env never saw that tail
        raise AssertionError(f"final message is not one JSON object ({e}); it ends with {text[-80:]!r}; "
                             f"see {log_path}\n{text[:2000]}")
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        raise AssertionError(f"final message does not match the schema: {e.message} at "
                             f"{list(e.absolute_path)}; see {log_path}")
    return obj


# ---- claude.ai usage limits: "You've hit your session limit · resets 6:10pm (UTC)" ----
# Runs sleep until the reset (plus a margin) and then continue on their own.

def limit_hit(text: str) -> bool:
    return bool(LIMIT_RE.search(text))


def seconds_until_reset(text: str) -> float:
    m = RESET_RE.search(text)
    assert m, f"usage limit hit but no parsable reset time in: {text[-300:]}"
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    h = h % 12 + (12 if ap == "pm" else 0)
    now = dt.datetime.now(dt.timezone.utc)
    reset = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if reset <= now:
        if now - reset < dt.timedelta(hours=3):   # the reset just passed (clock skew / late message)
            return LIMIT_MARGIN_S
        reset += dt.timedelta(days=1)
    return (reset - now).total_seconds() + LIMIT_MARGIN_S


def wait_for_reset(text: str, who: str) -> float:
    """Sleep until the quota resets; returns the seconds slept."""
    s = seconds_until_reset(text)
    print(f"[quota] {who}: usage limit hit; sleeping {s / 60:.1f} min until the reset", flush=True)
    time.sleep(s)
    print(f"[quota] {who}: resuming after the reset", flush=True)
    return s
