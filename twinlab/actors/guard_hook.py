"""Hook installed in the actor's sandboxed config dir (Claude Code settings.json; Codex hooks.json).
- PreToolUse: denies a WebSearch query, WebFetch URL or Bash command that contains a blocklisted string.
- PostToolUse: if a tool's output contains a blocklisted string, every string of the output that contains one is
  replaced by a fixed message before the model sees it (Claude Code `updatedToolOutput`, same shape as the output).
  Outgoing checks alone let content in: actors fetched papers and database records with commands that named no
  blocked string (qwen3.8-max 037-02, 052-01, 075-01, 2026-11-05).
Both messages tell the actor the source is excluded and to go on; judge.audit_actor invalidates a run only after
BLOCKED_ATTEMPTS_INVALID denied requests or when a blocked string reached the actor unreplaced.
The blocklist is stored only as hashes (needles.json: per UTF-8 byte length, the crc32 and sha256 of each needle), so
reading this directory reveals no source string (mimo-v2.6-pro 058-02 2026-11-05 decoded the earlier base64 list and
read the paper title and DOI from it)."""
import hashlib
import json
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDLES = json.load(open(os.path.join(HERE, "needles.json")))
BY_LEN = {int(n): (set(v["crc32"]), set(v["sha256"])) for n, v in NEEDLES.items()}
DENY_MSG = "This source is excluded by the task protocol. Do not try to reach it, and continue with other routes."
REDACTED_MSG = ("[content withheld: this source is excluded by the task protocol. Do not try to reach it, and continue "
                "with other routes.]")


def log(rec):
    rec["t"] = time.time()
    with open(os.path.join(HERE, "guard.log"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def hit(s: str) -> bool:
    """True if s contains a blocklisted string (byte substring match on UTF-8, crc32 prefilter, sha256 confirm)."""
    b = s.encode("utf-8", errors="surrogatepass")
    for n, (crcs, shas) in BY_LEN.items():
        for i in range(len(b) - n + 1):
            w = b[i:i + n]
            if zlib.crc32(w) in crcs and hashlib.sha256(w).hexdigest() in shas:
                return True
    return False


def redact(x):
    """Same structure; every string containing a blocklisted string replaced by REDACTED_MSG."""
    if isinstance(x, dict):
        return {k: redact(v) for k, v in x.items()}
    if isinstance(x, list):
        return [redact(v) for v in x]
    if isinstance(x, str) and hit(x):
        return REDACTED_MSG
    return x


def out(event: str, **fields):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, **fields}}))


ev = json.load(sys.stdin)
event = ev["hook_event_name"]
tool, inp = ev["tool_name"], ev["tool_input"]
if event == "PreToolUse":
    probe = {"WebSearch": lambda: inp["query"], "WebFetch": lambda: inp["url"],
             "Bash": lambda: inp["command"]}.get(tool)
    if probe is not None:
        blocked = hit(probe())
        log({"event": event, "tool": tool, "blocked": blocked, "input": probe()[:300]})
        if blocked:
            out(event, permissionDecision="deny", permissionDecisionReason=DENY_MSG)
elif event == "PostToolUse":
    resp = ev["tool_response"]
    new = redact(resp)   # strings are checked one by one, so the output keeps its shape (else Claude Code ignores it)
    if new != resp:
        log({"event": event, "tool": tool, "redacted": True})
        if "CODEX_HOME" in os.environ:
            # Codex 0.166 ignores updatedToolOutput; "decision: block" replaces the tool result with the reason
            # ("PostToolUse hook blocked the tool result", probe 2026-11-06)
            print(json.dumps({"decision": "block", "reason": REDACTED_MSG}))
        else:
            out(event, updatedToolOutput=new)
else:
    raise AssertionError(f"guard hook installed for an unexpected event {event!r}")
