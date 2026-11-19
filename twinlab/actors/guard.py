"""Install the source guard (guard_hook.py) into an actor's sandboxed config dir. The needles are written only as
hashes (per UTF-8 byte length: crc32 for a fast prefilter, sha256 to confirm), never as text: anything in this
directory is readable by the actor (2026-11-05 mimo-v2.6-pro 058-02 decoded the earlier base64 list)."""
import hashlib
import json
import shutil
import zlib
from pathlib import Path

HOOK_FILE = "guard.py"
NEEDLES_FILE = "needles.json"
HOOK_SOURCE = Path(__file__).parent / "guard_hook.py"
# The tool result Claude Code records for a call the hook denied (probe 2026-11-11, actor stream):
# "PreToolUse:<tool> hook error: <DENY_MSG>", is_error true. The judge counts these in the actor stream, which the
# harness writes outside the sandbox; hooks/guard.log is inside it and the actor can edit or delete it.
DENY_MSG = "This source is excluded by the task protocol. Do not try to reach it, and continue with other routes."
CLAUDE_PRE_TOOLS = ("WebSearch", "WebFetch", "Bash")   # Claude Code tools whose request the hook checks
DENIED_RESULT = "PreToolUse:{tool} hook error: " + DENY_MSG
assert f'DENY_MSG = "{DENY_MSG}"' in HOOK_SOURCE.read_text(), "guard.DENY_MSG differs from guard_hook.DENY_MSG"


def needles_record(needles: list) -> dict:
    by_len = {}
    for s in sorted(set(needles)):
        b = s.encode("utf-8")
        assert b, "empty blocklist entry"
        d = by_len.setdefault(str(len(b)), {"crc32": [], "sha256": []})
        d["crc32"].append(zlib.crc32(b))
        d["sha256"].append(hashlib.sha256(b).hexdigest())
    return by_len


def install(hooks_dir: Path, case) -> None:
    """hooks_dir must not exist yet. Writes guard.py and needles.json (case blocklist + canary + benchmark name)."""
    hooks_dir.mkdir(parents=True)
    shutil.copy(HOOK_SOURCE, hooks_dir / HOOK_FILE)
    (hooks_dir / NEEDLES_FILE).write_text(json.dumps(needles_record(case.guard_needles)))
