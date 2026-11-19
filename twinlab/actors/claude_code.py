"""Claude Code as the actor, with a fresh, context-free config inside the sandbox (sandbox.py).

- The per-run config dir is mounted where the logged-in config lives (settings.CLAUDE_CONFIG_DIR). It has no
  CLAUDE.md, skills, plugins, agents, memory, or history. A Claude model gets only the login keys of .claude.json,
  and the real .credentials.json is bind-mounted on top so auth (and token refresh) writes back to the user's
  credential file. A non-Claude model (machine.yaml model_backends) gets no login: its backend token authenticates,
  so Claude Code need not be signed in.
- Only the `lab` MCP server is configured (--strict-mcp-config).
- guard_hook.py (guard.py installs it, needles stored only as hashes): PreToolUse denies WebSearch/WebFetch/Bash
  requests naming the source; PostToolUse withholds any tool output string that contains a blocklisted string.
- --disable-slash-commands removes all skills (claude.ai account skills are synced into any config dir).
"""
import collections
import json
import sys
import uuid

from .. import settings
from ..claude_cli import claude_env_for, limit_hit, model_backend, wait_for_reset
from ..lab import LAB_REQUEST_TOTAL_S
from . import guard
from .base import ActorAgent
from .sandbox import actor_env, bwrap_prefix

# Only these keys of the logged-in .claude.json are copied into the actor's config dir (auth + onboarding), for a
# Claude model only.
CLAUDE_JSON_KEEP = [
    "oauthAccount", "userID", "hasCompletedOnboarding", "lastOnboardingVersion", "installMethod",
    "autoUpdates", "machineID", "migrationVersion", "firstStartTime",
    "opusProMigrationComplete", "sonnet1m45MigrationComplete",
]
# Blacklist, not a whitelist: tools that reach outside the run (other Claude sessions on the
# machine, remote triggers, notifications) or schedule work after it. Subagent tools (Task, Workflow) stay; their
# results come back as tool results and do not need SendMessage.
DISALLOWED_TOOLS = ["ListAgents", "SendMessage", "RemoteTrigger", "PushNotification",
                    "CronCreate", "CronDelete", "CronList", "ScheduleWakeup"]
EXTRA_FLAGS = ["--disable-slash-commands", "--disallowed-tools", ",".join(DISALLOWED_TOOLS)]
EMPTY_TOOL_NAME ='"name":"","input":'        # a tool_use block whose name the model left empty
EMPTY_TOOL_NAME_ERROR = "Param Incorrect"     # the provider's 400 for that history (via LiteLLM)
MCP_TIMEOUT_MARGIN_S = 300        # the actor's MCP tool timeout exceeds a lab request's total limit by this much
MCP_CONFIG_FILE = "lab_mcp.json"  # in the per-run config dir, rewritten with the lab url on every invocation
SYNTHETIC_MODEL = "<synthetic>"   # Claude Code's placeholder model on locally generated assistant messages


class ClaudeCodeActor(ActorAgent):
    name = "claude-code"

    def prepare(self, case, model):
        guard.install(self.cfg / "hooks", case)
        if model_backend(model) is None:
            host = json.loads((settings.CLAUDE_CONFIG_DIR / ".claude.json").read_text())
            slim = {k: host[k] for k in CLAUDE_JSON_KEEP if k in host}
            assert "oauthAccount" in slim, \
                f"{model}: the .claude.json in {settings.CLAUDE_CONFIG_DIR} has no oauthAccount; sign in Claude Code"
            (self.cfg / ".claude.json").write_text(json.dumps(slim, indent=1))
            (self.cfg / ".credentials.json").touch()     # bind-mount target for the real credentials
            print(f"[actor] {model}: Claude model, login of {settings.CLAUDE_CONFIG_DIR} mounted", flush=True)
        else:
            (self.cfg / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True}, indent=1))
            print(f"[actor] {model}: backend token, no Claude Code login mounted", flush=True)
        hook = [{"type": "command",   # path inside the sandbox
                 "command": f"{sys.executable} {settings.CLAUDE_CONFIG_DIR}/hooks/{guard.HOOK_FILE}"}]
        hook_settings = {"hooks": {"PreToolUse": [{"matcher": "|".join(guard.CLAUDE_PRE_TOOLS), "hooks": hook}],
                                   "PostToolUse": [{"matcher": "*", "hooks": hook}]}}
        (self.cfg / "settings.json").write_text(json.dumps(hook_settings, indent=1))
        case.write_actor_workspace(self.ws)

    def new_session_id(self):
        return str(uuid.uuid4())

    def command(self, message, model, mcp_url, session_id, new_session):
        binds = [(self.cfg, settings.CLAUDE_CONFIG_DIR)]
        if model_backend(model) is None:
            creds = settings.CLAUDE_CONFIG_DIR / ".credentials.json"
            binds.append((creds, creds))
        # the hook and the settings that install it, read-only (hooks/guard.log stays writable for the hook)
        protected = [(self.cfg / rel, settings.CLAUDE_CONFIG_DIR / rel) for rel in
                     ("settings.json", f"hooks/{guard.HOOK_FILE}", f"hooks/{guard.NEEDLES_FILE}")]
        # the lab url (with the run's token) goes into a file, not argv: without own_proc, other runs' actors can
        # read every host command line
        (self.cfg / MCP_CONFIG_FILE).write_text(json.dumps({"mcpServers": {"lab": {"type": "http", "url": mcp_url}}}))
        self.model = model   # env() is called right after command() and needs the model's backend
        return bwrap_prefix(binds, self.ws, protected) + [
            str(settings.CLAUDE_BIN), *settings.CLAUDE_LAUNCH_FLAGS, *EXTRA_FLAGS, "-p", message,
            "--model", model, *(["--effort", self.effort] if self.effort is not None else []),
            "--output-format", "stream-json", "--verbose",
            "--mcp-config", str(settings.CLAUDE_CONFIG_DIR / MCP_CONFIG_FILE), "--strict-mcp-config",
            *(["--session-id", session_id] if new_session else ["--resume", session_id]),
        ]

    def env(self):
        # same path inside the sandbox, but the per-run config dir; a non-Claude model gets its backend
        # (claude_cli.model_backend): its token takes precedence over the mounted login
        # MCP_TOOL_TIMEOUT: a lab request may take LAB_REQUEST_TOTAL_S with env retries; the CLI gave up after 300 s
        # (E3 007-02 rth #1, 2026-11-06: the actor never got an answer the ledger recorded)
        return actor_env({**claude_env_for(self.model, settings.CLAUDE_CONFIG_DIR),
                          "MCP_TOOL_TIMEOUT": str((LAB_REQUEST_TOTAL_S + MCP_TIMEOUT_MARGIN_S) * 1000)}, ())

    def session_id_from_stream(self, stream):
        raise AssertionError("Claude Code sessions are started with a known id")

    def served_models(self, stream):
        """Models that actually produced the assistant messages. The requested --model can be silently served by
        another model (both 009 actor runs of 2026-09-27 asked for claude-opus-5-5 and were answered by
        claude-opus-5), so it is checked, not assumed. The last line may still be being written while the actor
        runs; a line that does not parse yet is left for the next check. Only the main thread counts: a subagent the
        actor starts itself may use another model (032-03 web-fetch on Haiku), and is not a substitution."""
        c = collections.Counter()
        for line in stream.read_text(errors="replace").splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (m.get("type") == "assistant" and not m.get("parent_tool_use_id")
                    and m["message"].get("model") != SYNTHETIC_MODEL):
                c[m["message"].get("model")] += 1
        return c

    def usage_limit_hit(self, stream_tail):
        return limit_hit(stream_tail)

    def wait_for_usage_reset(self, stream_tail):
        return wait_for_reset(stream_tail, "actor")

    def terminal_api_error(self, stream_tail):
        # mimo-v2.6-pro emitted a tool call with an empty name (042-02, 058-02 runs of 2026-11-05); Claude Code keeps
        # it in the history and the provider rejects every later request with "Param Incorrect" (a resume of 042-02
        # failed the same way): the model's own output ended the session
        if EMPTY_TOOL_NAME_ERROR in stream_tail and any(
                EMPTY_TOOL_NAME in s.read_text(errors="replace") for s in self.run_dir.glob("actor_stream_*.jsonl")):
            return "empty_tool_name_in_history"
        return None

    @staticmethod
    def tool_events(stream):
        events = []
        for line in stream.read_text().splitlines():
            m = json.loads(line)
            if m["type"] not in ("assistant", "user") or isinstance(m["message"]["content"], str):
                continue
            for c in m["message"]["content"]:
                if c["type"] == "tool_use":
                    events.append({"kind": "call", "name": c["name"], "input": c["input"]})
                elif c["type"] == "tool_result":
                    cc = c["content"]
                    if isinstance(cc, list):
                        cc = " ".join(x.get("text", f"<{x['type']}>") for x in cc)
                    events.append({"kind": "result", "text": str(cc), "lab": False})   # lab results: by text
        return events
