"""Codex CLI as the actor (`codex exec --json`), inside the same sandbox as Claude Code (sandbox.py).

- CODEX_HOME is a per-run dir mounted at ~/.codex inside the sandbox. Its config.toml holds only the model provider
  of machine.yaml, the web-search setting, and the `lab` MCP server; nothing of the user's own Codex config.
- The provider's API key is passed as the environment variable named by the provider's env_key.
- config.toml points model_catalog_json at a copy of machine.yaml codex.model_catalog: official model metadata
  (reasoning defaults, context window, encrypted reasoning) for names Codex does not know, e.g. the gateway's
  azure_openai/<slug>; without it Codex falls back to metadata that sends no reasoning settings (2026-11-05).
- Codex's own sandbox is off (--dangerously-bypass-approvals-and-sandbox): bubblewrap is the sandbox.
- guard_hook.py (guard.py installs it, needles stored only as hashes; CODEX_HOME/hooks.json, same payload as Claude
  Code: tool_name "Bash", tool_input.command): PreToolUse denies shell commands that name the source record,
  PostToolUse withholds shell output that contains a blocklisted string. Codex runs hooks only after their trust is
  persisted, so every invocation passes --dangerously-bypass-hook-trust (the hook is ours). The hosted web_search tool
  does not go through hooks (probe 2026-11-05); since 2026-11-06 Codex talks to
  the provider through websearch_relay.py (one per run, base_url overridden per invocation), which drops answers
  whose web searches reached the source and adds a protocol notice instead. Runs prepared before that have no
  websearch_guard.json and resume without the relay; their leaks are caught by the contamination audit.
- stream_idle_timeout_ms is raised (STREAM_IDLE_TIMEOUT_MS) because the relay hands Codex an answer only once it is
  complete.
- The event stream does not say which model answered, so the served model is not checked (REPORTS_SERVED_MODEL).
- Codex assigns the session id (thread_id of the first event); later invocations use `codex exec resume <id>`.
"""
import json
import os
import re
import shutil
import sys

from .. import settings
from . import guard, websearch_relay
from .base import ActorAgent
from .sandbox import actor_env, bwrap_prefix

SANDBOX_CODEX_HOME = settings.HOME / ".codex"
MODEL_CATALOG_NAME = "model_catalog.json"   # copy of settings.CODEX_MODEL_CATALOG in the per-run CODEX_HOME
# One lab call waits for the environment agent (minutes, or hours if the environment's claude.ai quota runs out
# and is waited out); Codex must not give up on the tool call before that.
MCP_TOOL_TIMEOUT_S = 24 * 3600
MCP_STARTUP_TIMEOUT_S = 60
# the url line of [mcp_servers.lab] in config.toml, as prepare() writes it or as command() last set it
MCP_URL_LINE = re.compile(r'^url = "http://127\.0\.0\.1:\d+/mcp[^"]*".*$', re.M)
STRIP_ENV_PREFIXES = ("CLAUDE", "ANTHROPIC", "CTX_", "OPENAI", "CODEX")
STRIP_ENV_PROXY = ("http_proxy", "https_proxy", "no_proxy", "all_proxy")
# item types of `codex exec --json` that are not tool use
NON_TOOL_ITEMS = {"agent_message", "reasoning", "todo_list", "error"}
# provider errors that every later request of the session repeats: the run ends and is judged as it stands
# (Azure allows 50 images per request; 058-02 luna got 51 lab images)
TERMINAL_API_ERRORS = {"Exceeded maximum number of images": "max_images_per_request"}
STREAM_IDLE_TIMEOUT_MS = 3600 * 1000   # Codex default 300000; the relay sends nothing until the answer is complete


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    assert isinstance(v, str), f"unsupported TOML value {v!r}"
    return json.dumps(v)   # a JSON string is a valid TOML basic string


class CodexActor(ActorAgent):
    name = "codex"
    REPORTS_SERVED_MODEL = False
    relay = None   # websearch_relay.Relay, started at the first invocation of this process

    def relay_args(self):
        if not (self.run_dir / websearch_relay.STATE_FILE).exists():
            print(f"[ws-relay] {self.run_dir} was prepared before the web-search relay: Codex talks to the "
                  f"provider directly", flush=True)
            return []
        assert settings.CODEX_PROXY is None, "the web-search relay connects to the provider directly"
        if self.relay is None:
            self.relay = websearch_relay.Relay(self.run_dir, settings.CODEX_PROVIDER["base_url"])
        name = settings.CODEX_PROVIDER["name"]
        return ["-c", f"model_providers.{name}.base_url="
                      f"{json.dumps(self.relay.base_url(settings.CODEX_PROVIDER['base_url']))}",
                "-c", f"model_providers.{name}.stream_idle_timeout_ms={STREAM_IDLE_TIMEOUT_MS}"]

    def prepare(self, case, model):
        assert settings.CODEX_BIN.exists(), f"machine.yaml codex.bin: {settings.CODEX_BIN} does not exist"
        self.cfg.mkdir(parents=True)
        p = settings.CODEX_PROVIDER
        shutil.copyfile(settings.CODEX_MODEL_CATALOG, self.cfg / MODEL_CATALOG_NAME)   # visible inside the sandbox
        lines = [f"model_provider = {toml_value(p['name'])}",
                 f"web_search = {toml_value(settings.CODEX_WEB_SEARCH)}",
                 f"model_catalog_json = {toml_value(str(SANDBOX_CODEX_HOME / MODEL_CATALOG_NAME))}",
                 "",
                 f"[model_providers.{p['name']}]",
                 *(f"{k} = {toml_value(v)}" for k, v in p.items()),
                 "",
                 "[mcp_servers.lab]",
                 'url = "http://127.0.0.1:0/mcp"',   # replaced on every invocation (command())
                 f"tool_timeout_sec = {MCP_TOOL_TIMEOUT_S}",
                 f"startup_timeout_sec = {MCP_STARTUP_TIMEOUT_S}"]
        (self.cfg / "config.toml").write_text("\n".join(lines) + "\n")
        guard.install(self.cfg / "hooks", case)
        hook = [{"type": "command",   # path inside the sandbox
                 "command": f"{sys.executable} {SANDBOX_CODEX_HOME}/hooks/{guard.HOOK_FILE}"}]
        (self.cfg / "hooks.json").write_text(json.dumps({"hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": hook}],
            "PostToolUse": [{"matcher": "Bash", "hooks": hook}]}}, indent=1))
        websearch_relay.write_state(self.run_dir, case.guard_needles)
        case.write_actor_workspace(self.ws)

    def new_session_id(self):
        return None

    def command(self, message, model, mcp_url, session_id, new_session):
        # effort None: runs started before 2026-11-05 recorded none and keep the catalog default on resume
        effort = [] if self.effort is None else ["-c", f"model_reasoning_effort={json.dumps(self.effort)}"]
        # the hook and hooks.json, read-only (hooks/guard.log stays writable for the hook)
        protected = [(self.cfg / rel, SANDBOX_CODEX_HOME / rel) for rel in
                     ("hooks.json", f"hooks/{guard.HOOK_FILE}", f"hooks/{guard.NEEDLES_FILE}")]
        # the lab url (with the run's token) goes into config.toml, not argv: without own_proc, other runs' actors
        # can read every host command line
        cfg_toml = self.cfg / "config.toml"
        text, n = MCP_URL_LINE.subn(f"url = {json.dumps(mcp_url)}", cfg_toml.read_text())
        assert n == 1, f"{cfg_toml}: expected one [mcp_servers.lab] url line, found {n}"
        cfg_toml.write_text(text)
        return bwrap_prefix([(self.cfg, SANDBOX_CODEX_HOME)], self.ws, protected) + [
            str(settings.CODEX_BIN), "exec", *([] if new_session else ["resume", session_id]),
            "--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust", "-m", model, *effort, *self.relay_args(), message,
        ]

    def env(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(STRIP_ENV_PREFIXES) and k.lower() not in STRIP_ENV_PROXY}
        env["CODEX_HOME"] = str(SANDBOX_CODEX_HOME)
        env[settings.CODEX_PROVIDER["env_key"]] = settings.CODEX_API_KEY
        if settings.CODEX_PROXY is not None:
            env.update(HTTP_PROXY=settings.CODEX_PROXY, HTTPS_PROXY=settings.CODEX_PROXY)
        return actor_env(env, (settings.CODEX_PROVIDER["env_key"],))

    def session_id_from_stream(self, stream):
        first = json.loads(stream.read_text().splitlines()[0])
        assert first["type"] == "thread.started", f"codex stream does not start with thread.started: {first}"
        return first["thread_id"]

    def served_models(self, stream):
        raise AssertionError("codex streams do not report the served model")

    def usage_limit_hit(self, stream_tail):
        return False   # API-key billing: no subscription usage limit to wait out

    def wait_for_usage_reset(self, stream_tail):
        raise AssertionError("codex has no usage limit to wait for")

    def terminal_api_error(self, stream_tail):
        for needle, label in TERMINAL_API_ERRORS.items():
            if needle in stream_tail:
                return label
        return None

    @staticmethod
    def tool_events(stream):
        events = []
        for line in stream.read_text().splitlines():
            m = json.loads(line)
            if m["type"] != "item.completed":
                continue
            it = m["item"]
            t = it["type"]
            if t in NON_TOOL_ITEMS:
                continue
            if t == "command_execution":
                events.append({"kind": "call", "name": "shell", "input": {"command": it["command"]}})
                events.append({"kind": "result", "text": it["aggregated_output"], "lab": False})
            elif t == "mcp_tool_call":
                lab = it["server"] == "lab"
                events.append({"kind": "call", "name": f"mcp__{it['server']}__{it['tool']}",
                               "input": it["arguments"]})
                if it["error"] is not None:
                    text = f"ERROR: {it['error']}"
                else:
                    text = " ".join(x.get("text", f"<{x['type']}>") for x in it["result"]["content"])
                events.append({"kind": "result", "text": text, "lab": lab})
            elif t == "web_search":
                events.append({"kind": "call", "name": "web_search", "input": {"query": it["query"]}})
            elif t == "file_change":
                events.append({"kind": "call", "name": "file_change", "input": it["changes"]})
            else:
                raise AssertionError(f"unknown codex item type {t!r} in {stream}: {line[:300]}")
        return events
