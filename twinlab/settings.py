"""Machine settings: where the agent CLIs live, which logged-in config they use, the proxy, and the sandbox.

They are read from machine.yaml at the repository root. That file is not committed (it holds a proxy password and
an API key); copy machine.example.yaml and fill it in. Nothing else in the package depends on the computer.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MACHINE_FILE = ROOT / "machine.yaml"

assert MACHINE_FILE.is_file(), f"{MACHINE_FILE} is missing: copy machine.example.yaml to machine.yaml and fill it in"
_m = yaml.safe_load(MACHINE_FILE.read_text())


def _path(p: str) -> Path:
    return Path(p).expanduser()


HOME = Path.home()

# ---- Claude Code: the environment agent and the judge always; the actor when --actor-agent claude-code ----
CLAUDE_BIN = _path(_m["claude"]["bin"])
CLAUDE_INSTALL_DIR = _path(_m["claude"]["install_dir"])
# A logged-in Claude Code config dir. The environment agent and the judge run on it directly; the actor gets a fresh
# copy with only the login keys (see actors/claude_code.py), mounted at the same path inside its sandbox.
CLAUDE_CONFIG_DIR = _path(_m["claude"]["config_dir"])
CLAUDE_LAUNCH_FLAGS = list(_m["claude"]["launch_flags"])
# zsh for Claude Code's Bash tool (env, judge, actor). A zsh whose loader prints a warning (anaconda's
# libtinfow "no version information available") puts that line into every tool result the models read.
SHELL_BIN = _path(_m["claude"]["shell"])
PROXY = _m["claude"]["proxy"]
NO_PROXY = _m["claude"]["no_proxy"]
# Non-Claude models run inside Claude Code through an Anthropic-compatible endpoint: model id -> base_url,
# auth_token, proxy (null = direct), config_dir, cli_env (extra CLI variables, e.g. retries). Claude models (claude-*) need no entry and run on the logged-in
# config above; any other model must have one (claude_cli.model_backend).
# An entry with `endpoint: deepseek` takes base_url and auth_token from the environment (_deepseek_endpoint).
# api_format: "openai" = an OpenAI-format endpoint behind the local LiteLLM proxy (scripts/llm_proxy.py), probed on
# its /v1/chat/completions route; "anthropic" = Claude Code talks to the endpoint directly, probed on /v1/messages.
DEEPSEEK_ANTHROPIC_DEFAULT = "https://api.deepseek.com/anthropic"


def _deepseek_endpoint(model: str, b: dict) -> dict:
    """DEEPSEEK_BASE_URL_OPENAI set: that endpoint is reached through the local proxy (b["llm_proxy"]: url,
    master_key), which scripts/llm_proxy.py was configured with. Unset: Claude Code talks to DEEPSEEK_BASE_URL_ANTHROPIC
    (default DEEPSEEK_ANTHROPIC_DEFAULT) with DEEPSEEK_API_KEY."""
    import os
    openai_base = os.environ.get("DEEPSEEK_BASE_URL_OPENAI")
    if openai_base:
        r = {"base_url": b["llm_proxy"]["url"], "auth_token": b["llm_proxy"]["master_key"], "api_format": "openai"}
        print(f"[settings] {model}: DEEPSEEK_BASE_URL_OPENAI={openai_base} -> through the local proxy {r['base_url']}",
              flush=True)
    else:
        r = {"base_url": os.environ.get("DEEPSEEK_BASE_URL_ANTHROPIC", DEEPSEEK_ANTHROPIC_DEFAULT),
             "auth_token": os.environ["DEEPSEEK_API_KEY"], "api_format": "anthropic"}
        print(f"[settings] {model}: Anthropic format, directly at {r['base_url']}", flush=True)
    return r


def _backend(model: str, b: dict) -> dict:
    b = {**b, "config_dir": _path(b["config_dir"])}
    if b.get("endpoint") == "deepseek":
        return {**b, **_deepseek_endpoint(model, b)}
    assert "endpoint" not in b, f"machine.yaml model_backends.{model}: unknown endpoint {b['endpoint']!r}"
    return {"api_format": "openai", **b}   # an explicit base_url: a LiteLLM proxy or gateway with an OpenAI route


MODEL_BACKENDS = {m: _backend(m, b) for m, b in _m["model_backends"].items()}

# ---- Codex: the actor when --actor-agent codex ----
CODEX_BIN = _path(_m["codex"]["bin"])
CODEX_PROVIDER = dict(_m["codex"]["provider"])   # one [model_providers.<name>] table of Codex's config.toml
CODEX_API_KEY = _m["codex"]["api_key"]           # exported to the actor as CODEX_PROVIDER["env_key"]
CODEX_WEB_SEARCH = _m["codex"]["web_search"]     # Codex config value: "live", "cached" or "disabled"
CODEX_PROXY = _m["codex"]["proxy"]               # null: the provider is reached directly
# model metadata catalog (Codex model_catalog_json) for model names Codex does not know, e.g. the gateway's
# azure_openai/<slug> (2026-11-05: without it Codex 0.141 sent no reasoning settings)
CODEX_MODEL_CATALOG = _path(_m["codex"]["model_catalog"])

# ---- Actor reasoning effort ----
# the agent CLI's default effort per actor model (measured from its requests); used, passed explicitly and recorded
# when no --actor-effort is given (every model's reasoning effort is recorded)
ACTOR_DEFAULT_EFFORT = dict(_m["actor_default_effort"])

# ---- Actor sandbox (bubblewrap) ----
BWRAP = _path(_m["sandbox"]["bwrap"])
# read-only inside the sandbox: the agent binaries and the Python the actor may use for analysis
SANDBOX_RO_BINDS = [_path(p) for p in _m["sandbox"]["ro_binds"]]
# true: the actor gets its own PID namespace and /proc, so host command lines (case id, paths) are invisible.
# false where bwrap cannot mount a new proc ("Can't mount proc on /newroot/proc", e.g. inside a Docker container);
# the actor then sees host command lines, though not other processes' files (/proc/<pid>/cwd, environ).
# Optional key: when absent, a one-line bwrap call decides (machine.yaml files from before 2026-11-11 have none).
OWN_PROC_TEST = ["--unshare-pid", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"]


def _own_proc() -> bool:
    if "own_proc" in _m["sandbox"]:
        v = _m["sandbox"]["own_proc"]
        assert isinstance(v, bool), "machine.yaml sandbox.own_proc must be true or false"
        return v
    import subprocess
    p = subprocess.run([str(BWRAP), *OWN_PROC_TEST], capture_output=True, text=True)
    print(f"[settings] sandbox.own_proc not set; bwrap {'can' if p.returncode == 0 else 'cannot'} mount a new /proc "
          f"here -> own_proc {p.returncode == 0}{'' if p.returncode == 0 else ' (' + p.stderr.strip() + ')'}",
          flush=True)
    return p.returncode == 0


SANDBOX_OWN_PROC = _own_proc()

for _p in (CLAUDE_BIN, CLAUDE_CONFIG_DIR, SHELL_BIN, BWRAP, *SANDBOX_RO_BINDS,
           *(b["config_dir"] for b in MODEL_BACKENDS.values())):   # CODEX_BIN is checked when Codex runs
    assert _p.exists(), f"machine.yaml: {_p} does not exist"
