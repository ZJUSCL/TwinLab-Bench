"""The bubblewrap sandbox every actor runs in.

Inside the sandbox the host filesystem is read-only, and $HOME, /tmp, the repository and the run output root are
empty tmpfs mounts. The actor sees only:
- the agent binaries and Python (settings.SANDBOX_RO_BINDS, read-only),
- its own per-run config dir (bound where the agent CLI expects its config); the guard hook files and the settings
  that install it are bound read-only on top, so the actor cannot switch the guard off,
- the per-run workspace, mounted at /tmp/workspace and used as the working directory.
With machine.yaml sandbox.own_proc it has its own PID namespace and /proc, so host processes (the harness command
line names the case, other runs' agents) are invisible; without it their command lines are visible, but their files
(/proc/<pid>/cwd, root, environ) are not, because bwrap's user namespace denies them. Its environment is a whitelist (actor_env): API keys and paths the
harness inherited from the user's shell stay outside.
The network is shared with the host (the actor needs its API and the lab MCP server on 127.0.0.1).
"""
import sys
from pathlib import Path

from .. import settings

WORKSPACE = "/tmp/workspace"   # the actor's cwd inside the sandbox
RUN_DIR_PREFIX = "run_"        # scripts/run_case.py: <out>/<case_id>/<timestamp>/run_<i>/actor_workspace
OUT_ROOT_DEPTH = 3             # ws.parents[3] is <out>

# Variables the agent CLIs need. Everything else the harness inherited (DEEPSEEK_API_KEY and other keys exported in
# the user's shell, PWD/OLDPWD with the launch directory) is dropped. CLAUDE*/ANTHROPIC* are the ones claude_env_for
# sets itself (inherited ones are stripped there).
ENV_KEEP = ("PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ", "SHELL", "TERM",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy", "MCP_TOOL_TIMEOUT", "CODEX_HOME")
ENV_KEEP_PREFIXES = ("CLAUDE", "ANTHROPIC")


def out_root(ws: Path) -> Path:
    run_dir = ws.parent
    assert run_dir.name.startswith(RUN_DIR_PREFIX), f"{ws}: not <out>/<case>/<time>/run_<i>/actor_workspace"
    return ws.parents[OUT_ROOT_DEPTH]


def bwrap_prefix(binds: list, ws: Path, protected: list) -> list:
    """Command prefix that runs a program in the sandbox. binds: [(host_path, sandbox_path)], writable.
    protected: [(host_file, sandbox_path)] bound read-only after binds (files inside a writable bind)."""
    # the guard hook runs sys.executable inside the sandbox; a venv in the repository or under $HOME is hidden there
    # unless a ro_binds entry covers it, and the hook would then fail without stopping the actor
    exe = Path(sys.executable)
    hidden = [settings.HOME, Path("/tmp"), settings.ROOT.resolve(), out_root(ws).resolve()]
    assert not any(exe.is_relative_to(h) for h in hidden) or any(exe.is_relative_to(p) for p in settings.SANDBOX_RO_BINDS), \
        (f"the Python running the benchmark ({exe}) is hidden inside the actor sandbox ($HOME, /tmp and the "
         f"repository are empty there): add its directory (e.g. the venv) to machine.yaml sandbox.ro_binds")
    args = [str(settings.BWRAP), "--die-with-parent", "--unshare-ipc", "--ro-bind", "/", "/", "--dev", "/dev"]
    if settings.SANDBOX_OWN_PROC:
        args += ["--unshare-pid", "--proc", "/proc"]
    args += ["--tmpfs", str(settings.HOME)]
    for d in (settings.ROOT, out_root(ws)):   # under $HOME they are already hidden
        if not d.resolve().is_relative_to(settings.HOME):
            args += ["--tmpfs", str(d.resolve())]
    for p in settings.SANDBOX_RO_BINDS:
        args += ["--ro-bind", str(p), str(p)]
    for src, dst in binds:
        args += ["--bind", str(src), str(dst)]
    for src, dst in protected:
        assert Path(src).is_file(), f"protected sandbox file {src} does not exist"
        args += ["--ro-bind", str(src), str(dst)]
    args += ["--tmpfs", "/tmp", "--bind", str(ws), WORKSPACE, "--chdir", WORKSPACE]
    return args


def actor_env(env: dict, extra_keep: tuple) -> dict:
    """The whitelisted part of an actor's process environment. extra_keep: further names this actor sets itself
    (e.g. Codex's provider key variable)."""
    out = {k: v for k, v in env.items() if k in ENV_KEEP or k in extra_keep or k.startswith(ENV_KEEP_PREFIXES)}
    out.update(HOME=str(settings.HOME), PWD=WORKSPACE)
    dropped = sorted(set(env) - set(out))
    print(f"[sandbox] actor env keeps {sorted(out)}; dropped {len(dropped)} inherited variables", flush=True)
    return out
