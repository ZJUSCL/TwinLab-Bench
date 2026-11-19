"""Guard in front of the environment agent's `python3` / `python` (installed per run as <run_dir>/env_bin/python3,
put first on the PATH of the env's Claude Code process; check_env replays with the same PATH).

What it stops, and only this (narrow, do not overreach): a calculator invocation whose stdout is
discarded (`>/dev/null`, `&>/dev/null`, `>/dev/null 2>&1`, closed stdout) AND that performs a billed action (prints a
CHARGE line with a non-free key). 044-01 replays 20261104-025345 and -025709: the Sonnet env ran
`python3 calculators.py grow --lines F01 >/dev/null 2>&1; echo skip` after the requested grow, used a patient vial
nobody saw, and 025709 was judged env_valid false.

How: the stdout target is read from the process itself (fd 1 is /dev/null or closed), not parsed from the command
text, so loops, subshells and variables do not matter. Such an invocation is run with its stdout captured after a
snapshot of calc_state/ and outbox/ (where every case calculator keeps its state, as Lab.snapshot_calc_state
assumes). Without a billable CHARGE line its output is dropped as the env asked and nothing else changes. With one,
calc_state/ and outbox/ are put back to the snapshot, the output (CHARGE lines included) is never printed, so the
harness bills nothing for it, and the refusal text goes to the Bash tool's own output (the nearest shell ancestor
whose stdout is not /dev/null) and to stderr; exit code REFUSED_RC. The env reads the refusal in the same turn and
continues. Every discarded-output invocation is logged as one JSON line to $TWINLAB_GUARD_LOG (Lab puts the new
lines of each request into its ledger record).

Not stopped (left to the CHARGE-line billing, the ledger flags and check_env): pipes into head/tail/grep/sed,
`2>/dev/null`, output written to a file, `--help`, any non-calculator python.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GUARD_LOG_VAR = "TWINLAB_GUARD_LOG"
REFUSED_RC = 3
REFUSAL_MARK = "LAB HARNESS REFUSED"   # first words of the refusal text (env_check / Lab look for it)
STATE_DIRS = ("calc_state", "outbox")   # twinlab.lab CALC_STATE, OUTBOX
HELP_ARGS = {"-h", "--help"}
SHELLS = {"zsh", "bash", "sh", "dash"}
# interpreter options that take a separate argument (python -W arg, -X opt)
OPTS_WITH_ARG = {"-W", "-X", "--check-hash-based-pycs"}


GUARD_BIN = "env_bin"                  # <run_dir>/env_bin: the shims
GUARD_LOG = "env_guard.jsonl"          # <run_dir>/env_guard.jsonl: one line per discarded-output calculator run
SHIM_NAMES = ("python3", "python")
ROOT = Path(__file__).resolve().parents[1]


def install(run_dir: Path):
    """Write the shims into <run_dir>/env_bin for the interpreter names that exist on the harness PATH."""
    b = run_dir / GUARD_BIN
    b.mkdir()
    names = [n for n in SHIM_NAMES if shutil.which(n)]
    assert "python3" in names, "no python3 on PATH to guard"
    for n in names:
        p = b / n
        p.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(ROOT)!r})\n"
                     f"from twinlab.calc_guard import main\nmain()\n")
        p.chmod(0o755)
    print(f"[calc_guard] installed {names} in {b}", flush=True)


def process_env(run_dir: Path, log: Path) -> dict:
    """Variables for a process (env agent, or check_env's replay) whose python3 goes through the guard."""
    b = run_dir / GUARD_BIN
    assert (b / "python3").is_file(), f"no guard installed in {b}"
    return {"PATH": f"{b}{os.pathsep}{os.environ['PATH']}", GUARD_LOG_VAR: str(log)}


def script_index(args: list):
    """Index in args of the script python would run, or None for -c / -m / - / no script."""
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-c", "-m", "-") or a.startswith(("-c", "-m")) and not a.startswith("--"):
            return None
        if a in OPTS_WITH_ARG:
            i += 2
            continue
        if a.startswith("-") and a != "-":
            i += 1
            continue
        return i
    return None


def stdout_discarded() -> bool:
    try:
        st = os.fstat(1)
    except OSError:   # stdout closed (>&-)
        return True
    dn = os.stat("/dev/null")
    return (st.st_dev, st.st_ino) == (dn.st_dev, dn.st_ino)


def real_interpreter(name: str) -> str:
    me = Path(sys.argv[0]).resolve().parent   # env_bin
    for d in os.environ["PATH"].split(os.pathsep):
        if not d or Path(d).resolve() == me:
            continue
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise AssertionError(f"[calc_guard] no real {name} on PATH after {me}")


def snapshot(cwd: Path, dst: Path) -> dict:
    had = {}
    for d in STATE_DIRS:
        had[d] = (cwd / d).is_dir()
        if had[d]:
            shutil.copytree(cwd / d, dst / d, symlinks=True)
    return had


def restore(cwd: Path, src: Path, had: dict):
    for d in STATE_DIRS:
        if (cwd / d).exists():
            shutil.rmtree(cwd / d)
        if had[d]:
            shutil.copytree(src / d, cwd / d, symlinks=True)


def tell_shell(text: str):
    """Write text to the stdout of the nearest shell ancestor whose stdout is not /dev/null (the Bash tool's output
    file in a run, the replay's pipe in check_env); never to a non-shell (claude's own stdout is its JSON answer)."""
    pid = os.getppid()
    while pid > 1:
        try:
            exe = os.path.basename(os.readlink(f"/proc/{pid}/exe"))
            fd1 = os.readlink(f"/proc/{pid}/fd/1")
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except OSError:
            return
        if exe.rstrip("0123456789.-") not in SHELLS:
            return
        if fd1 != "/dev/null":
            with open(f"/proc/{pid}/fd/1", "a") as f:
                f.write(text)
            return
        pid = ppid


def main():
    name = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    real = real_interpreter(name)
    i = script_index(args)
    if (i is None or os.path.basename(args[i]) != "calculators.py" or HELP_ARGS & set(args[i + 1:])
            or not stdout_discarded()):
        os.execv(real, [real, *args])
    # a calculator run whose whole stdout is thrown away: run it captured, keep it only if it billed nothing
    from .env_check import CHARGE_ITEM_RE, CHARGE_RE, FREE_ACTIONS
    cwd = Path.cwd()
    log = Path(os.environ[GUARD_LOG_VAR])
    tmp = Path(tempfile.mkdtemp(prefix="guard_"))
    had = snapshot(cwd, tmp)
    p = subprocess.run([real, *args], stdout=subprocess.PIPE, text=True)
    billed = [f"{k} x{n}" for line in CHARGE_RE.finditer(p.stdout) for k, n, _ in CHARGE_ITEM_RE.findall(line.group(1))
              if k not in FREE_ACTIONS]
    cmd = " ".join([name, *args])
    rec = {"t": time.time(), "cwd": str(cwd), "command": cmd, "rc": p.returncode, "billed": billed}
    if not billed:
        shutil.rmtree(tmp)
        rec["decision"] = "passed (no billed action; output discarded as asked)"
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        sys.exit(p.returncode)
    restore(cwd, tmp, had)
    shutil.rmtree(tmp)
    rec["decision"] = "refused (billed action with discarded output; state restored, nothing billed)"
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    msg = (f"\n{REFUSAL_MARK} this command and did not run it: `{cmd}` performs a billed lab action "
           f"({', '.join(billed)}) and its output was sent to /dev/null, so neither you nor the actor would see the "
           f"result. Nothing was performed, nothing was charged, the lab state is unchanged. Run a billed action only "
           f"when the actor's request asks for it, never discard its output, and report what it prints.\n")
    tell_shell(msg)
    sys.stderr.write(msg)
    sys.exit(REFUSED_RC)
