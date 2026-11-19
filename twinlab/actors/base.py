"""What every actor agent provides, and the loop that runs one.

An actor agent (claude_code.py, codex.py) knows how to set up its per-run config, build the command line of one
invocation, and read its own output stream. run_actor() is the same for all agents:

  invocation 0      the task brief
  invocation 1..    if the actor stops without submitting a report: a nudge (at most MAX_NUDGES), or, after a
                    usage-limit wait, a "you were interrupted" message; always resuming the same session
  kill              timeout, environment fatal error, served model != requested model, or no report within
                    REPORT_GRACE_S after the lab closed

Each invocation writes run_dir/actor_stream_<tag>_<k>.jsonl (the agent's raw JSON event stream).
"""
import collections
import json
import subprocess
import time
from pathlib import Path

from ..prompts import MESSAGES

MAX_NUDGES = 2          # resumes when the actor stops without submitting a report
REPORT_GRACE_S = 600    # after the lab closes (budget exhausted), kill the actor after this
POLL_S = 2
ACTOR_TIME_LOG = "actor_time.jsonl"   # wall seconds per actor invocation, summed by resume


class ActorAgent:
    """Interface. Subclasses fill in every method; run_actor() only uses these."""
    name = ""
    REPORTS_SERVED_MODEL = True   # False: the stream does not say which model answered, so it cannot be checked

    def __init__(self, run_dir: Path, effort):
        self.run_dir = run_dir
        self.effort = effort                      # reasoning effort flag of the agent CLI; None = its default
        self.cfg = run_dir / "actor_config"       # the agent's config dir, mounted inside the sandbox
        self.ws = run_dir / "actor_workspace"     # the actor's working directory, /tmp/workspace in the sandbox

    def prepare(self, case, model: str):
        """Fresh run: create the config dir and the workspace for an actor that runs `model`."""
        raise NotImplementedError

    def check_prepared(self):
        """Resumed run: the dirs from prepare() must exist."""
        assert self.cfg.is_dir() and self.ws.is_dir(), f"no actor dirs in {self.run_dir}"

    def new_session_id(self):
        """Session id to start with, or None if the agent assigns one (read back with session_id_from_stream)."""
        raise NotImplementedError

    def command(self, message: str, model: str, mcp_url: str, session_id, new_session: bool) -> list:
        raise NotImplementedError

    def env(self) -> dict:
        raise NotImplementedError

    def session_id_from_stream(self, stream: Path) -> str:
        raise NotImplementedError

    def served_models(self, stream: Path) -> collections.Counter:
        raise NotImplementedError

    def usage_limit_hit(self, stream_tail: str) -> bool:
        raise NotImplementedError

    def wait_for_usage_reset(self, stream_tail: str) -> float:
        raise NotImplementedError

    def terminal_api_error(self, stream_tail: str):
        """A provider limit that every further request of this session would hit again (e.g. too many images in
        the history): the actor's run ends there and is judged as it stands (058-02 luna).
        Returns a short label, or None."""
        raise NotImplementedError

    @staticmethod
    def tool_events(stream: Path) -> list:
        """The actor's tool use in order, for the judge: {"kind": "call", "name", "input"} and
        {"kind": "result", "text", "lab"}. Lab tools are named mcp__lab__<tool>."""
        raise NotImplementedError


def run_actor(agent: ActorAgent, prompt: str, model: str, mcp_url: str, timeout_s: int, lab, session_id,
              tag: str, start_with_resume: bool) -> dict:
    """Run the actor until it submits its report, gives up, or is killed. Polls lab.fatal and kills the actor if
    the environment crashed. start_with_resume=True continues an existing session with `prompt` as the message.
    mcp_url: from mcp_server.serve (holds the run's token)."""
    deadline = time.time() + timeout_s
    invocations = []
    k, nudges, message = 0, 0, prompt
    while True:
        new_session = k == 0 and not start_with_resume
        cmd = agent.command(message, model, mcp_url, session_id, new_session)
        stream = agent.run_dir / f"actor_stream_{tag}_{k}.jsonl"
        t_start = time.time()
        with open(stream, "w") as fo, open(agent.run_dir / f"actor_stderr_{tag}_{k}.log", "w") as fe:
            p = subprocess.Popen(cmd, env=agent.env(), stdin=subprocess.DEVNULL, stdout=fo, stderr=fe)
            status = "exited"
            served = collections.Counter()   # first answer is checked as soon as it exists: substitution fails fast
            while p.poll() is None:
                if agent.REPORTS_SERVED_MODEL and not served:
                    served = agent.served_models(stream)
                if served and set(served) != {model}:
                    status = "killed_model_mismatch"
                elif lab.fatal is not None:
                    status = "killed_env_fatal"
                elif time.time() > deadline:
                    status = "killed_timeout"
                elif (lab.closed_at is not None and lab.report is None
                      and time.time() > lab.closed_at + REPORT_GRACE_S):
                    status = "killed_no_report_after_budget"
                if status != "exited":
                    p.kill()
                    p.wait()
                    break
                time.sleep(POLL_S)
        # actor wall time, appended right away so a crash below still leaves it for resume (which deducts it from
        # the time limit); usage-limit waits fall between invocations and are not counted
        with open(agent.run_dir / ACTOR_TIME_LOG, "a") as ft:
            ft.write(json.dumps({"tag": tag, "k": k, "seconds": round(time.time() - t_start, 1)}) + "\n")
        if session_id is None:
            session_id = agent.session_id_from_stream(stream)
        served = agent.served_models(stream) if agent.REPORTS_SERVED_MODEL else None
        invocations.append({"tag": tag, "k": k, "rc": p.returncode, "status": status, "stream": stream.name,
                            "served_models": None if served is None else dict(served)})
        print(f"[actor] invocation {k}: rc={p.returncode} status={status} "
              f"served={'not reported by ' + agent.name if served is None else dict(served)}", flush=True)
        if served is not None:
            assert set(served) <= {model}, (f"actor was served {dict(served)} although {model} was requested; "
                                            f"the run does not measure {model} (see {stream})")
        k += 1
        if status != "exited" or lab.report is not None:
            break
        tail = stream.read_text(errors="replace")[-4000:]
        if p.returncode != 0 and agent.usage_limit_hit(tail):   # wait for the reset, then continue the session
            deadline += agent.wait_for_usage_reset(tail)
            message = MESSAGES["interrupted_resume"]
            continue
        terminal = agent.terminal_api_error(tail) if p.returncode != 0 else None
        if terminal is not None:
            invocations[-1]["status"] = f"ended_api_limit:{terminal}"
            print(f"[actor] session ended by a provider limit ({terminal}); judged as it stands", flush=True)
            break
        assert p.returncode == 0, (f"actor {agent.name} exited rc={p.returncode}; "
                                   f"see {agent.run_dir}/actor_stderr_{tag}_{k - 1}.log")
        if nudges >= MAX_NUDGES:
            break
        nudges += 1
        message = MESSAGES["budget_exhausted"] if lab.closed_at is not None else MESSAGES["nudge"]
    return {"session_id": session_id, "invocations": invocations}
