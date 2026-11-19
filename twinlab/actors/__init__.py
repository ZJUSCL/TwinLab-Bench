"""The evaluated agent (the ACTOR): one class per agent CLI, all run by actors.base.run_actor in the same sandbox.

  claude-code   actors/claude_code.py
  codex         actors/codex.py

A human tester (scripts/human_baseline.py) also plays the actor; its run records agent "human" and has no stream.
"""
from pathlib import Path

from .base import ActorAgent, run_actor
from .claude_code import ClaudeCodeActor
from .codex import CodexActor

AGENTS = {a.name: a for a in (ClaudeCodeActor, CodexActor)}
HUMAN = "human"


def make_agent(name: str, run_dir: Path, effort) -> ActorAgent:
    assert name in AGENTS, f"unknown actor agent {name!r}; choose one of {sorted(AGENTS)}"
    return AGENTS[name](run_dir, effort)


def tool_events(run_dir: Path, agent_name: str) -> list:
    """All tool use of the actor in a run, in order, across its stream files."""
    if agent_name == HUMAN:
        return []
    agent = AGENTS[agent_name]
    return [e for f in sorted(run_dir.glob("actor_stream_*.jsonl")) for e in agent.tool_events(f)]


__all__ = ["AGENTS", "HUMAN", "make_agent", "run_actor", "tool_events"]
