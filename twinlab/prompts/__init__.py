"""The texts models read, kept as files so they can be edited without touching code.

  env_system.md    rules of the environment agent (case-specific instructions and the ground truth are appended)
  judge_system.md  grading and environment-audit instructions of the judge
  actor_brief.md   task brief template for the actor (filled from case.yaml by Case.actor_prompt)
  messages.yaml    short fixed messages: nudges, resume notices, lab notices
"""
from pathlib import Path

import yaml

DIR = Path(__file__).parent


def text(name: str) -> str:
    return (DIR / name).read_text(encoding="utf-8")


MESSAGES = yaml.safe_load(text("messages.yaml"))
