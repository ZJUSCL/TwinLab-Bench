"""Environment agent: a persistent Claude Code session (resumed per actor request) that knows the full case,
reads the original notebook text and its figures, and renders raw observations for the actor.

Its rules are prompts/env_system.md; the case's environment_instructions and the ground truth (case view, record
files, literature notes) are appended to them. It answers every request through the StructuredOutput tool of
--json-schema with an object in response_schema() (closed); the Lab
checks that answer and does all accounting.
"""
import ast
import json
import os
import shutil
import uuid
from pathlib import Path

from . import calc_guard, prompts
from .case import Case
from .claude_cli import (STRUCTURED_ANSWER_MARK, close_objects, config_dir_for, parse_json_answer,
                         run_claude_json_env, structured_answer, structured_answer_instructions)

VERDICTS = ["performed", "partially_performed", "not_simulatable", "answered_question", "refused_question"]
CONTINUED_STEP = "continued_step"   # zero-cost action key for the rest of an action split by a checkpoint
TOOLS = "Read,Glob,Grep"             # plus Bash when the case has a calculator
CALC_USAGE = "calculator_usage.txt"  # the calculator's module docstring, written into env_ws (rule 14)
CALL_TIMEOUT_S = 3600   # deepseek-flash behind a gateway that cools down for up to about an hour (CLI retries, 2026-11-04)
# LEGACY header written by claude_cli.json_answer_instructions: runs started 2026-10-26..10-30 answer with JSON text
ANSWER_FORMAT_MARK = "# ANSWER FORMAT"
# new runs (and the system prompt's last section): submit through --json-schema's StructuredOutput tool
# every unperformed item names its kind (rule 6): lab declines read as "could not be simulated" to the actor and the
# judges (deepseek-flash env audit 2026-11-04)
NOT_PERFORMED_PATTERN = "^(SIMULATION LIMIT|LAB): "
NUMBERS_CHECK = ("Before you submit, check that every number in observation matches the calculator output or the "
                 "record line it comes from; a submitted answer cannot be corrected afterwards.\n")


def response_schema(action_keys: list, consumable_items: list, checkpoint_ids: list) -> dict:
    consumed = ({"type": "array",
                 "items": {"type": "object",
                           "properties": {"item": {"type": "string", "enum": consumable_items},
                                          "n": {"type": "integer", "minimum": 1}},
                           "required": ["item", "n"]}}
                if consumable_items else {"type": "array", "maxItems": 0})
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": VERDICTS},
            "action_types": {"type": "array", "items": {"type": "string", "enum": action_keys}},
            "consumed": consumed,
            "stopped_at_checkpoint": {"type": "string", "enum": ["", *checkpoint_ids],
                                      "description": "checkpoint id where execution stopped (rule 11), or ''"},
            "deferred_steps": {"type": "array", "items": {"type": "string"},
                               "description": "requested steps after the checkpoint, NOT performed (rule 11)"},
            "observation": {"type": "string", "description": "exactly what the actor will read"},
            "not_performed_items": {"type": "array", "items": {"type": "string", "pattern": NOT_PERFORMED_PATTERN},
                                    "description": "items of the request that were not done, each starting with "
                                                   "'SIMULATION LIMIT: ' or 'LAB: ' (rule 6)"},
            "parameter_resolution": {
                "type": "array",
                "items": {"type": "object",
                          "properties": {"parameter": {"type": "string", "description": "procedure.param key"},
                                         "sample": {"type": "string",
                                                    "description": "which sample(s) this entry applies to, or 'all'"},
                                         "requested": {"type": "array", "items": {"type": "string"}},
                                         "used": {"type": "string"},
                                         "mode": {"type": "string",
                                                  "enum": ["recorded", "extrapolated", "lab_limitation"]}},
                          "required": ["parameter", "sample", "requested", "used", "mode"]},
            },
            "images": {"type": "array", "items": {"type": "string"}},
            "grounding": {"type": "string",
                          "description": "INTERNAL, never shown to the actor: which notebook entries / table rows "
                                         "this answer is based on"},
        },
        "required": ["verdict", "action_types", "consumed", "stopped_at_checkpoint", "deferred_steps", "observation",
                     "not_performed_items", "parameter_resolution", "images", "grounding"],
    }


class EnvAgent:
    def __init__(self, case: Case, run_dir: Path, model: str, resume: bool):
        self.call_timeout_s = CALL_TIMEOUT_S   # overwritten per attempt by the lab (lab._env_step_with_retry)
        self.case = case
        self.model = model
        self.ws = run_dir / "env_ws"
        self.log_path = run_dir / "env_calls.jsonl"
        self.tmp_dir = run_dir / "env_tmp"   # the env's /tmp (claude_cli.private_tmp_prefix)
        self.system_prompt_file = run_dir / "env_system_prompt.md"
        # continued_step only exists for checkpointed cases; offering it elsewhere let the env bill a normal request
        # with it (001 Sonnet run, 2026-09-29), which the lab server rightly rejects
        action_keys = [*case.actions.keys(), *([CONTINUED_STEP] if case.checkpoints["enforce"] else [])]
        # likewise checkpoint ids: offered with enforce false, the env stopped at one anyway (033 replay, 2026-10-16)
        checkpoint_ids = [c["id"] for c in case.checkpoints["list"]] if case.checkpoints["enforce"] else []
        # open schema: what LEGACY text-JSON runs were told and validated against; closed: --json-schema runs
        self.schema_open = response_schema(action_keys, list(case.consumables), checkpoint_ids)
        self.schema = close_objects(self.schema_open)
        self.last_single_turn_ctx = 0
        calc = case.d["harness"]["env_calculator"]
        self.calculator = (case.assets_dir / calc) if calc else None
        assert self.calculator is None or self.calculator.is_file(), self.calculator
        self.tools = TOOLS + (",Bash" if self.calculator else "")
        print(f"[env] tools={self.tools}", flush=True)
        # python3 guard against billed calculator runs with discarded output (calc_guard.py, 2026-11-04); runs
        # started before it have no env_bin and resume without it, as they ran
        self.guard_log = run_dir / calc_guard.GUARD_LOG
        if self.calculator and not resume:
            calc_guard.install(run_dir)
        guarded = self.calculator is not None and (run_dir / calc_guard.GUARD_BIN).is_dir()
        self.extra_env = calc_guard.process_env(run_dir, self.guard_log) if guarded else {}
        print(f"[env] calculator guard {'on' if guarded else 'off'}", flush=True)
        if resume:   # reuse the existing workspace, system prompt, and session (see restore)
            assert self.ws.is_dir() and self.system_prompt_file.is_file(), f"nothing to resume in {run_dir}"
            # the answer path follows the system prompt the session started with: LEGACY text JSON (runs started
            # 2026-10-26..10-30) or StructuredOutput; runs started before 10-26 have neither marker
            sp = self.system_prompt_file.read_text(encoding="utf-8")
            self.legacy_text_json = ANSWER_FORMAT_MARK in sp
            assert self.legacy_text_json != (STRUCTURED_ANSWER_MARK in sp), \
                f"{self.system_prompt_file}: expected exactly one of {ANSWER_FORMAT_MARK!r} / " \
                f"{STRUCTURED_ANSWER_MARK!r} (run started before 2026-10-26?); not resumable"
            print(f"[env] resume: answer path = {'LEGACY text JSON' if self.legacy_text_json else 'StructuredOutput'}",
                  flush=True)
            return
        self.legacy_text_json = False
        self.ws.mkdir(parents=True)
        (self.ws / "case_env.yaml").write_text(case.env_view_yaml(), encoding="utf-8")
        for f in case.env_context_files():
            os.symlink(f, self.ws / f.name)
        os.symlink(case.figures_dir, self.ws / "figures")
        for src, dst in case.d["harness"]["env_data_dirs"].items():   # raw data: Grep/Read-able, not embedded
            s = case.assets_dir / src
            assert s.is_dir(), s
            os.symlink(s, self.ws / dst)
        if self.calculator:   # deterministic, data-anchored number generator the env must call (via Bash)
            os.symlink(self.calculator, self.ws / self.calculator.name)
            # its usage as a file, so the env never needs to open the source (rule 14, 2026-11-04)
            doc = ast.get_docstring(ast.parse(self.calculator.read_text(encoding="utf-8")))
            assert doc and len(doc) > 300, f"{self.calculator}: no usable module docstring"
            (self.ws / CALC_USAGE).write_text(doc + "\n", encoding="utf-8")
        kfiles = case.env_knowledge_files()
        (self.ws / "env_knowledge").mkdir()
        for f in kfiles:
            os.symlink(f, self.ws / "env_knowledge" / f.name)
        print(f"[env] knowledge files: {[f.name for f in kfiles]}", flush=True)
        # the conclusions the rubric expects, so the env can avoid stating them in answers (066-03 #5:
        # a textbook answer restated the K2 argument); data and experiments are never withheld because of it
        conclusions = "".join(f"\n- {v['full'].strip()}" for v in case.scoring["rubric"].values())
        parts = [prompts.text("env_system.md"),
                 "\n# CONCLUSIONS THE ACTOR MUST REACH ITSELF\n"
                 "The scoring expects the actor to reach the conclusions below by its own analysis. Never state them, "
                 "and never volunteer the reasoning that leads to them, in answers to questions (textbook, history or "
                 "staff answers). This list changes nothing about experiments: run every requested experiment and "
                 "report every observation exactly as the rules say; observations that let the actor reach these "
                 "conclusions are what the lab is for, so never withhold, soften or refuse data because of this list."
                 + conclusions + "\n",
                 "\n# CASE-SPECIFIC INSTRUCTIONS (part of the rules)\n", case.d["environment_instructions"].strip(),
                 "\n\n# EMBEDDED GROUND TRUTH\n",
                 f"\n## FILE: case_env.yaml\n```yaml\n{case.env_view_yaml()}\n```\n"]
        for f in case.env_context_files():
            parts.append(f"\n## FILE: {f.name}\n{f.read_text(encoding='utf-8')}\n")
        for f in kfiles:
            parts.append(f"\n## FILE: env_knowledge/{f.name}\n{f.read_text(encoding='utf-8')}\n")
        parts.append(structured_answer_instructions() + NUMBERS_CHECK)
        assert STRUCTURED_ANSWER_MARK in parts[-1]
        self.system_prompt_file.write_text("".join(parts), encoding="utf-8")
        print(f"[env] system prompt chars={len(''.join(parts))}", flush=True)
        self.session_id = str(uuid.uuid4())
        self.n_calls = 0
        self.session_created = False
        (self.ws / "session_record.md").write_text("# Session record (harness-maintained, what the actor was shown; "
                                                   "'recorded CHARGE:' lines are past charges, already billed)\n")

    def state(self) -> dict:
        return {"session_id": self.session_id, "n_calls": self.n_calls, "session_created": self.session_created}

    def restore(self, st: dict):
        self.session_id, self.n_calls, self.session_created = st["session_id"], st["n_calls"], st["session_created"]
        print(f"[env] resumed session {self.session_id} after {self.n_calls} calls", flush=True)

    def step(self, request: str, parameters: dict, ledger: dict, notes: list, noise: list) -> dict:
        note_txt = "".join(f"\n- {n}" for n in notes)
        prompt = (
            f"## Actor request #{ledger['n_requests'] + 1}\n{request}\n\n"
            f"## Structured parameters from the actor (several values = any is acceptable)\n"
            f"{json.dumps(parameters, indent=1) if parameters else '(none)'}\n"
            # Sonnet env dropped a structured key from parameter_resolution (003 A/B, 2026-10-11): name them all.
            # Root cause there: it stopped at a checkpoint before that step but left deferred_steps empty.
            + (f"parameter_resolution must contain at least one entry for EACH of these keys (rule 4): "
               f"{sorted(parameters)}. The only exception is a key whose step comes after the checkpoint where you "
               f"stop; that step must then be listed in deferred_steps (rule 11).\n" if parameters else "")
            + "\n"
            f"## Lab ledger (authoritative, maintained by the harness)\n{json.dumps(ledger, indent=1)}\n"
            f"\n## Noise multipliers for derived values in this request (use in order, rule 3)\n{noise}\n"
            + (f"\n## Harness notes since your last turn{note_txt}\n" if notes else "")
            + (self.legacy_answer_line() if self.legacy_text_json else
               "\nSubmit your answer with the StructuredOutput tool.")
        )
        if not self.session_created:
            session_args = ["--session-id", self.session_id]
            self.session_created = True   # set before the call: a failed first call may still have created it
        else:
            session_args = ["--resume", self.session_id]
        args = ["-p", prompt, "--safe-mode", "--tools", self.tools, "--strict-mcp-config",
                "--permission-mode", "bypassPermissions", "--model", self.model,
                "--system-prompt-file", str(self.system_prompt_file), "--output-format", "json",
                *([] if self.legacy_text_json else ["--json-schema", json.dumps(self.schema)]),
                *session_args]
        # the lab sets call_timeout_s per attempt (remaining share of LAB_REQUEST_TOTAL_S, lab._env_step_with_retry)
        out = run_claude_json_env(args, cwd=self.ws, timeout_s=self.call_timeout_s, log_path=self.log_path,
                                  tmp_dir=self.tmp_dir, extra_env=self.extra_env)
        self.n_calls += 1
        u = out["usage"]
        ctx = u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]
        print(f"[env] call {self.n_calls} context_tokens={ctx} turns={out['num_turns']}", flush=True)
        if out["num_turns"] <= 2 and self.last_single_turn_ctx and ctx < 0.7 * self.last_single_turn_ctx:
            print(f"[env] WARNING context shrank {self.last_single_turn_ctx} -> {ctx}: possible compaction; "
                  f"consistency now relies on session_record.md", flush=True)
        if out["num_turns"] <= 2:
            self.last_single_turn_ctx = ctx
        if self.legacy_text_json:
            return parse_json_answer(out, self.schema_open, self.log_path)
        return structured_answer(out, self.schema, self.log_path)

    def legacy_answer_line(self) -> str:
        """LEGACY text-JSON runs only. Sonnet env opened a new key after grounding and closed the object (`...","}`;
        001-03 run 20261029-202451 #1, 040-02 replay-20261029-213945 #2): name the keys, their order, and where the
        object ends."""
        keys = self.schema_open["required"]
        return ("\nEnd your turn with the JSON object described under ANSWER FORMAT: exactly these keys, in this order, "
                f"each once and no other key: {', '.join(keys)}. The object ends with }} immediately "
                f"after the closing quote of the {keys[-1]} string.")

    def transcript(self) -> Path:
        """The live Claude Code session transcript of the env agent (every tool call with its output)."""
        found = list((config_dir_for(self.model) / "projects").glob(f"*/{self.session_id}.jsonl"))
        assert len(found) == 1, f"env session transcript for {self.session_id}: expected 1 file, found {found}"
        return found[0]

    def archive_session(self, run_dir: Path) -> Path:
        """Copy the env agent's Claude Code session transcript (every tool call with its output, including the
        calculator commands) into the run directory, for the post-hoc check of the environment."""
        src = self.transcript()
        dst = run_dir / "env_session.jsonl"
        shutil.copyfile(src, dst)
        print(f"[env] session transcript archived: {src} -> {dst} ({dst.stat().st_size} bytes)", flush=True)
        return dst

    def resolve_image(self, rel: str) -> Path:
        p = (self.ws / rel).resolve()
        assert p.is_file() and self.case.figures_dir in p.parents, f"env attached a non-figure path: {rel}"
        return p
