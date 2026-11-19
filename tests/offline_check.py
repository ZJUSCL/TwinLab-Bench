"""Offline test of every case (no model calls): case loading, actor brief, report tool, environment schema and
system prompt, and the Lab's accounting and checkpoint rules against a scripted fake environment.
Run it after any change to twinlab/ or to a case.

python tests/offline_check.py                       # all cases
python tests/offline_check.py cases/0*/case.yaml -v # some cases, printing the brief and the lab responses
"""
import json
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # run from a checkout without installing
from twinlab.case import Case, case_name               # noqa: E402
from twinlab.env_agent import EnvAgent, response_schema   # noqa: E402
from twinlab.lab import Lab                            # noqa: E402
from twinlab.prompts import MESSAGES                   # noqa: E402


class FakeEnv:
    """Returns scripted environment answers in order and writes a Claude Code-shaped session transcript (the Lab
    bills from the CHARGE lines in it). A script entry is an answer, or (answer, [(command, output), ...])."""
    def __init__(self, ws, script):
        self.ws, self.script = ws, list(script)
        self.path = ws.parent / "fake_env_transcript.jsonl"
        self.path.write_text("")
        self.n_tools = 0
        self.guard_log = ws.parent / "env_guard.jsonl"   # twinlab/calc_guard.py log; unguarded unless a test sets it
        self.extra_env = {}

    def transcript(self):
        return self.path

    def write_turn(self, ledger, shell, notes):
        # harness notes travel in the same user message as the request header (twinlab.env_agent.EnvAgent.step)
        note_txt = "".join(f"\n- {n}" for n in notes)
        lines = [{"message": {"role": "user", "content": f"## Actor request #{ledger['n_requests'] + 1}\nreq"
                                                         + (f"\n## Harness notes since your last turn{note_txt}\n"
                                                            if notes else "")}}]
        for cmd, out in shell:
            self.n_tools += 1
            tid = f"t{self.n_tools}"
            lines.append({"message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": cmd}}]}})
            lines.append({"message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": out}]}})
        with self.path.open("a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))

    def step(self, request, params, ledger, notes, noise):
        self.notes_seen = list(notes)
        entry = self.script.pop(0)
        ans, shell = entry if isinstance(entry, tuple) else (entry, [])
        self.write_turn(ledger, shell, notes)
        return ans

    def state(self):
        return {}

    def restore(self, st):
        pass

    def resolve_image(self, rel):
        raise AssertionError("no images in the fake environment")


def answer(**kw) -> dict:
    base = {"verdict": "performed", "action_types": [], "consumed": [], "stopped_at_checkpoint": "",
            "deferred_steps": [], "observation": "obs", "not_performed_items": [], "parameter_resolution": [],
            "images": [], "grounding": "g"}
    base.update(kw)
    return base


# Traces that tell the actor the record was edited or assembled (realism pilot 2026-10-23, user: blinding must not
# show; synthetic content must look real without contradicting facts). Matched case-insensitively in the brief and
# every text file of the rendered actor workspace.
AUTHORING_TRACES = re.compile("|".join([
    r"\[(link )?removed\]", r"\[figure omitted\]", r"\[redacted\]", r"\bredacted\b",
    r"replaced (by|with) (roles?|codes?|placeholders?)", r"\b(are|is|were|was) not named\b",
    r"\banonymi[sz]ed\b", r"\bde-?identified\b", r"\bpseudonymi[sz]ed\b", r"see the task brief",
]), re.IGNORECASE)


def check_actor_view_traces(case: Case, brief: str):
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        case.write_actor_workspace(ws)
        texts = [("BRIEF", brief)] + [(str(f.relative_to(ws)), f.read_text(encoding="utf-8"))
                                      for f in sorted(ws.rglob("*")) if f.is_file() and case.is_text_file(f)]
    hits = [f"{name}:{i}: {m.group(0)!r}" for name, t in texts for i, line in enumerate(t.split("\n"), 1)
            for m in AUTHORING_TRACES.finditer(line)]
    assert not hits, f"authoring traces in the actor view: {hits[:8]}"
    named = [name for name, t in texts if case_name(case.id) in t]   # the guard would withhold these files' content
    assert not named, f"the case name {case_name(case.id)!r} is in the actor view: {named[:8]}"


def check(case_path: Path, verbose: bool):
    case = Case(case_path, "case")
    brief = case.actor_prompt()
    check_actor_view_traces(case, brief)
    sch = response_schema(list(case.actions), list(case.consumables), [c["id"] for c in case.checkpoints["list"]])
    ev = case.env_view_yaml()
    for s in case.d["harness"]["env_hidden_sections"]:
        assert f"\n{s}:" not in ev, f"hidden section {s} in the environment view"
    with tempfile.TemporaryDirectory() as td:   # builds the env workspace and system prompt, no model call
        EnvAgent(case, Path(td), "no-model", resume=False)

    a0 = list(case.actions)[0]
    item = next(iter(case.consumables), None)
    cps = [c["id"] for c in case.checkpoints["list"]]
    script = [answer(action_types=[a0], consumed=[{"item": item, "n": 1}] if item else [])]
    if case.checkpoints["enforce"]:
        script.append(answer(action_types=[a0], stopped_at_checkpoint=cps[0], deferred_steps=["step B", "step C"]))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = td / "env_ws"
        ws.mkdir()
        (ws / "session_record.md").write_text("")
        lab = Lab(case, FakeEnv(ws, script), td, seed=0)
        texts = [lab.call("lab", {"request": f"req {i}"})["content"][0]["text"] for i in range(len(script))]
        assert lab.cost_used == sum(case.actions[a0]["cost"] for _ in script), lab.cost_used
        if item:
            assert lab.materials_left()[item] == case.consumables[item] - case.initial_consumed[item] - 1
        if not case.checkpoints["enforce"]:   # a checkpoint stop must be rejected when checkpoints are off
            lab.env.script = [answer(action_types=[a0], stopped_at_checkpoint="x", deferred_steps=["y"])]
            try:
                lab.call("lab", {"request": "bad"})
                raise SystemExit("expected an assertion for a checkpoint stop with checkpoints disabled")
            except AssertionError:
                pass
        # a whole-request refusal shows its per-item reasons, as partial execution does (2026-10-29)
        lab.env.script = [answer(verdict="not_simulatable", observation="", not_performed_items=["item X: reason R"])]
        t = lab.call("lab", {"request": "refused"})["content"][0]["text"]
        assert "\n- item X: reason R" in t, t
        # ... and the observation text, where the env often puts the decline reasons (062-02, 069-02; 2026-10-29)
        lab.env.script = [answer(verdict="not_simulatable", observation="REFUSED item Y: reason S",
                                 not_performed_items=["item Y"])]
        t = lab.call("lab", {"request": "refused with reasons in observation"})["content"][0]["text"]
        assert "REFUSED item Y: reason S" in t and "\n- item Y" in t, t
        lab.env.script = [answer(verdict="not_simulatable", observation="UNAVAILABLE: reason U")]
        t = lab.call("lab", {"request": "refused, no items"})["content"][0]["text"]
        assert "UNAVAILABLE: reason U" in t and MESSAGES["not_simulatable"] in t, t
        r = lab.call("submit_final_report", {k: "x" for k in case.final_report["required"]})
        assert not r["isError"], r
    if verbose:
        print(f"----- {case.id}: actor brief -----\n{brief}")
        print("report tool", json.dumps(case.report_tool_schema())[:300])
        print("env schema keys", list(sch["properties"]))
        for i, t in enumerate(texts):
            print(f"----- lab response {i} -----\n{t}")


(ROOT / "tmp").mkdir(exist_ok=True)   # check_calc_guard needs a temporary directory outside /tmp
ROLLBACK_CASE = ROOT / "cases/044-seahorse-patient-ocr-01/case.yaml"   # any case with a calculator works


class CalcEnv(FakeEnv):
    """Writes calc_state like a calculator command (vials used so far), then answers or fails a check."""
    def step(self, request, params, ledger, notes, noise):
        used, ok = self.script.pop(0)
        self.write_turn(ledger, [], notes)
        (self.ws / "calc_state").mkdir(exist_ok=True)
        (self.ws / "calc_state" / "lab.json").write_text(json.dumps({"used": used}))
        assert ok, "scripted environment failure after the calculator ran"
        return answer(action_types=["inspect"])


def check_resume_rollback():
    """Resume after a request that failed mid-way: calc_state must return to its state before that request, and a
    resume after a committed request must leave it alone (twinlab.lab snapshot/rollback, 2026-10-19)."""
    case = Case(ROLLBACK_CASE, "case")
    used = lambda ws: json.loads((ws / "calc_state" / "lab.json").read_text())["used"]   # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = td / "env_ws"
        ws.mkdir()
        (ws / "session_record.md").write_text("")
        env = CalcEnv(ws, [(1, True), (16, False)])
        lab = Lab(case, env, td, seed=0)
        lab.call("lab", {"request": "req 1"})
        assert used(ws) == 1
        try:   # as mcp_server does on an exception inside the Lab
            lab.call("lab", {"request": "req 2"})
            raise SystemExit("expected the scripted failure")
        except AssertionError:
            # shape of a JSON parse failure (001-03 run 20261029-202451): exception line, then an answer excerpt
            lab.fatal = ('Traceback (most recent call last):\n  File "x.py", line 1, in f\n'
                         'json.decoder.JSONDecodeError: Unterminated string\n\nDuring handling of the above '
                         'exception, another exception occurred:\n\nTraceback (most recent call last):\n'
                         '  File "x.py", line 2, in g\nAssertionError: final message is not one JSON object '
                         '(Unterminated string); it ends with \'done.","}\'; see log\n{"verdict":"performed"')
            lab.failed_request = {"name": "lab"}
        assert used(ws) == 16 and lab.n_requests == 1
        st = json.loads(json.dumps(lab.state()))
        lab2 = Lab(case, env, td, seed=0)
        lab2.restore(st)
        note = lab2.pending_notes[-1]
        assert "The failed check was: AssertionError: final message is not one JSON object" in note \
            and 'done.","}' in note and '{"verdict"' not in note, note
        assert used(ws) == 1, f"calc_state not rolled back: used {used(ws)}"
        kept = list(td.glob("calc_state.discarded_request2_*"))
        assert len(kept) == 1 and json.loads((kept[0] / "lab.json").read_text())["used"] == 16, kept
        # resume after a committed request (actor crash, no fatal): nothing to roll back
        env.script = [(5, True)]
        lab2.call("lab", {"request": "req 2 again"})
        st = json.loads(json.dumps(lab2.state()))
        Lab(case, env, td, seed=0).restore(st)
        assert used(ws) == 5, f"committed calc_state was changed: used {used(ws)}"


class FlakyEnv(CalcEnv):
    """CalcEnv whose script entries may be 'net': the calculator ran, then the call died on a gateway error."""
    def step(self, request, params, ledger, notes, noise):
        if self.script[0][1] == "net":
            used, _ = self.script.pop(0)
            self.write_turn(ledger, [], notes)
            (self.ws / "calc_state").mkdir(exist_ok=True)
            (self.ws / "calc_state" / "lab.json").write_text(json.dumps({"used": used}))
            raise AssertionError("claude exited 1; see log\nstdout: API Error: 500 litellm.InternalServerError: "
                                 "OpenAIException - Connection error.")
        self.notes_seen = list(notes)
        return super().step(request, params, ledger, notes, noise)


def check_transient_env_retry():
    """A lab request whose env call dies on a network/gateway error is retried in place (total cap
    LAB_REQUEST_TOTAL_S): calc_state goes back to before the request, the env gets a DISCARDED note, the request is
    committed once; a non-transient failure still propagates on the first attempt."""
    import twinlab.lab as labmod
    case = Case(ROLLBACK_CASE, "case")
    used = lambda ws: json.loads((ws / "calc_state" / "lab.json").read_text())["used"]   # noqa: E731
    pause = labmod.LAB_RETRY_PAUSE_S
    labmod.LAB_RETRY_PAUSE_S = 0
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ws = td / "env_ws"
            ws.mkdir()
            (ws / "session_record.md").write_text("")
            env = FlakyEnv(ws, [(1, True), (9, "net"), (8, "net"), (2, True), (7, False)])
            lab = Lab(case, env, td, seed=0)
            lab.call("lab", {"request": "req 1"})
            lab.call("lab", {"request": "req 2"})
            assert used(ws) == 2 and lab.n_requests == 2, (used(ws), lab.n_requests)
            assert sum("was DISCARDED" in n and "network/gateway" in n for n in env.notes_seen) == 2, env.notes_seen
            assert len(list(td.glob("calc_state.discarded_request2_*"))) == 2
            try:
                lab.call("lab", {"request": "req 3"})
                raise SystemExit("expected the scripted non-transient failure to propagate")
            except AssertionError as e:
                assert "scripted environment failure" in str(e) and lab.n_requests == 2
    finally:
        labmod.LAB_RETRY_PAUSE_S = pause


class OutboxEnv(FakeEnv):
    """Writes a file into env_ws/outbox like a calculator export, then answers or fails a check."""
    def step(self, request, params, ledger, notes, noise):
        name, ok = self.script.pop(0)
        self.write_turn(ledger, [], notes)
        if name:
            (self.ws / "outbox" / "export").mkdir(parents=True, exist_ok=True)
            (self.ws / "outbox" / "export" / name).write_text(name)
        assert ok, "scripted environment failure after the calculator ran"
        return answer(action_types=["inspect"])


def check_outbox_delivery():
    """Files a calculator puts in env_ws/outbox reach the actor workspace after an accepted request and are listed in
    the result; a request that never committed delivers nothing (twinlab.lab outbox, 074, 2026-10-29)."""
    case = Case(ROLLBACK_CASE, "case")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = td / "env_ws"
        ws.mkdir()
        (ws / "session_record.md").write_text("")
        (td / "actor_workspace").mkdir()
        env = OutboxEnv(ws, [("a.zip", True), (None, True), ("b.zip", False), ("c.zip", True)])
        lab = Lab(case, env, td, seed=0)
        r = lab.call("lab", {"request": "export 1"})["content"][0]["text"]
        assert (td / "actor_workspace/export/a.zip").read_text() == "a.zip" and "- export/a.zip (5 bytes)" in r, r
        assert not (ws / "outbox").exists()
        r = lab.call("lab", {"request": "nothing"})["content"][0]["text"]
        assert "FILES DELIVERED" not in r, r
        try:
            lab.call("lab", {"request": "export 2"})
            raise SystemExit("expected the scripted failure")
        except AssertionError:
            lab.fatal, lab.failed_request = "AssertionError: scripted", {"name": "lab"}
        st = json.loads(json.dumps(lab.state()))
        lab2 = Lab(case, env, td, seed=0)
        lab2.restore(st)
        assert not (ws / "outbox").exists() and not (td / "actor_workspace/export/b.zip").exists()
        assert len(list(td.glob("outbox.discarded_request3_*"))) == 1
        r = lab2.call("lab", {"request": "export 2 again"})["content"][0]["text"]
        assert (td / "actor_workspace/export/c.zip").exists() and "export/c.zip" in r and "b.zip" not in r, r


PER_ITEM_CASE = ROOT / "cases/045-oyster-family-heat-03/case.yaml"     # challenge: cost 1, per_item
PER_REQUEST_CASE = ROOT / "cases/001-coral-lcm-porites-03/case.yaml"  # library_prep: cost 1.5, per request
QUESTION_CASE = ROOT / "cases/045-oyster-family-heat-03/case.yaml"   # challenge: cost 1, per_item


def billing_lab(case_path, td, script):
    case = Case(case_path, "case")
    ws = td / "env_ws"
    ws.mkdir()
    (ws / "session_record.md").write_text("")
    return Lab(case, FakeEnv(ws, script), td, seed=0)


def ledger_records(td):
    return [json.loads(x) for x in (td / "lab_ledger.jsonl").read_text().splitlines()]


def check_charge_line_billing():
    """The Lab bills from the calculator's CHARGE lines in the environment transcript (2026-10-29): a per-item action
    the env listed once for a run of nine groups costs nine (045-03 run 20261029-214420), a per-request action run
    twice in one request costs once (001-01 run 20261027-043031, rule 10), one actor question = one CHARGE line costs
    once even if the env lists it twice (070-01 run 20261026-074442), and a request without billable CHARGE lines
    keeps the env's listing."""
    run9 = ("python3 calculators.py run --challenge H4 --families all", "H4 set up: 9 groups\nCHARGE: challenge x9")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lab = billing_lab(PER_ITEM_CASE, td, [
            (answer(action_types=["challenge"]), [run9]),                       # env under-lists
            (answer(action_types=["challenge"] * 9), [run9]),                   # env right
            (answer(action_types=["challenge"]), [run9, ("python3 calculators.py run --challenge H2 --families A,B",
                                                         "CHARGE: challenge x2")]),   # two runs add up: 11
            (answer(action_types=["inspect"]), [("python3 calculators.py results --challenge H4",
                                                 "rows\nCHARGE: inspect x1")]),       # free only
        ])
        costs = []
        for i in range(4):
            lab.call("lab", {"request": f"req {i + 1}"})
            costs.append(lab.cost_used)
        assert costs == [9, 18, 29, 29], costs
        recs = ledger_records(td)
        assert recs[0]["billing_from_charge_lines"] == {"env_listed": ["challenge"], "charge_lines": {"challenge": 9}}
        assert recs[0]["env"]["action_types"] == ["challenge"] * 9, recs[0]["env"]["action_types"]
        assert "billing_from_charge_lines" not in recs[1] and "billing_from_charge_lines" not in recs[3], recs[1]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lab = billing_lab(PER_ITEM_CASE, td, [(answer(action_types=["challenge"]), [run9]), answer()])
        lab.call("lab", {"request": "run"})
        lab.call("lab", {"request": "next"})
        assert any("Request #1 was billed from the calculator's CHARGE lines {'challenge': 9}" in x
                   for x in lab.env.notes_seen), lab.env.notes_seen
    with tempfile.TemporaryDirectory() as td:   # per-request action in two runs: once (rule 10)
        td = Path(td)
        prep = lambda s: (f"python3 calculators.py prep --setting {s}", "library done\nCHARGE: library_prep x1")  # noqa: E731
        lab = billing_lab(PER_REQUEST_CASE, td, [
            (answer(action_types=["library_prep"], consumed=[{"item": "library_preps", "n": 2}]), [prep("a"), prep("b")]),
        ])
        lab.call("lab", {"request": "two library preps"})
        assert lab.cost_used == 1.5 and "billing_from_charge_lines" not in ledger_records(td)[0], lab.cost_used
    with tempfile.TemporaryDirectory() as td:   # one multi-topic question = one call = one charge (070-01)
        td = Path(td)
        ask = ("python3 calculators.py ask --topic cleaning,gap", "HAOYU (cleaning): x\nHAOYU (gap): y\nCHARGE: challenge x1")
        lab = billing_lab(QUESTION_CASE, td, [(answer(action_types=["challenge", "challenge"]), [ask])])
        lab.call("lab", {"request": "one question on two topics"})
        assert lab.cost_used == 1, lab.cost_used
    with tempfile.TemporaryDirectory() as td:   # no billable CHARGE line (calculators without them): env listing
        td = Path(td)
        lab = billing_lab(PER_ITEM_CASE, td, [(answer(action_types=["challenge"] * 3), [("cat notes.md", "notes")])])
        lab.call("lab", {"request": "no calculator"})
        assert lab.cost_used == 3, lab.cost_used
    with tempfile.TemporaryDirectory() as td:   # session_record.md CHARGE lines are not billed again when the env
        td = Path(td)                           # tails the record in a calculator command (047-02 233832 #15)
        lab = billing_lab(PER_ITEM_CASE, td, [(answer(action_types=["challenge"] * 9, observation="H4\nCHARGE: challenge x9",
                                                      grounding="ran it\n  CHARGE: challenge x9"), [run9])])
        lab.call("lab", {"request": "run"})
        rec = (td / "env_ws" / "session_record.md").read_text()
        # the observation's CHARGE line is cut before the actor sees it (2026-11-04), the grounding's is recorded
        assert rec.count("recorded CHARGE: challenge x9") == 1 and "\nCHARGE:" not in rec, rec
        assert lab.pending_notes and "billing line" in lab.pending_notes[-1], lab.pending_notes
        tail2 = ("tail -5 session_record.md; python3 calculators.py run --challenge H2 --families A,B",
                 rec + "\nH2 set up\nCHARGE: challenge x2")
        lab.env.script.append((answer(action_types=["challenge"] * 2), [tail2]))
        lab.call("lab", {"request": "run H2"})
        assert lab.cost_used == 11, lab.cost_used
        assert "billing_from_charge_lines" not in ledger_records(td)[1], ledger_records(td)[1]
    from twinlab.env_check import charges_from_outputs
    tagged = ["CHARGE: question x1 [once per request]", "CHARGE: question x1 [once per request]"]
    assert charges_from_outputs(tagged, {"question"}, "t") == {"question": 1}
    assert charges_from_outputs(["CHARGE: a x2, b x3", "CHARGE: a x1"], {"a", "b"}, "t") == {"a": 3, "b": 3}
    assert charges_from_outputs(["CHARGE: a x2", "CHARGE: none"], set(), "t") == {"a": 1}
    # anchored at the line start (2026-10-30): a printed docstring's mid-line charge text is not billed (004-02
    # 20261029-234203 #1), a real line (also indented) is, and `uniq -c` prefixes are not read
    doc = "  quote --actions <k1,k2>       lab hours of the named steps only (CHARGE: quote x1, cost 0)"
    assert charges_from_outputs([doc], {"quote"}, "t") == {}, "docstring charge text was billed"
    assert charges_from_outputs([doc + "\nCHARGE: quote x1\n  CHARGE: read x2"], {"read"}, "t") == {"quote": 1, "read": 2}
    assert charges_from_outputs(["     42 CHARGE: modify_wells x1"], {"modify_wells"}, "t") == {}
    from twinlab.env_check import filtered_calculator_commands
    cmds = ["cd ws && python3 calculators.py run --x 1 2>&1 | grep -v '^#'",         # filtered
            "python3 calculators.py run --x 1 | uniq -c",                             # filtered
            "python3 calculators.py run --x 1 > out.txt",                             # redirected
            "for p in A B; do timeout 60 python3 calculators.py t --p $p; done | grep x",   # loop output filtered
            "python3 calculators.py run --x 1 2>&1", "python3 calculators.py a || true",   # not filtered
            "tail -3 session_record.md | cut -c1-80; python3 calculators.py read --p A",  # pipe on another command
            "python3 calculators.py take --spec '{\"L\": \"X|Y\"}' && python3 calculators.py b",   # quoted |
            "cat notes.md | grep x", "grep -n RECORD calculators.py | head"]         # no calculator invocation
    assert filtered_calculator_commands(cmds) == cmds[:4], filtered_calculator_commands(cmds)
    with tempfile.TemporaryDirectory() as td:   # filtered output: bill max(env listing, CHARGE lines) per key
        td = Path(td)
        grep9 = ("python3 calculators.py run --challenge H4 --families all | grep set", "H4 set up: 9 groups")
        piped2 = ("python3 calculators.py run --challenge H2 --families A,B | tail -3", "CHARGE: challenge x2")
        lab = billing_lab(PER_ITEM_CASE, td, [
            (answer(action_types=["challenge"] * 9), [grep9]),            # grep dropped the CHARGE line: listing 9
            (answer(action_types=["challenge"]), [piped2]),               # listing 1 < CHARGE 2: 2
            (answer(action_types=["challenge"] * 3), [piped2]),           # listing 3 > CHARGE 2: 3
            answer(),
        ])
        costs = []
        for i in range(4):
            lab.call("lab", {"request": f"req {i + 1}"})
            costs.append(lab.cost_used)
        assert costs == [9, 11, 14, 14], costs
        recs = ledger_records(td)
        assert recs[0]["billing_from_charge_lines"]["filtered_commands"] == [grep9[0]], recs[0]
        assert recs[1]["billing_from_charge_lines"]["billed_max_of_listing_and_charge_lines"] == {"challenge": 2}
        assert "filtered_commands" in recs[2]["billing_from_charge_lines"], recs[2]
        assert any("Request #3 ran calculator commands through pipes" in x for x in lab.env.notes_seen), \
            lab.env.notes_seen


def check_resume_after_kill_billing():
    """A run stopped inside request #2 without a fatal error (process killed): the resubmitted request #2 is billed
    from its own CHARGE lines only, not also from the interrupted attempt's (twinlab.lab restore, 2026-11-03)."""
    run9 = ("python3 calculators.py run --challenge H4 --families all", "H4 set up: 9 groups\nCHARGE: challenge x9")
    run2 = ("python3 calculators.py run --challenge H2 --families A,B", "H2 set up\nCHARGE: challenge x2")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lab = billing_lab(PER_ITEM_CASE, td, [(answer(action_types=["challenge"] * 9), [run9])])
        lab.call("lab", {"request": "run H4"})
        assert lab.cost_used == 9, lab.cost_used
        lab.save_state()
        # request #2 starts (snapshot taken, env runs the calculator), then the process dies: no commit, no fatal
        lab.snapshot_calc_state(2)
        lab.env.write_turn(lab.ledger(), [run2], [])
        st = json.loads(lab.state_path.read_text())
        assert st["fatal"] is None and st["n_requests"] == 1, st
        lab2 = Lab(lab.case, lab.env, td, seed=0)
        lab2.restore(st)
        assert any("request #2 was DISCARDED" in x for x in lab2.pending_notes), lab2.pending_notes
        lab2.env.script = [(answer(action_types=["challenge"] * 2), [run2])]
        lab2.call("lab", {"request": "run H2 again"})
        assert lab2.cost_used == 11, f"interrupted attempt billed again: cost {lab2.cost_used}"
        assert "billing_from_charge_lines" not in ledger_records(td)[-1], ledger_records(td)[-1]


HIDDEN_CASE = ROOT / "cases/038-orau-ultrafiltration-02/case.yaml"   # actor_hidden_params: rinse_test.cleaning, ...


def check_lab_limitation_notice():
    """LAB LIMITATION notice: full text also after other output, options for catalog parameters only, hidden
    parameter names and values never shown (2026-11-03)."""
    case = Case(HIDDEN_CASE, "case")
    hidden = case.d["harness"]["actor_hidden_params"][0]
    visible = next(iter(case.catalog_for_actor()))
    lim = lambda p: {"parameter": p, "requested": "off-menu value Z", "used": "", "mode": "lab_limitation",   # noqa: E731
                     "sample": "all"}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = td / "env_ws"
        ws.mkdir()
        (ws / "session_record.md").write_text("")
        lab = Lab(case, FakeEnv(ws, [answer(observation="obs text", parameter_resolution=[lim(visible), lim(hidden)]),
                                     answer(verdict="not_simulatable", observation="",
                                            parameter_resolution=[lim(visible)])]), td, seed=0)
        t = lab.call("lab", {"request": "both"})["content"][0]["text"]
        tail = ("is not available in this lab; nothing was done or consumed for that step. Available options: "
                f"{case.catalog_for_actor()[visible]}")
        assert f"obs text\n\nLAB LIMITATION (not an experimental result): {visible} = off-menu value Z {tail}" in t, t
        assert ("\n\nLAB LIMITATION (not an experimental result): this setting = off-menu value Z is not available "
                "in this lab; nothing was done or consumed for that step. This setting has no option list in "
                "lab_catalog.") in t, t
        assert hidden not in t and hidden.split(".", 1)[1] not in t and "[]" not in t, t
        for v in case.menu_index()[hidden]["values"]:
            assert v not in t, f"hidden value {v!r} shown"
        t = lab.call("lab", {"request": "only limited"})["content"][0]["text"]
        assert t.endswith(f"\n\nLAB LIMITATION (not an experimental result): {visible} = off-menu value Z {tail}"), t


def check_provenance_lines_cut():
    """Calculator provenance lines the env copied into an observation are cut before the leak check (084-01 run
    20261103-072629 #1); a mid-line copy still stops the request; the generic leak word spares parameter keys such
    as dnase_source (003)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        obs = "PACKAGE P1: Kestrel x 3\n  _source recorded: campaign log and roll table\nrows 77"
        lab = billing_lab(PER_ITEM_CASE, td, [answer(observation=obs), answer(observation="next"),
                                              answer(observation="value 3 (_source recorded: log)")])
        t = lab.call("lab", {"request": "r1"})["content"][0]["text"]
        assert "PACKAGE P1: Kestrel x 3\nrows 77" in t and "_source" not in t, t
        assert ledger_records(td)[0]["env"]["stripped_provenance_lines"] == ["  _source recorded: campaign log and roll table"]
        lab.call("lab", {"request": "r2"})
        assert any("provenance line" in x for x in lab.env.notes_seen), lab.env.notes_seen
        try:
            lab.call("lab", {"request": "r3"})
            raise SystemExit("expected the leak check to stop a mid-line provenance copy")
        except AssertionError as e:
            assert "'_source'" in str(e), e
    from twinlab.lab import GENERIC_LEAK
    generic = re.compile(GENERIC_LEAK, re.I)   # the generic part alone (045-01 also lists '_source' itself)
    assert generic.search("column_cleanup.dnase_source = old kit; vortex_source true") is None
    assert generic.search("value 3 (_source recorded: log)") is not None


def check_partial_without_items_relabelled():
    """partially_performed with no not_performed_items (031-02 run 20261014-124236 #19 aborted the run) is relabelled
    performed and recorded in the ledger; with nothing done and nothing reported it still stops (2026-11-04)."""
    run1 = ("python3 calculators.py run --challenge H4 --families A", "H4 set up: 1 group\nCHARGE: challenge x1")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lab = billing_lab(PER_ITEM_CASE, td, [
            (answer(verdict="partially_performed", action_types=["challenge"], observation="all done"), [run1]),
            answer(verdict="partially_performed", observation="  ")])
        t = lab.call("lab", {"request": "r1"})["content"][0]["text"]
        rec = ledger_records(td)[0]
        assert rec["env"]["verdict"] == "performed" and rec["cost"] == 1, rec
        assert rec["env"]["verdict_relabelled"] == "partially_performed -> performed (no not_performed_items)", rec
        assert "all done" in t and MESSAGES["not_performed"] not in t, t
        try:
            lab.call("lab", {"request": "r2"})
            raise SystemExit("expected a stop: partially_performed with nothing done and nothing reported")
        except AssertionError as e:
            assert "no not_performed_items, no action_types and no observation" in str(e), e


def check_lost_output_flags():
    """An observation cut off with an ellipsis, and a billed calculator output whose decimals do not reach the
    observation (087-02 replay 20261103-141338 #1), are flagged in the ledger record (no retry) and by
    output_problems for check_env; a complete observation is not flagged (2026-11-04)."""
    from twinlab.env_check import output_problems
    refine = ("python3 calculators.py run --challenge H4 --families A",
              "RESULT A: mortality 0.42, 0.37\nguest-guest distances: C17-C18 1.36, C18-C19 1.36\n"
              "  _source recorded: x 9.99\nCHARGE: challenge x1")
    nmr = ("python3 calculators.py run --challenge H2 --families B", "NMR ratio 1 : 0.16\nCHARGE: challenge x1")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lab = billing_lab(PER_ITEM_CASE, td, [
            (answer(action_types=["challenge"] * 2,
                    observation="RESULT A: mortality 0.42, 0.37\ndistances: C17-C18 1.36, C17-C19 ... "),
             [refine, nmr]),
            (answer(action_types=["challenge"] * 2,
                    observation="mortality 0.42, 0.37; C17-C18 1.36, C18-C19 1.36; NMR 1 : 0.16"), [refine, nmr])])
        lab.call("lab", {"request": "cut"})
        lab.call("lab", {"request": "complete"})
        recs = ledger_records(td)
        kinds = [p["kind"] for p in recs[0]["env"]["output_flags"]]
        assert kinds == ["observation_truncated", "billed_output_unreported"], recs[0]["env"]["output_flags"]
        assert "0.16" in recs[0]["env"]["output_flags"][1]["what"], recs[0]["env"]["output_flags"]
        assert "output_flags" not in recs[1]["env"], recs[1]["env"]
        assert recs[0]["cost"] == 2, recs[0]   # flag only: billing unchanged
    # free CHARGE lines and integer-only outputs are not result checks; numbers from the request text do not count
    assert output_problems(["DAY 6\nCHARGE: challenge x1", "q 0.5\nCHARGE: inspect x1"], "ok", "r") == []
    assert output_problems(["set 0.003 run\nCHARGE: challenge x1"], "ok", "amplitude 0.003") == []
    assert output_problems([], "fine …", "r")[0]["kind"] == "observation_truncated"


GUARD_FAKE_CALC = '''import json, sys
from pathlib import Path
st = Path("calc_state"); st.mkdir(exist_ok=True)
f = st / "n.json"; n = json.loads(f.read_text())["n"] if f.exists() else 0
cmd = sys.argv[1]
if cmd == "grow":
    f.write_text(json.dumps({"n": n + 1})); print(f"grown {n + 1} OD 0.731"); print("CHARGE: grow x1")
elif cmd == "status":
    print(f"vials used {n}")
elif cmd == "quote":
    print("quote 2.5 h"); print("CHARGE: inspect x1")
'''


class GuardEnv(FakeEnv):
    """FakeEnv of a guarded run: a script entry may carry guard log records the calculator guard wrote."""
    def step(self, request, params, ledger, notes, noise):
        entry = self.script[0]
        if isinstance(entry, tuple) and len(entry) == 3:
            self.script[0] = entry[:2]
            with self.guard_log.open("a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in entry[2]))
        return super().step(request, params, ledger, notes, noise)


def check_benchmark_name_guard():
    """The benchmark's own name is blocked in any spelling: the guard hook denies the search and
    withholds tool output, and the post-run audit flags tool results but not the actor's own text."""
    from twinlab.actors import guard
    from twinlab.judge import audit_actor
    case = Case(ROLLBACK_CASE, "case")
    assert "TwinLab-Bench" in case.guard_needles and "twinlab bench" in case.guard_needles
    assert case_name("002-resazurin-clam-02") == "resazurin-clam" and case_name("007-resazurin-oyster-02-xp-01") == "resazurin-oyster"
    assert case_name(case.id) in case.guard_needles
    with tempfile.TemporaryDirectory() as td:
        hooks = Path(td) / "hooks"
        guard.install(hooks, case)

        def hook(ev: dict) -> str:
            return subprocess.run([sys.executable, str(hooks / guard.HOOK_FILE)], input=json.dumps(ev),
                                  capture_output=True, text=True, check=True).stdout

        pre = {"hook_event_name": "PreToolUse", "tool_name": "WebSearch"}
        out = hook({**pre, "tool_input": {"query": "twinlab-bench dataset answers"}})
        assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny", out
        assert hook({**pre, "tool_input": {"query": "qubit assay linear range"}}) == ""
        post = hook({"hook_event_name": "PostToolUse", "tool_name": "WebFetch", "tool_input": {},
                     "tool_response": {"text": "code at huggingface.co/datasets/x/TwinLab-Bench"}})
        assert "withheld" in json.loads(post)["hookSpecificOutput"]["updatedToolOutput"]["text"], post
        run = Path(td) / "run"
        run.mkdir()
        (run / "run_config.json").write_text(json.dumps({"actor_agent": "claude-code"}))
        stream = run / "actor_stream_r0_0.jsonl"
        own = {"type": "assistant", "message": {"content": [{"type": "text", "text": "maybe a TwinLab-Bench task"}]}}
        stream.write_text(json.dumps(own) + "\n")
        assert not audit_actor(case, run)["contaminated"]
        result = {"type": "user", "message": {"content": [{"type": "tool_result", "content": "found TwinLab_Bench"}]}}
        stream.write_text(json.dumps(result) + "\n")
        assert audit_actor(case, run)["contaminated"]
        # a request alone is not contamination; the third stopped request is
        call = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch",
                                                              "input": {"query": "TwinLab-Bench answers"}}]}}
        stream.write_text(json.dumps(call) + "\n")
        log = run / "actor_config/hooks/guard.log"
        log.parent.mkdir(parents=True)
        for n_blocked, expected in ((0, False), (2, False), (3, True)):
            log.write_text("".join(json.dumps({"tool": "WebSearch", "blocked": True}) + "\n" for _ in range(n_blocked)))
            a = audit_actor(case, run)
            assert a["contaminated"] is expected and a["n_blocked_attempts"] == n_blocked, a


def check_calc_guard():
    """twinlab/calc_guard.py (2026-11-04, 044-01 replays 025345/025709 `grow --lines F01 >/dev/null 2>&1; echo skip`):
    a calculator run whose whole stdout goes to /dev/null and that prints a billable CHARGE line is refused before it
    counts (calc_state restored, CHARGE line never printed, refusal text in the shell's own output); free runs with
    /dev/null, pipes into head, --help and plain runs are untouched. Run through check_env's replay_command (the same
    guarded PATH a run uses). In the Lab the refused command is neither billed nor treated as a filtered command, the
    plain rerun is billed once, and the guard record goes into the ledger."""
    from twinlab import calc_guard
    from twinlab.env_check import CHARGE_RE, replay_command
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:   # not under /tmp: replay_command binds nothing here
        run = Path(td)
        ws = run / "env_ws"
        ws.mkdir()
        (ws / "calculators.py").write_text(GUARD_FAKE_CALC)
        calc_guard.install(run)
        glog = run / "env_guard_replay.jsonl"   # replay_command's guard log (ws.parent)
        used = lambda: json.loads((ws / "calc_state/n.json").read_text())["n"] if (ws / "calc_state/n.json").exists() else 0  # noqa: E731
        sh = lambda c: replay_command(c, ws, ws)   # noqa: E731
        out = sh("python3 calculators.py grow --lines F01 >/dev/null 2>&1; echo skip")
        assert calc_guard.REFUSAL_MARK in out and "skip" in out and "grow x1" in out, out
        assert CHARGE_RE.search(out) is None and "0.731" not in out, out
        assert used() == 0 and not (ws / "calc_state").exists(), "refused run changed calc_state"
        rec = json.loads(glog.read_text().splitlines()[-1])
        assert rec["decision"].startswith("refused") and rec["billed"] == ["grow x1"], rec
        out = sh("python3 calculators.py grow; python3 calculators.py status >/dev/null; echo ok")
        assert "CHARGE: grow x1" in out and calc_guard.REFUSAL_MARK not in out and used() == 1, out
        assert json.loads(glog.read_text().splitlines()[-1])["decision"].startswith("passed"), glog.read_text()
        out = sh("python3 calculators.py quote &>/dev/null; echo q")   # free CHARGE key: passes
        assert calc_guard.REFUSAL_MARK not in out, out
        n_log = len(glog.read_text().splitlines())
        out = sh("python3 calculators.py grow | head -5; python3 calculators.py --help >/dev/null; python3 -c 'print(7)'")
        assert "CHARGE: grow x1" in out and used() == 2 and calc_guard.REFUSAL_MARK not in out, out
        assert len(glog.read_text().splitlines()) == n_log, "pipes, --help and -c must not reach the guard's check"
        out = sh("for i in 1 2; do python3 calculators.py grow; done >/dev/null; echo loop")
        assert out.count(calc_guard.REFUSAL_MARK) == 2 and used() == 2, out
    # Lab: refused attempt + plain rerun in one request
    refused_out = ("\nLAB HARNESS REFUSED this command and did not run it: `python3 calculators.py run --challenge H4 "
                   "--families A` performs a billed lab action (challenge x1) ...\nskip")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        case = Case(PER_ITEM_CASE, "case")
        ws = td / "env_ws"
        ws.mkdir()
        (ws / "session_record.md").write_text("")
        rec = {"command": "python3 calculators.py run --challenge H4 --families A", "billed": ["challenge x1"],
               "decision": "refused (billed action with discarded output; state restored, nothing billed)"}
        env = GuardEnv(ws, [(answer(action_types=["challenge"], observation="mortality 0.42"),
                             [("python3 calculators.py run --challenge H4 --families A >/dev/null 2>&1; echo skip",
                               refused_out),
                              ("python3 calculators.py run --challenge H4 --families A",
                               "RESULT A: mortality 0.42\nCHARGE: challenge x1")], [rec])])
        env.extra_env = {"PATH": "guarded"}
        lab = Lab(case, env, td, seed=0)
        lab.call("lab", {"request": "run H4 on A"})
        r = ledger_records(td)[0]
        assert r["cost"] == 1 and "billing_from_charge_lines" not in r, r
        assert r["env"]["guard_discarded_output_runs"] == [rec], r["env"]
        assert not any("pipes or filters" in x for x in lab.pending_notes), lab.pending_notes


def check_json_answer_forms():
    """Env/judge final message forms accepted by twinlab.claude_cli.parse_json_answer: bare object, one fenced block,
    prose then object, prose then one fenced block (069-02, 2026-10-29), any of these plus stray closing braces
    (003-03 judge, 2026-10-29); anything else fails."""
    from twinlab.claude_cli import parse_json_answer
    schema = {"type": "object", "required": ["a"]}
    obj = '{"a": 1, "b": "x {y}"}'
    for text in (obj, f"```json\n{obj}\n```", f"Nothing could be done.\n\n{obj}",
                 f"Nothing could be done. Nothing was charged.\n\n```json\n{obj}\n```",
                 f"{obj}\n}}", f"{obj}}}]", f"```json\n{obj}\n```\n}}", f"Nothing could be done.\n\n{obj}\n}}"):
        assert parse_json_answer({"result": text}, schema, Path("-")) == {"a": 1, "b": "x {y}"}, text
    for text in (f"{obj}\nmore", f"{obj}\n}}\nThe score is 9.", f"{obj}}} done",f"Note {{x}}.\n```json\n{obj}\n```", f"x\n```json\n{obj}\n```\ntrailing"):
        try:
            parse_json_answer({"result": text}, schema, Path("-"))
            raise SystemExit(f"accepted {text!r}")
        except AssertionError:
            pass
    # a malformed tail is named on the error's first line (001-03 run 20261029-202451 ended '...","}')
    try:
        parse_json_answer({"result": '{"a": 1, "g": "done.","}'}, schema, Path("-"))
        raise SystemExit("accepted a dangling key")
    except AssertionError as e:
        assert str(e).splitlines()[0].count('done.","}') == 1, str(e)


def open_objects(schema, where="root"):
    """Paths of objects in a JSON schema that list properties but are not closed (additionalProperties false)."""
    bad = []
    if isinstance(schema, list):
        for i, x in enumerate(schema):
            bad += open_objects(x, f"{where}[{i}]")
    elif isinstance(schema, dict):
        if "properties" in schema and schema.get("additionalProperties") is not False:
            bad.append(where)
        for k, v in schema.items():
            if k == "properties":
                for pk, pv in v.items():
                    bad += open_objects(pv, f"{where}.{pk}")
            elif k not in ("required", "enum"):
                bad += open_objects(v, f"{where}.{k}")
    return bad


def check_structured_answers():
    """--json-schema answers (restored 2026-10-30): env and judge schemas closed at every level, structured_output
    re-validated and required, StructuredOutput rejections counted from transcript lines, and the env answer path
    chosen by the system prompt's marker (LEGACY text JSON for runs started 2026-10-26..10-30)."""
    import twinlab.env_agent as env_mod
    from twinlab.claude_cli import (STRUCTURED_ANSWER_MARK, json_answer_instructions, structured_answer,
                                    structured_output_rejections)
    from twinlab.judge import judge_schema
    case = Case(ROLLBACK_CASE, "case")
    js = judge_schema(case)
    assert not open_objects(js), open_objects(js)
    assert js["properties"]["rubric"]["additionalProperties"] is False
    # a closed schema rejects an extra key; a missing structured_output fails
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        env = EnvAgent(case, td, "no-model", resume=False)
        assert not open_objects(env.schema), open_objects(env.schema)
        assert env.schema["properties"]["parameter_resolution"]["items"]["additionalProperties"] is False
        assert "additionalProperties" not in json.dumps(env.schema_open)
        assert structured_answer({"structured_output": answer()}, env.schema, Path("-")) == answer()
        for bad in ({"structured_output": dict(answer(), bash_note="x")}, {"structured_output": None},
                    {"result": json.dumps(answer())}):
            try:
                structured_answer(bad, env.schema, Path("-"))
                raise SystemExit(f"accepted {str(bad)[:80]}")
            except AssertionError:
                pass
        # rejections: a StructuredOutput tool_use (recorded twice, as the CLI does) rejected once, then accepted;
        # an is_error result of another tool does not count
        so = lambda i: {"type": "tool_use", "id": i, "name": "StructuredOutput", "input": {}}   # noqa: E731
        res = lambda i, err: {"type": "tool_result", "tool_use_id": i, "content": "Output does not match",  # noqa: E731
                              "is_error": err}
        lines = [{"message": {"role": "assistant", "content": [so("a")]}},
                 {"type": "api-request-blob", "message": {"role": "assistant", "content": [so("a")]}},
                 {"message": {"role": "user", "content": [res("a", True)]}},
                 {"message": {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "Bash"}]}},
                 {"message": {"role": "user", "content": [res("b", True)]}},
                 {"message": {"role": "assistant", "content": [so("c")]}},
                 {"message": {"role": "user", "content": [res("c", False)]}}, {"type": "tools-used"}]
        assert structured_output_rejections([json.dumps(x) for x in lines]) == ["Output does not match"]

        calls = []
        real_run = env_mod.run_claude_json_env

        def fake_run(args, cwd, timeout_s, log_path, tmp_dir, extra_env):
            # env calls carry the calculator guard's PATH (twinlab/calc_guard.py) for cases with a calculator
            assert extra_env["PATH"].startswith(str(Path(td) / "env_bin")), extra_env
            calls.append(args)
            out = {"usage": {"input_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                   "num_turns": 2}
            if "--json-schema" in args:
                return dict(out, result="", structured_output=answer())
            return dict(out, result="```json\n" + json.dumps(dict(answer(), bash_note="x")) + "\n```")

        env_mod.run_claude_json_env = fake_run
        try:
            led = {"n_requests": 0}
            sp = env.system_prompt_file.read_text(encoding="utf-8")
            assert STRUCTURED_ANSWER_MARK in sp and env_mod.ANSWER_FORMAT_MARK not in sp and not env.legacy_text_json
            assert env.step("r", {}, led, [], []) == answer()
            a = calls[-1]
            assert json.loads(a[a.index("--json-schema") + 1]) == env.schema and "--session-id" in a
            assert a[1].endswith("Submit your answer with the StructuredOutput tool."), a[1][-200:]
            st = env.state()
            # resume of a structured run stays structured
            env2 = EnvAgent(case, td, "no-model", resume=True)
            env2.restore(st)
            assert not env2.legacy_text_json and env2.step("r", {}, led, [], []) == answer()
            assert "--json-schema" in calls[-1] and "--resume" in calls[-1]
            # resume of a LEGACY run (system prompt with # ANSWER FORMAT): text JSON, open schema, key-order line
            env.system_prompt_file.write_text(sp.split(STRUCTURED_ANSWER_MARK)[0] + json_answer_instructions(
                env.schema_open), encoding="utf-8")
            env3 = EnvAgent(case, td, "no-model", resume=True)
            env3.restore(st)
            assert env3.legacy_text_json
            assert env3.step("r", {}, led, [], []) == dict(answer(), bash_note="x")
            assert "--json-schema" not in calls[-1] and "described under ANSWER FORMAT" in calls[-1][1]
            # neither marker (run started before 2026-10-26): not resumable
            env.system_prompt_file.write_text(sp.split(STRUCTURED_ANSWER_MARK)[0], encoding="utf-8")
            try:
                EnvAgent(case, td, "no-model", resume=True)
                raise SystemExit("resumed a run without an answer marker")
            except AssertionError:
                pass
        finally:
            env_mod.run_claude_json_env = real_run


def check_score_rescale_100():
    """0-100 scale (twinlab/scale.py): items x 100/M, total one decimal half-up, core missed -> 0."""
    from types import SimpleNamespace
    from twinlab.judge import score
    from twinlab.scale import rescale

    def rub(**pts):
        return {k: {"points": v, "component": "conclusion"} for k, v in pts.items()}

    r85 = rub(A=3, B=2.5, C=3)                              # M = 8.5, core A
    s = rescale({"A": 2, "B": 2.5, "C": 3}, 7.5, r85, ["A"])
    assert (s["max_native"], s["total_100"], s["derived_100"]) == (8.5, 88.2, 0), s
    s = rescale({"A": 3, "B": 2.5, "C": 3}, 8.5, r85, ["A"])
    assert (s["total_100"], s["derived_100"], s["points_100"]) == (100.0, 100.0, {"A": 35.3, "B": 29.4, "C": 35.3}), s
    r1025 = rub(A=4, B=3.25, C=3)                           # M = 10.25
    s = rescale({"A": 4, "B": 3.25, "C": 2.75}, 10.0, r1025, ["A"])
    assert (s["total_100"], s["derived_100"]) == (97.6, 97.6), s     # 97.5609...
    s = rescale({"A": 4, "B": 3.25, "C": 2.75}, 9.0, r1025, ["A"])   # one trap (-1) scaled by the same factor
    assert s["total_100"] == 87.8, s                                 # 87.8048...
    s = rescale({"A": 4, "B": 0.5}, 0.5, rub(A=4, B=4), ["A"])       # 6.25 -> half-up 6.3, not banker's 6.2
    assert s["total_100"] == 6.3, s
    # judge.score adds the 0-100 fields next to the native ones (unchanged)
    case = SimpleNamespace(scoring={"rubric": r85, "pass_rule": {"required_full": ["A"]},
                                    "penalties": {"per_trap_triggered": -1, "consumable_overuse": []}},
                           consumables=[], initial_consumed={}, rubric_sha256="x", budget_unit="u")
    lab = SimpleNamespace(consumed={}, report={"x": 1}, cost_used=3.0, n_requests=2)
    flag = {"flag": False}
    j = {"rubric": {"A": {"points": 3}, "B": {"points": 2}, "C": {"points": 3}}, "traps": {"t": {"triggered": True}},
         "env_leak": flag, "env_inconsistency": flag, "env_coverage_gap": flag}
    s = score(case, j, lab)
    assert (s["total"], s["pass"], s["max_native"], s["total_100"], s["derived_100"]) == (7, True, 8.5, 82.4, 82.4), s
    j["rubric"]["A"]["points"] = 2.5
    s = score(case, j, lab)
    assert (s["total"], s["pass"], s["total_100"], s["derived_100"]) == (6.5, False, 76.5, 0), s


def main():
    verbose = "-v" in sys.argv
    check_score_rescale_100()
    print("PASS 0-100 score rescale (x 100/M, one decimal half-up, penalties scaled, core missed -> 0)", flush=True)
    check_json_answer_forms()
    print("PASS env/judge JSON answer forms (LEGACY text path)", flush=True)
    check_structured_answers()
    print("PASS structured answers (closed schemas, re-validation, rejection count, answer path by marker)", flush=True)
    check_resume_rollback()
    check_transient_env_retry()
    print("PASS resume rollback of calc_state", flush=True)
    check_outbox_delivery()
    print("PASS outbox delivery to the actor workspace", flush=True)
    check_charge_line_billing()
    print("PASS billing from CHARGE lines (per-item, per-request split, one question, no CHARGE, anchored, filtered, session_record)", flush=True)
    check_resume_after_kill_billing()
    print("PASS resume after a killed request bills the resubmission only", flush=True)
    check_lab_limitation_notice()
    print("PASS LAB LIMITATION notice (full text after output, catalog options only, hidden names not shown)", flush=True)
    check_provenance_lines_cut()
    print("PASS calculator provenance lines cut from observations, mid-line copies still stop", flush=True)
    check_partial_without_items_relabelled()
    print("PASS partially_performed without not_performed_items relabelled performed, empty one still stops", flush=True)
    check_lost_output_flags()
    print("PASS lost env output flagged (ellipsis end, billed decimals unreported), billing unchanged", flush=True)
    check_calc_guard()
    print("PASS calculator guard: billed run with /dev/null stdout refused before it counts, free/pipe/help untouched, "
          "Lab bills the plain rerun once", flush=True)
    check_benchmark_name_guard()
    print("PASS benchmark name: web search denied, tool output withheld, post-run audit flags it, own text ignored", flush=True)
    paths = [Path(p) for p in sys.argv[1:] if p != "-v"] or sorted((ROOT / "cases").glob("*/case.yaml"))
    failed = []
    for p in paths:
        try:
            check(p, verbose)
            print(f"PASS {p.parent.name}", flush=True)
        except Exception as e:
            failed.append(p.parent.name)
            print(f"FAIL {p.parent.name}: {type(e).__name__}: {e}", flush=True)
            if verbose:
                traceback.print_exc()
    print(f"\n{len(paths) - len(failed)}/{len(paths)} cases pass" + (f"; failing: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
