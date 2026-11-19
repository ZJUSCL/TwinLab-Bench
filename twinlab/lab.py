"""The lab: the four tools the actor calls (lab, lab_catalog, lab_status, submit_final_report).

The Lab runs in the harness process, outside the actor's sandbox (mcp_server.py exposes it over MCP). For each `lab`
request it asks the environment agent what happens, checks the answer, charges the case's action costs, keeps the
budget and consumables, and returns the observation text and images to the actor. Everything deterministic about a
run (costs, budget, noise draws, what was shown) is decided here, not by the environment model.
"""
import base64
import collections
import io
import json
import math
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image

from .case import Case
from .env_agent import CONTINUED_STEP, EnvAgent
from .env_check import (CALC_INVOKE_RE, FREE_ACTIONS, charges_from_outputs, filtered_calculator_commands, load_turns,
                        outside_file_access, output_problems, recorded_charge_lines, strip_stdout_devnull)
from .prompts import MESSAGES

Image.MAX_IMAGE_PIXELS = None   # slide overviews are ~143 MP stitched scans from a trusted source
MAX_IMAGE_EDGE = 1568           # images returned to the actor are downscaled to this long edge
IMAGE_JPEG_QUALITY = 85
EXC_LINE_RE = re.compile(r"[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Expired)\b")   # traceback exception header line (TimeoutExpired: 008-02 DS run 20261104-073446)
CALC_STATE = "calc_state"                   # the calculator's state dir inside env_ws (cases' calculators.py)
CALC_SNAPSHOT = "calc_state_before_request" # run_dir copy of calc_state taken before each lab request (resume rollback)
# env_ws/outbox: files a case calculator hands to the actor (e.g. logger exports of real data, 074, 2026-10-29); after
# an accepted request the Lab moves them into the actor's workspace (same relative path) and lists them in the result
OUTBOX = "outbox"
# one lab request, env retries included, may take this long (total cap 1800 s); the actor's MCP tool
# timeout (actors/claude_code.py) is set above it
LAB_REQUEST_TOTAL_S = 1800
LAB_RETRY_PAUSE_S = 30          # wait before resubmitting after a transient env failure
LAB_RETRY_MIN_LEFT_S = 120      # no new attempt with less time than this left
# failures worth retrying: gateway/network errors seen in env call records (11-07: "Connection error", 500 via LiteLLM)
TRANSIENT_MARKS = ("Connection error", "ECONNRESET", "ETIMEDOUT", "ECONNREFUSED", "socket hang up",
                   "InternalServerError", "ServiceUnavailable", "APIConnectionError", "overloaded",
                   "API Error: 500", "API Error: 502", "API Error: 503", "API Error: 504", "API Error: 529",
                   "Request timed out", "timed out")
ACTOR_WS = "actor_workspace"                # run_dir/actor_workspace, as in twinlab/actors/base.py

def build_tools(case: Case) -> list:
    return [
        {
            "name": "lab",
            "description": (
                "Submit ONE request to the physical lab: any bench operation, any measurement, a look at existing "
                "material, or a general question to a senior lab member. Describe exactly what should be done and "
                "every parameter. Returns raw observations (numbers, descriptions, instrument output, images), the "
                "cost charged, and the remaining budget."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "request": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "description": "Optional {\"procedure.param\": [values]} from lab_catalog. Several values = "
                                       "any of them is acceptable; the lab reports which one it used.",
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["request"],
            },
        },
        {
            "name": "lab_catalog",
            "description": "List the lab's procedures and the parameter options its equipment and reagents support.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "lab_status",
            "description": "Show the budget used/remaining and how much of each limited material is left.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "submit_final_report",
            "description": "Submit your final report. Call exactly once, when you are finished.",
            "inputSchema": case.report_tool_schema(),
        },
    ]


# internal file and section names that must never appear in an observation (cases add their own leak_patterns)
GENERIC_LEAK = (r"hidden_truth|notebook_source|case_env|observation_table|env_knowledge|session_record"
                r"|quantitative_model|lab_history|parameter_menus|expert_reference"
                # mechanism word (005-03 run 20261029-212041, "The calculator did not print ..."); cases only listed
                # the plural. "calculator sheet" is real notebook text in 059's archive entries, shown verbatim.
                r"|\bcalculator\b(?! sheet)"
                # calculator provenance word; \b keeps parameter keys such as dnase_source (003) out
                r"|\b_source\b")
# Calculators print a provenance line ("  _source recorded: ...") for the environment. The env copied one into an
# observation (084-01 run 20261103-072629 #1, leak stop); whole lines that start with it are cut from the observation
# before the leak check, logged, kept in the ledger and reported to the env. Mid-line copies still fail the check.
PROVENANCE_LINE_RE = re.compile(r"^[ \t]*_source\b.*(?:\n|$)", re.M)
CALC_WORD_RE = re.compile(r"\bcalculators?\b(?! sheet)", re.I)
# one sentence (up to . ! ? or end of line) that names the calculator; leading spaces taken with it
CALC_WORD_SENTENCE_RE = re.compile(r"[ \t]*[^.!?\n]*\bcalculators?\b(?! sheet)[^.!?\n]*[.!?]?", re.I)
BILLING_LINE_RE = re.compile(r"^[ \t]*(?:recorded )?(?:CHARGE|BILLING):.*(?:\n|$)", re.M)


class Lab:
    def __init__(self, case: Case, env: EnvAgent, run_dir: Path, seed: int):
        self.case, self.env = case, env
        h = case.d["harness"]
        self.rng = random.Random(seed)
        self.noise_sigma = math.sqrt(math.log(1 + h["replicate_cv"] ** 2))
        self.noise_n = h["noise_multipliers_per_request"]
        self.catalog = case.catalog_for_actor()
        self.menu = case.menu_index()
        # actor_hidden_params: in the menu but not in the actor's catalog; never echoed back (033-02 leak, 2026-10-19)
        # the env reports either 'procedure.param' or the bare 'param'; a bare name is hidden only if no visible param uses it
        hidden_full = set(self.menu) - set(self.catalog)
        bare = lambda keys: {k.split(".", 1)[-1] for k in keys}   # noqa: E731
        self.hidden_params = hidden_full | (bare(hidden_full) - bare(self.catalog))
        print(f"[lab] hidden params (not echoed to actor): {sorted(self.hidden_params)}", flush=True)
        self.tools = build_tools(case)
        self.leak_re = re.compile("|".join([GENERIC_LEAK, *case.leak_patterns]), re.I)
        self.unit = case.budget_unit
        self.per_item = {k for k, a in case.actions.items() if a.get("per_item")}
        self.cost_budget = case.budget_total
        self.cost_used = 0.0
        self.consumed = dict(case.initial_consumed)
        self.n_requests = 0
        self.pending_notes = []
        self.report = None
        self.fatal = None
        self.closed_at = None      # set when lab days are exhausted; actor must report within REPORT_GRACE_S
        self.failed_request = None # the request being processed when a fatal error happened (for resume)
        self.last_deferred_steps = []   # steps the env deferred at a checkpoint in the last request (replay uses it)
        self.run_dir = run_dir
        self.state_path = run_dir / "lab_state.json"
        # case files fingerprint from run_config.json (runs started from 2026-11-05 on; expert replays and older runs
        # have none and are not checked)
        cfg_path = run_dir / "run_config.json"
        self.case_sha = json.loads(cfg_path.read_text()).get("case_files_sha256") if cfg_path.exists() else None
        print(f"[lab] case files sha {'checked before each request' if self.case_sha else 'not recorded: unchecked'}",
              flush=True)
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.ledger_path = run_dir / "lab_ledger.jsonl"
        self.ledger_path.touch()   # a run without lab requests still has a (empty) ledger for judge and env_check
        self.img_cache = run_dir / "img_cache"
        self.img_cache.mkdir(exist_ok=True)   # exists when resuming a run

    # ---------------- persistence (resume) ----------------
    def state(self) -> dict:
        s = self.rng.getstate()
        return {"cost_used": self.cost_used, "consumed": self.consumed,
                "n_requests": self.n_requests, "pending_notes": self.pending_notes, "report": self.report,
                "closed_at": self.closed_at, "fatal": self.fatal, "failed_request": self.failed_request,
                "rng_state": [s[0], list(s[1]), s[2]], "env": self.env.state()}

    def save_state(self):
        # the MCP server runs one thread per request and the actor may call tools in parallel: serialize the
        # snapshot + write, and use a per-thread temp name so two writers never race on one temp file
        with self.state_lock:
            tmp = self.state_path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self.state(), ensure_ascii=False))
            tmp.replace(self.state_path)

    def restore(self, st: dict):
        for k in ("cost_used", "consumed", "n_requests", "pending_notes", "report", "closed_at"):
            setattr(self, k, st[k])
        s = st["rng_state"]
        self.rng.setstate((s[0], tuple(s[1]), s[2]))
        self.env.restore(st["env"])
        rolled_back = self.rollback_calc_state(st["n_requests"], st["fatal"] is not None)
        if rolled_back and st["fatal"] is None:
            # the run stopped inside request #n without a fatal error (process killed, quota): its attempt is in the
            # env transcript under the same request number as the resubmission, so without this note
            # (DISCARDED_RE in env_check.load_turns) its CHARGE lines were billed again with the new attempt
            n = st["n_requests"] + 1
            print(f"[lab] request #{n} was interrupted without a fatal error; its attempt is discarded", flush=True)
            self.pending_notes.append(
                f"Your answer to actor request #{n} was DISCARDED by the harness because the run stopped before "
                f"it was recorded; nothing of it happened and nothing was consumed (the calculator state is back "
                f"to before that request). The actor will resubmit if it still needs it.")
        if st["fatal"] is not None:
            n = st["n_requests"] + 1
            # the exception line, not the traceback's last line: a JSON parse failure appends an excerpt of the bad
            # answer after it, so the last line used to be that excerpt (001-03 run 20261029-202451: the env got its
            # own answer cut at 600 chars, not the error, and repeated the same malformed tail on resume)
            exc_lines = [l for l in st["fatal"].strip().splitlines() if EXC_LINE_RE.match(l)]
            assert exc_lines, f"no exception line in the stored fatal traceback:\n{st['fatal'][-1000:]}"
            print(f"[lab] discard note quotes: {exc_lines[-1][:600]!r}", flush=True)
            self.pending_notes.append(
                f"Your answer to actor request #{n} was DISCARDED by the harness after a technical check failed; "
                f"nothing of it happened and nothing was consumed. The actor will resubmit if it still needs it. "
                f"The failed check was: {exc_lines[-1][:600]}. Stopping at a checkpoint "
                f"keeps verdict = performed; deferred steps are not not_performed_items.")
        print(f"[lab] restored: requests={self.n_requests} cost={self.cost_used} "
              f"previous_fatal={st['fatal'] is not None}", flush=True)

    def snapshot_calc_state(self, n: int):
        """Copy env_ws/calc_state before request #n runs, so a resume can undo a request that never completed
        (044, 2026-10-19: a discarded request had already used 15 vials in calc_state)."""
        src, snap = self.env.ws / CALC_STATE, self.run_dir / CALC_SNAPSHOT
        assert not self._outbox_files(), f"undelivered files {self._outbox_files()} before request #{n}"
        if snap.exists():
            shutil.rmtree(snap)
        snap.mkdir()
        had = src.is_dir()
        if had:
            shutil.copytree(src, snap / CALC_STATE)
        (snap / "request.json").write_text(json.dumps({"request": n, "had_state": had}))

    def rollback_calc_state(self, n_committed: int, fatal: bool):
        """On resume: if the last snapshot belongs to a request that never committed (#n_committed + 1), put
        calc_state back to that snapshot; the state the aborted request left is moved to run_dir for inspection."""
        snap = self.run_dir / CALC_SNAPSHOT
        if not (snap / "request.json").exists():
            print("[lab] no calc_state snapshot (run predates it, or no lab request yet); nothing rolled back",
                  flush=True)
            return False
        meta = json.loads((snap / "request.json").read_text())
        if meta["request"] == n_committed:
            print(f"[lab] calc_state snapshot is of committed request #{n_committed}; nothing rolled back", flush=True)
            if fatal:
                print(f"[lab] WARNING the fatal error came after request #{n_committed} was committed: calc_state and "
                      f"the ledger keep it, but the resume notice says it was not processed", flush=True)
            return False
        assert meta["request"] == n_committed + 1, f"calc_state snapshot of request {meta}, committed {n_committed}"
        self._discard_outbox(meta["request"])
        cur = self.env.ws / CALC_STATE
        if cur.exists():
            # numbered: in-place env retries can discard the same request twice within one second
            k = len(list(self.run_dir.glob(f"{CALC_STATE}.discarded_request{meta['request']}_*")))
            kept = (self.run_dir /
                    f"{CALC_STATE}.discarded_request{meta['request']}_{time.strftime('%Y%m%d-%H%M%S')}_{k}")
            cur.rename(kept)
            print(f"[lab] calc_state left by discarded request #{meta['request']} moved to {kept.name}", flush=True)
        if meta["had_state"]:
            shutil.copytree(snap / CALC_STATE, cur)
        print(f"[lab] calc_state rolled back to before request #{meta['request']} "
              f"({'restored snapshot' if meta['had_state'] else 'no state existed then'})", flush=True)
        return True

    def _outbox_files(self) -> list:
        """Files (not directories) waiting in env_ws/outbox: a calculator may create an empty subdirectory without
        writing anything (luna high 085-02 20261106-101252 #6, outbox/impedance/ empty), which is not a delivery."""
        out = self.env.ws / OUTBOX
        return sorted(p for p in out.rglob("*") if p.is_file()) if out.exists() else []

    def _discard_outbox(self, n: int):
        """Files a calculator wrote for a request that did not happen are kept in run_dir for inspection only."""
        out = self.env.ws / OUTBOX
        if self._outbox_files():
            kept = self.run_dir / f"{OUTBOX}.discarded_request{n}_{time.strftime('%Y%m%d-%H%M%S')}"
            out.rename(kept)
            print(f"[lab] outbox of request #{n} not delivered, moved to {kept.name}", flush=True)

    def _deliver_outbox(self, n: int) -> list:
        """Move env_ws/outbox/* into the actor's workspace; returns [(relative path, bytes)]."""
        out = self.env.ws / OUTBOX
        files = sorted(p for p in out.rglob("*") if p.is_file()) if out.exists() else []
        if not files:
            return []
        ws = self.run_dir / ACTOR_WS
        assert ws.is_dir(), f"no actor workspace {ws} to deliver {len(files)} file(s) to"
        got = []
        for p in files:
            rel = p.relative_to(out)
            assert self.leak_re.search(str(rel)) is None, f"delivered file name leaks an internal identifier: {rel}"
            dst = ws / rel
            assert not dst.exists(), f"request #{n} would overwrite {rel} in the actor workspace"
            dst.parent.mkdir(parents=True, exist_ok=True)
            size = p.stat().st_size
            shutil.move(str(p), str(dst))
            got.append((str(rel), size))
        shutil.rmtree(out)
        print(f"[lab] request #{n}: delivered {got} to the actor workspace", flush=True)
        return got

    def ledger(self) -> dict:
        return {
            "n_requests": self.n_requests,
            "budget_unit": self.unit, "cost_used": self.cost_used, "cost_budget": self.cost_budget,
            "materials_left": self.materials_left(),
            "final_report_submitted": self.report is not None,
            "lab_closed": self.closed_at is not None,
        }

    def materials_left(self) -> dict:
        return {k: self.case.consumables[k] - u for k, u in self.consumed.items()}

    def _log(self, rec: dict):
        rec["t"] = time.time()
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _record(self, n: int, request: str, params: dict, shown: str, grounding: str):
        """External memory for the env agent (survives context compaction): exactly what the actor saw."""
        # line-start CHARGE lines are written as 'recorded CHARGE: ...' so that a `tail session_record.md` in a
        # calculator command is not billed again (047-02 run 20261029-233832 #15)
        entry = (f"\n## Request #{n}\nActor request: {request}\nStructured parameters: {json.dumps(params)}\n"
                 f"Shown to actor:\n{shown}\nInternal grounding: {grounding}\n")
        with open(self.env.ws / "session_record.md", "a", encoding="utf-8") as f:
            f.write(recorded_charge_lines(entry))

    # ---------------- tools ----------------
    def call(self, name: str, args: dict) -> dict:
        with self.lock:
            if name == "lab_status":
                return text_result(json.dumps(self.ledger(), indent=1))
            if name == "lab_catalog":
                return text_result(json.dumps(self.catalog, indent=1))
            if name == "submit_final_report":
                if self.report is not None:
                    return text_result("A final report was already submitted.", is_error=True)
                self.report = args
                self._log({"type": "final_report", "report": args, "ledger": self.ledger()})
                return text_result("Final report received. You are done; stop now.")
            assert name == "lab", name
            if "request" not in args:   # interface error: answer immediately, no env call, no cost
                # (047-04/033-02/042-02 luna runs 2026-11-05 sent final-report fields to `lab` and the run aborted)
                print(f"[lab] call without 'request', keys {sorted(args)}", flush=True)
                return text_result(f"The lab tool needs a 'request' (free text) and optional 'parameters'; got keys "
                                   f"{sorted(args)}. The final report goes to submit_final_report.", is_error=True)
            if self.report is not None:
                return text_result("The final report was already submitted; the lab is closed.", is_error=True)
            if self.cost_budget == 0:
                # zero-experiment baseline: a budget-0 case never reaches the environment; the
                # request is logged and refused (no env call, no cost)
                print(f"[lab] zero-budget session: request refused", flush=True)
                self._log({"type": "lab_zero_budget", "request": args["request"]})
                return text_result(MESSAGES["zero_budget"], is_error=True)
            if self.closed_at is not None:
                self._log({"type": "lab_after_close", "request": args["request"]})
                return text_result(MESSAGES["budget_exhausted"], is_error=True)
            params = args.get("parameters") or {}
            unknown = [k for k in params if k not in self.menu]
            if unknown:   # interface error: answer immediately, no env call, no cost
                return text_result(f"Unknown parameter keys {unknown}. Valid keys: {sorted(self.catalog)} (see lab_catalog).",
                             is_error=True)
            return self._lab(args["request"], params)

    # menu values that describe a range for the env to fill in ("any volume ...", "other current up to 1.8 A, 1 to
    # 600 s": 018-03 Opus DS run 20261106-065634 aborted when the env filled in 0.25 A for 600 s)
    OPEN_MENU_PREFIX = ("any ", "other ")

    def _check_resolution(self, so: dict, params: dict, request: str):
        """The environment must honor 'any of these' selections that contain a recorded value, and may only declare
        lab_limitation for values that are off the menu."""
        def match(used: str, values) -> bool:   # tolerate extra detail appended by the env ("..., 2 min")
            # and whitespace differences ('B06,B04' for the actor's 'B06, B04': 046-01 DS run 20261104-152713 aborted)
            u = "".join(used.lower().split())
            return any(u == w or u.startswith(w) or w in u for w in ("".join(v.lower().split()) for v in values))

        by_param = {}
        for r in so["parameter_resolution"]:
            by_param.setdefault(r["parameter"], []).append(r)
            if r["mode"] == "lab_limitation" and r["parameter"] in self.menu:
                m = self.menu[r["parameter"]]
                if r["used"] in r["requested"] and r["used"] in m["values"]:
                    # a menu value used as asked but labelled lab_limitation: the label is the harness's to know
                    # (004-02 Sonnet-actor DS run 20261105-085137 aborted on it, the "other duration" menu entry)
                    new = "recorded" if r["used"] in m["recorded"] else "extrapolated"
                    print(f"[lab] RELABEL {r['parameter']}={r['used']} ({r['sample']}): lab_limitation -> {new}",
                          flush=True)
                    r["mode"] = new
                    continue
                assert not set(r["requested"]) & m["values"], \
                    f"env declared lab_limitation for a menu value: {r}"
        for k, req in params.items():
            if k not in by_param and so["deferred_steps"] and k.split(".")[0] not in so["action_types"]:
                continue   # the step this parameter belongs to was deferred at a checkpoint (resolved when resubmitted)
            if k not in by_param and so["deferred_steps"]:
                # the env stopped at a checkpoint and put the part this parameter governs into deferred_steps, while
                # the action key itself was charged for the earlier part (007-02 rtf Opus DS run 20261106-065634:
                # load_plate.well_content belongs to adding oysters, deferred after plate_loaded). Logged, not fatal.
                print(f"[lab] structured parameter {k}={req} unresolved, request stopped at checkpoint with deferred "
                      f"steps {so['deferred_steps']}", flush=True)
                continue
            assert k in by_param, f"env did not resolve structured parameter {k}"
            rs, m = by_param[k], self.menu[k]
            # the actor's structured choice must be applied to at least one sample (other samples may carry values
            # the actor assigned in free text, e.g. an added extraction blank)
            off_menu = not any(match(v, m["values"]) for v in req)
            # a menu value that describes a range ("any volume with liquid height >= 1.05 mm, ...") is filled in by
            # the env with a concrete value (007-02 kimi run 20261105-103633 aborted on 400/700 uL)
            open_choice = any(v.lower().startswith(self.OPEN_MENU_PREFIX) for v in req)
            if off_menu or open_choice:
                # the actor phrased a value outside the catalog (e.g. a hidden parameter, or compound ids as
                # days.trial: 047-04 luna run 20261105-115105 aborted); the env mapped it to what it judged
                # equivalent. Allowed, logged for the judge's audit (env lean lenient, user 11-04).
                print(f"[lab] off-menu {k}={req} mapped to {[r['used'] for r in rs]}", flush=True)
            else:
                # either direction: the env may add detail to the actor's value, or the actor's phrasing may add
                # detail to the menu value the env applied ("fracture surface" -> menu "fracture": 032-03 mimo
                # rerun 20261106-041036 aborted)
                # (prefix only and not for numbers, so that "16" is never taken as "1")
                def extends(v: str, used: str) -> bool:
                    a, b = "".join(v.lower().split()), "".join(used.lower().split())
                    return a.startswith(b) and not b.replace(".", "").isdigit()
                applied = any(match(r["used"], req) or any(extends(v, r["used"]) for v in req)
                              or r["mode"] == "lab_limitation" for r in rs)
                if not applied:
                    # the actor wrote a condition in the request text that the structured value cannot carry
                    # ("11 cycles, 10 if the cDNA yield is above 5 ng", listed only "11"; the env used 10 on the
                    # printed yields: 001-03 mimo runs 20261106-102441/102505 aborted). Allowed when the value used
                    # appears as a token in the request text; logged for the judge's audit.
                    in_text = [r["used"] for r in rs if re.search(
                        r"(?<![\w.])" + re.escape(str(r["used"]).strip()) + r"(?![\w.])", request, re.I)]
                    assert in_text, f"env applied none of the actor's values {k}={req}: {rs}"
                    print(f"[lab] {k}={req}: env applied {in_text}, a value named in the request text, not the "
                          f"structured one", flush=True)
            if set(req) & m["recorded"] and not open_choice:
                for r in rs:   # a recorded value used as asked but labelled extrapolated: the label is the harness's
                    # to know (032-03 DS run 20261104-075742 #6 aborted on it, used == requested == recorded)
                    if r["mode"] == "extrapolated" and r["used"] in m["recorded"] and r["used"] in req:
                        print(f"[lab] RELABEL {k}={r['used']} ({r['sample']}): extrapolated -> recorded", flush=True)
                        r["mode"] = "recorded"
                if not any(r["mode"] == "recorded" and match(r["used"], set(req) & m["recorded"]) for r in rs):
                    # the actor listed a recorded and an off-record value and the env applied the off-record one the
                    # actor also asked for (004-02 mimo run 20261105-112012 aborted): allowed, logged for the judge
                    assert any(r["used"] in req for r in rs), f"env ignored a recorded value in {k}={req}: {rs}"
                    print(f"[lab] {k}={req}: env applied {[r['used'] for r in rs]}, not the recorded value",
                          flush=True)

    def _bill_from_charge_lines(self, n: int, so: dict):
        """Bill request #n from the CHARGE lines its calculator commands printed (the environment's own shell output
        in its session transcript), not from the environment's action_types: the Sonnet env listed a per-item action
        once for a run of many items (045-03 20261029-214420: challenge x9 listed once, 3 runs; 003-03
        20261029-204001 #4, #11). Free actions and continued_step stay as the env listed them. Returns the record of
        the replacement, or None when the CHARGE lines agree or the calculator printed no billable CHARGE line."""
        turns = load_turns(self.env.transcript())
        assert n in turns, f"request #{n}: no turn in the environment transcript"
        outs = [b["output"] for b in turns[n]["bash"] if "calculators.py" in b["command"]]
        charges = charges_from_outputs(outs, self.per_item, f"lab #{n}")
        listed = collections.Counter(a for a in so["action_types"] if a not in FREE_ACTIONS)
        cmds = [b["command"] for b in turns[n]["bash"]]
        if self.env.extra_env:   # guarded: a /dev/null stdout either billed nothing or was refused (calc_guard.py)
            cmds = [strip_stdout_devnull(c) for c in cmds]
        filtered = filtered_calculator_commands(cmds)
        # the env piped/redirected calculator output (002-02 20261029: a grep dropped CHARGE lines; uniq -c merged
        # them): the CHARGE lines it saw may be incomplete, so bill per key the larger of its listing and those lines
        billed = listed | charges if filtered else charges
        if filtered:
            print(f"[lab] WARNING request #{n}: {len(filtered)} calculator command(s) piped or redirected "
                  f"{[c[:160] for c in filtered]}; CHARGE lines {dict(charges)}, env listed {dict(listed)} -> "
                  f"billed max {dict(billed)}", flush=True)
            self.pending_notes.append(f"Request #{n} ran calculator commands through pipes or filters; run them "
                                      f"without grep/sed/head/tail/uniq or redirection, the harness bills from "
                                      f"their CHARGE lines.")
        if not billed or (listed == billed and not filtered):
            return None
        fix = {"env_listed": list(so["action_types"]), "charge_lines": dict(charges)}
        if filtered:
            fix["filtered_commands"] = [c[:600] for c in filtered]
            fix["billed_max_of_listing_and_charge_lines"] = dict(billed)
        if so["verdict"] not in ("performed", "partially_performed"):
            # the calculator charged but the env reports nothing done: not billed; check_env reports it
            print(f"[lab] WARNING request #{n}: verdict {so['verdict']} but CHARGE lines {dict(charges)}; "
                  f"not billed", flush=True)
            return None
        if listed == billed:   # filtered, but the listing already covers the CHARGE lines: recorded, unchanged
            return fix
        so["action_types"] = [a for a in so["action_types"] if a in FREE_ACTIONS] + list(billed.elements())
        print(f"[lab] BILLING request #{n}: env listed {fix['env_listed']}, CHARGE lines {dict(charges)} -> "
              f"action_types {so['action_types']}", flush=True)
        self.pending_notes.append(f"Request #{n} was billed from the calculator's CHARGE lines {dict(billed)}, not "
                                  f"from your action_types {fix['env_listed']}: a per-item action is listed once per "
                                  f"item (CHARGE: key xN = the key N times), a per-request action once (rule 10).")
        return fix

    def check_case_files(self):
        """The case files must be the ones the run started with (env_ws links to cases/; the env system prompt and
        calculator_usage.txt were frozen at the start). A changed case stops the run: it is invalid and rerun, never
        resumed on mixed versions."""
        if self.case_sha is None:
            return
        now = self.case.files_sha256()
        assert now == self.case_sha, (f"case files changed since this run started ({self.case_sha[:16]} -> "
                                      f"{now[:16]}): the run is invalid; rerun it, do not resume")

    def _env_step_with_retry(self, request: str, params: dict, before: dict, noise: list) -> dict:
        """env.step, retried after a transient failure (network, gateway 5xx, call timeout) until LAB_REQUEST_TOTAL_S
        has passed since the request started (11-07 batches stopped on env calls that died of gateway connection errors). Each retry undoes what the failed
        attempt did, exactly as a resume does (calc_state back to the snapshot, undelivered outbox moved aside, a
        DISCARDED note to the env), then resubmits the same request in the same env session. Any other error, or a
        transient one after the deadline, propagates (the run stops and can be resumed)."""
        n = self.n_requests + 1
        deadline = time.time() + LAB_REQUEST_TOTAL_S
        attempt = 0
        while True:
            attempt += 1
            self.env.call_timeout_s = max(1, int(deadline - time.time()))
            try:
                return self.env.step(request, params, before, self.pending_notes, noise)
            except (AssertionError, subprocess.TimeoutExpired) as e:
                text = str(e)
                transient = isinstance(e, subprocess.TimeoutExpired) or any(m in text for m in TRANSIENT_MARKS)
                left = deadline - time.time()
                print(f"[lab] request #{n} env attempt {attempt} failed (transient={transient}, {left:.0f} s left of "
                      f"{LAB_REQUEST_TOTAL_S}): {text[:300]!r}", flush=True)
                if not transient or left < LAB_RETRY_MIN_LEFT_S:
                    raise
                time.sleep(LAB_RETRY_PAUSE_S)
                assert self.rollback_calc_state(self.n_requests, False), f"no calc_state snapshot of request #{n}"
                self.pending_notes.append(
                    f"Your answer to actor request #{n} was DISCARDED by the harness because the call failed on a "
                    f"network/gateway error before it was recorded; nothing of it happened and nothing was consumed "
                    f"(the calculator state is back to before that request). The same request follows again.")
                self.snapshot_calc_state(n)

    def _lab(self, request: str, params: dict) -> dict:
        self.check_case_files()
        before = self.ledger()
        noise = [round(self.rng.lognormvariate(-self.noise_sigma ** 2 / 2, self.noise_sigma), 3)
                 for _ in range(self.noise_n)]
        self.snapshot_calc_state(self.n_requests + 1)
        glog = self.env.guard_log
        g0 = len(glog.read_text().splitlines()) if glog.exists() else 0
        so = self._env_step_with_retry(request, params, before, noise)
        # calculator runs with discarded stdout in this request (twinlab/calc_guard.py): billed ones were refused
        # before they counted (state restored, CHARGE lines never shown); kept in the ledger record
        guard = [json.loads(x) for x in glog.read_text().splitlines()[g0:]] if glog.exists() else []
        if guard:
            so["guard_discarded_output_runs"] = guard
            for g in guard:
                print(f"[lab] GUARD request #{self.n_requests + 1}: {g['decision']}: {g['command'][:200]} "
                      f"billed={g['billed']}", flush=True)
        self._check_resolution(so, params, request)
        self.pending_notes = []
        n = self.n_requests + 1
        if so["verdict"] == "answered_question" and so["not_performed_items"]:
            # rule 6: a public-knowledge answer next to declined items is partially_performed (029 #4, deepseek-flash
            # env audit 2026-11-04); same handling as the relabel below
            print(f"[lab] RELABEL request #{self.n_requests + 1}: answered_question with not_performed_items "
                  f"{so['not_performed_items']} -> partially_performed", flush=True)
            so["verdict"] = "partially_performed"
            so["verdict_relabelled"] = "answered_question -> partially_performed (not_performed_items)"
        if so["verdict"] == "performed" and so["not_performed_items"]:
            # same meaning as partially_performed (rule 7); the env picked the wrong label
            print(f"[lab] request #{self.n_requests + 1}: verdict performed with not_performed_items "
                  f"{so['not_performed_items']} -> relabelled partially_performed", flush=True)
            so["verdict"] = "partially_performed"
        if so["verdict"] == "partially_performed" and not so["not_performed_items"]:
            # a checkpoint stop keeps verdict performed (rule 11); Sonnet env mislabelled it (006 A/B, 2026-10-11).
            # Without deferred steps: every item was done but labelled partial (031-02 124236 #19, which aborted the
            # run before 2026-11-04); relabel, but only when something was actually done or reported
            assert so["action_types"] or so["observation"].strip(), \
                f"partially_performed with no not_performed_items, no action_types and no observation: {so}"
            reason = "only deferred steps" if so["deferred_steps"] else "no not_performed_items"
            print(f"[lab] RELABEL request #{n}: partially_performed with {reason} -> performed", flush=True)
            so["verdict"] = "performed"
            so["verdict_relabelled"] = f"partially_performed -> performed ({reason})"   # kept in the ledger record
        v = so["verdict"]
        billing_fix = self._bill_from_charge_lines(n, so)
        provenance = [m.group(0).rstrip("\n") for m in PROVENANCE_LINE_RE.finditer(so["observation"])]
        if provenance:
            so["observation"] = PROVENANCE_LINE_RE.sub("", so["observation"])
            so["stripped_provenance_lines"] = provenance   # kept in the ledger record for check_env and the judge
            print(f"[lab] request #{n}: cut {len(provenance)} calculator provenance line(s) from the observation: "
                  f"{[p[:160] for p in provenance]}", flush=True)
            self.pending_notes.append(f"Request #{n}: your observation contained {len(provenance)} calculator "
                                      f"provenance line(s) (the line starting with the provenance marker); the "
                                      f"harness cut them before the actor saw it. Never copy those lines.")
        # calculator billing lines are the harness's (the ledger header shows the charge): deepseek-flash envs copied
        # them into every observation of 040-03 and 046-01 (env audits c/d, 2026-11-04); cut and told, like provenance
        billing_lines = [m.group(0).rstrip("\n") for m in BILLING_LINE_RE.finditer(so["observation"])]
        if billing_lines:
            so["observation"] = BILLING_LINE_RE.sub("", so["observation"])
            so["stripped_billing_lines"] = billing_lines
            print(f"[lab] request #{n}: cut {len(billing_lines)} billing line(s) from the observation", flush=True)
            self.pending_notes.append(f"Request #{n}: your observation contained {len(billing_lines)} calculator "
                                      f"billing line(s) (CHARGE:/BILLING:); the harness cut them before the actor "
                                      f"saw it. The harness bills; never copy those lines.")
        # lost output (087-02 replay 20261103-141338 #1, observation cut at 'C17-C19 ...', billed NMR never shown):
        # flagged in the log and the ledger record for check_env and the reader, no retry (2026-11-04)
        turn = load_turns(self.env.transcript())[n]
        signs = output_problems([b["output"] for b in turn["bash"]
                                 if CALC_INVOKE_RE.search(b["command"])], so["observation"], request)
        outside = outside_file_access(turn, self.env.ws)
        if outside:   # rule 14: kept in the ledger record, and the env is told before its next turn (no block)
            so["outside_file_access"] = outside
            print(f"[lab] WARNING request #{n}: env read outside its workspace or the calculator source: {outside}",
                  flush=True)
            self.pending_notes.append(f"Request #{n}: you opened files outside your working directory or read the "
                                      f"calculator source ({len(outside)} tool calls, e.g. {outside[0][:120]}). "
                                      f"Rule 14: work only with the files in your working directory and only run "
                                      f"the calculator.")
        if signs:
            so["output_flags"] = signs
            print(f"[lab] WARNING request #{n}: env observation may have lost calculator output: {signs}", flush=True)
        # the bare mechanism word "calculator" stopped two DS runs (058-02 20261104-152714, 008-02 20261105-004156,
        # e.g. "no calculator command covers ..."): the sentences that hold it are cut and the env is told, instead of
        # a fatal stop; every other internal identifier still stops the run below
        mech = [s for s in CALC_WORD_SENTENCE_RE.findall(so["observation"])]
        items_mech = [x for x in so["not_performed_items"] if CALC_WORD_RE.search(x)]
        if mech or items_mech:
            so["observation"] = CALC_WORD_SENTENCE_RE.sub("", so["observation"])
            so["not_performed_items"] = [CALC_WORD_SENTENCE_RE.sub("", x) for x in so["not_performed_items"]]
            so["stripped_calculator_sentences"] = mech + items_mech
            print(f"[lab] request #{n}: cut {len(mech) + len(items_mech)} sentence(s) naming the calculator", flush=True)
            self.pending_notes.append(f"Request #{n}: your answer named the calculator; the harness cut those "
                                      f"sentences before the actor saw them. Describe results and limits in lab "
                                      f"terms only (rule 1).")
        # the actor's copies of the case files carry the case's search-term replacements, the environment's copies do
        # not: an env that quotes a record verbatim would hand the actor the original term (034-03 luna run
        # 20261105-111507 #1 quoted the issue-thread title, audit hit). Same replacements on what the actor sees.
        for term, repl in self.case.text_replacements.items():
            if term in so["observation"] or any(term in x for x in so["not_performed_items"]):
                so["observation"] = so["observation"].replace(term, repl)
                so["not_performed_items"] = [x.replace(term, repl) for x in so["not_performed_items"]]
                print(f"[lab] request #{n}: replaced {term!r} -> {repl!r} (actor_text_replacements)", flush=True)
        leak = self.leak_re.search(so["observation"] + " ".join(so["not_performed_items"]))
        assert leak is None, f"env observation contains internal identifier {leak.group(0)!r}: {so['observation']}"
        if v in ("performed", "partially_performed"):
            if not so["action_types"]:
                # e.g. re-reporting results the actor already paid for (a response lost in transit): zero cost
                assert not so["consumed"], f"env consumed material in request #{n} but listed no action_types"
                print(f"[lab] request #{n}: performed with no action_types (zero-cost re-report)", flush=True)
        else:
            assert not so["consumed"], so
            if so["action_types"] and v != "answered_question" and all(a in FREE_ACTIONS for a in so["action_types"]):
                # a free status look listed with a refusal costs nothing (058-02 DS run 20261104-152713 #1 aborted on
                # not_simulatable + ['inspect']); dropped and kept visible in the ledger record
                print(f"[lab] RELABEL request #{n}: free action(s) {so['action_types']} dropped from verdict {v}",
                      flush=True)
                so["action_types_dropped"] = so["action_types"]
                so["action_types"] = []
            assert not so["action_types"] or v == "answered_question", so
        if so["deferred_steps"] and not so["stopped_at_checkpoint"]:
            # steps held back for a reason other than a checkpoint (a material not yet delivered) are steps not done:
            # mimo-pro 001-03 20261106-102441 stopped on verdict performed + deferred_steps without a checkpoint
            print(f"[lab] RELABEL request #{n}: deferred_steps without a checkpoint -> not_performed_items "
                  f"(verdict {v})", flush=True)
            so["deferred_steps_relabelled"] = so["deferred_steps"]
            so["not_performed_items"] = so["not_performed_items"] + so["deferred_steps"]
            so["deferred_steps"] = []
            if v == "performed":
                v = so["verdict"] = "partially_performed"
        if v == "performed":
            assert not so["not_performed_items"], f"verdict performed but items not performed: {so}"
        if v == "partially_performed":
            assert so["not_performed_items"], f"partially_performed without not_performed_items: {so}"
        cp_on = self.case.checkpoints["enforce"]
        assert cp_on or (not so["stopped_at_checkpoint"] and not so["deferred_steps"]), \
            f"checkpoints disabled but env stopped at a checkpoint: {so}"
        assert bool(so["stopped_at_checkpoint"]) or not so["deferred_steps"], \
            f"deferred_steps without stopped_at_checkpoint: {so}"
        assert cp_on or CONTINUED_STEP not in so["action_types"], f"continued_step without checkpoints: {so}"
        real = [a for a in so["action_types"] if a != CONTINUED_STEP]   # continued_step: already charged, cost 0
        dup = {a for a in real if real.count(a) > 1 and not self.case.actions[a].get("per_item")}
        if dup:   # rule 10 bills a per-request action once: collapse instead of aborting the run (038-02 DS run
            # 20261104-075742 stopped on ['inspect', 'inspect', 'ams_date']); kept visible in the ledger record
            kept = []
            for a in so["action_types"]:
                if a not in dup or a not in kept:
                    kept.append(a)
            print(f"[lab] RELABEL request #{n}: per-request action(s) {sorted(dup)} listed more than once -> once: "
                  f"{so['action_types']} -> {kept}", flush=True)
            so["action_types_collapsed"] = {"from": so["action_types"], "to": kept}
            so["action_types"] = kept
            real = [a for a in kept if a != CONTINUED_STEP]
        cost = float(sum(self.case.actions[a]["cost"] for a in real))
        # before n_requests/cost change: when this fired after them (luna high 085-02 20261106-101252 #6), the saved
        # state counted the failed request as committed and the resume found its outbox undelivered
        assert v in ("performed", "partially_performed") or not self._outbox_files(), \
            f"request #{n}: the calculator wrote files {self._outbox_files()} for the actor but the verdict is {v}"
        # budget enforcement: the budget may overshoot on the last request, then the lab closes (see below)
        problems = []
        if self.case.free_requests_after_budget and cost > 0 and self.cost_used >= self.cost_budget:
            problems.append(f"the {self.unit} budget is used up; only requests that cost nothing are accepted")
        new_consumed = dict(self.consumed)
        for c in so["consumed"]:
            new_consumed[c["item"]] += c["n"]
            if new_consumed[c["item"]] > self.case.consumables[c["item"]]:
                problems.append(f"not enough {c['item']} left for this request")
        self.n_requests = n
        self.last_deferred_steps = []
        if problems:
            # undo what the env's calculator calls did to calc_state for this request (snapshot taken before it ran):
            # a rejected request must leave no trace (048-03 kept the powder it would have used; 004-02 sol #11 left
            # its take/sink in calc_state; found 2026-11-06). Also moves an undelivered outbox aside.
            assert self.rollback_calc_state(n - 1, False), f"no calc_state snapshot of rejected request #{n}"
            msg = "Request rejected by the lab manager: " + "; ".join(problems) + ". Nothing was done."
            self.pending_notes.append(f"Request #{n} was REJECTED for budget reasons ({'; '.join(problems)}); "
                                      f"none of it happened. Forget the observation you produced for it.")
            self._log({"type": "lab", "n": n, "request": request, "env": so, "rejected": problems, **({"billing_from_charge_lines": billing_fix} if billing_fix else {}),
                       "ledger": self.ledger()})
            shown = f"[Lab request #{n}] {msg}\nLedger: {json.dumps(self.ledger())}"
            self._record(n, request, params, shown, "REJECTED; nothing happened")
            return text_result(shown)
        self.cost_used += cost
        self.last_deferred_steps = list(so["deferred_steps"])
        self.consumed = new_consumed
        images = [self._image_block(self.env.resolve_image(p)) for p in so["images"]]
        left = self.materials_left()
        header = (f"[Lab request #{n}] charged {cost:g} {self.unit} | remaining: "
                  f"{self.cost_budget - self.cost_used:g} {self.unit}"
                  + (f", materials {json.dumps(left)}" if left else ""))
        limited = any(r["mode"] == "lab_limitation" for r in so["parameter_resolution"])
        # whole-request refusal: the env often writes the decline reasons (incl. the calculator's REFUSED/UNAVAILABLE
        # reasons) into observation, not into the items (062-02, 069-02), so show it before the notice (2026-10-29)
        refusal_obs = so["observation"].strip() + "\n\n" if so["observation"].strip() else ""
        if v == "not_simulatable" and limited:
            body = ""          # the LAB LIMITATION notice below says everything
        elif v == "not_simulatable" and so["not_performed_items"]:
            # whole request refused: show the per-item reasons exactly as for partial execution (070-02, 2026-10-29)
            body = refusal_obs + MESSAGES["not_performed"] + "".join(f"\n- {x}" for x in so["not_performed_items"])
        elif v == "not_simulatable":
            body = refusal_obs + MESSAGES["not_simulatable"]
        else:
            used = [r for r in so["parameter_resolution"]
                    if r["mode"] != "lab_limitation" and r["parameter"] not in self.hidden_params]
            body = ""
            if used:
                body = "Parameters used: " + "; ".join(
                    f"{r['parameter']} = {r['used']}" + ("" if r["sample"] == "all" else f" ({r['sample']})")
                    for r in used) + "\n\n"
            body += so["observation"]
            if so["not_performed_items"]:
                body += "\n\n" + MESSAGES["not_performed"] + "".join(f"\n- {x}" for x in so["not_performed_items"])
            if so["stopped_at_checkpoint"] and so["deferred_steps"]:   # nothing to say when nothing was deferred
                body += ("\n\n" + MESSAGES["checkpoint_stop"].format(cp=so["stopped_at_checkpoint"])
                         + "".join(f"\n- {x}" for x in so["deferred_steps"]))
            if so["stopped_at_checkpoint"] and self.case.checkpoint_mode == "memo":
                body += "\n\n" + MESSAGES["checkpoint_memo"]
        for r in so["parameter_resolution"]:
            if r["mode"] == "lab_limitation":
                # both branches give the full notice (before 2026-11-03 a non-empty body cut it after the value).
                # Options only for catalog parameters: a hidden parameter (or a name the env made up) has no catalog
                # entry, and "Available options: []" read as "nothing is possible"; hidden names and values are
                # never shown (033-02 leak), the actor's own requested value is.
                name = "this setting" if r["parameter"] in self.hidden_params else r["parameter"]
                opts = (f" Available options: {self.catalog[r['parameter']]}" if r["parameter"] in self.catalog
                        else " This setting has no option list in lab_catalog.")
                body += (("\n\n" if body else "") +
                         f"LAB LIMITATION (not an experimental result): {name} = {r['requested']} "
                         f"is not available in this lab; nothing was done or consumed for that step.{opts}")
        delivered = self._deliver_outbox(n)
        if delivered:
            body += "\n\n" + MESSAGES["files_delivered"] + "".join(f"\n- {r} ({s} bytes)" for r, s in delivered)
        if self.cost_used >= self.cost_budget and self.case.free_requests_after_budget:
            if cost > 0:   # this request used the budget up; later paid requests are rejected above, free ones run
                body += "\n\n" + MESSAGES["budget_used_up_free_only"].format(unit=self.unit)
                print(f"[lab] budget used up at request #{n} ({self.cost_used} {self.unit}); "
                      f"only cost-0 requests from now on", flush=True)
        elif self.cost_used >= self.cost_budget:
            self.closed_at = time.time()
            body += "\n\n" + MESSAGES["budget_exhausted"]
            print(f"[lab] budget exhausted at request #{n} ({self.cost_used} {self.unit}); lab closed", flush=True)
        self._log({"type": "lab", "n": n, "request": request, "env": so, "noise": noise, "cost": cost, **({"billing_from_charge_lines": billing_fix} if billing_fix else {}),
                   "shown": header + "\n\n" + body, "ledger": self.ledger(), "closed": self.closed_at is not None})
        self._record(n, request, params, header + "\n\n" + body, so["grounding"])
        print(f"[lab #{n}] {v} {so['action_types']} cost={cost} used={self.cost_used} imgs={len(images)} "
              f"not_performed={len(so['not_performed_items'])} checkpoint={so['stopped_at_checkpoint'] or '-'}",
              flush=True)
        return {"content": [{"type": "text", "text": header + "\n\n" + body}, *images], "isError": False}

    def _image_block(self, path: Path) -> dict:
        cached = self.img_cache / (path.stem + ".jpg")
        if not cached.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=IMAGE_JPEG_QUALITY)
            cached.write_bytes(buf.getvalue())
        return {"type": "image", "data": base64.b64encode(cached.read_bytes()).decode(), "mimeType": "image/jpeg"}


def text_result(s: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": s}], "isError": is_error}

