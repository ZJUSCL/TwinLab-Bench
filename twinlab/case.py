"""Case loading and the different views of a case (actor / environment / judge).

Everything case-specific lives in the case YAML; this module only reads it. Required generic fields:
  resources.budget: {unit, total, expert_reference, multiplier}   (total == expert_reference * multiplier)
  resources.consumables: {item: available_count}                   (may be empty)
  resources.initial_state: {consumed: {item: n}, existing_material: str}
  actions: {key: {cost: float, note/params/...}}
  final_report: {instructions: str, fields: {name: description}, required: [names]}
  environment_instructions: str                                     (case-specific rules for the env agent)
  checkpoints: {enforce: bool, list: [{id, after}]}
  harness: assets_dir, figures_dir, env_context_files, env_knowledge_glob, env_data_dirs, env_calculator,
           actor_files, workspace_note, replicate_cv, noise_multipliers_per_request, env_hidden_sections,
           env_hidden_meta_keys, actor_hidden_params, leak_patterns
"""
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml

from . import prompts
from .prompts import MESSAGES

TEXT_SUFFIXES = {".md", ".txt", ".csv", ".tsv", ".rmd", ".qmd", ".py", ".r", ".json", ".yaml", ".yml"}


# case = use checkpoints.enforce from the YAML; memo = on, plus a neutral write-up request at every stop (pilot 2026-11-01)
CHECKPOINT_MODES = ("case", "on", "off", "memo")
# review cases score conclusion/verification; campaign cases score outcome/process/knowledge
RUBRIC_COMPONENTS = ("conclusion", "verification", "other", "outcome", "process", "knowledge")
AFTER_EXHAUSTED_FREE_ONLY = "free_requests_only"   # resources.budget.after_exhausted (optional)
# {case dir: {old rubric_sha256: new rubric_sha256}} for scoring edits that cannot change any judgment (written by the
# edit script); a run judged on the old hash is treated as judged on the new one
RUBRIC_SHA_EQUIV = Path(__file__).resolve().parents[1] / "cases/rubric_sha_equiv.json"
# The benchmark's own name in any spelling: its public repository holds every hidden answer. Checked
# by the actor guard, the Codex web-search relay and the post-run audit, next to each case's own blocklist.
_NAME_FORMS = [f"{t}{sep}{b}" for t in ("TwinLab", "Twinlab", "twinlab") for sep in ("-", "_", " ", "") for b in ("Bench", "bench")]
BENCHMARK_NAME_NEEDLES = _NAME_FORMS + sorted({n.upper() for n in _NAME_FORMS})


def case_name(case_id: str) -> str:
    """The case id without number and version: 002-resazurin-clam-02 -> resazurin-clam."""
    m = re.fullmatch(r"\d{3}-(.+?)(?:-\d{2})?(?:-(?:s2|xp)-\d{2})?", case_id)
    assert m, f"unexpected case id {case_id!r}"
    return m.group(1)


def accepted_rubric_shas(case_dir: str, cur_sha: str) -> set:
    equiv = json.loads(RUBRIC_SHA_EQUIV.read_text())["cases"].get(case_dir, {})
    return {cur_sha} | {old for old, new in equiv.items() if new == cur_sha}


class Case:
    def __init__(self, path: Path, checkpoints: str):
        assert checkpoints in CHECKPOINT_MODES, checkpoints
        self.path = path.resolve()
        self.raw_text = self.path.read_text(encoding="utf-8")
        self.d = yaml.safe_load(self.raw_text)
        self.id = self.d["meta"]["case_id"]
        h = self.d["harness"]
        self.assets_dir = (self.path.parent / h["assets_dir"]).resolve()
        assert self.assets_dir.is_dir(), self.assets_dir
        self.figures_dir = self.assets_dir / h["figures_dir"]
        assert self.figures_dir.is_dir(), self.figures_dir
        cc = self.d["meta"]["contamination_controls"]
        self.canary = cc["canary"]
        self.blocklist = cc["web_blocklist"]
        assert all(isinstance(b, str) for b in self.blocklist), f"web_blocklist entries must be strings: {self.blocklist}"
        # actor guard, relay, audit; the case's name matches with or without its number and version (substring)
        self.guard_needles = list(self.blocklist) + [self.canary] + BENCHMARK_NAME_NEEDLES + [case_name(self.id)]
        # identifying term in actor-visible text -> natural replacement (no visible redaction marker)
        self.text_replacements = cc["actor_text_replacements"]
        assert isinstance(self.text_replacements, dict), self.text_replacements
        self.actions = self.d["actions"]
        for k, a in self.actions.items():
            assert "cost" in a, f"action {k} has no cost"
        self.resources = self.d["resources"]
        self.scoring = self.d["scoring"]
        for k, item in self.scoring["rubric"].items():
            assert item.get("component") in RUBRIC_COMPONENTS, \
                f"rubric item {k} needs component: one of {RUBRIC_COMPONENTS}"
        # frozen-rubric check: every run records the hash of the scoring section it was judged with
        self.rubric_sha256 = hashlib.sha256(json.dumps(self.scoring, sort_keys=True).encode()).hexdigest()[:16]
        # hashes recorded before a judging-neutral scoring edit (2026-11-04: pass_rule.min_total removed) count as current
        self.rubric_sha_accepted = accepted_rubric_shas(self.path.parent.name, self.rubric_sha256)
        b = self.resources["budget"]
        assert abs(b["total"] - b["expert_reference"] * b["multiplier"]) < 1e-9, b
        self.budget_unit = b["unit"]
        self.budget_total = b["total"]
        # opt-in per case (047-04: a separate allocation of cost-0 ALM runs must stay usable after the paid trials are
        # spent; 2026-11-05 qwen run 202153 was closed out of it): when the budget is used up the lab keeps accepting
        # requests that cost nothing instead of closing. Cases without the key close as before.
        self.free_requests_after_budget = b.get("after_exhausted") == AFTER_EXHAUSTED_FREE_ONLY
        assert b.get("after_exhausted") in (None, AFTER_EXHAUSTED_FREE_ONLY), b
        self.consumables = self.resources["consumables"]
        init = self.resources["initial_state"]["consumed"]
        assert set(init) <= set(self.consumables), f"initial_state.consumed has unknown items {set(init) - set(self.consumables)}"
        self.initial_consumed = {k: init.get(k, 0) for k in self.consumables}
        self.final_report = self.d["final_report"]
        assert set(self.final_report["required"]) <= set(self.final_report["fields"]), self.final_report
        self.checkpoints = self.d["checkpoints"]
        if checkpoints != "case":           # protocol override (2x2 design: case type x protocol)
            self.d["checkpoints"]["enforce"] = checkpoints in ("on", "memo")
        self.checkpoint_mode = checkpoints
        # the judge reads the YAML text; it must show the enforce value this run used, not the file's default
        # (007-02 run 20261101-071116 judged env_inconsistency under --checkpoints off, 2026-11-03)
        self.judge_text = self.raw_text
        if checkpoints != "case":
            self.judge_text, n = re.subn(
                r"^  enforce: (true|false)\b.*$",
                f"  enforce: {'true' if self.checkpoints['enforce'] else 'false'}"
                f"   # set for this run by --checkpoints {checkpoints} (the case file's default is overridden)",
                self.raw_text, flags=re.M)
            assert n == 1, f"expected exactly one 'checkpoints.enforce' line in {self.path}, found {n}"
            assert yaml.safe_load(self.judge_text)["checkpoints"]["enforce"] == self.checkpoints["enforce"]
        print(f"[case] {self.id}: checkpoints {'ENFORCED' if self.checkpoints['enforce'] else 'off'} "
              f"(mode={checkpoints})", flush=True)
        if self.checkpoints["enforce"]:
            assert self.checkpoints["list"], "checkpoints.enforce is true but no checkpoints are listed"
        self.leak_patterns = h["leak_patterns"]

    # ---------------- environment view ----------------
    def env_view_yaml(self) -> str:
        """Case without scoring/predictions (and without contamination meta). Hidden truth stays: the
        environment agent is omniscient by design, it only must not reveal it."""
        d = copy.deepcopy(self.d)
        h = d["harness"]
        for k in h["env_hidden_sections"]:
            d.pop(k)
        for k in h["env_hidden_meta_keys"]:
            d["meta"].pop(k)
        d.pop("harness")
        # already in the env system prompt as CASE-SPECIFIC INSTRUCTIONS (env_agent.py); it was there twice,
        # median 2.6k chars per case (redundancy audit 2026-10-26)
        d.pop("environment_instructions")
        return yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=120)

    def env_context_files(self) -> list:
        return [self.assets_dir / f for f in self.d["harness"]["env_context_files"]]

    def env_knowledge_files(self) -> list:
        return sorted(self.assets_dir.glob(self.d["harness"]["env_knowledge_glob"]))

    def files_sha256(self) -> str:
        """One sha256 over every case file a run reads after it starts: case.yaml, the calculator, the environment's
        record, knowledge, data and figure files (env_ws links to them). Recorded in run_config.json and checked
        before every lab request and on resume, so a case edited mid-run stops the run instead of mixing versions
       ."""
        files = {"case.yaml": self.path}
        calc = self.d["harness"]["env_calculator"]
        if calc:
            files[f"calculator/{calc}"] = self.assets_dir / calc
        for f in self.env_context_files():
            files[f"context/{f.name}"] = f
        for f in self.env_knowledge_files():
            files[f"env_knowledge/{f.name}"] = f
        dirs = {"figures": self.figures_dir, **{f"data/{dst}": self.assets_dir / src
                                                for src, dst in self.d["harness"]["env_data_dirs"].items()}}
        for prefix, d in dirs.items():
            for f in sorted(d.rglob("*")):
                if not f.is_dir():
                    files[f"{prefix}/{f.relative_to(d).as_posix()}"] = f
        h = hashlib.sha256()
        for name in sorted(files):
            h.update(f"{name}\0{hashlib.sha256(files[name].read_bytes()).hexdigest()}\n".encode())
        return h.hexdigest()

    # ---------------- parameter menus ----------------
    def catalog_for_actor(self) -> dict:
        """{'procedure.param': [values]} without recorded flags or extrapolation bases. Values are sorted neutrally
        (numeric ascending, else alphabetical) because the case lists recorded candidates first."""
        def order(v):
            return (0, float(v), "") if v.replace(".", "", 1).isdigit() else (1, 0.0, v.lower())
        hidden = set(self.d["harness"]["actor_hidden_params"])
        out = {f"{p}.{k}": sorted((c["value"] for c in cands if c.get("actor_visible", True)), key=order)
               for p, params in self.d["parameter_menus"].items() for k, cands in params.items()}
        assert hidden <= set(out), f"unknown hidden params {hidden - set(out)}"
        return {k: v for k, v in out.items() if k not in hidden}

    def menu_index(self) -> dict:
        """{'procedure.param': {'values': set, 'recorded': set}}; asserts every candidate is recorded or has a basis."""
        idx = {}
        for p, params in self.d["parameter_menus"].items():
            for k, cands in params.items():
                for c in cands:
                    assert c.get("recorded") or c.get("basis"), f"menu candidate without basis: {p}.{k}={c['value']}"
                idx[f"{p}.{k}"] = {"values": {c["value"] for c in cands},
                                   "recorded": {c["value"] for c in cands if c.get("recorded")}}
        return idx

    # ---------------- actor view ----------------
    def actor_files(self) -> dict:
        """{source_dir: dest_subdir_in_workspace}"""
        return {self.assets_dir / src: dst for src, dst in self.d["harness"]["actor_files"].items()}

    def is_text_file(self, p: Path) -> bool:
        return p.suffix.lower() in TEXT_SUFFIXES

    def write_actor_workspace(self, ws: Path):
        """Copy the files the actor may read (harness.actor_files) into a new workspace dir; text files are
        sanitized so no link points at the source record."""
        ws.mkdir()
        n_text = n_bin = 0
        for src_dir, dst in self.actor_files().items():
            assert src_dir.is_dir(), src_dir
            for f in sorted(src_dir.rglob("*")):
                if f.is_dir():
                    continue
                out = ws / dst / f.relative_to(src_dir)
                out.parent.mkdir(parents=True, exist_ok=True)
                if self.is_text_file(f):
                    out.write_text(self.sanitize_for_actor(f.read_text(encoding="utf-8", errors="strict")),
                                   encoding="utf-8")
                    n_text += 1
                else:
                    shutil.copyfile(f, out)
                    n_bin += 1
        print(f"[actor] workspace: {n_text} text files (sanitized), {n_bin} binary files", flush=True)

    def sanitize_for_actor(self, text: str) -> str:
        """Remove links that point at the source record and replace identifying terms with the case's natural
        replacements (meta.contamination_controls.actor_text_replacements). No redaction marker is left: a marker
        such as "[removed]" tells the reader the record was edited (realism pilot 2026-10-23)."""
        out = []
        for line in text.split("\n"):
            new = self._strip_blocked_links(line)
            if new != line and re.fullmatch(r"(\d+\.|[-*+>])?", new.strip()):
                continue                     # the line held only the removed link
            out.append(new)
        text = "\n".join(out)
        for term, repl in self.text_replacements.items():
            text = text.replace(term, repl)
        for b in self.blocklist:
            assert b not in text, (f"blocklisted string {b!r} survived sanitization; give it a natural replacement in "
                                   f"meta.contamination_controls.actor_text_replacements")
        return text

    def _strip_blocked_links(self, line: str) -> str:
        """Markdown image -> nothing, markdown link -> its text, <img> -> nothing, <a> -> its inner text, bare URL ->
        its file name if it ends in one (keeps code like read.csv("x.csv") readable), else nothing."""
        line = re.sub(r"!\[[^\]]*\]\((https?://[^)]*)\)",
                      lambda m: "" if self._blocked(m.group(1)) else m.group(0), line)
        line = re.sub(r"\[([^\]]*)\]\((https?://[^)]*)\)",
                      lambda m: m.group(1) if self._blocked(m.group(2)) else m.group(0), line)
        line = re.sub(r"<img[^>]*src=\"(https?://[^\"]*)\"[^>]*>",
                      lambda m: "" if self._blocked(m.group(1)) else m.group(0), line)
        line = re.sub(r"<a [^>]*href=\"(https?://[^\"]*)\"[^>]*>(.*?)</a>",
                      lambda m: m.group(2) if self._blocked(m.group(1)) else m.group(0), line)

        def bare(m):
            url = m.group(2).rstrip(".,;:")
            tail = m.group(2)[len(url):]
            if not self._blocked(url):
                return m.group(0)
            name = url.rstrip("/").rsplit("/", 1)[-1]
            return (m.group(1) + name + tail) if re.fullmatch(r"[\w.-]+\.[A-Za-z0-9]{1,5}", name) else tail
        return re.sub(r"([ \t]?)(https?://[^\s\"'<>)\]]+)", bare, line)

    def _blocked(self, url: str) -> bool:
        return any(b in url for b in self.blocklist)

    def report_tool_schema(self) -> dict:
        fr = self.final_report
        return {"type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in fr["fields"].items()},
                "required": fr["required"]}

    def actor_prompt(self) -> str:
        """The task brief: prompts/actor_brief.md filled with this case's brief, resources, and report instructions."""
        items = [f"{k} ({v - self.initial_consumed[k]} of {v} left)" for k, v in self.consumables.items()]
        checkpoint_note = ""
        if self.checkpoints["enforce"]:
            stops = "".join(f"\n  - {c['after']}" for c in self.checkpoints["list"])
            checkpoint_note = "\n" + MESSAGES["brief_checkpoints"].format(stops=stops) + "\n"
        return prompts.text("actor_brief.md").format(
            actor_brief=self.d["actor_brief"].strip(),
            materials=f"- Limited materials: {', '.join(items)}.\n" if items else "",
            budget_total=self.budget_total,
            budget_unit=self.budget_unit,
            existing_material=self.resources["initial_state"]["existing_material"].strip(),
            workspace_note=self.d["harness"]["workspace_note"].strip(),
            checkpoint_note=checkpoint_note,
            report_instructions=self.final_report["instructions"].strip(),
        )
