"""Content manifest of the benchmark cases, so a copy of the benchmark can be checked byte for byte.

python scripts/case_manifest.py --write     # (re)write cases/official_manifest.json
python scripts/case_manifest.py --check     # fail if any hash differs from the stored manifest

Per case: sha256 of case.yaml, the rubric hash the judge records (Case.rubric_sha256), the calculator file, the actor
workspace (built by Case.write_actor_workspace, i.e. after sanitization, exactly what the actor gets) and the env-side
files (env context files, env_knowledge, env data dirs and the figures dir, under the names the env workspace uses).
A tree hash is sha256 over "<relative path>\\0<sha256 of the file>\\n" lines in sorted path order.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # run from a checkout without installing
from twinlab.case import Case   # noqa: E402

MANIFEST = ROOT / "cases/official_manifest.json"


def case_ids() -> list:
    ids = sorted(p.parent.name for p in (ROOT / "cases").glob("*/case.yaml"))
    assert ids, ROOT / "cases"
    return ids


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_sha(entries: dict) -> str:
    """entries: {relative name: file path}"""
    h = hashlib.sha256()
    for name in sorted(entries):
        h.update(f"{name}\0{file_sha(entries[name])}\n".encode())
    return h.hexdigest()


def files_under(d: Path, prefix: str) -> dict:
    assert d.is_dir(), d
    return {f"{prefix}/{f.relative_to(d).as_posix()}": f for f in sorted(d.rglob("*")) if not f.is_dir()}


def case_entry(case_id: str) -> dict:
    c = Case(ROOT / "cases" / case_id / "case.yaml", "case")
    assert c.id == case_id, (c.id, case_id)
    calc = c.d["harness"]["env_calculator"]
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        c.write_actor_workspace(ws)
        actor = files_under(ws, "ws")
        actor_sha = tree_sha(actor)
    env = {}
    for f in c.env_context_files():
        assert f.is_file() and f.name not in env, f
        env[f.name] = f
    for f in c.env_knowledge_files():
        env[f"env_knowledge/{f.name}"] = f
    for src, dst in c.d["harness"]["env_data_dirs"].items():
        env.update(files_under(c.assets_dir / src, dst))
    env.update(files_under(c.figures_dir, "figures"))
    return {
        "case_yaml_sha256": file_sha(c.path),
        "rubric_sha256": c.rubric_sha256,
        "calculator": calc,
        "calculator_sha256": file_sha(c.assets_dir / calc) if calc else None,
        "actor_workspace_sha256": actor_sha,
        "actor_workspace_files": len(actor),
        "env_files_sha256": tree_sha(env),
        "env_files": len(env),
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cur = {i: case_entry(i) for i in case_ids()}
    if a.write:
        MANIFEST.write_text(json.dumps({"n_cases": len(cur), "cases": cur}, indent=1) + "\n")
        print(f"wrote {MANIFEST} ({len(cur)} cases)")
        return
    stored = json.loads(MANIFEST.read_text())["cases"]
    assert set(stored) == set(cur), {"only in manifest": sorted(set(stored) - set(cur)),
                                     "only on disk": sorted(set(cur) - set(stored))}
    diff = {i: {k: (stored[i][k], v) for k, v in e.items() if stored[i][k] != v} for i, e in cur.items()}
    diff = {i: d for i, d in diff.items() if d}
    for i, d in diff.items():
        print(f"CHANGED {i}: {json.dumps(d)}")
    assert not diff, f"{len(diff)} case(s) differ from {MANIFEST.name}"
    print(f"manifest check: {len(cur)} cases match {MANIFEST.name}")


if __name__ == "__main__":
    main()
