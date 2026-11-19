"""Record the frozen rubric hash of every case, or check that none changed since the freeze.

python scripts/freeze_rubrics.py --write     # record (only when deliberately re-freezing)
python scripts/freeze_rubrics.py --check     # assert current rubrics match the freeze
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # run from a checkout without installing
from twinlab.case import Case   # noqa: E402

FREEZE = ROOT / "cases/rubric_freeze.json"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.write:
        cur = {p.parent.name: Case(p, "case").rubric_sha256 for p in sorted((ROOT / "cases").glob("*/case.yaml"))}
        FREEZE.write_text(json.dumps({"frozen_at": time.strftime("%Y-%m-%d"), "cases": cur}, indent=1))
        print(json.dumps(cur, indent=1))
        return
    frozen = json.loads(FREEZE.read_text())["cases"]
    # only frozen cases are checked: cases still being built by parallel sessions may not load yet (2026-10-12)
    cur = {k: Case(ROOT / "cases" / k / "case.yaml", "case").rubric_sha256 for k in frozen}
    changed = {k: (frozen.get(k), v) for k, v in cur.items() if frozen.get(k) != v}
    print("changed since freeze:", changed or "none")
    assert not changed, ("rubric changed after freeze: put the change in a new variant, or record it in meta.yaml, "
                         "re-judge all runs of that case, and re-freeze with --write")


if __name__ == "__main__":
    main()
