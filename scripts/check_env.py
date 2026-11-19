"""Deterministic after-the-fact check of the environment agent of one run (no model calls); see twinlab/env_check.py.

python scripts/check_env.py --run-dir runs/<case>/<ts>/run_0 --scratch <dir outside runs/ and cases/>
                            [--transcript <env session .jsonl>]   # default: <run-dir>/env_session.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.env_check import check_env   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--transcript", type=Path, help="only for runs before 2026-10-13, which have no env_session.jsonl")
    a = ap.parse_args()
    check_env(a.run_dir, a.scratch, a.transcript if a.transcript else a.run_dir / "env_session.jsonl")


if __name__ == "__main__":
    main()
