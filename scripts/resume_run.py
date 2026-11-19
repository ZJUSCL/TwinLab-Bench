"""Resume a run that stopped on a harness/environment/actor/judge error, with the current (fixed) code.

python scripts/resume_run.py --run-dir runs/<case>/<timestamp>/run_<i>

See twinlab.run.resume_run for what is restored. The case's summary.json is rewritten afterwards.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.run import resume_run, summarize   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    resume_run(run_dir)
    summarize(run_dir.parent)


if __name__ == "__main__":
    main()
