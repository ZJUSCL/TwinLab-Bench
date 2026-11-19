"""Human baseline: a person plays the actor on a case through a terminal, against the same lab, environment agent,
budget, judge and scoring as the model actor.

python scripts/human_baseline.py --case cases/008-qpcr-polyic-02/case.yaml --tester T1 \
    --env-model claude-sonnet-5-5 --judge-model claude-opus-5-5 --out runs

Commands at the prompt:
  brief                      show the task brief again
  catalog                    list the lab's procedures and parameter options (same as the lab_catalog tool)
  status                     budget used/remaining and limited materials left (same as lab_status)
  lab                        submit ONE lab request; type the request, end with a line containing only "."
                             then optionally give parameters as JSON ({"procedure.param": ["value"]}) or leave empty
  schema                     show the fields of the final report
  report <file.json>         submit the final report from a JSON file with those fields (once; ends the session)
Images returned by the lab are saved under <run_dir>/human_images/ and their paths are printed.
The workspace with the case files is printed at start; analyse them with any tools you like.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
from twinlab.actors import HUMAN                    # noqa: E402
from twinlab.case import Case                       # noqa: E402
from twinlab.env_agent import EnvAgent              # noqa: E402
from twinlab.lab import Lab                         # noqa: E402
from twinlab.run import finish_run, run_config      # noqa: E402

END_OF_REQUEST = "."


def show(result: dict, img_dir: Path, n_img: list) -> None:
    for c in result["content"]:
        if c["type"] == "text":
            print(c["text"])
        else:
            assert c["type"] == "image", c["type"]
            n_img[0] += 1
            p = img_dir / f"img_{n_img[0]:03d}.jpg"
            p.write_bytes(base64.b64decode(c["data"]))
            print(f"[image saved] {p}")
    if result["isError"]:
        print("[error]")


def read_request() -> tuple:
    print(f'Request (end with a line containing only "{END_OF_REQUEST}"):')
    lines = []
    while True:
        line = input()
        if line.strip() == END_OF_REQUEST:
            break
        lines.append(line)
    raw = input("Parameters JSON (empty for none): ").strip()
    params = json.loads(raw) if raw else {}
    return "\n".join(lines).strip(), params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--tester", required=True, help="anonymous tester label, e.g. T1")
    ap.add_argument("--env-model", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    case = Case(a.case, "case")
    run_dir = (a.out / case.id / f"human-{a.tester}-{time.strftime('%Y%m%d-%H%M%S')}" / "run_0").resolve()
    run_dir.mkdir(parents=True)
    cfg = run_config(case, 0, HUMAN, f"human:{a.tester}", a.env_model, a.judge_model, None, "case", None)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=1))
    (run_dir / "actor_state.json").write_text(json.dumps({"session_id": None, "invocations": []}, indent=1))
    env = EnvAgent(case, run_dir, a.env_model, resume=False)
    lab = Lab(case, env, run_dir, 0)
    ws = run_dir / "actor_workspace"
    case.write_actor_workspace(ws)
    brief = case.actor_prompt()
    (run_dir / "actor_prompt.md").write_text(brief, encoding="utf-8")
    img_dir = run_dir / "human_images"
    img_dir.mkdir()
    n_img = [0]
    session_log = run_dir / "human_session.jsonl"
    print(brief)
    print(f"\n=== Your case files: {ws}\n=== In this terminal the tools are commands: "
          "brief, catalog, status, lab, schema, report <file.json>. Type 'help' for details.")
    t0 = time.time()
    while lab.report is None:
        cmd = input("\nlab> ").strip()
        if not cmd:
            continue
        with open(session_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time() - t0, "cmd": cmd}) + "\n")
        if cmd == "help":
            print(__doc__)
        elif cmd == "brief":
            print(brief)
        elif cmd == "catalog":
            show(lab.call("lab_catalog", {}), img_dir, n_img)
        elif cmd == "status":
            show(lab.call("lab_status", {}), img_dir, n_img)
        elif cmd == "schema":
            print(json.dumps(case.report_tool_schema(), indent=1))
        elif cmd == "lab":
            request, params = read_request()
            with open(session_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time() - t0, "request": request, "parameters": params},
                                   ensure_ascii=False) + "\n")
            show(lab.call("lab", {"request": request, "parameters": params}), img_dir, n_img)
            lab.save_state()
            assert lab.fatal is None, f"environment error; see {run_dir}. Resume with scripts/resume_run.py is not " \
                                      f"supported for human runs yet. Error:\n{lab.fatal}"
        elif cmd.startswith("report "):
            report = json.loads(Path(cmd.split(" ", 1)[1].strip()).expanduser().read_text())
            missing = [k for k in case.report_tool_schema()["required"] if k not in report]
            if missing:
                print(f"[error] report is missing required fields {missing}; see 'schema'")
                continue
            show(lab.call("submit_final_report", report), img_dir, n_img)
        else:
            print("unknown command; type 'help'")
    lab.save_state()
    st = json.loads((run_dir / "actor_state.json").read_text())
    st["wall_time_s"] = round(time.time() - t0)
    (run_dir / "actor_state.json").write_text(json.dumps(st, indent=1))
    print("\nScoring (judge running)...")
    res = finish_run(case, run_dir, lab, cfg)
    print(json.dumps(res["score"], indent=1))


if __name__ == "__main__":
    main()
