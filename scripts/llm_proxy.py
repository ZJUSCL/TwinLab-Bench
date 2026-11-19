"""Optional helper: run a LiteLLM proxy that lets Claude Code (actor, environment, judge) talk to a non-Claude model.

TwinLab does not depend on LiteLLM. Claude Code speaks the Anthropic /v1/messages format; a model behind an
OpenAI-style endpoint needs a translator. Any Anthropic-compatible endpoint works
(machine.yaml `model_backends.<model>.base_url`); this script only manages one if LiteLLM happens to be installed.

python scripts/llm_proxy.py write-config --model deepseek-flash --upstream-model deepseek-flash \
    --api-base https://api.deepseek.com/v1 --key-env DEEPSEEK_API_KEY --master-key <token> --config ~/.config/twinlab/litellm_ds_chat.yaml
python scripts/llm_proxy.py start  --config ~/.config/twinlab/litellm_ds_chat.yaml --port 4001 --workers 32
python scripts/llm_proxy.py status --port 4001 --model deepseek-flash   # liveness + served model and fingerprint
python scripts/llm_proxy.py stop   --port 4001
python scripts/llm_proxy.py stream-test --port 4001 --model deepseek-flash   # decode speed through the proxy

Settings that matter (each found the hard way, 2026-11-04):
- use_chat_completions_url_for_anthropic_messages: true. LiteLLM 1.108 otherwise routes /v1/messages through the
  gateway's /v1/responses, which rejects multi-turn tool histories.
- several workers (--workers 32): one process became the bottleneck at ~70 parallel runs (streams ~20x slower).
- LITELLM_LOCAL_MODEL_COST_MAP=True: no price-table download at start-up.
- drop_params: true. Retries belong to Claude Code (CLAUDE_CODE_MAX_RETRIES in machine.yaml cli_env), not here.
- The upstream key is written only into the config file outside the repository (mode 600), never into the repo.
Claude Code itself needs, per backend (machine.yaml cli_env): CLAUDE_CODE_MAX_CONTEXT_TOKENS (else it assumes 200k
and auto-compacts) and CLAUDE_CODE_MAX_OUTPUT_TOKENS (thinking counts against the default 32k).
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

STATE_DIR = Path("~/.config/twinlab").expanduser()
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost, never through a web proxy


def litellm_bin() -> str:
    # the dedicated venv first: a `litellm` on PATH may lack the proxy extras (litellm[proxy])
    venv = STATE_DIR / "litellm-venv/bin/litellm"
    exe = str(venv) if venv.is_file() else shutil.which("litellm")
    assert exe and Path(exe).is_file(), ("LiteLLM is not installed (no `litellm` on PATH and no ~/.config/twinlab/litellm-venv);"
                                 " install it or point machine.yaml model_backends at another Anthropic-compatible "
                                 "endpoint")
    return exe


def pid_file(port: int) -> Path:
    return STATE_DIR / f"litellm_{port}.pid"


def write_config(a):
    key = os.environ[a.key_env]
    cfg = Path(a.config).expanduser()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    text = (f"model_list:\n- model_name: {a.model}\n  litellm_params:\n    model: openai/{a.upstream_model}\n"
            f"    api_base: {a.api_base}\n    api_key: {key}\n"
            "litellm_settings:\n  drop_params: true\n  use_chat_completions_url_for_anthropic_messages: true\n"
            f"general_settings:\n  master_key: {a.master_key}\n")
    cfg.write_text(text)
    cfg.chmod(0o600)
    print(f"written {cfg} (mode 600; model {a.model} -> openai/{a.upstream_model} at {a.api_base})")


def start(a):
    exe = litellm_bin()
    assert not alive(a.port), f"something already answers on port {a.port}"
    log = Path(a.log or STATE_DIR / f"litellm_{a.port}.log").expanduser()
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "NO_PROXY": "localhost,127.0.0.1",
           "no_proxy": "localhost,127.0.0.1"}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    p = subprocess.Popen([exe, "--config", str(Path(a.config).expanduser()), "--port", str(a.port),
                          "--num_workers", str(a.workers)], stdout=open(log, "a"), stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    pid_file(a.port).write_text(str(p.pid))
    for _ in range(120):
        if alive(a.port):
            print(f"LiteLLM up on {a.port} (pid {p.pid}, {a.workers} workers, log {log})")
            return
        assert p.poll() is None, f"LiteLLM exited rc={p.returncode}; see {log}"
        time.sleep(1)
    raise AssertionError(f"LiteLLM did not answer on {a.port} within 120 s; see {log}")


def stop(a):
    pf = pid_file(a.port)
    assert pf.is_file(), f"no pid file {pf}; was the proxy started by this script?"
    pid = int(pf.read_text())
    os.killpg(pid, signal.SIGTERM)   # the uvicorn master and its workers share the session started above
    pf.unlink()
    print(f"sent SIGTERM to process group {pid} (port {a.port})")


def alive(port: int) -> bool:
    try:
        return OPENER.open(f"http://localhost:{port}/health/liveliness", timeout=5).status == 200
    except OSError:
        return False


def status(a):
    up = alive(a.port)
    print(f"port {a.port}: {'alive' if up else 'NOT answering'}")
    if not up:
        sys.exit(1)
    body = {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    req = urllib.request.Request(f"http://localhost:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {a.master_key}"})
    d = json.loads(OPENER.open(req, timeout=120).read())
    print(f"served model {d['model']}, system_fingerprint {d.get('system_fingerprint')}")


def stream_test(a):
    body = {"model": a.model, "max_tokens": 3000, "stream": True,
            "messages": [{"role": "user", "content": "Write a 1500-word essay on the history of the thermometer."}]}
    req = urllib.request.Request(f"http://localhost:{a.port}/v1/messages", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-api-key": a.master_key,
                                  "anthropic-version": "2023-06-01"})
    t0, chars = time.time(), 0
    with OPENER.open(req, timeout=600) as r:
        for raw in r:
            chars += len(raw)
    dt = time.time() - t0
    print(f"{dt:.1f} s, {chars / dt:.0f} SSE bytes/s (an idle proxy streams ~10k bytes/s for this prompt)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-config")
    for k in ("--model", "--upstream-model", "--api-base", "--key-env", "--config"):
        w.add_argument(k, required=True)
    w.add_argument("--master-key", required=True, help="the proxy's own token (machine.yaml auth_token)")
    s = sub.add_parser("start")
    s.add_argument("--config", required=True)
    s.add_argument("--port", required=True, type=int)
    s.add_argument("--workers", required=True, type=int)
    s.add_argument("--log")
    for name in ("stop",):
        sub.add_parser(name).add_argument("--port", required=True, type=int)
    for name in ("status", "stream-test"):
        p = sub.add_parser(name)
        p.add_argument("--port", required=True, type=int)
        p.add_argument("--model", required=True)
        p.add_argument("--master-key", required=True)
    a = ap.parse_args()
    {"write-config": write_config, "start": start, "stop": stop, "status": status,
     "stream-test": stream_test}[a.cmd](a)


if __name__ == "__main__":
    main()
