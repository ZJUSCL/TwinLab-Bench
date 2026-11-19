"""Withdraw-and-notice relay for Codex's hosted web_search.

The hosted web_search runs on the provider's servers inside one model response, so no Codex hook sees it (probe
2026-11-05). Codex sends the full conversation on every request (store=false), so a relay between Codex and the
gateway can keep a response out of the conversation altogether:

- Every Responses API request is forwarded and the whole streamed answer is buffered before Codex gets it.
- If a web_search_call item of the answer (query, queries, opened URL) or a url_citation of an answer message holds a
  blocklisted string, the answer is dropped: Codex never receives it, so neither the call nor its results enter the
  conversation. The relay appends NOTICE_TEXT (a developer message, naming no blocked string and not repeating the
  query) at the end of the request and asks again.
- Each notice is inserted again at the same position of every later request of the run (Codex itself does not know
  it), so the protocol note stays in the conversation; positions are kept in run_dir/websearch_guard.json, which
  also survives resume.
- After MAX_WITHDRAWALS_PER_TURN withdrawals for one request, the next attempt is sent without the web_search tool.
- Every withdrawal is logged in websearch_guard.json (events) and printed; judge.audit_actor reports them.

The audit checks the same parts of the Codex rollout (web_search_call items), so a run whose blocked searches were
all withdrawn is not contaminated by them. websearch_guard.json lives in run_dir, outside the sandbox mounts (the
actor never sees the plain needles).
"""
import http.server
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STATE_FILE = "websearch_guard.json"
MAX_WITHDRAWALS_PER_TURN = 3
UPSTREAM_TIMEOUT_S = 3600
NOTICE_TEXT = ("Lab protocol notice: your previous web search was withdrawn because it reached material that this "
               "task's protocol excludes (the source record this task is based on: its publications, datasets, "
               "authors' write-ups or code repositories). Searching for that source is a protocol violation. "
               "Continue the task with the lab and the files you have.")
SKIP_HEADERS = {"host", "content-length", "accept-encoding", "connection"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def write_state(run_dir: Path, needles: list):
    """Fresh run (CodexActor.prepare): the needles the relay checks, no notices yet."""
    p = run_dir / STATE_FILE
    assert not p.exists(), p
    p.write_text(json.dumps({"needles": needles, "notices": [], "events": []}, indent=1, ensure_ascii=False))


def sse_events(raw: bytes) -> list:
    """data payloads of an SSE body (the gateway writes `data:` without a space)."""
    out = []
    for block in raw.decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n\n"):
        data = [ln[5:].lstrip(" ") for ln in block.split("\n") if ln.startswith("data:")]
        if data and data[0] != "[DONE]":
            out.append(json.loads("\n".join(data)))
    return out


def outside_parts(events: list) -> list:
    """What the answer brought in from the web: web_search_call items and url_citation annotations of messages."""
    parts = []
    for e in events:
        if e.get("type") != "response.output_item.done":
            continue
        it = e["item"]
        if it["type"] == "web_search_call":
            parts.append(("web_search_call", it))
        elif it["type"] == "message":
            for c in it.get("content") or []:
                for a in c.get("annotations") or []:
                    parts.append(("url_citation", a))
    return parts


class Relay:
    def __init__(self, run_dir: Path, upstream_base: str):
        self.state_path = run_dir / STATE_FILE
        self.state = json.loads(self.state_path.read_text())
        u = urllib.parse.urlsplit(upstream_base)
        self.upstream_origin = f"{u.scheme}://{u.netloc}"
        self.lock = threading.Lock()   # one conversation: requests are handled one at a time
        relay = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):   # noqa: A002 (base class signature)
                pass

            def do_POST(self):
                relay.handle_post(self)

            def do_GET(self):
                relay.passthrough(self, None)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        print(f"[ws-relay] 127.0.0.1:{self.port} -> {self.upstream_origin}, {len(self.state['needles'])} needles, "
              f"{len(self.state['notices'])} notices carried", flush=True)

    def base_url(self, upstream_base: str) -> str:
        return f"http://127.0.0.1:{self.port}{urllib.parse.urlsplit(upstream_base).path}"

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1, ensure_ascii=False))

    def hits(self, parts) -> list:
        found = []
        for kind, obj in parts:
            s = json.dumps(obj, ensure_ascii=False)
            found += [{"kind": kind, "needle_index": i} for i, n in enumerate(self.state["needles"]) if n in s]
        return found

    def with_notices(self, codex_input: list) -> list:
        out = list(codex_input)
        for k, n in enumerate(sorted(self.state["notices"], key=lambda n: n["at"])):
            assert n["at"] <= len(codex_input), \
                f"Codex input shrank below a withdrawal notice position ({n['at']} > {len(codex_input)}): compaction?"
            out.insert(n["at"] + k, {"type": "message", "role": "developer",
                                     "content": [{"type": "input_text", "text": n["text"]}]})
        return out

    def upstream(self, handler, path, data: bytes):
        headers = {k: v for k, v in handler.headers.items() if k.lower() not in SKIP_HEADERS}
        req = urllib.request.Request(self.upstream_origin + path, data=data, method=handler.command, headers=headers)
        try:
            r = OPENER.open(req, timeout=UPSTREAM_TIMEOUT_S)
        except urllib.error.HTTPError as e:
            r = e
        return r.status if hasattr(r, "status") else r.code, r.headers, r.read()

    @staticmethod
    def reply(handler, status, headers, body: bytes):
        handler.send_response(status)
        for k, v in headers.items():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                handler.send_header(k, v)
        handler.send_header("content-length", str(len(body)))
        handler.send_header("connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()

    def passthrough(self, handler, data):
        self.reply(handler, *self.upstream(handler, handler.path, data))

    def handle_post(self, handler):
        data = handler.rfile.read(int(handler.headers["content-length"]))
        if not handler.path.endswith("/responses"):
            self.passthrough(handler, data)
            return
        body = json.loads(data)
        assert body.get("stream") is True and "previous_response_id" not in body, \
            f"unexpected Codex request shape: stream={body.get('stream')} keys={sorted(body)}"
        codex_input = body["input"]
        assert isinstance(codex_input, list), type(codex_input)
        with self.lock:
            withdrawn = 0
            while True:
                send = dict(body, input=self.with_notices(codex_input))
                if withdrawn >= MAX_WITHDRAWALS_PER_TURN:
                    send["tools"] = [t for t in body.get("tools") or [] if not t.get("type", "").startswith("web_search")]
                status, headers, raw = self.upstream(handler, handler.path, json.dumps(send).encode())
                if status != 200:
                    self.reply(handler, status, headers, raw)
                    return
                found = self.hits(outside_parts(sse_events(raw)))
                if not found:
                    self.reply(handler, status, headers, raw)
                    return
                assert withdrawn < MAX_WITHDRAWALS_PER_TURN, \
                    f"blocked web content without the web_search tool: {found}"
                withdrawn += 1
                parts = outside_parts(sse_events(raw))
                ev = {"t": time.time(), "input_len": len(codex_input), "attempt": withdrawn, "hits": found,
                      "web_search_calls": [p[1].get("action") for p in parts if p[0] == "web_search_call"],
                      "tool_removed_next": withdrawn >= MAX_WITHDRAWALS_PER_TURN}
                self.state["events"].append(ev)
                if all(n["at"] != len(codex_input) for n in self.state["notices"]):   # one notice per position
                    self.state["notices"].append({"at": len(codex_input), "text": NOTICE_TEXT})
                self.save()
                print(f"[ws-relay] withdrew an answer at input item {len(codex_input)} (attempt {withdrawn}): "
                      f"{[h['kind'] for h in found]}; web_search removed next: {ev['tool_removed_next']}",
                      flush=True)
