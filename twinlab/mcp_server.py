"""Expose a Lab to the actor as an MCP server (Streamable HTTP on 127.0.0.1, JSON-RPC, no SDK).

The actor's agent CLI (Claude Code or Codex) connects to http://127.0.0.1:<port>/mcp/<token> from inside its sandbox. A
tools/call is answered as an SSE stream with keep-alive comments, because one environment call can take minutes.
Any exception inside the Lab marks the run fatal (lab.fatal) and the lab answers "unavailable" from then on; the
run stops and can be resumed after the cause is fixed (scripts/resume_run.py).
"""
import json
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .lab import Lab, text_result

SSE_KEEPALIVE_S = 10   # keep-alive comment interval while an environment call runs
MCP_PATH_PREFIX = "/mcp/"
MCP_TOKEN_BYTES = 16


def serve(lab: Lab):
    """Start the server in a daemon thread. Returns (server, url); call server.shutdown() when the actor is done.
    The url path holds a random token: the network is shared with the host, so without it an actor could find and
    call the lab of another run on the machine (run_batch.py runs many cases at once)."""
    path = f"{MCP_PATH_PREFIX}{secrets.token_urlsafe(MCP_TOKEN_BYTES)}"
    srv = make_server(lab, path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}{path}"


def make_server(lab: Lab, path: str) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _wrong_path(self):
            if self.path == path:
                return False
            print(f"[lab] refused a request to {self.path[:len(MCP_PATH_PREFIX) + 4]}... (not this run's MCP path)",
                  flush=True)
            self.send_response(404)
            self.end_headers()
            return True

        def do_GET(self):
            if self._wrong_path():
                return
            self.send_response(405)
            self.end_headers()

        def do_DELETE(self):
            if self._wrong_path():
                return
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            if self._wrong_path():
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs = body if isinstance(body, list) else [body]
            if not isinstance(body, list) and body.get("method") == "tools/call":
                return self._sse_call(body)
            replies = [r for r in (self._handle(m) for m in msgs) if r is not None]
            if not replies:
                self.send_response(202)
                self.end_headers()
                return
            data = json.dumps(replies if isinstance(body, list) else replies[0]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse_call(self, m: dict):
            """tools/call answered as an SSE stream with keep-alive comments every SSE_KEEPALIVE_S, so a slow
            environment call (minutes) does not look like an idle connection to the client. If the client is gone
            the call still completes and is recorded in the lab state."""
            out = {}
            worker = threading.Thread(target=lambda: out.setdefault("reply", self._handle(m)), daemon=True)
            worker.start()
            alive = True
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.flush()
            except OSError:
                alive = False
            while worker.is_alive():
                worker.join(SSE_KEEPALIVE_S)
                if alive and worker.is_alive():
                    try:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    except OSError:
                        alive = False
                        print("[lab] client disconnected during a tools/call; result will be kept", flush=True)
            if alive:
                try:
                    self.wfile.write(f"event: message\ndata: {json.dumps(out['reply'])}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    print("[lab] client disconnected before the result was sent; result kept", flush=True)

        def _handle(self, m: dict):
            if "id" not in m:          # notification
                return None
            method, mid = m["method"], m["id"]
            if method == "initialize":
                res = {"protocolVersion": m["params"]["protocolVersion"], "capabilities": {"tools": {}},
                       "serverInfo": {"name": "lab", "version": "0.1"}}
            elif method == "ping":
                res = {}
            elif method == "tools/list":
                res = {"tools": lab.tools}
            elif method == "tools/call":
                if lab.fatal is not None:
                    res = text_result("The lab is unavailable.", is_error=True)
                else:
                    if isinstance(m["params"], str):
                        # 053-01 DS-actor run 20261104-114756: Claude Code sent params as a JSON string
                        # ('{"name":"lab_status","arguments":{}}'), which stopped the run as a harness error
                        print(f"[lab] tools/call params arrived as a JSON string; decoded: {m['params'][:200]}",
                              flush=True)
                        m["params"] = json.loads(m["params"])
                    try:
                        res = lab.call(m["params"]["name"], m["params"].get("arguments") or {})
                        lab.save_state()
                    except Exception as e:  # environment/harness bug: stop the run (fail fast), resumable
                        lab.fatal = traceback.format_exc()
                        lab.failed_request = m["params"]
                        lab.save_state()
                        print(f"[lab] FATAL environment error: {e}", flush=True)
                        res = text_result("The lab is unavailable.", is_error=True)
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
            return {"jsonrpc": "2.0", "id": mid, "result": res}

    return ThreadingHTTPServer(("127.0.0.1", 0), H)
