"""Print a compact view of one actor stream: tool calls, tool results (truncated), and text.

python scripts/show_stream.py runs/.../run_0/actor_stream_r0_0.jsonl [max_chars]

The stream format (Claude Code or Codex) is taken from the run's run_config.json.
"""
import json
import sys
from pathlib import Path

DEFAULT_MAX_CHARS = 600


def show_claude_code(path: Path, n: int):
    for line in path.read_text().splitlines():
        m = json.loads(line)
        t = m["type"]
        if t == "system" and m.get("subtype") == "init":
            print(f"[init] model={m.get('model')} tools={m.get('tools')} mcp={m.get('mcp_servers')} "
                  f"skills={m.get('skills')} slash={m.get('slash_commands')}")
        elif t in ("assistant", "user"):
            content = m["message"]["content"]
            if isinstance(content, str):
                print(f"[{t} text] {content[:n]}")
                continue
            for c in content:
                if c["type"] == "text":
                    print(f"[{t} text] {c['text'][:n]}")
                elif c["type"] == "tool_use":
                    print(f"[tool_use {c['name']}] {json.dumps(c['input'])[:n]}")
                elif c["type"] == "tool_result":
                    cc = c["content"]
                    if isinstance(cc, list):
                        cc = " ".join(x.get("text", f"<{x['type']}>") for x in cc)
                    print(f"[tool_result err={c.get('is_error')}] {str(cc)[:n]}")
        elif t == "result":
            print(f"[result] turns={m.get('num_turns')} cost={m.get('total_cost_usd')} err={m.get('is_error')}")


def show_codex(path: Path, n: int):
    for line in path.read_text().splitlines():
        m = json.loads(line)
        t = m["type"]
        if t == "thread.started":
            print(f"[thread] {m['thread_id']}")
        elif t == "turn.completed":
            print(f"[turn completed] usage={json.dumps(m['usage'])}")
        elif t == "item.completed":
            it = m["item"]
            if it["type"] in ("agent_message", "reasoning"):
                print(f"[{it['type']}] {it.get('text', '')[:n]}")
            elif it["type"] == "command_execution":
                print(f"[shell rc={it['exit_code']}] {it['command'][:n]}\n  -> {it['aggregated_output'][:n]}")
            elif it["type"] == "mcp_tool_call":
                res = it["error"] if it["error"] is not None else " ".join(
                    x.get("text", f"<{x['type']}>") for x in it["result"]["content"])
                print(f"[mcp {it['server']}.{it['tool']}] {json.dumps(it['arguments'])[:n]}\n  -> {str(res)[:n]}")
            else:
                print(f"[{it['type']}] {json.dumps(it)[:n]}")
        elif t in ("error", "turn.failed"):
            print(f"[{t}] {json.dumps(m)[:n]}")


def main():
    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MAX_CHARS
    cfg = path.parent / "run_config.json"
    assert cfg.is_file(), f"no run_config.json next to {path}; cannot tell the stream format"
    agent = json.loads(cfg.read_text())["actor_agent"]
    {"claude-code": show_claude_code, "codex": show_codex}[agent](path, n)


if __name__ == "__main__":
    main()
