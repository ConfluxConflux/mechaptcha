#!/usr/bin/env python3
"""
Extract human-readable chat logs from Claude Code JSONL session files.
Outputs one markdown file per session, plus a combined file.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path.home() / ".claude/projects/-Users-jmc-Desktop-claude-mechaptcha"
OUTPUT_DIR = Path(__file__).parent / "chatlogs"


def extract_text(content_blocks):
    """Extract plain text from a message's content block list."""
    parts = []
    for block in content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block["text"].strip())
            elif block.get("type") == "image":
                src = block.get("source", {})
                if src.get("type") == "base64":
                    parts.append("[image]")
                else:
                    parts.append(f"[image: {src.get('url', '')}]")
        elif isinstance(block, str):
            parts.append(block.strip())
    return "\n\n".join(p for p in parts if p)


def parse_session(jsonl_path):
    """Parse a single session JSONL into a list of (role, text, timestamp) tuples."""
    messages = []
    seen_ids = set()

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = obj.get("type")
            if role not in ("user", "assistant"):
                continue

            msg = obj.get("message", {})
            msg_id = msg.get("id") or obj.get("uuid") or obj.get("promptId")

            # Deduplicate streamed assistant chunks by message id
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)

            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            # For assistant messages, only keep text blocks (skip tool_use, thinking)
            if role == "assistant":
                content = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]

            text = extract_text(content)
            if not text:
                continue

            ts = obj.get("timestamp") or msg.get("created_at") or ""
            messages.append((role, text, ts))

    return messages


def format_session(session_id, messages, index):
    lines = [f"# Session {index}: {session_id}\n"]
    for role, text, ts in messages:
        label = "**User**" if role == "user" else "**Claude**"
        ts_str = f" _{ts}_" if ts else ""
        lines.append(f"## {label}{ts_str}\n\n{text}\n")
        lines.append("---\n")
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    jsonl_files = sorted(PROJECT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {PROJECT_DIR}")
        sys.exit(1)

    all_sessions = []
    for i, path in enumerate(jsonl_files, 1):
        messages = parse_session(path)
        if not messages:
            print(f"  [{i}] {path.stem} — no messages, skipping")
            continue

        print(f"  [{i}] {path.stem} — {len(messages)} messages")
        formatted = format_session(path.stem, messages, i)
        out_path = OUTPUT_DIR / f"session_{i:02d}_{path.stem[:8]}.md"
        out_path.write_text(formatted)
        all_sessions.append((i, path.stem, formatted))

    # Write combined file
    combined = f"# MeCHaptcha Claude Code Chat Logs\n\nExtracted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\nSessions: {len(all_sessions)}\n\n---\n\n"
    combined += "\n\n---\n\n".join(body for _, _, body in all_sessions)
    combined_path = OUTPUT_DIR / "all_sessions_combined.md"
    combined_path.write_text(combined)

    print(f"\nDone. {len(all_sessions)} sessions written to {OUTPUT_DIR}/")
    print(f"Combined log: {combined_path}")


if __name__ == "__main__":
    main()
