"""Render a compact Markdown report from eval/eval_neurobait.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_markdown(result: dict) -> str:
    summary = result.get("summary", {})
    lines = [
        "# NeuroBait Eval Report",
        "",
        f"- Input: `{result.get('input', '-')}`",
        f"- Conversations: `{result.get('conversations', 0)}`",
        f"- Assistant turns: `{summary.get('assistant_turns', 0)}`",
        f"- Avg words/turn: `{summary.get('avg_words', 0)}`",
        f"- Avg sentences/turn: `{summary.get('avg_sentences', 0)}`",
        f"- Action cue turn rate: `{summary.get('action_turn_rate', 0)}`",
        f"- Label leak turn rate: `{summary.get('label_leak_turn_rate', 0)}`",
        "",
        "## Review Notes",
        "",
        "- Target label leak rate is `0`.",
        "- Action cues are heuristic; use qualitative review for final pass.",
        "- English transfer should be reviewed separately on EN held-out examples.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Markdown from NeuroBait eval JSON.")
    parser.add_argument("input", type=Path, help="Eval JSON path")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path")
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_suffix(".md")
    output.write_text(render_markdown(result), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
