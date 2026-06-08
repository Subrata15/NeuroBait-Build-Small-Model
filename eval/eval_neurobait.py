"""Lightweight NeuroBait dataset/eval helpers.

GPU generation is intentionally left to Modal in a later pass. This script gives
us a local, dependency-light way to validate held-out conversations and compute
persona heuristics for references or generated outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


LABEL_RE = re.compile(r"\b(Micro-action|Hook|Stakes|Validasi|Resep Engagement)\b", re.I)
ACTION_CUES = (
    "ambil",
    "buka",
    "berdiri",
    "catat",
    "klik",
    "letakkan",
    "pakai",
    "pilih",
    "singkir",
    "tulis",
    "taruh",
    "open",
    "pick",
    "write",
    "stand",
    "move",
    "set",
)


@dataclass
class TurnScore:
    conversation_id: int
    turn_index: int
    role: str
    words: int
    sentences: int
    has_action_cue: bool
    has_label_leak: bool
    text: str


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def assistant_turns(rows: Iterable[dict]) -> Iterable[tuple[int, int, str]]:
    for conv_id, row in enumerate(rows):
        for turn_index, message in enumerate(row.get("messages", [])):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str):
                    yield conv_id, turn_index, content


def count_sentences(text: str) -> int:
    chunks = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return max(1, len(chunks)) if text.strip() else 0


def score_text(conv_id: int, turn_index: int, text: str) -> TurnScore:
    lowered = text.lower()
    words = re.findall(r"\b[\w'-]+\b", text)
    return TurnScore(
        conversation_id=conv_id,
        turn_index=turn_index,
        role="assistant",
        words=len(words),
        sentences=count_sentences(text),
        has_action_cue=any(cue in lowered for cue in ACTION_CUES),
        has_label_leak=bool(LABEL_RE.search(text)),
        text=text,
    )


def summarize(scores: list[TurnScore]) -> dict:
    if not scores:
        return {
            "assistant_turns": 0,
            "avg_words": 0,
            "avg_sentences": 0,
            "action_turn_rate": 0,
            "label_leak_turn_rate": 0,
        }
    return {
        "assistant_turns": len(scores),
        "avg_words": round(mean(s.words for s in scores), 2),
        "avg_sentences": round(mean(s.sentences for s in scores), 2),
        "action_turn_rate": round(mean(1 if s.has_action_cue else 0 for s in scores), 4),
        "label_leak_turn_rate": round(mean(1 if s.has_label_leak else 0 for s in scores), 4),
    }


def evaluate_file(input_path: Path, output_path: Path | None = None) -> dict:
    rows = load_jsonl(input_path)
    scores = [score_text(*turn) for turn in assistant_turns(rows)]
    result = {
        "input": str(input_path),
        "conversations": len(rows),
        "summary": summarize(scores),
        "turns": [asdict(score) for score in scores],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score NeuroBait JSONL conversations.")
    parser.add_argument("input", type=Path, help="JSONL file with messages[] conversations")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate_file(args.input, args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
