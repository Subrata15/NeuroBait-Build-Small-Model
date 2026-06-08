"""Validate NeuroBait JSONL training/eval files before Modal image build."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SYSTEM_PREFIX = "Kamu adalah NeuroBait"
VALID_ROLES = {"system", "user", "assistant"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def validate_row(row: dict, path: Path, index: int) -> list[str]:
    errors = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return [f"{path}:{index}: messages must be a list with at least 3 turns"]

    roles = [message.get("role") for message in messages]
    if roles[0] != "system":
        errors.append(f"{path}:{index}: first role must be system")
    if any(role not in VALID_ROLES for role in roles):
        errors.append(f"{path}:{index}: invalid role sequence {roles}")

    system_content = messages[0].get("content", "")
    if not isinstance(system_content, str) or not system_content.startswith(SYSTEM_PREFIX):
        errors.append(f"{path}:{index}: missing official NeuroBait system prompt")

    previous = "system"
    for turn_no, message in enumerate(messages[1:], start=1):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{path}:{index}: turn {turn_no} has empty content")
        expected = "user" if previous in {"system", "assistant"} else "assistant"
        if role != expected:
            errors.append(f"{path}:{index}: turn {turn_no} expected {expected}, got {role}")
        previous = role

    if roles[-1] != "assistant":
        errors.append(f"{path}:{index}: final role should be assistant")
    return errors


def validate_file(path: Path, expected_count: int | None = None) -> dict:
    rows = load_jsonl(path)
    errors = []
    turns = Counter()
    for index, row in enumerate(rows, start=1):
        errors.extend(validate_row(row, path, index))
        for message in row.get("messages", []):
            turns[message.get("role")] += 1

    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"{path}: expected {expected_count} rows, found {len(rows)}")

    return {
        "path": str(path),
        "rows": len(rows),
        "turns": dict(turns),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NeuroBait dataset JSONL files.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--expected-train", type=int, default=270)
    parser.add_argument("--expected-eval", type=int, default=30)
    args = parser.parse_args()

    results = [
        validate_file(args.train, args.expected_train),
        validate_file(args.eval, args.expected_eval),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))

    errors = [error for result in results for error in result["errors"]]
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
