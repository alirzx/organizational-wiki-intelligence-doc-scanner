from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_review(records: list[dict]) -> list[dict]:
    allowed = {"uncertain", "guard_rejected"}
    result = []
    for record in records:
        if record.get("decision") not in allowed:
            continue
        clean = {key: value for key, value in record.items() if "embedding" not in key.lower() and "vector" not in key.lower()}
        result.append(clean)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("input", type=Path); parser.add_argument("output", type=Path); args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in export_review(records)), encoding="utf-8")


if __name__ == "__main__": main()
