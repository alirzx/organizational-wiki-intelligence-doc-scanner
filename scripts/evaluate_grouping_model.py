from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_scenarios(rows: list[dict]) -> dict:
    scenarios: dict[str, list[dict]] = {}
    for row in rows: scenarios.setdefault(row.get("scenario", "unspecified"), []).append(row)
    details = {}
    for name, items in sorted(scenarios.items()):
        correct_membership = sum(item.get("predicted_group") == item.get("expected_group") for item in items)
        correct_type = sum(item.get("predicted_type") == item.get("expected_type") for item in items)
        incorrect_merges = sum(bool(item.get("incorrect_merge")) for item in items)
        fragmentation = sum(bool(item.get("fragmented")) for item in items)
        details[name] = {"membership_accuracy": correct_membership / len(items), "type_accuracy": correct_type / len(items),
                         "incorrect_merges": incorrect_merges, "fragmentation": fragmentation,
                         "passed": incorrect_merges == 0 and fragmentation == 0 and correct_membership == len(items)}
    return {"scenarios": details, "pass_rate": sum(v["passed"] for v in details.values()) / max(1, len(details))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("input", type=Path); parser.add_argument("output", type=Path); args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.write_text(json.dumps(evaluate_scenarios(rows), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
