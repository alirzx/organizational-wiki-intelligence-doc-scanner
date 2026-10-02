from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


ALLOWED_LABELS = {"SAME_GROUP", "NEW_GROUP"}
ALLOWED_SPLITS = {"train", "validation", "test", "unassigned"}
from app.text_processing.types import CandidateReason
ALLOWED_REASONS = {reason.value for reason in CandidateReason}


def validate_record(record: dict) -> list[str]:
    if not isinstance(record,dict):
        return ['invalid:record']
    errors: list[str] = []
    for key in ("pair_id", "document_id", "source_id", "block_a", "block_b", "label"):
        if key not in record:
            errors.append(f"missing:{key}")
    for key in ('pair_id','document_id','source_id'):
        if not isinstance(record.get(key),str) or not record[key]:
            errors.append(f'invalid:{key}')
    if record.get("label") not in ALLOWED_LABELS:
        errors.append("invalid:label")
    if record.get("split", "unassigned") not in ALLOWED_SPLITS:
        errors.append("invalid:split")
    similarity = record.get('semantic_similarity')
    if similarity is not None and (not isinstance(similarity,(int,float)) or not math.isfinite(similarity) or not -1 <= similarity <= 1):
        errors.append('invalid:semantic_similarity')
    reasons = record.get('candidate_reasons',['manual'])
    if not isinstance(reasons,list) or not reasons or any(not isinstance(r,str) or r not in ALLOWED_REASONS for r in reasons):
        errors.append("invalid:candidate_reasons")
    for name in ("block_a", "block_b"):
        block = record.get(name, {})
        if not isinstance(block,dict):
            errors.append(f'invalid:{name}'); continue
        if not all(isinstance(block.get(key),int) for key in ('page_number','content_ordinal','page_width','page_height')):
            errors.append(f'invalid:{name}:dimensions'); continue
        if not isinstance(block.get('text'),str) or not isinstance(block.get('page_id'),str) or not block.get('page_id'):
            errors.append(f'invalid:{name}:text_or_page')
        if block.get("page_number", 0) < 1 or block.get("content_ordinal", -1) < 0:
            errors.append(f"invalid:{name}:order")
        if block.get("page_width", 0) < 1 or block.get("page_height", 0) < 1:
            errors.append(f"invalid:{name}:dimensions")
        confidence = block.get("confidence")
        if confidence is not None and (not isinstance(confidence,(int,float)) or not 0 <= confidence <= 1):
            errors.append(f"invalid:{name}:confidence")
        bbox = block.get("bbox", {})
        if not isinstance(bbox,dict) or any(not isinstance(bbox.get(k),(int,float)) or not math.isfinite(bbox[k]) for k in ('x1','y1','x2','y2')):
            errors.append(f'invalid:{name}:bbox'); continue
        if bbox.get("x2", 0) <= bbox.get("x1", 0) or bbox.get("y2", 0) <= bbox.get("y1", 0):
            errors.append(f"invalid:{name}:bbox")
    return errors


def assign_document_splits(records: list[dict], validation_ratio: float = .2, seed: int = 42) -> list[dict]:
    documents = sorted({record["source_id"] for record in records})
    random.Random(seed).shuffle(documents)
    validation_count = max(1, round(len(documents) * validation_ratio)) if len(documents) > 1 else 0
    validation = set(documents[:validation_count])
    return [record | {"split": "validation" if record["source_id"] in validation else "train"} for record in records]


def build_dataset(records: list[dict], *, validation_ratio: float = .2, seed: int = 42) -> tuple[list[dict], dict]:
    seen: dict[str, str] = {}
    errors: list[str] = []
    accepted: list[dict] = []
    for index, record in enumerate(records, 1):
        record = dict(record)
        if not record.get("pair_id") and record.get("block_a") and record.get("block_b"):
            raw = f'{record.get("document_id")}:{record["block_a"].get("block_id")}:{record["block_b"].get("block_id")}'
            record["pair_id"] = "pair_" + sha256(raw.encode()).hexdigest()[:24]
        validation = validate_record(record)
        if validation:
            errors.extend(f"line {index}:{item}" for item in validation); continue
        previous = seen.get(record["pair_id"])
        if previous and previous != record["label"]:
            errors.append(f"line {index}:conflict:{record['pair_id']}"); continue
        if previous:
            errors.append(f"line {index}:duplicate:{record['pair_id']}"); continue
        seen[record["pair_id"]] = record["label"]
        accepted.append(record)
    split = assign_document_splits(accepted, validation_ratio, seed)
    summary = {"records": len(split), "errors": errors, "labels": dict(Counter(item["label"] for item in split)),
               "splits": dict(Counter(item["split"] for item in split)),
               "documents": len({item["source_id"] for item in split})}
    return split, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pos", type=Path, nargs="?"); parser.add_argument("output_pos", type=Path, nargs="?")
    parser.add_argument("--input", dest="input_opt", type=Path); parser.add_argument("--output", dest="output_opt", type=Path)
    parser.add_argument("--summary", type=Path); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    input_path, output_path = args.input_opt or args.input_pos, args.output_opt or args.output_pos
    if input_path is None or output_path is None: parser.error("input and output are required")
    input_files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    records = [json.loads(line) for path in input_files for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset, summary = build_dataset(records, seed=args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in dataset), encoding="utf-8")
    (args.summary or output_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
