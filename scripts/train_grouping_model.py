from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.text_processing.classifiers.model_package import write_model_package
from app.text_processing.feature_extractor import feature_schema


def _row_features(record: dict) -> list[float]:
    a, b = record["block_a"], record["block_b"]
    box_a, box_b = a["bbox"], b["bbox"]
    raw = [
        abs(box_b["y1"] - box_a["y2"]) / max(1, a["page_height"]),
        abs(box_b["x1"] - box_a["x1"]) / max(1, a["page_width"]),
        len(a["text"]), len(b["text"]),
        float(a["page_number"] == b["page_number"]),
        float(a.get("column_id") == b.get("column_id") and a.get("column_id") is not None),
    ]
    size = len(feature_schema().features)
    return (raw + [0.0] * size)[:size]


def compute_metrics(labels: list[int], predictions: list[int], scores: list[float], candidate_recall: float = 1.0) -> tuple[dict, dict]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, average_precision_score, roc_auc_score
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    metrics = {
        "new_group_precision": float(precision[0]), "new_group_recall": float(recall[0]), "new_group_f1": float(f1[0]),
        "same_group_precision": float(precision[1]), "same_group_recall": float(recall[1]), "same_group_f1": float(f1[1]),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "pr_auc": float(average_precision_score(labels, scores)), "candidate_recall": float(candidate_recall),
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    matrix = {"labels": ["NEW_GROUP", "SAME_GROUP"], "matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}
    return metrics, matrix


def train(records: list[dict], output: Path, *, seed: int = 42) -> Path:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("install requirements-grouping.txt to train") from exc
    train_rows = [item for item in records if item.get("split") == "train"]
    valid_rows = [item for item in records if item.get("split") == "validation"]
    if not train_rows or not valid_rows:
        raise ValueError("document-disjoint train and validation rows are required")
    labels = lambda rows: [1 if row["label"] == "SAME_GROUP" else 0 for row in rows]
    schema = feature_schema()
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, random_state=seed, class_weight="balanced")
    model.fit([_row_features(row) for row in train_rows], labels(train_rows),
              eval_set=[([_row_features(row) for row in valid_rows], labels(valid_rows))],
              callbacks=[lgb.early_stopping(25, verbose=False)])
    scores = model.predict_proba([_row_features(row) for row in valid_rows])[:, 1].tolist()
    predictions = [int(score >= .5) for score in scores]
    metrics, matrix = compute_metrics(labels(valid_rows), predictions, scores)
    importance = io.StringIO(); writer = csv.writer(importance); writer.writerow(["feature", "gain", "split"])
    booster = model.booster_
    gains, splits = booster.feature_importance("gain"), booster.feature_importance("split")
    for name, gain, split in zip(schema.names, gains, splits, strict=True): writer.writerow([name, float(gain), int(split)])
    model_file = output.parent / ".grouping-model.tmp.txt"; booster.save_model(str(model_file))
    model_text = model_file.read_text(encoding="utf-8"); model_file.unlink()
    sources = sorted({row["source_id"] for row in records})
    dataset_digest = sha256("\n".join(json.dumps(r, sort_keys=True) for r in records).encode()).hexdigest()
    return write_model_package(
        output, model_text=model_text, schema=schema, metrics=metrics, confusion_matrix=matrix,
        feature_importance_csv=importance.getvalue(), package_id=f"grouping-{dataset_digest[:12]}",
        training={"dataset_sha256": dataset_digest, "split_strategy": "document_holdout",
                  "train_documents": len({r['source_id'] for r in train_rows}),
                  "validation_documents": len({r['source_id'] for r in valid_rows})},
        semantic={"optional": True, "missing_semantic_trained": True, "provider": "ollama", "model": "embeddinggemma",
                  "model_fingerprint": None, "dimensions": 768, "prompt_profile": "sentence_similarity_v1"},
        thresholds={"uncertain_lower": .55, "merge": .75, "selection_metric": "macro_f1"},
        library_version=lgb.__version__, seed=seed, hyperparameters=model.get_params(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset_pos", type=Path, nargs="?"); parser.add_argument("output_pos", type=Path, nargs="?")
    parser.add_argument("--dataset", dest="dataset_opt", type=Path); parser.add_argument("--output-dir", dest="output_opt", type=Path); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); dataset, output = args.dataset_opt or args.dataset_pos, args.output_opt or args.output_pos
    if dataset is None or output is None: parser.error("dataset and output directory are required")
    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    train(records, output, seed=args.seed)


if __name__ == "__main__": main()
