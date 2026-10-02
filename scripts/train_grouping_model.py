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
from app.text_processing.feature_extractor import feature_schema, extract_features
from app.text_processing.canonical import block_from_record
from app.text_processing.types import CandidatePair, CandidateReason


def _row_features(record: dict) -> list[float]:
    a = block_from_record(record['block_a'], record['document_id'])
    b = block_from_record(record['block_b'], record['document_id'])
    candidate = CandidatePair(record['pair_id'], a.block_id, b.block_id, a.page_number != b.page_number,
                              b.page_number-a.page_number,
                              tuple(CandidateReason(r) for r in record.get('candidate_reasons', ['manual'])))
    return list(extract_features(a, b, candidate, semantic_similarity=record.get('semantic_similarity')).values)


def compute_metrics(labels: list[int], predictions: list[int], scores: list[float], candidate_recall: float | None = None) -> tuple[dict, dict]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, average_precision_score, roc_auc_score
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    metrics = {
        "new_group_precision": float(precision[0]), "new_group_recall": float(recall[0]), "new_group_f1": float(f1[0]),
        "same_group_precision": float(precision[1]), "same_group_recall": float(recall[1]), "same_group_f1": float(f1[1]),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "pr_auc": float(average_precision_score(labels, scores)), "candidate_recall": candidate_recall,
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    matrix = {"labels": ["NEW_GROUP", "SAME_GROUP"], "matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}
    return metrics, matrix


def train(records: list[dict], output: Path, *, seed: int = 42) -> Path:
    from scripts.build_grouping_dataset import validate_record
    if any(validate_record(row) for row in records):
        raise ValueError('training rows violate the grouping pair contract')
    split_sources = {split: {row['source_id'] for row in records if row.get('split') == split}
                     for split in ('train', 'validation', 'test')}
    if any(split_sources[a] & split_sources[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('training, validation and test sources must be disjoint')
    split_documents = {split:{row['document_id'] for row in records if row.get('split') == split}
                       for split in ('train','validation','test')}
    if any(split_documents[a] & split_documents[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('training, validation and test documents must be disjoint')
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
    import numpy as np
    # Explicit missing-semantic examples make optional provider fallback a trained behavior.
    augmented = train_rows + [row | {'semantic_similarity':None} for row in train_rows if row.get('semantic_similarity') is not None]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, random_state=seed, class_weight="balanced", n_jobs=1, verbosity=-1)
    model.fit(np.asarray([_row_features(row) for row in augmented]), labels(augmented),
              eval_set=[(np.asarray([_row_features(row) for row in valid_rows]), labels(valid_rows))],
              callbacks=[lgb.early_stopping(25, verbose=False)])
    scores = model.predict_proba(np.asarray([_row_features(row) for row in valid_rows]))[:, 1].tolist()
    predictions = [int(score >= .75) for score in scores]
    metrics, matrix = compute_metrics(labels(valid_rows), predictions, scores)
    importance = io.StringIO(); writer = csv.writer(importance); writer.writerow(["feature", "gain", "split"])
    booster = model.booster_
    gains, splits = booster.feature_importance("gain"), booster.feature_importance("split")
    for name, gain, split in zip(schema.names, gains, splits, strict=True): writer.writerow([name, float(gain), int(split)])
    model_text = booster.model_to_string()
    dataset_digest = sha256("\n".join(json.dumps(r, sort_keys=True) for r in records).encode()).hexdigest()
    return write_model_package(
        output, model_text=model_text, schema=schema, metrics=metrics, confusion_matrix=matrix,
        feature_importance_csv=importance.getvalue(), package_id=f"grouping-{schema.sha256[:8]}-{dataset_digest[:12]}",
        training={"dataset_sha256": dataset_digest, "split_strategy": "document_holdout",
                  "train_documents": len({r['source_id'] for r in train_rows}),
                  "validation_documents": len({r['source_id'] for r in valid_rows}),
                  'input_granularity': 'canonical_text_blocks'},
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
