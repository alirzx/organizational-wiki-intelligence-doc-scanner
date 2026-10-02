from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from app.text_processing.types import FeatureSchema


REPORT_FILES = ("model.txt", "metrics.json", "confusion-matrix.json", "feature-importance.csv")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_model_package(
    destination: str | Path, *, model_text: str, schema: FeatureSchema,
    metrics: dict, confusion_matrix: dict, feature_importance_csv: str,
    training: dict, semantic: dict, thresholds: dict, package_id: str,
    library_version: str = "unknown", seed: int = 42, hyperparameters: dict | None = None,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        # Native tree offsets are byte counts; Windows CRLF translation corrupts them.
        (temporary / "model.txt").write_text(model_text, encoding="utf-8", newline='\n')
        (temporary / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "confusion-matrix.json").write_text(json.dumps(confusion_matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "feature-importance.csv").write_text(feature_importance_csv, encoding="utf-8")
        checksums = {name: file_sha256(temporary / name) for name in REPORT_FILES}
        metadata = {
            "schema_version": "wiki-hami.grouping-model.v1", "package_id": package_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "classifier": {"family": "lightgbm", "library_version": library_version,
                           "format": "lightgbm-native-text", "seed": seed, "hyperparameters": hyperparameters or {}},
            "files": {"model": "model.txt", "metrics": "metrics.json", "confusion_matrix": "confusion-matrix.json",
                      "feature_importance": "feature-importance.csv", "sha256": checksums},
            "labels": {"NEW_GROUP": 0, "SAME_GROUP": 1},
            "feature_schema": {"version": schema.version, "sha256": schema.sha256,
                               "features": [item.__dict__ for item in schema.features]},
            "normalization": {"version": schema.normalizer_version, "config_sha256": sha256(schema.normalizer_version.encode()).hexdigest()},
            "semantic": semantic, "thresholds": thresholds, "training": training, "metrics": metrics,
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if destination.exists():
            backup = destination.with_name(destination.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            destination.replace(backup)
        temporary.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_model_package(path: str | Path, schema: FeatureSchema | None = None) -> dict:
    path = Path(path)
    manifest_path = path / "metadata.json"
    if not manifest_path.is_file():
        raise ValueError("metadata.json is missing")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("classifier", {}).get("format") != "lightgbm-native-text":
        raise ValueError("only native LightGBM text packages are accepted")
    if metadata.get("thresholds", {}).get("uncertain_lower", 1) > metadata.get("thresholds", {}).get("merge", 0):
        raise ValueError("threshold order is invalid")
    for name, expected in metadata.get("files", {}).get("sha256", {}).items():
        target = path / name
        if target.suffix.lower() in {".pkl", ".pickle", ".joblib"} or not target.is_file() or file_sha256(target) != expected:
            raise ValueError(f"invalid package file: {name}")
    if schema and metadata.get("feature_schema", {}).get("sha256") != schema.sha256:
        raise ValueError("feature schema mismatch")
    return metadata
