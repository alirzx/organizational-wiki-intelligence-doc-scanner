import json
from pathlib import Path
import shutil

from app.text_processing.classifiers.model_package import REPORT_FILES, validate_model_package, write_model_package
from app.text_processing.feature_extractor import feature_schema


def test_native_model_package_has_reports_checksums_and_contract():
    schema = feature_schema()
    tmp_path = Path(".artifacts/test-model-package")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    path = write_model_package(
        tmp_path / "package", model_text="tree\n", schema=schema,
        metrics={"same_group_precision": 1., "same_group_recall": 1., "new_group_precision": 1., "new_group_recall": 1., "macro_f1": 1., "candidate_recall": 1.},
        confusion_matrix={"matrix": [[1, 0], [0, 1]]}, feature_importance_csv="feature,gain,split\n",
        training={"dataset_sha256": "a" * 64, "split_strategy": "document_holdout", "train_documents": 2, "validation_documents": 1},
        semantic={"optional": True, "missing_semantic_trained": True, "provider": "ollama", "model": "embeddinggemma", "model_fingerprint": None, "dimensions": 768, "prompt_profile": "sentence_similarity_v1"},
        thresholds={"uncertain_lower": .55, "merge": .75}, package_id="pkg-test",
    )
    metadata = validate_model_package(path, schema)
    assert all((path / name).is_file() for name in REPORT_FILES)
    assert metadata["labels"] == {"NEW_GROUP": 0, "SAME_GROUP": 1}
    assert not list(path.glob("*.pkl")) and not list(path.glob("*.joblib"))
    shutil.rmtree(tmp_path)
