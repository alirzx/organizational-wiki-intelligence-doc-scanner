from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from app.text_processing.classifiers.base import BlockRelationshipClassifier, GroupingModelIncompatible, GroupingModelUnavailable
from app.text_processing.relationship import prediction_from_score
from app.text_processing.types import CandidatePair, FeatureSchema, FeatureVector, RelationshipPrediction, TextBlock


class LightGBMRelationshipClassifier(BlockRelationshipClassifier):
    """Validated native LightGBM package loader. No pickle/joblib formats are accepted."""

    def __init__(self, package_dir: str | Path, schema: FeatureSchema):
        self.path = Path(package_dir)
        self.schema = schema
        self._booster = None
        if self.path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
            raise GroupingModelIncompatible("serialized Python model formats are forbidden")
        manifest_path = self.path / "metadata.json"
        legacy = False
        if not manifest_path.is_file() and (self.path / "manifest.json").is_file():
            manifest_path = self.path / "manifest.json"
            legacy = True
        if not manifest_path.is_file():
            raise GroupingModelUnavailable(f"missing model manifest: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GroupingModelIncompatible("invalid model manifest") from exc
        self._legacy = legacy
        self._validate_package()

    @property
    def package_id(self) -> str:
        return str(self.manifest["package_id"])

    def _validate_package(self) -> None:
        if not self._legacy:
            from app.text_processing.classifiers.model_package import validate_model_package
            try:
                validate_model_package(self.path, self.schema)
            except (OSError, ValueError) as exc:
                raise GroupingModelIncompatible(str(exc)) from exc
            classifier = self.manifest.get("classifier", {})
            if classifier.get("format") != "lightgbm-native-text":
                raise GroupingModelIncompatible("only native LightGBM text models are accepted")
            labels = self.manifest.get("labels", {})
            if labels != {"NEW_GROUP": 0, "SAME_GROUP": 1}:
                raise GroupingModelIncompatible("label mapping is incompatible")
            semantic = self.manifest.get("semantic", {})
            if semantic.get("optional") and not semantic.get("missing_semantic_trained"):
                raise GroupingModelIncompatible("package was not trained for missing semantics")
            self.model_path = self.path / self.manifest["files"]["model"]
            return
        required = {"package_id", "model_file", "model_sha256", "feature_schema_hash", "labels", "normalizer_version"}
        if not required <= self.manifest.keys():
            raise GroupingModelIncompatible("manifest is missing required fields")
        if self.manifest["feature_schema_hash"] != self.schema.sha256:
            raise GroupingModelIncompatible("feature schema hash mismatch")
        if self.manifest["normalizer_version"] != self.schema.normalizer_version:
            raise GroupingModelIncompatible("normalizer version mismatch")
        if self.manifest["labels"] != ["NEW_GROUP", "SAME_GROUP"]:
            raise GroupingModelIncompatible("label order is incompatible")
        model_path = self.path / self.manifest["model_file"]
        if model_path.suffix.lower() not in {".txt", ".json"} or not model_path.is_file():
            raise GroupingModelIncompatible("native LightGBM model file is missing")
        digest = sha256(model_path.read_bytes()).hexdigest()
        if digest != self.manifest["model_sha256"]:
            raise GroupingModelIncompatible("model checksum mismatch")
        self.model_path = model_path

    def _load(self):
        if self._booster is None:
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise GroupingModelUnavailable("install the grouping optional dependency") from exc
            self._booster = lgb.Booster(model_file=str(self.model_path))
        return self._booster

    def predict(self, block_a: TextBlock, block_b: TextBlock, candidate: CandidatePair, features: FeatureVector, *, semantic_available: bool) -> RelationshipPrediction:
        features.validate(self.schema)
        raw = self._load().predict([list(features.values)])
        score = float(raw[0])
        return prediction_from_score(candidate, score, semantic_available=semantic_available, model_package_id=self.package_id)
