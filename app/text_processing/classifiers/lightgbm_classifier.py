from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from threading import RLock

from app.text_processing.classifiers.base import BlockRelationshipClassifier, GroupingModelIncompatible, GroupingModelUnavailable
from app.text_processing.relationship import prediction_from_score
from app.text_processing.types import CandidatePair, FeatureSchema, FeatureVector, RelationshipPrediction, TextBlock


class LightGBMRelationshipClassifier(BlockRelationshipClassifier):
    """Validated native LightGBM package loader. No pickle/joblib formats are accepted."""

    def __init__(self, package_dir: str | Path, schema: FeatureSchema, *, merge_threshold: float | None = None,
                 uncertain_lower: float | None = None, batch_size: int = 2048, num_threads: int = 1):
        self.path = Path(package_dir)
        self.schema = schema
        self._booster = None
        self.merge_override = merge_threshold
        self.uncertain_override = uncertain_lower
        self.batch_size = batch_size
        self.num_threads = num_threads
        self._lock = RLock()
        if batch_size < 1 or num_threads < 1:
            raise ValueError('inference batch size and thread count must be positive')
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
        self._thresholds()

    def _thresholds(self):
        packaged = self.manifest.get('thresholds',{})
        merge = getattr(self,'merge_override',None)
        lower = getattr(self,'uncertain_override',None)
        merge = packaged.get('merge',.75) if merge is None else merge
        lower = packaged.get('uncertain_lower',.55) if lower is None else lower
        if not 0 <= lower <= merge <= 1:
            raise GroupingModelIncompatible('resolved threshold order is invalid')
        return merge,lower

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
            try:
                self._booster = lgb.Booster(model_file=str(self.model_path))
            except Exception as exc:
                raise GroupingModelIncompatible('native model load failed') from exc
        return self._booster

    def predict(self, block_a: TextBlock, block_b: TextBlock, candidate: CandidatePair, features: FeatureVector, *, semantic_available: bool) -> RelationshipPrediction:
        features.validate(self.schema)
        return self.predict_many({block_a.block_id:block_a,block_b.block_id:block_b},[candidate],
                                 {candidate.pair_id:features},semantic_available=semantic_available)[0]

    def predict_many(self, blocks, candidates, features, *, semantic_available):
        if not candidates:
            return []
        import numpy as np
        merge,lower = self._thresholds()
        results = []
        lock = getattr(self,'_lock',None)
        from contextlib import nullcontext
        with lock if lock is not None else nullcontext():
            booster = self._load()
            for offset in range(0,len(candidates),getattr(self,'batch_size',2048)):
                batch = candidates[offset:offset+getattr(self,'batch_size',2048)]
                for candidate in batch:
                    features[candidate.pair_id].validate(self.schema)
                matrix = np.asarray([features[c.pair_id].values for c in batch],dtype=np.float64)
                try:
                    raw = np.asarray(booster.predict(matrix,num_threads=getattr(self,'num_threads',1)),dtype=float)
                except Exception as exc:
                    raise GroupingModelIncompatible('native prediction failed') from exc
                if raw.shape != (len(batch),) or not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():
                    raise GroupingModelIncompatible('invalid native prediction scores')
                for candidate,score in zip(batch,raw,strict=True):
                    available = bool(features[candidate.pair_id].values[self.schema.semantic_indices[1]])
                    results.append(prediction_from_score(candidate,float(score),merge_threshold=merge,uncertain_lower=lower,
                                                        semantic_available=available,model_package_id=self.package_id))
        return results
