from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from .baselines import (
    BINARY_FEATURE_NAMES,
    BaselineModelError,
    FittedDecisionTreeBaseline,
    FittedLogisticRegressionBaseline,
    FittedRandomForestBaseline,
    LOG1P_FEATURE_NAMES,
    WEIGHTING_REGIMES,
    _validate_feature_matrix,
)
from .dataset import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .schema import FEATURE_SCHEMA_VERSION
from .training_data import EXPECTED_FEATURE_NAMES, PROJECT_ROOT


MODEL_ARTIFACT_SCHEMA_VERSION = "1.0.0"
SERIALIZATION_FORMAT = "joblib_sklearn_pipeline_v1"
DEFAULT_TRUSTED_MODELS_DIR = PROJECT_ROOT / "models"
SUPPORTED_ARTIFACT_MODEL_FAMILIES: tuple[str, ...] = (
    "RandomForestClassifier",
    "DecisionTreeClassifier",
    "LogisticRegression",
)
SERIALIZATION_SECURITY_NOTICE = (
    "SHA-256 verification confirms file integrity against the manifest but does NOT "
    "make arbitrary pickle/joblib deserialization safe against untrusted inputs. "
    "Load ONLY operator-approved artifacts from a restricted trusted directory."
)

_FITTED_FAMILY_MAP = {
    FittedRandomForestBaseline: "RandomForestClassifier",
    FittedDecisionTreeBaseline: "DecisionTreeClassifier",
    FittedLogisticRegressionBaseline: "LogisticRegression",
}


class ModelArtifactError(ValueError):
    """Raised when a model artifact manifest, file, provenance, or inference input is invalid."""


@dataclass(frozen=True)
class InferencePrediction:
    """Deterministic per-event prediction distinguishing uncalibrated model score from probability."""

    model_id: str
    model_version: str
    model_family: str
    feature_schema_version: str
    decision_threshold: float
    predicted_label: int
    predicted_label_name: str
    model_score: float
    is_calibrated_probability: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_family": self.model_family,
            "feature_schema_version": self.feature_schema_version,
            "decision_threshold": self.decision_threshold,
            "predicted_label": self.predicted_label,
            "predicted_label_name": self.predicted_label_name,
            "model_score": self.model_score,
            "is_calibrated_probability": self.is_calibrated_probability,
        }


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Verified, read-only model artifact and deterministic inference engine."""

    manifest: Mapping[str, object]
    pipeline: Pipeline
    artifact_dir: Path
    model_file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def model_id(self) -> str:
        return str(self.manifest["model_id"])

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def model_family(self) -> str:
        return str(self.manifest["model_family"])

    @property
    def decision_threshold(self) -> float:
        return float(self.manifest["decision_threshold"])

    @property
    def is_calibrated_probability(self) -> bool:
        return bool(self.manifest["score_semantics"]["calibrated_probability"])

    def predict_batch(
        self,
        features: Sequence[Sequence[Any]] | np.ndarray,
    ) -> tuple[InferencePrediction, ...]:
        """Run deterministic batch inference using the frozen preprocessing pipeline and threshold."""
        try:
            matrix = _validate_feature_matrix(features)
        except BaselineModelError as exc:
            raise ModelArtifactError(str(exc)) from exc

        proba = self.pipeline.predict_proba(matrix)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        if classes != [0, 1]:
            raise ModelArtifactError(f"Unexpected classifier class ordering: {classes!r}")
        scores = np.asarray(proba[:, 1], dtype=np.float64)
        if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
            raise ModelArtifactError("Model produced non-finite or out-of-range scores.")

        threshold = self.decision_threshold
        calibrated = self.is_calibrated_probability
        predictions: list[InferencePrediction] = []
        for score_val in scores.tolist():
            pred_label = 1 if score_val >= threshold else 0
            predictions.append(
                InferencePrediction(
                    model_id=self.model_id,
                    model_version=self.model_version,
                    model_family=self.model_family,
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    decision_threshold=threshold,
                    predicted_label=pred_label,
                    predicted_label_name="attack_associated" if pred_label == 1 else "normal",
                    model_score=float(score_val),
                    is_calibrated_probability=calibrated,
                )
            )
        return tuple(predictions)

    def predict_one(
        self,
        features: Sequence[Any] | Mapping[str, Any],
    ) -> InferencePrediction:
        """Run deterministic single-event inference from an ordered 13-vector or feature mapping."""
        if isinstance(features, Mapping):
            actual_keys = set(features)
            expected_keys = set(EXPECTED_FEATURE_NAMES)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                unexpected = sorted(actual_keys - expected_keys)
                raise ModelArtifactError(
                    f"Feature mapping schema mismatch (missing={missing}, unexpected={unexpected})."
                )
            ordered_row = [features[name] for name in EXPECTED_FEATURE_NAMES]
            return self.predict_batch([ordered_row])[0]
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
            raise ModelArtifactError("features must be an ordered 13-element sequence or feature mapping.")
        return self.predict_batch([features])[0]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelArtifactError(f"Unable to read file for SHA-256 verification: {path}") from exc
    return digest.hexdigest()


def _safe_basename(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or "\\" in value
        or value in (".", "..")
    ):
        raise ModelArtifactError(f"Invalid {description}: {value!r}")
    return value


def _validate_threshold(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not (0.0 <= float(value) <= 1.0)
    ):
        raise ModelArtifactError(f"Invalid decision_threshold: {value!r}")
    return float(value)


def _validate_trusted_dir(artifact_dir: str | Path, trusted_root: str | Path) -> Path:
    raw_dir = Path(artifact_dir)
    if raw_dir.is_symlink():
        raise ModelArtifactError(f"Symlink artifact directories are prohibited: {raw_dir}")
    resolved_root = Path(trusted_root).resolve()
    resolved_dir = raw_dir.resolve()
    if not resolved_dir.is_relative_to(resolved_root) or resolved_dir == resolved_root:
        raise ModelArtifactError(
            f"Artifact directory {resolved_dir} must reside strictly inside trusted root {resolved_root}."
        )
    for forbidden in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
        resolved_forbidden = forbidden.resolve()
        if resolved_dir.is_relative_to(resolved_forbidden):
            raise ModelArtifactError(
                f"Artifact directory must not reside inside dataset directory {resolved_forbidden}."
            )
    return resolved_dir


def _preprocessing_spec_for_family(model_family: str) -> dict[str, object]:
    if model_family == "LogisticRegression":
        return {
            "profile": "linear_models",
            "imputation": "median",
            "apply_log1p": True,
            "log1p_features": list(LOG1P_FEATURE_NAMES),
            "binary_indicator_features": list(BINARY_FEATURE_NAMES),
            "scaler": "StandardScaler",
            "fit_scope": "training_data_only",
        }
    return {
        "profile": "tree_models",
        "imputation": "median",
        "apply_log1p": False,
        "scaler": None,
        "fit_scope": "training_data_only",
    }


def _validate_training_provenance(provenance: object) -> dict[str, object]:
    if not isinstance(provenance, Mapping):
        raise ModelArtifactError("training_dataset_provenance must be a mapping.")
    dataset_name = provenance.get("dataset_name")
    dataset_version = provenance.get("dataset_version")
    scenarios = provenance.get("scenarios")
    scenario_hashes = provenance.get("scenario_labelled_sha256")
    if (
        not isinstance(dataset_name, str)
        or not dataset_name.strip()
        or not isinstance(dataset_version, str)
        or not dataset_version.strip()
        or not isinstance(scenarios, list)
        or not scenarios
        or not isinstance(scenario_hashes, dict)
        or set(scenario_hashes) != set(scenarios)
        or any(not _is_sha256(h) for h in scenario_hashes.values())
    ):
        raise ModelArtifactError("Incomplete or invalid training_dataset_provenance.")
    if provenance.get("unknown_records_excluded") is not True:
        raise ModelArtifactError("training_dataset_provenance must confirm unknown_records_excluded=true.")
    if provenance.get("benchmark_outer_test_predictions_excluded") is not True:
        raise ModelArtifactError(
            "training_dataset_provenance must confirm benchmark_outer_test_predictions_excluded=true."
        )
    return dict(provenance)


def save_model_artifact(
    fitted_model: (
        FittedRandomForestBaseline
        | FittedDecisionTreeBaseline
        | FittedLogisticRegressionBaseline
    ),
    artifact_dir: str | Path,
    *,
    trusted_root: str | Path = DEFAULT_TRUSTED_MODELS_DIR,
    model_id: str,
    model_version: str,
    decision_threshold: float,
    training_dataset_provenance: Mapping[str, object],
    training_config_sha256: str,
    intended_use: str,
    limitations: Sequence[str],
    created_at_utc: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Atomically save a fitted baseline pipeline and versioned provenance manifest."""
    model_family = _FITTED_FAMILY_MAP.get(type(fitted_model))
    if model_family is None or model_family not in SUPPORTED_ARTIFACT_MODEL_FAMILIES:
        raise ModelArtifactError(f"Unsupported fitted model type: {type(fitted_model).__name__}")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelArtifactError("model_id must be a non-empty string.")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ModelArtifactError("model_version must be a non-empty string.")
    threshold = _validate_threshold(decision_threshold)
    if not _is_sha256(training_config_sha256):
        raise ModelArtifactError("training_config_sha256 must be a 64-character hex SHA-256.")
    if not isinstance(intended_use, str) or not intended_use.strip():
        raise ModelArtifactError("intended_use must be a non-empty string.")
    if (
        not isinstance(limitations, Sequence)
        or isinstance(limitations, (str, bytes))
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise ModelArtifactError("limitations must be a non-empty sequence of strings.")
    if fitted_model.weighting_regime not in WEIGHTING_REGIMES:
        raise ModelArtifactError(f"Unsupported weighting_regime: {fitted_model.weighting_regime!r}")
    if fitted_model.pipeline.named_steps["classifier"].class_weight is not None:
        raise ModelArtifactError("Estimator class_weight must be None.")

    validated_provenance = _validate_training_provenance(training_dataset_provenance)
    resolved_dir = _validate_trusted_dir(artifact_dir, trusted_root)

    manifest_path = resolved_dir / "artifact_manifest.json"
    model_filename = "model.joblib"
    model_path = resolved_dir / model_filename
    if manifest_path.exists() or model_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing model artifact in {resolved_dir}."
        )

    resolved_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_model_name = tempfile.mkstemp(
        prefix=".model.", suffix=".tmp", dir=resolved_dir
    )
    os.close(fd)
    tmp_model_path = Path(tmp_model_name)
    try:
        joblib.dump(fitted_model.pipeline, tmp_model_path)
        os.replace(tmp_model_path, model_path)
    finally:
        if tmp_model_path.exists():
            tmp_model_path.unlink()

    model_sha256 = _sha256_file(model_path)
    model_size = model_path.stat().st_size
    timestamp_utc = (
        datetime.now(timezone.utc).isoformat()
        if created_at_utc is None
        else str(created_at_utc)
    )

    manifest: dict[str, object] = {
        "artifact_schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id.strip(),
        "model_version": model_version.strip(),
        "model_family": model_family,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ordered_feature_names": list(EXPECTED_FEATURE_NAMES),
        "preprocessing": _preprocessing_spec_for_family(model_family),
        "hyperparameters": dict(fitted_model.hyperparameters),
        "weighting_regime": fitted_model.weighting_regime,
        "random_state": fitted_model.random_state,
        "decision_threshold": threshold,
        "score_semantics": {
            "type": "uncalibrated_model_score",
            "calibrated_probability": False,
            "note": (
                "Positive-class (1 = attack_associated) score from estimator predict_proba; "
                "not a calibrated probability unless independently verified."
            ),
        },
        "training_dataset_provenance": validated_provenance,
        "training_config_sha256": training_config_sha256.lower(),
        "model_file": {
            "filename": model_filename,
            "sha256": model_sha256,
            "size_bytes": model_size,
            "serialization_format": SERIALIZATION_FORMAT,
        },
        "creation_metadata": {
            "created_at_utc": timestamp_utc,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "intended_use": intended_use.strip(),
        "limitations": [item.strip() for item in limitations],
        "security_notice": SERIALIZATION_SECURITY_NOTICE,
    }

    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd_m, tmp_manifest_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=resolved_dir
    )
    tmp_manifest_path = Path(tmp_manifest_name)
    try:
        with os.fdopen(fd_m, "wb") as handle:
            handle.write(manifest_bytes)
        os.replace(tmp_manifest_path, manifest_path)
    finally:
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink()

    return manifest_path, manifest


def load_model_artifact(
    artifact_dir: str | Path,
    *,
    trusted_root: str | Path = DEFAULT_TRUSTED_MODELS_DIR,
    operator_approved: bool = False,
    expected_model_id: str | None = None,
    expected_model_version: str | None = None,
    expected_model_sha256: str | None = None,
) -> LoadedModelArtifact:
    """Fail-closed loader for operator-approved, hash-verified trusted model artifacts."""
    if operator_approved is not True:
        raise ModelArtifactError(
            "Explicit operator_approved=True is required before deserializing a model artifact."
        )
    resolved_dir = _validate_trusted_dir(artifact_dir, trusted_root)
    manifest_path = resolved_dir / "artifact_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ModelArtifactError(f"Missing or invalid artifact manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"Corrupted artifact manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ModelArtifactError("Artifact manifest must be a JSON object.")

    if manifest.get("artifact_schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError(
            f"Unsupported artifact_schema_version: {manifest.get('artifact_schema_version')!r}"
        )
    model_id = manifest.get("model_id")
    model_version = manifest.get("model_version")
    model_family = manifest.get("model_family")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelArtifactError("Artifact manifest is missing model_id.")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ModelArtifactError("Artifact manifest is missing model_version.")
    if expected_model_id is not None and model_id != expected_model_id:
        raise ModelArtifactError(
            f"Model ID mismatch: manifest has {model_id!r}, expected {expected_model_id!r}."
        )
    if expected_model_version is not None and model_version != expected_model_version:
        raise ModelArtifactError(
            f"Model version mismatch: manifest has {model_version!r}, expected {expected_model_version!r}."
        )
    if model_family not in SUPPORTED_ARTIFACT_MODEL_FAMILIES:
        raise ModelArtifactError(f"Unsupported model_family: {model_family!r}")

    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ModelArtifactError(
            f"Unsupported feature_schema_version: {manifest.get('feature_schema_version')!r}"
        )
    if manifest.get("ordered_feature_names") != list(EXPECTED_FEATURE_NAMES):
        raise ModelArtifactError("Artifact ordered_feature_names do not match FEATURE_SCHEMA v1.0.0.")

    _validate_threshold(manifest.get("decision_threshold"))
    _validate_training_provenance(manifest.get("training_dataset_provenance"))
    if not _is_sha256(manifest.get("training_config_sha256")):
        raise ModelArtifactError("Artifact manifest has invalid training_config_sha256.")

    score_semantics = manifest.get("score_semantics")
    if not isinstance(score_semantics, dict) or not isinstance(
        score_semantics.get("calibrated_probability"), bool
    ):
        raise ModelArtifactError("Artifact manifest is missing score_semantics.")
    if not isinstance(manifest.get("intended_use"), str) or not manifest["intended_use"].strip():
        raise ModelArtifactError("Artifact manifest is missing intended_use.")
    if not isinstance(manifest.get("limitations"), list) or not manifest["limitations"]:
        raise ModelArtifactError("Artifact manifest is missing limitations.")
    if not isinstance(manifest.get("creation_metadata"), dict):
        raise ModelArtifactError("Artifact manifest is missing creation_metadata.")

    model_file_meta = manifest.get("model_file")
    if not isinstance(model_file_meta, dict):
        raise ModelArtifactError("Artifact manifest is missing model_file metadata.")
    if model_file_meta.get("serialization_format") != SERIALIZATION_FORMAT:
        raise ModelArtifactError(
            f"Unsupported serialization_format: {model_file_meta.get('serialization_format')!r}"
        )
    model_filename = _safe_basename(model_file_meta.get("filename"), "model filename")
    expected_sha = model_file_meta.get("sha256")
    if not _is_sha256(expected_sha):
        raise ModelArtifactError("Artifact manifest has invalid model_file.sha256.")
    if expected_model_sha256 is not None and expected_sha != expected_model_sha256.lower():
        raise ModelArtifactError("Manifest model SHA-256 does not match expected_model_sha256.")

    model_path = (resolved_dir / model_filename).resolve()
    if model_path.parent != resolved_dir or model_path.is_symlink() or not model_path.is_file():
        raise ModelArtifactError(f"Missing or unsafe model file: {model_path}")
    actual_size = model_path.stat().st_size
    if model_file_meta.get("size_bytes") != actual_size:
        raise ModelArtifactError(
            f"Model file size mismatch: expected {model_file_meta.get('size_bytes')}, got {actual_size}."
        )
    actual_sha = _sha256_file(model_path)
    if actual_sha != expected_sha:
        raise ModelArtifactError("Model file SHA-256 mismatch against artifact manifest.")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ModelArtifactError(f"Failed to deserialize verified model file: {exc}") from exc

    if not isinstance(pipeline, Pipeline) or "classifier" not in pipeline.named_steps:
        raise ModelArtifactError("Deserialized artifact is not a valid scikit-learn Pipeline.")
    classifier = pipeline.named_steps["classifier"]
    if type(classifier).__name__ != model_family:
        raise ModelArtifactError(
            f"Deserialized classifier {type(classifier).__name__!r} does not match manifest {model_family!r}."
        )
    if getattr(classifier, "class_weight", None) is not None:
        raise ModelArtifactError("Deserialized classifier must have class_weight=None.")
    if list(getattr(classifier, "classes_", [])) != [0, 1]:
        raise ModelArtifactError("Deserialized classifier must be fitted on binary classes [0, 1].")

    return LoadedModelArtifact(
        manifest=manifest,
        pipeline=pipeline,
        artifact_dir=resolved_dir,
        model_file_sha256=actual_sha,
    )
