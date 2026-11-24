from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from .deployment_training import (
    DEFAULT_DEPLOYMENT_MODEL_ID,
    DEFAULT_DEPLOYMENT_MODEL_VERSION,
)
from .features import extract_features
from .model_artifact import (
    DEFAULT_TRUSTED_MODELS_DIR,
    LoadedModelArtifact,
    ModelArtifactError,
    load_model_artifact,
)
from .schema import EventDataError, FEATURE_SCHEMA_VERSION
from .training_data import EXPECTED_FEATURE_NAMES


DEFAULT_DEPLOYED_MODEL_SHA256 = (
    "6a0d260db3d79c48eed76728fbf10caa092977a7ae57434e1379eaeaecd9e4ed"
)
DEFAULT_MAX_ADVISORY_EVENTS = 50_000
MAX_ADVISORY_EVENTS_HARD_CAP = 250_000

CSV_DOMAIN_LIMITATION_NOTE = (
    "Advisory inference extracts the exact 13-feature FEATURE_SCHEMA v1.0.0 from "
    "validated CSV evidence via backend.ml.features.extract_features(). Because the "
    "model was trained on synthetic AIT-LDS v2.1 Apache HTTP access logs, CSV logs "
    "from non-Apache or non-HTTP sources may experience domain shift; treat outputs "
    "strictly as read-only advisory signals alongside deterministic rules R-A through R-D."
)


class MLAdvisoryError(ValueError):
    """Structured fail-closed error for optional ML advisory inference."""

    def __init__(self, code: str, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def to_detail(self) -> dict[str, object]:
        return {
            "ml_advisory_error": self.code,
            "message": self.message,
            "deterministic_analysis_available": True,
        }


@dataclass(frozen=True)
class MLAdvisoryRuntimeConfig:
    """Server-side operator-controlled configuration for optional ML advisory inference."""

    enabled: bool
    operator_approved: bool
    trusted_root: Path
    model_id: str
    model_version: str
    expected_model_sha256: str
    max_events: int

    @property
    def artifact_dir(self) -> Path:
        return self.trusted_root / self.model_id / self.model_version


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime_config() -> MLAdvisoryRuntimeConfig:
    """Read server-side operator authorization and pinned model identity from environment."""
    enabled = _env_truthy("FORENSIC_INSIGHT_ML_ENABLED", default=False)
    operator_approved = _env_truthy("FORENSIC_INSIGHT_ML_OPERATOR_APPROVED", default=False)

    raw_root = os.environ.get("FORENSIC_INSIGHT_ML_MODELS_DIR", "").strip()
    trusted_root = Path(raw_root) if raw_root else DEFAULT_TRUSTED_MODELS_DIR

    model_id = (
        os.environ.get("FORENSIC_INSIGHT_ML_MODEL_ID", "").strip()
        or DEFAULT_DEPLOYMENT_MODEL_ID
    )
    model_version = (
        os.environ.get("FORENSIC_INSIGHT_ML_MODEL_VERSION", "").strip()
        or DEFAULT_DEPLOYMENT_MODEL_VERSION
    )
    expected_sha256 = (
        os.environ.get("FORENSIC_INSIGHT_ML_MODEL_SHA256", "").strip().lower()
        or DEFAULT_DEPLOYED_MODEL_SHA256
    )

    raw_max_events = os.environ.get("FORENSIC_INSIGHT_ML_MAX_EVENTS", "").strip()
    if raw_max_events:
        try:
            max_events = int(raw_max_events)
        except ValueError as exc:
            raise MLAdvisoryError(
                "invalid_server_config",
                "FORENSIC_INSIGHT_ML_MAX_EVENTS must be a positive integer.",
                status_code=503,
            ) from exc
    else:
        max_events = DEFAULT_MAX_ADVISORY_EVENTS

    if max_events <= 0 or max_events > MAX_ADVISORY_EVENTS_HARD_CAP:
        raise MLAdvisoryError(
            "invalid_server_config",
            f"FORENSIC_INSIGHT_ML_MAX_EVENTS must be between 1 and {MAX_ADVISORY_EVENTS_HARD_CAP}.",
            status_code=503,
        )

    return MLAdvisoryRuntimeConfig(
        enabled=enabled,
        operator_approved=operator_approved,
        trusted_root=trusted_root,
        model_id=model_id,
        model_version=model_version,
        expected_model_sha256=expected_sha256,
        max_events=max_events,
    )


class MLAdvisoryService:
    """Lazy, operator-authorized, hash-verified read-only ML inference service."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache_key: tuple[Path, str, str, str] | None = None
        self._cached_artifact: LoadedModelArtifact | None = None

    def clear_cache(self) -> None:
        with self._lock:
            self._cache_key = None
            self._cached_artifact = None

    def get_authorized_artifact(
        self,
        config: MLAdvisoryRuntimeConfig | None = None,
    ) -> LoadedModelArtifact:
        """Verify server-side operator approval and pinned model SHA-256 before returning model."""
        cfg = load_runtime_config() if config is None else config

        if not cfg.enabled:
            raise MLAdvisoryError(
                "ml_not_enabled",
                "Optional ML advisory inference is disabled by server configuration "
                "(set FORENSIC_INSIGHT_ML_ENABLED=true and FORENSIC_INSIGHT_ML_OPERATOR_APPROVED=true).",
                status_code=503,
            )
        if not cfg.operator_approved:
            raise MLAdvisoryError(
                "ml_not_approved",
                "Optional ML advisory inference requires explicit server-side operator approval "
                "(FORENSIC_INSIGHT_ML_OPERATOR_APPROVED=true).",
                status_code=503,
            )
        if not _is_sha256(cfg.expected_model_sha256):
            raise MLAdvisoryError(
                "invalid_server_config",
                "Pinned FORENSIC_INSIGHT_ML_MODEL_SHA256 must be a valid 64-character hex SHA-256.",
                status_code=503,
            )

        resolved_dir = cfg.artifact_dir.resolve()
        desired_key = (
            resolved_dir,
            cfg.model_id,
            cfg.model_version,
            cfg.expected_model_sha256,
        )

        with self._lock:
            if self._cached_artifact is not None and self._cache_key == desired_key:
                # Re-verify on-disk model integrity before serving from cache so post-load tampering fails closed.
                model_meta = self._cached_artifact.manifest.get("model_file")
                filename = (
                    str(model_meta.get("filename", ""))
                    if isinstance(model_meta, Mapping)
                    else ""
                )
                model_path = self._cached_artifact.artifact_dir / filename
                manifest_path = self._cached_artifact.artifact_dir / "artifact_manifest.json"
                if (
                    not manifest_path.is_file()
                    or not model_path.is_file()
                    or _sha256_file(model_path) != cfg.expected_model_sha256
                ):
                    self._cached_artifact = None
                    self._cache_key = None
                else:
                    return self._cached_artifact

            try:
                loaded = load_model_artifact(
                    cfg.artifact_dir,
                    trusted_root=cfg.trusted_root,
                    operator_approved=True,
                    expected_model_id=cfg.model_id,
                    expected_model_version=cfg.model_version,
                    expected_model_sha256=cfg.expected_model_sha256,
                )
            except ModelArtifactError as exc:
                raise MLAdvisoryError(
                    "model_unavailable_or_invalid",
                    f"Trusted ML model artifact failed verification: {exc}",
                    status_code=503,
                ) from exc

            self._cached_artifact = loaded
            self._cache_key = desired_key
            return loaded

    def analyze_verified_csv_evidence(
        self,
        *,
        case_id: int,
        verified_evidence: Sequence[tuple[int, str, Path, Path]],
        only_attack_associated: bool = False,
        config: MLAdvisoryRuntimeConfig | None = None,
    ) -> dict[str, object]:
        """Run bounded, read-only ML advisory inference over integrity-verified CSV working copies.

        verified_evidence items are (evidence_db_id, evidence_code, original_path, working_path).
        Original evidence files are never modified or read for feature extraction.
        """
        cfg = load_runtime_config() if config is None else config
        artifact = self.get_authorized_artifact(cfg)

        all_predictions: list[dict[str, object]] = []
        total_events_scored = 0
        total_attack_associated = 0
        total_normal = 0
        per_evidence_summary: list[dict[str, object]] = []

        for ev_db_id, ev_code, original_path, working_path in sorted(
            verified_evidence, key=lambda item: item[1]
        ):
            try:
                with open(working_path, "r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        raise MLAdvisoryError(
                            "unsupported_input",
                            f"Evidence {ev_code} has no CSV headers.",
                            status_code=422,
                        )
                    normalized_fieldnames = [name.strip() for name in reader.fieldnames]
                    rows: list[dict[str, str]] = []
                    for raw_row in reader:
                        total_events_scored += 1
                        if total_events_scored > cfg.max_events:
                            raise MLAdvisoryError(
                                "event_limit_exceeded",
                                f"ML advisory request exceeds the maximum limit of {cfg.max_events} events.",
                                status_code=413,
                            )
                        rows.append(
                            {
                                norm_key: (raw_row.get(orig_key) or "").strip()
                                for orig_key, norm_key in zip(
                                    reader.fieldnames, normalized_fieldnames
                                )
                            }
                        )
            except MLAdvisoryError:
                raise
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                raise MLAdvisoryError(
                    "unsupported_input",
                    f"Unable to read verified CSV working copy for {ev_code}: {exc}",
                    status_code=422,
                ) from exc

            if not rows:
                raise MLAdvisoryError(
                    "unsupported_input",
                    f"Evidence {ev_code} contains no event rows.",
                    status_code=422,
                )

            try:
                feature_records = extract_features(
                    rows,
                    case_id=case_id,
                    evidence_id=ev_code,
                    original_evidence_reference=str(original_path),
                )
            except EventDataError as exc:
                raise MLAdvisoryError(
                    "feature_extraction_failed",
                    f"Evidence {ev_code} could not be mapped to FEATURE_SCHEMA v{FEATURE_SCHEMA_VERSION}: {exc}",
                    status_code=422,
                ) from exc

            matrix = np.asarray(
                [
                    [
                        math.nan if rec.features[col] is None else float(rec.features[col])
                        for col in EXPECTED_FEATURE_NAMES
                    ]
                    for rec in feature_records
                ],
                dtype=np.float64,
            )

            try:
                batch_preds = artifact.predict_batch(matrix)
            except ModelArtifactError as exc:
                raise MLAdvisoryError(
                    "inference_failed",
                    f"ML advisory inference failed for {ev_code}: {exc}",
                    status_code=422,
                ) from exc

            ev_attack = 0
            ev_normal = 0
            for rec, pred in zip(feature_records, batch_preds):
                if pred.predicted_label == 1:
                    ev_attack += 1
                    total_attack_associated += 1
                else:
                    ev_normal += 1
                    total_normal += 1

                if only_attack_associated and pred.predicted_label != 1:
                    continue

                ev = rec.event
                all_predictions.append(
                    {
                        "model_id": pred.model_id,
                        "model_version": pred.model_version,
                        "model_family": pred.model_family,
                        "feature_schema_version": pred.feature_schema_version,
                        "decision_threshold": pred.decision_threshold,
                        "predicted_label": pred.predicted_label,
                        "predicted_label_name": pred.predicted_label_name,
                        "model_score": pred.model_score,
                        "score_type": "uncalibrated_model_score",
                        "is_calibrated_probability": pred.is_calibrated_probability,
                        "source_event_reference": {
                            "case_id": case_id,
                            "evidence_db_id": ev_db_id,
                            "evidence_id": ev.evidence_id,
                            "event_id": ev.event_id,
                            "timestamp": ev.timestamp.isoformat(),
                            "source_ip": ev.source_ip,
                            "event_type": ev.event_type,
                            "resource": ev.resource,
                            "response_status": ev.response_status,
                            "transfer_size": ev.transfer_size,
                        },
                    }
                )

            per_evidence_summary.append(
                {
                    "evidence_db_id": ev_db_id,
                    "evidence_id": ev_code,
                    "events_scored": len(feature_records),
                    "attack_associated_count": ev_attack,
                    "normal_count": ev_normal,
                }
            )

        manifest_limitations = [
            str(item) for item in artifact.manifest.get("limitations", [])
        ]
        combined_limitations = [*manifest_limitations, CSV_DOMAIN_LIMITATION_NOTE]
        score_semantics = dict(artifact.manifest.get("score_semantics", {}))

        return {
            "case_id": case_id,
            "advisory_only": True,
            "deterministic_rules_unchanged": True,
            "only_attack_associated": bool(only_attack_associated),
            "model": {
                "model_id": artifact.model_id,
                "model_version": artifact.model_version,
                "model_family": artifact.model_family,
                "model_sha256": artifact.model_file_sha256,
                "decision_threshold": artifact.decision_threshold,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "score_semantics": score_semantics,
                "intended_use": str(artifact.manifest.get("intended_use", "")),
                "limitations": combined_limitations,
            },
            "summary": {
                "evidence_items_scored": len(per_evidence_summary),
                "total_events_scored": total_events_scored,
                "attack_associated_count": total_attack_associated,
                "normal_count": total_normal,
                "returned_predictions_count": len(all_predictions),
                "per_evidence": per_evidence_summary,
            },
            "predictions": all_predictions,
            "limitations": combined_limitations,
        }


ml_advisory_service = MLAdvisoryService()
