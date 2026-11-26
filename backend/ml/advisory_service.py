from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import ipaddress
import math
import os
from pathlib import Path
import re
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
    "validated Apache-style HTTP CSV evidence via backend.ml.features.extract_features(). "
    "Evidence outside the validated Apache HTTP access-log domain is rejected before inference; "
    "treat advisory outputs strictly as read-only signals alongside deterministic rules R-A through R-D."
)

SUPPORTED_HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS", "PATCH", "TRACE", "CONNECT"}
)
_UNSUPPORTED_EVENT_TYPE_PREFIXES: tuple[str, ...] = (
    "PROCESS_",
    "FILE_",
    "USB_",
    "NETWORK_",
    "FIREWALL_",
    "REGISTRY_",
    "AUTH_",
    "LOGIN",
    "LOGOUT",
    "SSH_",
    "DNS_",
    "KERNEL_",
    "SERVICE_",
    "SYSLOG",
    "USER_",
    "DISK_",
    "DEVICE_",
)
_WINDOWS_OR_DEVICE_RESOURCE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|USB[\\/])")


def _is_valid_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _is_valid_http_status(value: str) -> bool:
    return len(value) == 3 and value.isascii() and value.isdigit() and 100 <= int(value) <= 599


def _is_valid_http_resource(value: str) -> bool:
    if not value or any(ch.isspace() for ch in value):
        return False
    if _WINDOWS_OR_DEVICE_RESOURCE_RE.match(value):
        return False
    return (
        value == "*"
        or value.startswith("/")
        or value.startswith("http://")
        or value.startswith("https://")
    )


def _is_explicitly_unsupported_event(event_type: str, resource: str) -> bool:
    upper_type = event_type.upper()
    if any(upper_type.startswith(prefix) for prefix in _UNSUPPORTED_EVENT_TYPE_PREFIXES):
        return True
    if _WINDOWS_OR_DEVICE_RESOURCE_RE.match(resource):
        return True
    return False


def validate_apache_http_domain_compatibility(
    evidence_code: str,
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Conservatively verify that all CSV rows conform to Apache-style HTTP access semantics.

    Fails closed before ML inference for clearly unsupported host/OS/USB/network telemetry,
    mixed-domain files, or ambiguous/unverifiable event schemas.
    """
    supported_count = 0
    unsupported_count = 0
    ambiguous_count = 0
    unsupported_event_types: set[str] = set()
    ambiguous_reasons: set[str] = set()

    for row in rows:
        event_type = (row.get("event_type") or row.get("event") or "").strip()
        resource = (row.get("resource") or row.get("description") or "").strip()
        source_ip = (row.get("source_ip") or row.get("source") or "").strip()
        status = (row.get("status") or "").strip()

        if _is_explicitly_unsupported_event(event_type, resource):
            unsupported_count += 1
            if event_type:
                unsupported_event_types.add(event_type[:40])
            continue

        method_ok = event_type.upper() in SUPPORTED_HTTP_METHODS
        status_ok = _is_valid_http_status(status)
        resource_ok = _is_valid_http_resource(resource)
        ip_ok = _is_valid_ip_address(source_ip)

        if method_ok and status_ok and resource_ok and ip_ok:
            supported_count += 1
        else:
            ambiguous_count += 1
            if not method_ok:
                ambiguous_reasons.add("non_http_event_type")
            if not status_ok:
                ambiguous_reasons.add("non_http_status_code")
            if not resource_ok:
                ambiguous_reasons.add("non_http_resource_path")
            if not ip_ok:
                ambiguous_reasons.add("non_ip_source_identifier")

    if unsupported_count == 0 and ambiguous_count == 0 and supported_count > 0:
        return

    extra: dict[str, object] = {
        "evidence_id": evidence_code,
        "supported_http_events": supported_count,
        "unsupported_domain_events": unsupported_count,
        "ambiguous_events": ambiguous_count,
        "supported_domain": "apache_http_access_log_v1",
    }
    if unsupported_event_types:
        extra["observed_unsupported_event_types"] = sorted(unsupported_event_types)[:8]
    if ambiguous_reasons:
        extra["incompatibility_reasons"] = sorted(ambiguous_reasons)

    if supported_count > 0:
        raise MLAdvisoryError(
            "mixed_event_domain",
            "This evidence mixes Apache HTTP access events with event types outside the ML "
            "model's validated Apache HTTP domain. ML advisory was blocked to prevent misleading "
            "predictions; deterministic rule-based forensic analysis remains available.",
            status_code=422,
            extra=extra,
        )
    if unsupported_count > 0:
        raise MLAdvisoryError(
            "unsupported_event_domain",
            "This evidence contains event types outside the ML model's validated Apache HTTP "
            "domain (such as process, filesystem, USB, or firewall telemetry). Deterministic "
            "rule-based forensic analysis remains available.",
            status_code=422,
            extra=extra,
        )
    raise MLAdvisoryError(
        "ambiguous_event_domain",
        "This evidence cannot be verified as compatible Apache-style HTTP access logs "
        "(requires standard HTTP methods, 3-digit HTTP status codes, URI request paths, "
        "and IP source addresses). Deterministic rule-based forensic analysis remains available.",
        status_code=422,
        extra=extra,
    )


class MLAdvisoryError(ValueError):
    """Structured fail-closed error for optional ML advisory inference."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 503,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = dict(extra) if extra is not None else {}

    def to_detail(self) -> dict[str, object]:
        detail: dict[str, object] = {
            "ml_advisory_error": self.code,
            "message": self.message,
            "deterministic_analysis_available": True,
        }
        detail.update(self.extra)
        return detail


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

            validate_apache_http_domain_compatibility(ev_code, rows)

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
