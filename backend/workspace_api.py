from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .audit import append_audit_event, verify_audit_chain
from .database import get_connection
from .ml.advisory_service import MLAdvisoryError, load_runtime_config
from .ml.schema import FEATURE_SCHEMA_VERSION


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent

VALID_CASE_STATUSES: tuple[str, ...] = (
    "Open",
    "In Analysis",
    "Under Review",
    "Closed",
)
VALID_FINDING_STATUSES: tuple[str, ...] = (
    "Requires Review",
    "Confirmed",
    "False Positive",
    "Escalated",
)
PRE_AUTH_SECURITY_NOTICE = (
    "Single-operator / local-deployment mode: user authentication and RBAC are not "
    "implemented in M1 (scheduled for Milestone M7). Restrict network exposure to trusted hosts."
)


class CaseUpdateRequest(BaseModel):
    status: Literal["Open", "In Analysis", "Under Review", "Closed"] | None = None
    description: str | None = Field(default=None, max_length=4000)


class FindingTriageUpdateRequest(BaseModel):
    status: Literal["Requires Review", "Confirmed", "False Positive", "Escalated"]
    triage_notes: str | None = Field(default=None, max_length=1000)


def _working_dir() -> Path:
    return Path(
        os.environ.get("FORENSIC_INSIGHT_WORKING_DIR", str(BASE_DIR / "uploads" / "working"))
    )


def _get_ml_status_snapshot() -> dict[str, object]:
    """Return read-only ML advisory status WITHOUT deserializing or loading model.joblib."""
    try:
        cfg = load_runtime_config()
    except MLAdvisoryError as exc:
        return {
            "enabled": False,
            "operator_approved": False,
            "ready_for_advisory": False,
            "advisory_only": True,
            "deserialized": False,
            "config_error": exc.message,
        }

    artifact_dir = cfg.artifact_dir
    manifest_present = (artifact_dir / "artifact_manifest.json").is_file()
    model_file_present = (artifact_dir / "model.joblib").is_file()
    artifact_present = manifest_present and model_file_present
    return {
        "enabled": cfg.enabled,
        "operator_approved": cfg.operator_approved,
        "model_id": cfg.model_id,
        "model_version": cfg.model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "expected_model_sha256": cfg.expected_model_sha256,
        "max_events": cfg.max_events,
        "artifact_present": artifact_present,
        "ready_for_advisory": bool(cfg.enabled and cfg.operator_approved and artifact_present),
        "advisory_only": True,
        "deserialized": False,
    }


def _resolve_evidence_row(connection, evidence_ref: str):
    if evidence_ref.isdigit():
        return connection.execute(
            """
            SELECT id, case_id, evidence_id, original_filename, stored_filename,
                   file_type, file_size, sha256, source_description, status, registered_date
            FROM evidence
            WHERE id = ?
            """,
            (int(evidence_ref),),
        ).fetchone()
    return connection.execute(
        """
        SELECT id, case_id, evidence_id, original_filename, stored_filename,
               file_type, file_size, sha256, source_description, status, registered_date
        FROM evidence
        WHERE evidence_id = ?
        """,
        (evidence_ref,),
    ).fetchone()


def _count_verified_csv_rows(working_path: Path) -> int:
    with open(working_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


@router.get("/ml/status")
def get_ml_advisory_status():
    """Return read-only ML advisory availability and pinned model metadata without deserializing."""
    return _get_ml_status_snapshot()


@router.get("/dashboard/summary")
def get_dashboard_summary():
    """Aggregate real metrics across cases, evidence, findings, and audit chain integrity."""
    connection = get_connection()
    try:
        total_cases = connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        status_rows = connection.execute(
            "SELECT COALESCE(status, 'Open') AS status, COUNT(*) AS cnt FROM cases GROUP BY COALESCE(status, 'Open')"
        ).fetchall()
        cases_by_status = {status: 0 for status in VALID_CASE_STATUSES}
        for row in status_rows:
            cases_by_status[str(row["status"])] = int(row["cnt"])

        total_evidence = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        total_findings = connection.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
        total_reports = connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

        sev_rows = connection.execute(
            "SELECT severity, COUNT(*) AS cnt FROM findings GROUP BY severity"
        ).fetchall()
        findings_by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for row in sev_rows:
            sev = str(row["severity"])
            if sev in findings_by_severity:
                findings_by_severity[sev] = int(row["cnt"])

        rule_rows = connection.execute(
            "SELECT rule_id, COUNT(*) AS cnt FROM findings WHERE rule_id IS NOT NULL GROUP BY rule_id"
        ).fetchall()
        findings_by_rule = {str(row["rule_id"]): int(row["cnt"]) for row in rule_rows}

        recent_cases_rows = connection.execute(
            """
            SELECT
                c.id,
                c.case_number,
                c.title,
                c.investigator,
                COALESCE(c.status, 'Open') AS status,
                c.created_at,
                (SELECT COUNT(*) FROM evidence e WHERE e.case_id = c.id) AS evidence_count,
                (SELECT COUNT(*) FROM findings f JOIN evidence e ON f.evidence_id = e.id WHERE e.case_id = c.id) AS findings_count,
                (SELECT COUNT(*) FROM findings f JOIN evidence e ON f.evidence_id = e.id WHERE e.case_id = c.id AND f.severity = 'HIGH') AS high_severity_count
            FROM cases c
            ORDER BY c.id DESC
            LIMIT 10
            """
        ).fetchall()

        recent_audit_rows = connection.execute(
            """
            SELECT event_id, timestamp, action_type, case_id, evidence_id, outcome
            FROM audit_events
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()

        audit_verification = verify_audit_chain(connection)

        return {
            "metrics": {
                "total_cases": int(total_cases),
                "cases_by_status": cases_by_status,
                "total_evidence": int(total_evidence),
                "total_findings": int(total_findings),
                "findings_by_severity": findings_by_severity,
                "findings_by_rule": findings_by_rule,
                "total_reports": int(total_reports),
            },
            "audit_chain": {
                "valid": bool(audit_verification["valid"]),
                "event_count": int(audit_verification["event_count"]),
                "issues_count": len(audit_verification["issues"]),
            },
            "ml_status": _get_ml_status_snapshot(),
            "recent_cases": [dict(row) for row in recent_cases_rows],
            "recent_audit_events": [dict(row) for row in recent_audit_rows],
            "security_notice": PRE_AUTH_SECURITY_NOTICE,
        }
    finally:
        connection.close()


@router.get("/cases/{case_id}")
def get_case_detail(case_id: int):
    """Return detailed case metadata and real aggregate counts from DB and verified CSVs."""
    from .main import verify_evidence_integrity

    connection = get_connection()
    try:
        case_row = connection.execute(
            """
            SELECT id, case_number, title, investigator, description,
                   COALESCE(status, 'Open') AS status, created_at
            FROM cases
            WHERE id = ?
            """,
            (case_id,),
        ).fetchone()
        if not case_row:
            raise HTTPException(status_code=404, detail="Case not found")

        evidence_rows = connection.execute(
            """
            SELECT id, evidence_id, stored_filename, sha256, file_type
            FROM evidence
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchall()

        severity_rows = connection.execute(
            """
            SELECT f.severity, COUNT(*) AS cnt
            FROM findings f
            JOIN evidence e ON f.evidence_id = e.id
            WHERE e.case_id = ?
            GROUP BY f.severity
            """,
            (case_id,),
        ).fetchall()
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        total_findings = 0
        for row in severity_rows:
            sev = str(row["severity"])
            cnt = int(row["cnt"])
            total_findings += cnt
            if sev in severity_counts:
                severity_counts[sev] = cnt

        custody_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM chain_of_custody coc
            JOIN evidence e ON coc.evidence_id = e.id
            WHERE e.case_id = ?
            """,
            (case_id,),
        ).fetchone()[0]

        reports_count = connection.execute(
            "SELECT COUNT(*) FROM reports WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]

        last_analysis_row = connection.execute(
            """
            SELECT timestamp
            FROM audit_events
            WHERE case_id = ? AND action_type = 'ANALYSIS_COMPLETED' AND outcome = 'SUCCESS'
            ORDER BY id DESC
            LIMIT 1
            """,
            (case_id,),
        ).fetchone()

        verified_csv_event_count = 0
        integrity_verified_count = 0
        integrity_failed_evidence: list[str] = []
        working_dir = _working_dir()

        for ev in evidence_rows:
            ev_code = str(ev["evidence_id"])
            original_path = Path(str(ev["stored_filename"]))
            working_path = working_dir / f"{ev_code}_working.csv"
            try:
                verify_evidence_integrity(ev_code, original_path, working_path, str(ev["sha256"]))
                integrity_verified_count += 1
                if str(ev["file_type"]).lower() == ".csv":
                    verified_csv_event_count += _count_verified_csv_rows(working_path)
            except HTTPException:
                integrity_failed_evidence.append(ev_code)
            except (OSError, UnicodeDecodeError, csv.Error):
                pass

        return {
            "case": dict(case_row),
            "summary": {
                "evidence_count": len(evidence_rows),
                "integrity_verified_evidence_count": integrity_verified_count,
                "integrity_failed_evidence": integrity_failed_evidence,
                "verified_csv_event_count": verified_csv_event_count,
                "findings_count": total_findings,
                "severity_counts": severity_counts,
                "custody_count": int(custody_count),
                "reports_count": int(reports_count),
                "last_analysis_at": last_analysis_row["timestamp"] if last_analysis_row else None,
            },
            "security_notice": PRE_AUTH_SECURITY_NOTICE,
        }
    finally:
        connection.close()


@router.patch("/cases/{case_id}")
def update_case_metadata(case_id: int, update: CaseUpdateRequest):
    """Update case lifecycle status or description with full audit trail."""
    if update.status is None and update.description is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'status' or 'description' must be provided.",
        )

    connection = get_connection()
    try:
        existing = connection.execute(
            "SELECT id, case_number, title, investigator, description, COALESCE(status, 'Open') AS status, created_at FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Case not found")

        new_status = update.status if update.status is not None else str(existing["status"])
        new_description = (
            update.description.strip()
            if update.description is not None
            else str(existing["description"] or "")
        )

        connection.execute(
            "UPDATE cases SET status = ?, description = ? WHERE id = ?",
            (new_status, new_description, case_id),
        )
        append_audit_event(
            connection,
            action_type="CASE_UPDATED",
            case_id=case_id,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={
                "previous_status": str(existing["status"]),
                "new_status": new_status,
                "description_updated": update.description is not None,
            },
        )
        connection.commit()

        updated = connection.execute(
            "SELECT id, case_number, title, investigator, description, status, created_at FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()
        return {
            "message": "Case updated successfully",
            "case": dict(updated),
        }
    except HTTPException:
        connection.rollback()
        raise
    finally:
        connection.close()


@router.get("/evidence/{evidence_ref}/preview")
def preview_evidence(
    evidence_ref: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """Read-only paginated preview from an integrity-verified CSV working copy."""
    from .main import verify_evidence_integrity

    connection = get_connection()
    try:
        ev = _resolve_evidence_row(connection, evidence_ref)
        if not ev:
            raise HTTPException(status_code=404, detail="Evidence not found")
    finally:
        connection.close()

    ev_code = str(ev["evidence_id"])
    original_path = Path(str(ev["stored_filename"]))
    working_path = _working_dir() / f"{ev_code}_working.csv"
    verify_evidence_integrity(ev_code, original_path, working_path, str(ev["sha256"]))

    file_type = str(ev["file_type"] or "").lower()
    if file_type != ".csv":
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "unsupported_format",
                "evidence_id": ev_code,
                "message": f"Tabular preview is only supported for .csv evidence (received {file_type or 'unknown'}).",
            },
        )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    rows_page: list[dict[str, str]] = []
    total_rows = 0

    try:
        with open(working_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = [h.strip() for h in (reader.fieldnames or [])]
            if not headers:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "validation_error": "missing_headers",
                        "evidence_id": ev_code,
                        "message": "Evidence CSV has no column headers.",
                    },
                )
            for idx, raw_row in enumerate(reader):
                total_rows += 1
                if start_idx <= idx < end_idx:
                    rows_page.append(
                        {
                            norm: str(raw_row.get(orig) or "")
                            for orig, norm in zip(reader.fieldnames or [], headers)
                        }
                    )
    except HTTPException:
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "malformed_csv",
                "evidence_id": ev_code,
                "message": f"Unable to preview CSV working copy: {exc}",
            },
        ) from exc

    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    return {
        "evidence_id": ev_code,
        "evidence_db_id": int(ev["id"]),
        "original_filename": str(ev["original_filename"]),
        "sha256": str(ev["sha256"]),
        "headers": headers,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "rows": rows_page,
    }


@router.post("/evidence/{evidence_ref}/verify")
def verify_single_evidence(evidence_ref: str):
    """Explicit on-demand SHA-256 verification of original and working evidence copies."""
    from .main import verify_evidence_integrity

    connection = get_connection()
    try:
        ev = _resolve_evidence_row(connection, evidence_ref)
        if not ev:
            raise HTTPException(status_code=404, detail="Evidence not found")

        ev_code = str(ev["evidence_id"])
        case_id = int(ev["case_id"])
        original_path = Path(str(ev["stored_filename"]))
        working_path = _working_dir() / f"{ev_code}_working.csv"
        recorded_sha256 = str(ev["sha256"])

        try:
            verify_evidence_integrity(ev_code, original_path, working_path, recorded_sha256)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
            append_audit_event(
                connection,
                action_type="EVIDENCE_INTEGRITY_FAILED",
                case_id=case_id,
                evidence_id=ev_code,
                outcome="FAILURE",
                metadata={"error": detail},
            )
            connection.commit()
            raise

        verified_at = datetime.now(timezone.utc).isoformat()
        append_audit_event(
            connection,
            action_type="EVIDENCE_INTEGRITY_VERIFIED",
            case_id=case_id,
            evidence_id=ev_code,
            outcome="SUCCESS",
            metadata={"sha256": recorded_sha256},
        )
        connection.commit()

        return {
            "evidence_id": ev_code,
            "case_id": case_id,
            "verified": True,
            "sha256": recorded_sha256,
            "verified_at": verified_at,
        }
    finally:
        connection.close()


@router.patch("/findings/{finding_id}/status")
def update_finding_triage_status(finding_id: str, payload: FindingTriageUpdateRequest):
    """Update analyst triage status and notes for a deterministic finding with audit logging."""
    connection = get_connection()
    try:
        existing = connection.execute(
            """
            SELECT f.id, f.finding_id, f.status, f.triage_notes, f.rule_id,
                   e.case_id, e.evidence_id AS evidence_code
            FROM findings f
            JOIN evidence e ON f.evidence_id = e.id
            WHERE f.finding_id = ?
            """,
            (finding_id,),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Finding not found")

        notes = payload.triage_notes.strip() if payload.triage_notes is not None else existing["triage_notes"]
        connection.execute(
            "UPDATE findings SET status = ?, triage_notes = ? WHERE finding_id = ?",
            (payload.status, notes, finding_id),
        )
        append_audit_event(
            connection,
            action_type="FINDING_STATUS_UPDATED",
            case_id=int(existing["case_id"]),
            evidence_id=str(existing["evidence_code"]),
            outcome="SUCCESS",
            metadata={
                "finding_id": finding_id,
                "rule_id": existing["rule_id"],
                "previous_status": str(existing["status"]),
                "new_status": payload.status,
                "has_triage_notes": bool(notes),
            },
        )
        connection.commit()

        updated = connection.execute(
            """
            SELECT f.id, f.finding_id, f.evidence_id, f.event_id, f.timestamp,
                   f.source, f.event_type, f.resource, f.reason, f.severity,
                   f.technical_observation, f.plain_language_explanation,
                   f.status, f.triage_notes, f.rule_id, f.rule_version, f.rule_fingerprint,
                   e.evidence_id AS evidence_code, e.case_id
            FROM findings f
            JOIN evidence e ON f.evidence_id = e.id
            WHERE f.finding_id = ?
            """,
            (finding_id,),
        ).fetchone()
        return {
            "message": "Finding triage status updated",
            "finding": dict(updated),
        }
    except HTTPException:
        connection.rollback()
        raise
    finally:
        connection.close()


@router.get("/cases/{case_id}/reports")
def list_case_reports(case_id: int):
    """Return persisted PDF report metadata history for a case."""
    connection = get_connection()
    try:
        case = connection.execute("SELECT id FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        rows = connection.execute(
            """
            SELECT id, case_id, filename, file_path AS path, sha256, generated_at
            FROM reports
            WHERE case_id = ?
            ORDER BY id DESC
            """,
            (case_id,),
        ).fetchall()
        reports = []
        for row in rows:
            item = dict(row)
            item["file_exists"] = Path(str(item["path"])).is_file()
            reports.append(item)
        return {
            "case_id": case_id,
            "total_reports": len(reports),
            "reports": reports,
        }
    finally:
        connection.close()
