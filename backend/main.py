import hashlib
import sqlite3
import csv
import statistics
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4
from collections import defaultdict
import traceback
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors

from .database import get_connection, initialize_database
from .audit import append_audit_event, get_audit_history, verify_audit_chain
from .rules import RULE_REGISTRY, calculate_rule_fingerprint, get_rule_metadata


initialize_database()


app = FastAPI(
    title="Forensic Insight",
    description="Digital forensic reporting and visualization platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CaseCreate(BaseModel):
    case_number: str
    title: str
    investigator: str
    description: str | None = None


class ChainOfCustodyEntry(BaseModel):
    person: str
    action: str
    remarks: str


# Phase 2: Threat Detection
BASE_DIR = Path(__file__).resolve().parent.parent
WORKING_DIR = Path(os.environ.get("FORENSIC_INSIGHT_WORKING_DIR", str(BASE_DIR / "uploads" / "working")))


def detect_threats(csv_path: Path, evidence_id: int) -> list:
    """
    Read CSV evidence and apply threat detection rules.
    Returns list of findings.
    """
    findings = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")
    
    if not rows:
        return findings
    
    # Extract numeric values for baseline comparison (RULE D)
    byte_values = []
    post_requests = defaultdict(list)  # For RULE C
    
    for idx, row in enumerate(rows):
        event_id = str(idx)
        timestamp = row.get('timestamp', '')
        source_ip = row.get('source_ip', row.get('source', ''))
        event_type = row.get('event_type', row.get('event', ''))
        resource = row.get('resource', row.get('description', ''))
        status = row.get('status', '')
        bytes_field = row.get('bytes', '')
        
        # Try to extract numeric bytes value
        try:
            if bytes_field and bytes_field.isdigit():
                byte_values.append(int(bytes_field))
        except:
            pass
        
        # Collect POST requests by source (for RULE C)
        if RULE_REGISTRY["R-C"]["detection_conditions"]["event_type_contains"] in event_type.upper():
            post_requests[source_ip].append({
                'timestamp': timestamp,
                'resource': resource,
                'index': idx
            })
    
    # Calculate baseline for RULE D (unusually large transfers)
    high_byte_threshold = None
    rule_d_conditions = RULE_REGISTRY["R-D"]["detection_conditions"]
    if len(byte_values) >= rule_d_conditions["minimum_numeric_samples"]:
        try:
            mean = statistics.mean(byte_values)
            stdev = statistics.stdev(byte_values)
            high_byte_threshold = mean + (rule_d_conditions["threshold_multiplier"] * stdev)
        except:
            pass
    
    # Apply detection rules
    for idx, row in enumerate(rows):
        event_id = str(idx)
        timestamp = row.get('timestamp', '')
        source_ip = row.get('source_ip', row.get('source', ''))
        event_type = row.get('event_type', row.get('event', ''))
        resource = row.get('resource', row.get('description', ''))
        status = row.get('status', '')
        bytes_field = row.get('bytes', '')
        
        # RULE A: HTTP 403 status
        rule_a = RULE_REGISTRY["R-A"]
        if rule_a["detection_conditions"]["status_contains"] in str(status):
            finding = {
                'rule_id': rule_a["rule_id"],
                'event_id': event_id,
                'timestamp': timestamp,
                'source': source_ip,
                'event_type': event_type,
                'resource': resource,
                'reason': 'HTTP 403 Forbidden status detected',
                'severity': rule_a["severity"],
                'technical_observation': f"Event triggered HTTP 403 status: {resource}",
                'plain_language_explanation': 'A device attempted to access a resource, but the server denied the request.',
            }
            findings.append(finding)
        
        # RULE B: POST to sensitive resources
        rule_b = RULE_REGISTRY["R-B"]
        sensitive_keywords = rule_b["detection_conditions"]["resource_keywords"]
        if rule_b["detection_conditions"]["event_type_contains"] in event_type.upper():
            resource_lower = resource.lower()
            for keyword in sensitive_keywords:
                if keyword in resource_lower:
                    finding = {
                        'rule_id': rule_b["rule_id"],
                        'event_id': event_id,
                        'timestamp': timestamp,
                        'source': source_ip,
                        'event_type': event_type,
                        'resource': resource,
                        'reason': f'POST request to restricted resource containing "{keyword}"',
                        'severity': rule_b["severity"],
                        'technical_observation': f"POST request detected to resource containing sensitive keyword: {resource}",
                        'plain_language_explanation': f'A device attempted to send data to a resource containing "{keyword}" in its name.',
                    }
                    findings.append(finding)
                    break
        
        # RULE C: Repeated POST requests from same source
        rule_c = RULE_REGISTRY["R-C"]
        if rule_c["detection_conditions"]["event_type_contains"] in event_type.upper() and source_ip in post_requests:
            posts = post_requests[source_ip]
            if len(posts) >= rule_c["detection_conditions"]["minimum_request_count"]:
                # Check if multiple posts occurred
                finding = {
                    'rule_id': rule_c["rule_id"],
                    'event_id': event_id,
                    'timestamp': timestamp,
                    'source': source_ip,
                    'event_type': event_type,
                    'resource': resource,
                    'reason': f'Multiple POST requests ({len(posts)}) from same source',
                    'severity': rule_c["severity"],
                    'technical_observation': f"Source {source_ip} made {len(posts)} POST requests",
                    'plain_language_explanation': f'A device sent multiple data-submission requests in a short timeframe.',
                }
                findings.append(finding)
                # Clear to avoid duplicates
                post_requests[source_ip] = []
        
        # RULE D: Unusually large data transfer
        if high_byte_threshold and bytes_field:
            try:
                bytes_val = int(bytes_field)
                if bytes_val > high_byte_threshold:
                    rule_d = RULE_REGISTRY["R-D"]
                    finding = {
                        'rule_id': rule_d["rule_id"],
                        'event_id': event_id,
                        'timestamp': timestamp,
                        'source': source_ip,
                        'event_type': event_type,
                        'resource': resource,
                        'reason': f'Data transfer exceeds baseline by 2+ standard deviations',
                        'severity': rule_d["severity"],
                        'technical_observation': f"Transfer size {bytes_val} bytes exceeds baseline {high_byte_threshold:.0f} bytes",
                        'plain_language_explanation': 'A data transfer was significantly larger than typical activity.',
                    }
                    findings.append(finding)
            except:
                pass
    
    return findings


@app.post("/cases/{case_id}/analyze")
async def analyze_evidence(case_id: int):
    """Run threat detection analysis on evidence for a case."""
    connection = get_connection()
    rules_fingerprint = calculate_rule_fingerprint()
    
    try:
        # Verify case exists
        case = connection.execute(
            "SELECT id FROM cases WHERE id = ?",
            (case_id,)
        ).fetchone()
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get evidence for this case
        evidence_list = connection.execute(
            "SELECT id, evidence_id, stored_filename, sha256, file_type FROM evidence WHERE case_id = ?",
            (case_id,)
        ).fetchall()
        
        if not evidence_list:
            raise HTTPException(status_code=404, detail="No evidence found for this case")

        # Verify every evidence item before any findings are changed or any detection runs.
        verified_evidence = []
        for evidence in evidence_list:
            ev_id = evidence[0]
            ev_code = evidence[1]
            stored_path = evidence[2]
            recorded_sha256 = evidence[3]
            file_type = evidence[4]

            original_path = Path(stored_path)
            working_path = WORKING_DIR / f"{ev_code}_working.csv"
            try:
                verify_evidence_integrity(ev_code, original_path, working_path, recorded_sha256)
                validate_evidence_for_analysis(ev_code, file_type, working_path)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
                action = "EVIDENCE_INTEGRITY_FAILED" if "integrity_error" in detail else "EVIDENCE_VALIDATION_FAILED"
                append_audit_event(
                    connection,
                    action_type=action,
                    case_id=case_id,
                    evidence_id=ev_code,
                    outcome="FAILURE",
                    metadata={
                        "status_code": exc.status_code,
                        "error": detail,
                        "rule_fingerprint": rules_fingerprint,
                    },
                )
                connection.commit()
                raise
            verified_evidence.append((ev_id, ev_code, working_path))

        # Integrity passed. Existing findings may now be replaced by a successful analysis.
        connection.execute(
            "DELETE FROM findings WHERE evidence_id IN (SELECT id FROM evidence WHERE case_id = ?)",
            (case_id,),
        )

        analysis_results = []
        # Process only integrity-verified evidence items.
        for ev_id, ev_code, working_path in verified_evidence:
            findings = detect_threats(working_path, ev_id)
            created_at = datetime.now(timezone.utc).isoformat()

            for finding in findings:
                finding_id = f"FD-{ev_id}-{uuid4().hex[:8].upper()}"
                connection.execute(
                    """
                    INSERT INTO findings (
                        finding_id,
                        evidence_id,
                        event_id,
                        timestamp,
                        source,
                        event_type,
                        resource,
                        reason,
                        severity,
                        technical_observation,
                        plain_language_explanation,
                        status,
                        created_at,
                        rule_id,
                        rule_version,
                        rule_fingerprint
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        finding_id,
                        ev_id,
                        finding.get("event_id"),
                        finding.get("timestamp"),
                        finding.get("source"),
                        finding.get("event_type"),
                        finding.get("resource"),
                        finding.get("reason"),
                        finding.get("severity"),
                        finding.get("technical_observation"),
                        finding.get("plain_language_explanation"),
                        "Requires Review",
                        created_at,
                        finding["rule_id"],
                        RULE_REGISTRY[finding["rule_id"]]["version"],
                        rules_fingerprint,
                    ),
                )

                analysis_results.append({
                    "finding_id": finding_id,
                    "event_id": finding.get("event_id"),
                    "severity": finding.get("severity"),
                    "reason": finding.get("reason"),
                    "rule_id": finding["rule_id"],
                    "rule_version": RULE_REGISTRY[finding["rule_id"]]["version"],
                    "rule_fingerprint": rules_fingerprint,
                })

        append_audit_event(
            connection,
            action_type="ANALYSIS_COMPLETED",
            case_id=case_id,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={
                "findings_count": len(analysis_results),
                "rule_fingerprint": rules_fingerprint,
            },
        )
        connection.commit()

        return {
            'message': 'Analysis complete',
            'findings_count': len(analysis_results),
            'findings': analysis_results
        }

    except HTTPException:
        raise
    except Exception as exc:
        connection.rollback()
        append_audit_event(
            connection,
            action_type="ANALYSIS_FAILED",
            case_id=case_id,
            evidence_id=None,
            outcome="FAILURE",
            metadata={
                "error": str(exc),
                "rule_fingerprint": rules_fingerprint,
            },
        )
        connection.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        connection.close()


@app.get("/cases/{case_id}/findings")
def get_case_findings(case_id: int):
    """Get all findings for a case."""
    connection = get_connection()
    
    # Verify case exists
    case = connection.execute(
        "SELECT id FROM cases WHERE id = ?",
        (case_id,)
    ).fetchone()
    
    if not case:
        connection.close()
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get findings for all evidence in this case
    rows = connection.execute(
        """
        SELECT
            f.id,
            f.finding_id,
            f.evidence_id,
            f.event_id,
            f.timestamp,
            f.source,
            f.event_type,
            f.resource,
            f.reason,
            f.severity,
            f.technical_observation,
            f.plain_language_explanation,
            f.status,
            f.rule_id,
            f.rule_version,
            f.rule_fingerprint,
            e.evidence_id as evidence_code
        FROM findings f
        JOIN evidence e ON f.evidence_id = e.id
        WHERE e.case_id = ?
        ORDER BY f.timestamp DESC, f.severity DESC
        """,
        (case_id,)
    ).fetchall()
    
    connection.close()
    
    # Count summary
    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for row in rows:
        severity = row['severity']
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    return {
        'case_id': case_id,
        'total_findings': len(rows),
        'severity_counts': severity_counts,
        'findings': [dict(row) for row in rows]
    }


@app.get("/findings/{finding_id}")
def get_finding_details(finding_id: str):
    """Get detailed information about a specific finding."""
    connection = get_connection()
    
    finding = connection.execute(
        """
        SELECT
            f.id,
            f.finding_id,
            f.evidence_id,
            f.event_id,
            f.timestamp,
            f.source,
            f.event_type,
            f.resource,
            f.reason,
            f.severity,
            f.technical_observation,
            f.plain_language_explanation,
            f.status,
            f.rule_id,
            f.rule_version,
            f.rule_fingerprint,
            e.evidence_id as evidence_code,
            e.case_id
        FROM findings f
        JOIN evidence e ON f.evidence_id = e.id
        WHERE f.finding_id = ?
        """,
        (finding_id,)
    ).fetchone()
    
    connection.close()
    
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")
    
    return dict(finding)


@app.get("/cases/{case_id}/events")
def get_case_events(case_id: int):
    """Get all events from all evidence files for a case (for Phase 3 visualizations)."""
    try:
        connection = get_connection()
        
        # Verify case exists
        case = connection.execute(
            "SELECT id FROM cases WHERE id = ?",
            (case_id,)
        ).fetchone()
        
        if not case:
            connection.close()
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get all evidence for this case
        evidence_list = connection.execute(
            "SELECT id, evidence_id, stored_filename FROM evidence WHERE case_id = ?",
            (case_id,)
        ).fetchall()
        
        connection.close()
        
        events = []
        
        # Read all events from all evidence files
        for evidence_row in evidence_list:
            try:
                # Convert Row to dict-like access
                ev_id = evidence_row[0]
                ev_code = evidence_row[1]
                stored_path = evidence_row[2]
                
                path = Path(stored_path)
                if not path.exists():
                    continue
                
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader is None:
                        continue
                    
                    for idx, row in enumerate(reader):
                        if row is None:
                            continue
                        
                        event = {
                            'event_id': str(idx),
                            'evidence_id': ev_id,
                            'evidence_code': ev_code,
                            'timestamp': row.get('timestamp', ''),
                            'source': row.get('source_ip', row.get('source', '')),
                            'event_type': row.get('event_type', row.get('event', '')),
                            'resource': row.get('resource', row.get('description', '')),
                            'status': row.get('status', ''),
                            'bytes': row.get('bytes', ''),
                            'findings': []
                        }
                        events.append(event)
            except Exception:
                # Log but continue processing other files
                continue
        
        # Get all findings for this case
        try:
            connection2 = get_connection()
            findings = connection2.execute(
                """
                SELECT f.event_id, f.evidence_id, f.finding_id, f.reason, f.severity
                FROM findings f
                JOIN evidence e ON f.evidence_id = e.id
                WHERE e.case_id = ?
                """,
                (case_id,)
            ).fetchall()
            connection2.close()
            
            # Build map: (event_id, evidence_id) -> findings
            for f_row in findings:
                event_id_str = str(f_row[0])  # event_id
                evidence_id_int = f_row[1]     # evidence_id
                finding_id = f_row[2]
                reason = f_row[3]
                severity = f_row[4]
                
                # Attach to matching event
                for event in events:
                    if event['event_id'] == event_id_str and event['evidence_id'] == evidence_id_int:
                        event['findings'].append({
                            'finding_id': finding_id,
                            'reason': reason,
                            'severity': severity
                        })
        except Exception:
            # If findings fetch fails, continue without them
            pass
        
        # Sort by timestamp
        try:
            events.sort(key=lambda e: e.get('timestamp', '9999-99-99') or '9999-99-99')
        except:
            pass
        
        return {
            'case_id': case_id,
            'total_events': len(events),
            'events': events
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving events: {str(e)}")


@app.get("/")
def root():
    """Serve frontend index.html"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "product": "Forensic Insight",
        "status": "running",
        "message": "Frontend not found. API endpoints available at /docs"
    }


@app.post("/cases")
def create_case(case: CaseCreate):
    created_at = datetime.now(timezone.utc).isoformat()
    # Keep existing schema behavior but avoid persisting NULL for new descriptions.
    normalized_description = (case.description or "").strip()

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO cases (
                case_number,
                title,
                investigator,
                description,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                case.case_number,
                case.title,
                case.investigator,
                normalized_description,
                created_at,
            ),
        )

        append_audit_event(
            connection,
            action_type="CASE_CREATED",
            case_id=cursor.lastrowid,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={"case_number": case.case_number, "title": case.title, "investigator": case.investigator},
        )
        connection.commit()

        return {
            "message": "Case created successfully",
            "case": {
                "id": cursor.lastrowid,
                "case_number": case.case_number,
                "title": case.title,
                "investigator": case.investigator,
                "description": normalized_description,
                "created_at": created_at,
            },
        }

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="A case with this case number already exists."
        )

    finally:
        connection.close()


@app.get("/cases")
def get_cases():
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            case_number,
            title,
            investigator,
            description,
            created_at
        FROM cases
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return {
        "cases": [dict(row) for row in rows]
    }


# Allowed file extensions for evidence upload
ALLOWED_EXTENSIONS = {".csv", ".json", ".txt", ".log"}
UPLOAD_DIR = Path(os.environ.get("FORENSIC_INSIGHT_UPLOAD_DIR", str(BASE_DIR / "uploads" / "original")))
WORKING_DIR = Path(os.environ.get("FORENSIC_INSIGHT_WORKING_DIR", str(BASE_DIR / "uploads" / "working")))


def calculate_sha256(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file contents."""
    return hashlib.sha256(file_bytes).hexdigest()


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 from a file incrementally."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as evidence_file:
        for chunk in iter(lambda: evidence_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence_integrity(
    evidence_id: str,
    original_path: Path,
    working_path: Path,
    recorded_sha256: str,
) -> None:
    """Verify original and working evidence against the ingestion-time SHA-256 baseline."""
    if not original_path.is_file():
        raise HTTPException(
            status_code=409,
            detail={
                "integrity_error": "missing_original",
                "evidence_id": evidence_id,
                "message": "Original evidence file is missing; processing blocked.",
                "recorded_sha256": recorded_sha256,
            },
        )

    original_sha256 = calculate_file_sha256(original_path)
    if original_sha256 != recorded_sha256:
        raise HTTPException(
            status_code=409,
            detail={
                "integrity_error": "original_hash_mismatch",
                "evidence_id": evidence_id,
                "message": "Original evidence hash does not match the recorded ingestion hash; processing blocked.",
                "recorded_sha256": recorded_sha256,
                "actual_sha256": original_sha256,
            },
        )

    if not working_path.is_file():
        raise HTTPException(
            status_code=409,
            detail={
                "integrity_error": "missing_working",
                "evidence_id": evidence_id,
                "message": "Working evidence copy is missing; processing blocked.",
                "recorded_sha256": recorded_sha256,
            },
        )

    working_sha256 = calculate_file_sha256(working_path)
    if working_sha256 != recorded_sha256:
        raise HTTPException(
            status_code=409,
            detail={
                "integrity_error": "working_hash_mismatch",
                "evidence_id": evidence_id,
                "message": "Working evidence hash does not match the recorded ingestion hash; processing blocked.",
                "recorded_sha256": recorded_sha256,
                "actual_sha256": working_sha256,
            },
        )


REQUIRED_ANALYTICAL_COLUMNS = {
    "timestamp": ("timestamp",),
    "source": ("source_ip", "source"),
    "event_type": ("event_type", "event"),
    "resource": ("resource", "description"),
    "status": ("status",),
    "bytes": ("bytes",),
}


def validate_evidence_for_analysis(
    evidence_id: str,
    file_type: str,
    working_path: Path,
) -> None:
    """Validate evidence that the existing rule-based detection engine can parse."""
    normalized_type = str(file_type or "").lower()
    if normalized_type != ".csv":
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "unsupported_format",
                "evidence_id": evidence_id,
                "message": (
                    f"Analysis is not supported for {normalized_type or 'unknown'} evidence. "
                    "Only CSV evidence is currently supported. JSON, TXT, and LOG files may be stored but require dedicated parsers."
                ),
                "file_type": normalized_type,
                "supported_formats": [".csv"],
            },
        )

    if not working_path.is_file():
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "missing_working_file",
                "evidence_id": evidence_id,
                "message": "The integrity-verified working evidence file is missing.",
            },
        )

    try:
        with open(working_path, "r", encoding="utf-8-sig", newline="") as evidence_file:
            reader = csv.reader(evidence_file, strict=True)
            rows = list(reader)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "unsupported_encoding",
                "evidence_id": evidence_id,
                "message": "Evidence is not valid UTF-8 text; only UTF-8 and UTF-8 with BOM are supported.",
            },
        )
    except csv.Error as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "malformed_csv",
                "evidence_id": evidence_id,
                "message": f"Evidence contains malformed CSV syntax: {exc}",
            },
        )

    if not rows:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "empty_file",
                "evidence_id": evidence_id,
                "message": "Evidence file is empty.",
            },
        )

    headers = [header.strip() for header in rows[0]]
    if not headers or all(header == "" for header in headers):
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "missing_headers",
                "evidence_id": evidence_id,
                "message": "Evidence has no usable CSV column headers.",
            },
        )

    duplicate_headers = sorted({header for header in headers if headers.count(header) > 1 and header})
    if duplicate_headers:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "duplicate_headers",
                "evidence_id": evidence_id,
                "message": "Evidence contains duplicate column headers.",
                "duplicate_headers": duplicate_headers,
            },
        )

    header_set = set(headers)

    logical_duplicate_groups = {
        logical_name: [name for name in alternatives if name in header_set]
        for logical_name, alternatives in REQUIRED_ANALYTICAL_COLUMNS.items()
    }
    duplicate_logical_columns = {
        logical_name: matches
        for logical_name, matches in logical_duplicate_groups.items()
        if len(matches) > 1
    }
    if duplicate_logical_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "duplicate_headers",
                "evidence_id": evidence_id,
                "message": "Evidence provides multiple accepted headers for the same required field.",
                "duplicate_logical_columns": duplicate_logical_columns,
            },
        )

    missing_columns = []
    column_indexes = {}
    for logical_name, alternatives in REQUIRED_ANALYTICAL_COLUMNS.items():
        matches = [name for name in alternatives if name in header_set]
        if not matches:
            missing_columns.append(" or ".join(alternatives))
        else:
            column_indexes[logical_name] = headers.index(matches[0])

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "missing_columns",
                "evidence_id": evidence_id,
                "message": "Evidence is missing required detection-engine columns.",
                "missing_columns": missing_columns,
                "required_columns": REQUIRED_ANALYTICAL_COLUMNS,
            },
        )

    if len(rows) == 1:
        raise HTTPException(
            status_code=422,
            detail={
                "validation_error": "empty_file",
                "evidence_id": evidence_id,
                "message": "Evidence contains headers but no data rows.",
            },
        )

    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(headers):
            raise HTTPException(
                status_code=422,
                detail={
                    "validation_error": "malformed_row",
                    "evidence_id": evidence_id,
                    "message": "Evidence contains a malformed CSV row.",
                    "row_number": row_number,
                    "expected_fields": len(headers),
                    "received_fields": len(row),
                },
            )

        values = {logical_name: row[index].strip() for logical_name, index in column_indexes.items()}
        missing_values = [name for name, value in values.items() if not value]
        if missing_values:
            raise HTTPException(
                status_code=422,
                detail={
                    "validation_error": "missing_required_values",
                    "evidence_id": evidence_id,
                    "message": "Evidence contains required values that are missing.",
                    "row_number": row_number,
                    "missing_values": missing_values,
                },
            )

        timestamp = values["timestamp"]
        timestamp_for_validation = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
        try:
            if "T" not in timestamp_for_validation and " " not in timestamp_for_validation:
                raise ValueError
            datetime.fromisoformat(timestamp_for_validation)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "validation_error": "invalid_timestamp",
                    "evidence_id": evidence_id,
                    "message": "Evidence contains an invalid timestamp.",
                    "row_number": row_number,
                    "field": "timestamp",
                    "received_value": timestamp,
                },
            )

        byte_value = values["bytes"]
        try:
            numeric_bytes = int(byte_value)
        except ValueError:
            numeric_bytes = -1
        if numeric_bytes < 0 or not byte_value.isdigit():
            raise HTTPException(
                status_code=422,
                detail={
                    "validation_error": "invalid_numeric_bytes",
                    "evidence_id": evidence_id,
                    "message": "Evidence contains a transfer size that is not a valid non-negative integer.",
                    "row_number": row_number,
                    "field": "bytes",
                    "received_value": byte_value,
                },
            )


@app.post("/cases/{case_id}/evidence")
async def upload_evidence(
    case_id: int,
    file: UploadFile,
    source_description: str = ""
):
    """Upload evidence file for a case."""
    
    # Verify case exists
    connection = get_connection()
    case = connection.execute(
        "SELECT id FROM cases WHERE id = ?",
        (case_id,)
    ).fetchone()
    
    if not case:
        connection.close()
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        connection.close()
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Maximum evidence upload size: 100 MiB (100 × 1024 × 1024 bytes).
    MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
    UPLOAD_CHUNK_SIZE = 1024 * 1024
    
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        WORKING_DIR.mkdir(parents=True, exist_ok=True)

        evidence_id = f"EV-{case_id}-{uuid4().hex[:8].upper()}"
        original_path = UPLOAD_DIR / f"{evidence_id}{file_ext}"
        working_path = WORKING_DIR / f"{evidence_id}_working.csv"
        temporary_path = None

        sha256_digest = hashlib.sha256()
        file_size = 0

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{evidence_id}_",
                suffix=".upload",
                dir=UPLOAD_DIR,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                while True:
                    chunk = await file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break

                    file_size += len(chunk)
                    if file_size > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "upload_error": "file_too_large",
                                "message": "Evidence upload exceeds the 100 MiB (100 × 1024 × 1024 bytes) maximum.",
                                "maximum_size_bytes": MAX_UPLOAD_SIZE_BYTES,
                                "received_size_bytes": file_size,
                            },
                        )

                    sha256_digest.update(chunk)
                    temporary_file.write(chunk)

            sha256_hash = sha256_digest.hexdigest()

            temporary_path.replace(original_path)
            temporary_path = None

            with open(original_path, "rb") as source_file, open(working_path, "wb") as working_file:
                while True:
                    chunk = source_file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    working_file.write(chunk)

            registered_date = datetime.now(timezone.utc).isoformat()

            cursor = connection.execute(
                """
                INSERT INTO evidence (
                    case_id,
                    evidence_id,
                    original_filename,
                    stored_filename,
                    file_type,
                    file_size,
                    sha256,
                    source_description,
                    status,
                    registered_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    evidence_id,
                    file.filename,
                    str(original_path),
                    file_ext,
                    file_size,
                    sha256_hash,
                    source_description,
                    "Registered",
                    registered_date,
                ),
            )

            evidence_db_id = cursor.lastrowid

            append_audit_event(
                connection,
                action_type="EVIDENCE_INGESTED",
                case_id=case_id,
                evidence_id=evidence_id,
                outcome="SUCCESS",
                metadata={
                    "original_filename": file.filename,
                    "file_type": file_ext,
                    "file_size": file_size,
                    "sha256": sha256_hash,
                },
            )
            connection.commit()

            return {
                "message": "Evidence registered successfully",
                "evidence": {
                    "id": evidence_db_id,
                    "evidence_id": evidence_id,
                    "case_id": case_id,
                    "original_filename": file.filename,
                    "file_type": file_ext,
                    "file_size": file_size,
                    "sha256": sha256_hash,
                    "source_description": source_description,
                    "status": "Registered",
                    "registered_date": registered_date,
                },
            }
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    except HTTPException:
        if 'original_path' in locals() and original_path.is_file():
            original_path.unlink(missing_ok=True)
        if 'working_path' in locals() and working_path.is_file():
            working_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        if 'original_path' in locals() and original_path.is_file():
            original_path.unlink(missing_ok=True)
        if 'working_path' in locals() and working_path.is_file():
            working_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        connection.close()


@app.get("/cases/{case_id}/evidence")
def get_case_evidence(case_id: int):
    """List all evidence for a case."""
    
    connection = get_connection()
    
    # Verify case exists
    case = connection.execute(
        "SELECT id FROM cases WHERE id = ?",
        (case_id,)
    ).fetchone()
    
    if not case:
        connection.close()
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get all evidence for this case
    rows = connection.execute(
        """
        SELECT
            id,
            evidence_id,
            case_id,
            original_filename,
            file_type,
            file_size,
            sha256,
            source_description,
            status,
            registered_date
        FROM evidence
        WHERE case_id = ?
        ORDER BY registered_date DESC
        """,
        (case_id,)
    ).fetchall()
    
    connection.close()
    
    return {
        "case_id": case_id,
        "evidence": [dict(row) for row in rows]
    }


@app.post("/evidence/{evidence_id}/chain-of-custody")
def record_chain_of_custody(evidence_id: int, entry: ChainOfCustodyEntry):
    """Record a chain of custody entry for evidence."""
    
    connection = get_connection()
    
    # Verify evidence exists
    evidence = connection.execute(
        "SELECT id FROM evidence WHERE id = ?",
        (evidence_id,)
    ).fetchone()
    
    if not evidence:
        connection.close()
        raise HTTPException(status_code=404, detail="Evidence not found")

    evidence_row = connection.execute(
        "SELECT id, case_id, evidence_id FROM evidence WHERE id = ?",
        (evidence_id,)
    ).fetchone()

    if not evidence_row:
        connection.close()
        raise HTTPException(status_code=404, detail="Evidence not found")

    case_id = evidence_row["case_id"]
    evidence_code = evidence_row["evidence_id"]

    person = (entry.person or "").strip()
    action = (entry.action or "").strip()
    remarks = (entry.remarks or "").strip()

    if not person or not action or not remarks:
        connection.close()
        raise HTTPException(
            status_code=422,
            detail="Person, action, and remarks are required for a chain-of-custody entry."
        )
    
    try:
        date_time = datetime.now(timezone.utc).isoformat()
        
        cursor = connection.execute(
            """
            INSERT INTO chain_of_custody (
                evidence_id,
                date_time,
                person,
                action,
                remarks
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                date_time,
                person,
                action,
                remarks,
            ),
        )

        append_audit_event(
            connection,
            action_type="CHAIN_OF_CUSTODY_UPDATED",
            case_id=case_id,
            evidence_id=evidence_code,
            outcome="SUCCESS",
            metadata={"entry_id": cursor.lastrowid, "person": person, "action": action, "remarks": remarks},
        )
        connection.commit()
        
        return {
            "message": "Chain of custody entry recorded",
            "entry": {
                "id": cursor.lastrowid,
                "evidence_id": evidence_id,
                "date_time": date_time,
                "person": person,
                "action": action,
                "remarks": remarks,
            },
        }
    
    finally:
        connection.close()


@app.get("/evidence/{evidence_id}/chain-of-custody")
def get_chain_of_custody(evidence_id: int):
    """Get chain of custody history for evidence."""
    
    connection = get_connection()
    
    # Verify evidence exists
    evidence = connection.execute(
        "SELECT id FROM evidence WHERE id = ?",
        (evidence_id,)
    ).fetchone()
    
    if not evidence:
        connection.close()
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    # Get all chain of custody entries
    rows = connection.execute(
        """
        SELECT
            id,
            evidence_id,
            date_time,
            person,
            action,
            remarks
        FROM chain_of_custody
        WHERE evidence_id = ?
        ORDER BY date_time ASC
        """,
        (evidence_id,)
    ).fetchall()
    
    connection.close()
    
    return {
        "evidence_id": evidence_id,
        "chain_of_custody": [dict(row) for row in rows]
    }


# Phase 5: Report Generation

def generate_pdf_report(case_id: int) -> tuple[str, str]:
    """
    Generate a comprehensive PDF forensic report from existing case data.
    Returns (pdf_path, pdf_sha256)
    """
    connection = get_connection()
    
    try:
        # Fetch case details
        case = connection.execute(
            "SELECT id, case_number, title, investigator, description, created_at FROM cases WHERE id = ?",
            (case_id,)
        ).fetchone()
        
        if not case:
            raise ValueError("Case not found")
        
        case_dict = dict(case)

        def normalize_text(value, placeholder: str = "Not recorded") -> str:
            normalized = (value or "").strip()
            return normalized if normalized else placeholder

        def normalize_display_filename(evidence: dict) -> str:
            filename = (evidence.get("original_filename") or "").strip()
            file_type = (evidence.get("file_type") or "").strip()

            if file_type and not file_type.startswith("."):
                file_type = f".{file_type}"

            if filename and file_type and not filename.lower().endswith(file_type.lower()):
                return filename + file_type

            return filename or "Not recorded"

        def short_sha256(value) -> str:
            sha_value = (value or "").strip()
            if not sha_value:
                return "Not recorded"
            return f"{sha_value[:16]}..." if len(sha_value) > 16 else sha_value
        
        # Fetch evidence
        evidence_rows = connection.execute(
            "SELECT id, evidence_id, original_filename, stored_filename, file_type, file_size, sha256, registered_date, status FROM evidence WHERE case_id = ?",
            (case_id,)
        ).fetchall()
        evidence_list = [dict(row) for row in evidence_rows]

        # Integrity gate: verify all associated evidence before any report artifact is created.
        for evidence in evidence_list:
            evidence_code = evidence["evidence_id"]
            original_path = Path(evidence["stored_filename"])
            working_path = WORKING_DIR / f"{evidence_code}_working.csv"
            verify_evidence_integrity(
                evidence_code,
                original_path,
                working_path,
                evidence["sha256"],
            )
        
        # Fetch findings
        findings_rows = connection.execute(
            """
            SELECT f.finding_id, f.evidence_id, f.event_id, f.timestamp, f.severity, f.reason, 
                   f.technical_observation, f.plain_language_explanation, f.status, f.source,
                   f.rule_id, f.rule_version, f.rule_fingerprint,
                   e.evidence_id as evidence_code
            FROM findings f
            JOIN evidence e ON f.evidence_id = e.id
            WHERE e.case_id = ?
            ORDER BY f.timestamp DESC, f.severity DESC
            """,
            (case_id,)
        ).fetchall()
        findings_list = [dict(row) for row in findings_rows]
        
        # Fetch chain of custody
        coc_rows = connection.execute(
            """
            SELECT c.id, c.date_time, c.person, c.action, c.remarks, e.evidence_id as evidence_code
            FROM chain_of_custody c
            JOIN evidence e ON c.evidence_id = e.id
            WHERE e.case_id = ?
            ORDER BY c.date_time ASC
            """,
            (case_id,)
        ).fetchall()
        coc_list = [dict(row) for row in coc_rows]
        
        # Derive timeline summary rows from case events.
        case_events_payload = get_case_events(case_id)
        summary_map = {}
        for event in case_events_payload.get("events", []):
            key = (
                event.get("timestamp") or "",
                event.get("source") or "",
                event.get("event_type") or "",
                event.get("resource") or "",
            )
            if key not in summary_map:
                summary_map[key] = {
                    "timestamp": key[0],
                    "source": key[1],
                    "event_type": key[2],
                    "resource": key[3],
                    "count": 0,
                }
            summary_map[key]["count"] += 1

        events_summary = list(summary_map.values())
        events_summary.sort(key=lambda item: (item.get("timestamp") or "zzzz", item.get("source") or ""))
        
        # Create PDF
        REPORTS_DIR = Path(os.environ.get("FORENSIC_INSIGHT_REPORTS_DIR", str(BASE_DIR / "reports")))
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        report_generated_utc = datetime.now(timezone.utc)
        report_generated_ist = report_generated_utc.astimezone(
            timezone(timedelta(hours=5, minutes=30))
        )

        report_filename = f"{case_dict['case_number']}_Report_{report_generated_utc.strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = REPORTS_DIR / report_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#0f3460'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0f3460'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )

        table_cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7,
            leading=8,
            wordWrap='CJK'
        )

        # 1. CASE IDENTIFICATION
        story.append(Paragraph("FORENSIC CASE REPORT", title_style))
        story.append(Spacer(1, 0.1*inch))

        # Report metadata block (report SHA-256 is computed post-build and returned by the API response).
        metadata_text = f"""
        <b>Report ID:</b> {report_filename}<br/>
        <b>Case Number:</b> {case_dict['case_number']}<br/>
        <b>Generated Timestamp (IST):</b> {report_generated_ist.strftime('%d %B %Y, %H:%M%S IST')}<br/>
        <b>Generated Timestamp (UTC):</b> {report_generated_utc.strftime('%Y-%m-%d %H:%M%S UTC')}<br/>
        <b>ForensicInsight Version:</b> {normalize_text(app.version, 'Not recorded')}<br/>
        <b>Report SHA-256:</b> Computed after PDF generation and returned by the /cases/{{case_id}}/report API response
        """
        story.append(Paragraph(metadata_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        case_description = normalize_text(case_dict.get('description'), 'Not provided')

        case_info = f"""
        <b>Case Number:</b> {case_dict['case_number']}<br/>
        <b>Case Title:</b> {case_dict['title']}<br/>
        <b>Investigator:</b> {case_dict['investigator']}<br/>
        <b>Report Generated:</b> {report_generated_ist.strftime('%d %B %Y, %H:%M%S IST')}<br/>
        <b>UTC:</b> {report_generated_utc.strftime('%Y-%m-%d %H:%M%S UTC')}<br/>
        <b>Description:</b> {case_description}
        """
        story.append(Paragraph(case_info, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # 2. EXECUTIVE/STAKEHOLDER SUMMARY
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        
        evidence_count = len(evidence_list)
        finding_count = len(findings_list)
        high_count = sum(1 for f in findings_list if f['severity'] == 'HIGH')
        medium_count = sum(1 for f in findings_list if f['severity'] == 'MEDIUM')
        low_count = sum(1 for f in findings_list if f['severity'] == 'LOW')
        
        summary_text = f"""
        This forensic analysis examined {evidence_count} evidence item(s) and generated {finding_count} finding(s) for review based on configured detection rules: {high_count} high-severity, {medium_count} medium-severity, and {low_count} low-severity.
        <br/><br/>
        The findings represent objective observations from the available evidence. This report is provided as a forensic analysis aid 
        and does not constitute a legal determination, court certification, or proof of liability. Professional legal review and 
        appropriate expert evaluation remain necessary for all conclusions.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # 3. EVIDENCE INVENTORY
        story.append(Paragraph("EVIDENCE INVENTORY", heading_style))
        
        if evidence_list:
            evidence_data = [
                ['Evidence ID', 'Original Filename', 'Type', 'Size', 'SHA-256', 'Registered', 'Status']
            ]
            for ev in evidence_list:
                display_filename = normalize_display_filename(ev)
                sha256_value = short_sha256(ev.get('sha256'))
                registered_date = (ev.get('registered_date') or '').strip()
                registered_value = registered_date[:10] if registered_date else 'Not recorded'

                evidence_data.append([
                    Paragraph(normalize_text(ev.get('evidence_id')), table_cell_style),
                    Paragraph(display_filename, table_cell_style),
                    Paragraph(normalize_text(ev.get('file_type')), table_cell_style),
                    Paragraph(str(ev['file_size']) if ev.get('file_size') is not None else 'Not recorded', table_cell_style),
                    Paragraph(sha256_value, table_cell_style),
                    Paragraph(registered_value, table_cell_style),
                    Paragraph(normalize_text(ev.get('status')), table_cell_style)
                ])

            available_width = doc.width
            evidence_col_widths = [
                available_width * 0.14,
                available_width * 0.26,
                available_width * 0.08,
                available_width * 0.09,
                available_width * 0.19,
                available_width * 0.12,
                available_width * 0.12,
            ]
            evidence_table = Table(evidence_data, colWidths=evidence_col_widths, repeatRows=1)
            evidence_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                ('RIGHTPADDING', (1, 1), (1, -1), 10),
                ('LEFTPADDING', (2, 1), (2, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            story.append(evidence_table)

            story.append(Spacer(1, 0.15*inch))
            story.append(Paragraph("EVIDENCE INTEGRITY", heading_style))

            integrity_data = [[
                'Evidence ID',
                'Original Filename',
                'Hash Algorithm',
                'Full SHA-256',
                'Verification Status'
            ]]

            for ev in evidence_list:
                full_hash = normalize_text(ev.get('sha256'))
                verification_status = 'Recorded' if full_hash != 'Not recorded' else 'Not recorded'
                integrity_data.append([
                    Paragraph(normalize_text(ev.get('evidence_id')), table_cell_style),
                    Paragraph(normalize_display_filename(ev), table_cell_style),
                    Paragraph('SHA-256', table_cell_style),
                    Paragraph(full_hash, table_cell_style),
                    Paragraph(verification_status, table_cell_style),
                ])

            integrity_col_widths = [
                available_width * 0.16,
                available_width * 0.23,
                available_width * 0.12,
                available_width * 0.35,
                available_width * 0.14,
            ]
            integrity_table = Table(integrity_data, colWidths=integrity_col_widths, repeatRows=1)
            integrity_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            story.append(integrity_table)
        else:
            story.append(Paragraph("No evidence items found.", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(PageBreak())
        
        # 4. CHAIN OF CUSTODY
        story.append(Paragraph("CHAIN OF CUSTODY", heading_style))
        story.append(Paragraph("Timestamps are shown exactly as stored in chain-of-custody records.", styles['Normal']))
        story.append(Spacer(1, 0.08*inch))
        
        if coc_list:
            coc_data = [
                ['Entry ID', 'Date/Time', 'Evidence ID', 'Person', 'Action', 'Remarks']
            ]
            for c in coc_list:
                date_time = (c.get('date_time') or '').strip()
                entry_id = c.get('id')
                evidence_code = normalize_text(c.get('evidence_code'))
                person = normalize_text(c.get('person'))
                action = normalize_text(c.get('action'))
                remarks = normalize_text(c.get('remarks'))
                date_time_display = date_time if date_time else 'Not recorded'
                if date_time_display not in ('Not recorded',):
                    # Preserve the exact stored value while adding visual line-break opportunities.
                    date_time_display = date_time_display.replace('T', 'T<br/>', 1)
                    date_time_display = date_time_display.replace('+', '<br/>+', 1)

                coc_data.append([
                    Paragraph(normalize_text(str(entry_id) if entry_id is not None else None), table_cell_style),
                    Paragraph(date_time_display, table_cell_style),
                    Paragraph(evidence_code, table_cell_style),
                    Paragraph(person, table_cell_style),
                    Paragraph(action, table_cell_style),
                    Paragraph(remarks, table_cell_style)
                ])

            coc_col_widths = [
                doc.width * 0.08,
                doc.width * 0.32,
                doc.width * 0.15,
                doc.width * 0.14,
                doc.width * 0.10,
                doc.width * 0.21,
            ]
            coc_table = Table(coc_data, colWidths=coc_col_widths, repeatRows=1)
            coc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            story.append(coc_table)
        else:
            story.append(Paragraph("No chain of custody entries found.", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(PageBreak())
        
        # 6. FINDINGS SUMMARY
        story.append(Paragraph("THREAT FINDINGS SUMMARY", heading_style))
        
        if findings_list:
            for finding in findings_list[:20]:  # Limit to first 20 findings per page
                finding_id = normalize_text(finding.get('finding_id'))
                reason = normalize_text(finding.get('reason'))
                severity = normalize_text(finding.get('severity'))
                severity_color = '#721c24' if severity == 'HIGH' else '#856404' if severity == 'MEDIUM' else '#0c5460'
                
                finding_heading_html = f"""
                <b style="color: {severity_color};">[{severity}]</b> {finding_id}
                """
                story.append(Paragraph(finding_heading_html, styles['Normal']))

                evidence_code = normalize_text(finding.get('evidence_code'))
                event_id = normalize_text(finding.get('event_id'))
                timestamp = normalize_text(finding.get('timestamp'))
                source = normalize_text(finding.get('source'))
                status = normalize_text(finding.get('status'))
                technical_observation = normalize_text(finding.get('technical_observation'))
                plain_language_explanation = normalize_text(finding.get('plain_language_explanation'))
                
                finding_details = f"""
                <b>Finding ID:</b> {finding_id} | <b>Severity:</b> {severity}<br/>
                <b>Evidence ID:</b> {evidence_code} | 
                <b>Event ID:</b> {event_id} | 
                <b>Timestamp:</b> {timestamp}<br/>
                <b>Source:</b> {source} | <b>Status:</b> {status}<br/>
                <b>Rule ID:</b> {normalize_text(finding.get('rule_id'))} |
                <b>Rule Version:</b> {normalize_text(finding.get('rule_version'))}<br/>
                <b>Rule Configuration SHA-256:</b> {normalize_text(finding.get('rule_fingerprint'))}<br/>
                <b>Detection Rule/Reason:</b> {reason}<br/>
                <b>Technical Observation:</b> {technical_observation}<br/>
                <b>Stakeholder Explanation:</b> {plain_language_explanation}
                """
                story.append(Paragraph(finding_details, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            if len(findings_list) > 20:
                story.append(Paragraph(f"<i>... and {len(findings_list) - 20} additional findings (see application for complete list)</i>", styles['Normal']))
        else:
            story.append(Paragraph("No findings identified.", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(PageBreak())
        
        # 6. VISUAL ANALYSIS SUMMARY
        story.append(Paragraph("ACTIVITY TIMELINE SUMMARY", heading_style))
        
        if events_summary:
            timeline_text = "Chronological summary of detected activities:<br/><br/>"
            for event in events_summary[:15]:
                timestamp = normalize_text(event.get('timestamp'))
                event_type = normalize_text(event.get('event_type'))
                resource = normalize_text(event.get('resource'))
                source = normalize_text(event.get('source'))
                count = event.get('count')
                count_text = str(count) if count is not None else 'Not recorded'
                timeline_text += f"{timestamp} - {event_type} to {resource} from {source} ({count_text} occurrence(s))<br/>"
            
            if len(events_summary) > 15:
                timeline_text += f"<br/><i>... and {len(events_summary) - 15} additional activities</i>"
            
            story.append(Paragraph(timeline_text, styles['Normal']))
        else:
            story.append(Paragraph("No activity timeline available.", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # 7. METHODOLOGY & DETECTION RULES
        story.append(Paragraph("METHODOLOGY", heading_style))

        evidence_ids_text = ", ".join(
            normalize_text(ev.get('evidence_id')) for ev in evidence_list
        ) if evidence_list else "Not recorded"
        
        methodology = f"""
        <b>Analysis Timestamp (IST):</b> {report_generated_ist.strftime('%d %B %Y, %H:%M:%S IST')}<br/>
        <b>Analysis Timestamp (UTC):</b> {report_generated_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
        <b>ForensicInsight Version:</b> {normalize_text(app.version, 'Not recorded')}<br/>
        <b>Input Evidence IDs:</b> {evidence_ids_text}<br/>
        <b>Rule Version:</b> Not versioned in current prototype<br/>
        <br/>
        <b>Detection Rules Applied:</b><br/>
        • <b>Rule A (R-A):</b> HTTP 403 Forbidden status responses (MEDIUM severity)<br/>
        • <b>Rule B (R-B):</b> POST requests to resources containing sensitive keywords: secret, admin, password, confidential, .zip (HIGH severity)<br/>
        • <b>Rule C (R-C):</b> Multiple POST requests from the same source (MEDIUM severity)<br/>
        • <b>Rule D (R-D):</b> Data transfers exceeding baseline + 2 standard deviations (MEDIUM severity)<br/>
        <br/>
        <b>Analysis Scope:</b> This analysis examined CSV evidence files containing event logs with timestamp, source, event_type, resource, 
        and other fields. Detection rules were applied to identify patterns consistent with configured criteria.
        """
        story.append(Paragraph(methodology, styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(PageBreak())
        
        # 8. LIMITATIONS
        story.append(Paragraph("LIMITATIONS AND DISCLAIMER", heading_style))
        
        limitations = """
        <b>Important Limitations:</b><br/>
        • <b>Identity Unknown:</b> The available evidence does not establish the identity of the person operating the system(s).<br/>
        • <b>Intent Not Determined:</b> The evidence shows actions taken but does not prove intent or motivation.<br/>
        • <b>Incomplete Context:</b> The forensic evidence provides a point-in-time view and may not capture full context of user actions.<br/>
        • <b>Non-Repudiation:</b> Digital evidence does not provide non-repudiation in all cases without additional corroborating evidence.<br/>
        • <b>Clock Accuracy:</b> Timestamp accuracy depends on reliability of system clocks at time of logging.<br/>
        • <b>Log Completeness:</b> Some evidence may be incomplete, corrupted, or missing due to log rotation or system errors.<br/>
        <br/>
        <b>Legal Position:</b> This report is provided as a forensic analysis aid only. It does not constitute:<br/>
        • Legal proof or admissibility in court proceedings<br/>
        • A determination of guilt, innocence, or responsibility<br/>
        • A certified expert opinion without professional review<br/>
        <br/>
        Applicable legal requirements, expert review by qualified personnel, and judicial evaluation remain necessary 
        for all legal determinations and proceedings.
        """
        story.append(Paragraph(limitations, styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # 9. APPENDIX A
        story.append(Paragraph("APPENDIX A - EVIDENCE REFERENCE", heading_style))
        appendix_text = "<b>Evidence ID | Original Filename</b><br/>"
        for ev in evidence_list:
            appendix_text += f"{normalize_text(ev.get('evidence_id'))} | {normalize_display_filename(ev)}<br/>"
        story.append(Paragraph(appendix_text, styles['Normal']))

        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("APPENDIX B - EVIDENCE INTEGRITY", heading_style))
        appendix_integrity = "<b>Evidence ID | Hash Algorithm | Full SHA-256 | Verification Status</b><br/>"
        for ev in evidence_list:
            full_hash = normalize_text(ev.get('sha256'))
            verification_status = 'Recorded' if full_hash != 'Not recorded' else 'Not recorded'
            appendix_integrity += f"{normalize_text(ev.get('evidence_id'))} | SHA-256 | {full_hash} | {verification_status}<br/>"
        story.append(Paragraph(appendix_integrity, styles['Normal']))

        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("APPENDIX C - FINDING REFERENCE", heading_style))
        appendix_findings = "<b>Finding ID | Severity | Evidence ID | Event ID</b><br/>"
        for finding in findings_list[:50]:
            appendix_findings += (
                f"{normalize_text(finding.get('finding_id'))} | "
                f"{normalize_text(finding.get('severity'))} | "
                f"{normalize_text(finding.get('evidence_code'))} | "
                f"{normalize_text(finding.get('event_id'))}<br/>"
            )
        if len(findings_list) > 50:
            appendix_findings += f"<br/><i>... and {len(findings_list) - 50} additional findings</i>"
        story.append(Paragraph(appendix_findings, styles['Normal']))
        
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("APPENDIX D - CHAIN OF CUSTODY", heading_style))
        appendix_coc = "<b>Entry ID | Date/Time | Evidence ID | Person | Action | Remarks</b><br/>"
        for c in coc_list:
            entry_id = c.get('id')
            entry_id_text = normalize_text(str(entry_id) if entry_id is not None else None)
            date_time = normalize_text(c.get('date_time'))
            evidence_code = normalize_text(c.get('evidence_code'))
            person = normalize_text(c.get('person'))
            action = normalize_text(c.get('action'))
            remarks = normalize_text(c.get('remarks'))
            appendix_coc += f"{entry_id_text} | {date_time} | {evidence_code} | {person} | {action} | {remarks}<br/>"
        story.append(Paragraph(appendix_coc, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Calculate SHA-256 of PDF
        with open(pdf_path, 'rb') as f:
            pdf_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        return str(pdf_path), pdf_sha256
    
    finally:
        connection.close()


@app.post("/cases/{case_id}/report")
def generate_case_report(case_id: int):
    """Generate a comprehensive PDF forensic report for a case."""
    connection = get_connection()
    try:
        pdf_path, pdf_sha256 = generate_pdf_report(case_id)
        append_audit_event(
            connection,
            action_type="REPORT_GENERATED",
            case_id=case_id,
            evidence_id=None,
            outcome="SUCCESS",
            metadata={"filename": Path(pdf_path).name, "sha256": pdf_sha256},
        )
        connection.commit()
        
        return {
            "message": "Report generated successfully",
            "report": {
                "filename": Path(pdf_path).name,
                "path": pdf_path,
                "sha256": pdf_sha256,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
        }
    except HTTPException as exc:
        append_audit_event(
            connection,
            action_type="REPORT_GENERATION_FAILED",
            case_id=case_id,
            evidence_id=None,
            outcome="FAILURE",
            metadata={"status_code": exc.status_code, "error": exc.detail},
        )
        connection.commit()
        raise
    except Exception as exc:
        traceback.print_exc()
        append_audit_event(
            connection,
            action_type="REPORT_GENERATION_FAILED",
            case_id=case_id,
            evidence_id=None,
            outcome="FAILURE",
            metadata={"error": str(exc)},
        )
        connection.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {exc}"
        )
    finally:
        connection.close()


@app.get("/cases/{case_id}/download")
def download_report(case_id: int, path: str):
    """Download a generated PDF report."""
    try:
        file_path = Path(path)
        
        # Validate the path is within the reports directory
        reports_dir = Path(os.environ.get("FORENSIC_INSIGHT_REPORTS_DIR", str(BASE_DIR / "reports"))).resolve()
        file_path = file_path.resolve()
        if not file_path.is_relative_to(reports_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(path=file_path, media_type="application/pdf", filename=file_path.name)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/audit")
def get_audit_history_endpoint(case_id: int | None = None, evidence_id: str | None = None):
    """Retrieve audit history; records are read-only through the application."""
    connection = get_connection()
    try:
        return {
            "history": get_audit_history(connection, case_id=case_id, evidence_id=evidence_id),
            "chain_verification": verify_audit_chain(connection),
        }
    finally:
        connection.close()


@app.get("/rules")
def get_rules_endpoint():
    """Return read-only metadata for the active detection rules."""
    return get_rule_metadata()


@app.get("/audit/verify")
def verify_audit_endpoint():
    connection = get_connection()
    try:
        return verify_audit_chain(connection)
    finally:
        connection.close()


# Serve static assets from frontend directory at /static
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
