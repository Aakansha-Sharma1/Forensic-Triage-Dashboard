import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import sqlite3
import threading


AUDIT_TABLE = "audit_events"

_AUDIT_LOCK = threading.RLock()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_metadata(metadata: dict | None) -> str:
    return json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def calculate_audit_hash(
    event_id: str,
    timestamp: str,
    action_type: str,
    case_id: int | None,
    evidence_id: int | str | None,
    outcome: str,
    metadata: dict | None,
    previous_hash: str,
) -> str:
    payload = {
        "event_id": event_id,
        "timestamp": timestamp,
        "action_type": action_type,
        "case_id": case_id,
        "evidence_id": evidence_id,
        "outcome": outcome,
        "metadata": metadata or {},
        "previous_hash": previous_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def initialize_audit_table(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            action_type TEXT NOT NULL,
            case_id INTEGER,
            evidence_id TEXT,
            outcome TEXT NOT NULL,
            metadata TEXT NOT NULL,
            previous_hash TEXT NOT NULL,
            current_hash TEXT NOT NULL UNIQUE
        )
        """
    )

    existing_columns = {
        row[1]: row[2]
        for row in connection.execute("PRAGMA table_info(audit_events)").fetchall()
    }
    if "case_id" not in existing_columns:
        connection.execute("ALTER TABLE audit_events ADD COLUMN case_id INTEGER")
    if "evidence_id" not in existing_columns:
        connection.execute("ALTER TABLE audit_events ADD COLUMN evidence_id TEXT")


def append_audit_event(
    connection,
    action_type: str,
    case_id: int | None,
    evidence_id: int | str | None,
    outcome: str,
    metadata: dict | None = None,
) -> dict:
    owns_transaction = not connection.in_transaction

    def append_record() -> dict:
        latest = connection.execute(
            "SELECT current_hash FROM audit_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        previous_hash = latest[0] if latest else "0" * 64

        event_id = f"AUD-{uuid4().hex.upper()}"
        timestamp = utc_timestamp()
        metadata_text = canonical_metadata(metadata)
        metadata_object = json.loads(metadata_text)
        normalized_evidence_id = str(evidence_id) if evidence_id is not None else None
        current_hash = calculate_audit_hash(
            event_id,
            timestamp,
            action_type,
            case_id,
            normalized_evidence_id,
            outcome,
            metadata_object,
            previous_hash,
        )

        connection.execute(
            """
            INSERT INTO audit_events (
                event_id, timestamp, action_type, case_id, evidence_id,
                outcome, metadata, previous_hash, current_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                timestamp,
                action_type,
                case_id,
                normalized_evidence_id,
                outcome,
                metadata_text,
                previous_hash,
                current_hash,
            ),
        )

        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "action_type": action_type,
            "case_id": case_id,
            "evidence_id": normalized_evidence_id,
            "outcome": outcome,
            "metadata": metadata_object,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
        }

    if owns_transaction:
        with _AUDIT_LOCK:
            connection.execute("BEGIN IMMEDIATE")
            try:
                record = append_record()
                connection.commit()
                return record
            except Exception:
                connection.rollback()
                raise

    # Caller-owned transactions are already serialized by SQLite's transaction lock.
    # Do not acquire _AUDIT_LOCK here: the caller may hold BEGIN IMMEDIATE while waiting
    # for the lock, while a self-owned writer could hold _AUDIT_LOCK while waiting for SQLite.
    try:
        return append_record()
    except Exception:
        raise


def _row_as_dict(row, column_names: tuple[str, ...]) -> dict:
    """Convert sqlite3.Row or tuple results into a stable dictionary."""
    if hasattr(row, "keys"):
        return {column: row[column] for column in column_names}
    return dict(zip(column_names, row))


def verify_audit_chain(connection) -> dict:
    column_names = (
        "id", "event_id", "timestamp", "action_type", "case_id",
        "evidence_id", "outcome", "metadata", "previous_hash", "current_hash",
    )
    rows = connection.execute(
        "SELECT id, event_id, timestamp, action_type, case_id, evidence_id, outcome, metadata, previous_hash, current_hash "
        "FROM audit_events ORDER BY id ASC"
    ).fetchall()

    issues = []
    expected_previous = "0" * 64

    for row in rows:
        record = _row_as_dict(row, column_names)
        try:
            metadata = json.loads(record["metadata"])
        except (json.JSONDecodeError, TypeError):
            metadata = None
            issues.append({
                "event_id": record["event_id"],
                "issue": "invalid_metadata",
            })

        if record["previous_hash"] != expected_previous:
            issues.append({
                "event_id": record["event_id"],
                "issue": "previous_hash_mismatch",
                "expected": expected_previous,
                "observed": record["previous_hash"],
            })

        evidence_id = record["evidence_id"]
        try:
            case_id = record["case_id"]
            calculated = calculate_audit_hash(
                record["event_id"],
                record["timestamp"],
                record["action_type"],
                case_id,
                evidence_id,
                record["outcome"],
                metadata,
                record["previous_hash"],
            )
        except Exception:
            calculated = ""

        if calculated != record["current_hash"]:
            issues.append({
                "event_id": record["event_id"],
                "issue": "current_hash_mismatch",
                "expected": calculated,
                "observed": record["current_hash"],
            })

        expected_previous = record["current_hash"]

    return {
        "valid": len(issues) == 0,
        "event_count": len(rows),
        "issues": issues,
    }


def protect_audit_records(connection) -> None:
    """Install SQLite triggers preventing application-level audit record changes/deletion."""
    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_update")
    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_delete")
    connection.execute(
        """
        CREATE TRIGGER prevent_audit_update
        BEFORE UPDATE ON audit_events
        BEGIN
            SELECT RAISE(ABORT, 'audit_events records are read-only');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER prevent_audit_delete
        BEFORE DELETE ON audit_events
        BEGIN
            SELECT RAISE(ABORT, 'audit_events records cannot be deleted');
        END
        """
    )


def get_audit_history(connection, case_id: int | None = None, evidence_id: int | str | None = None) -> list:
    """Return audit records in insertion order without exposing mutation APIs."""
    clauses = []
    parameters = []
    if case_id is not None:
        clauses.append("case_id = ?")
        parameters.append(case_id)
    if evidence_id is not None:
        clauses.append("evidence_id = ?")
        parameters.append(str(evidence_id))

    query = (
        "SELECT id, event_id, timestamp, action_type, case_id, evidence_id, outcome, metadata, previous_hash, current_hash "
        "FROM audit_events"
    )
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id ASC"

    column_names = (
        "id", "event_id", "timestamp", "action_type", "case_id",
        "evidence_id", "outcome", "metadata", "previous_hash", "current_hash",
    )
    rows = connection.execute(query, tuple(parameters)).fetchall()
    history = []
    for row in rows:
        record = _row_as_dict(row, column_names)
        try:
            record["metadata"] = json.loads(record["metadata"])
        except (json.JSONDecodeError, TypeError):
            record["metadata"] = None
        history.append(record)
    return history
