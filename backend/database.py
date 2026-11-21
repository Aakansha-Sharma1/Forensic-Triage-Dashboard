import sqlite3
import os
from pathlib import Path

from .audit import initialize_audit_table, protect_audit_records

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("FORENSIC_INSIGHT_DB_PATH", str(BASE_DIR / "forensic.db")))


def get_connection():
    database_path = Path(os.environ.get("FORENSIC_INSIGHT_DB_PATH", str(BASE_DIR / "forensic.db")))
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            investigator TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            evidence_id TEXT NOT NULL UNIQUE,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            source_description TEXT,
            status TEXT NOT NULL,
            registered_date TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases(id)
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chain_of_custody (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_id INTEGER NOT NULL,
            date_time TEXT NOT NULL,
            person TEXT NOT NULL,
            action TEXT NOT NULL,
            remarks TEXT,
            FOREIGN KEY (evidence_id) REFERENCES evidence(id)
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_id TEXT NOT NULL UNIQUE,
            evidence_id INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT,
            event_type TEXT NOT NULL,
            resource TEXT,
            reason TEXT NOT NULL,
            severity TEXT NOT NULL,
            technical_observation TEXT NOT NULL,
            plain_language_explanation TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (evidence_id) REFERENCES evidence(id)
        )
        """
    )

    initialize_audit_table(connection)
    protect_audit_records(connection)

    connection.commit()
    connection.close()