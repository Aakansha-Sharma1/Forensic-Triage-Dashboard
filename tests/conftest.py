import os
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def isolated_app(tmp_path_factory):
    root = tmp_path_factory.mktemp("forensic-insight-isolated")
    db_path = root / "forensic.db"
    upload_dir = root / "uploads" / "original"
    working_dir = root / "uploads" / "working"
    reports_dir = root / "reports"

    env_names = {
        "FORENSIC_INSIGHT_DB_PATH": str(db_path),
        "FORENSIC_INSIGHT_UPLOAD_DIR": str(upload_dir),
        "FORENSIC_INSIGHT_WORKING_DIR": str(working_dir),
        "FORENSIC_INSIGHT_REPORTS_DIR": str(reports_dir),
    }
    previous_env = {name: os.environ.get(name) for name in env_names}
    os.environ.update(env_names)

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from backend.database import initialize_database
    from backend.main import app

    initialize_database()

    try:
        with TestClient(app) as client:
            yield client, {
                "root": root,
                "db": db_path,
                "upload": upload_dir,
                "working": working_dir,
                "reports": reports_dir,
            }
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def reset_isolated_state(isolated_app):
    client, paths = isolated_app

    import sqlite3

    connection = sqlite3.connect(paths["db"])
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_update")
    connection.execute("DROP TRIGGER IF EXISTS prevent_audit_delete")
    for table in ("findings", "chain_of_custody", "evidence", "cases", "audit_events"):
        connection.execute(f"DELETE FROM {table}")
    connection.execute("DELETE FROM sqlite_sequence")
    connection.execute(
        "CREATE TRIGGER prevent_audit_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit_events records are read-only'); END"
    )
    connection.execute(
        "CREATE TRIGGER prevent_audit_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit_events records cannot be deleted'); END"
    )
    connection.commit()
    connection.close()

    for directory in (paths["upload"], paths["working"], paths["reports"]):
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    yield
