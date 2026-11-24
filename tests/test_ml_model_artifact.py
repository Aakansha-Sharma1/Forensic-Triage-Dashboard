import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backend.ml.baselines import train_random_forest_baseline
from backend.ml.model_artifact import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    ModelArtifactError,
    load_model_artifact,
    save_model_artifact,
)
from backend.ml.training_data import EXPECTED_FEATURE_NAMES


def _synthetic_training_data():
    rows = []
    labels = []
    for i in range(12):
        rows.append(
            [
                180 + i * 10,
                1 + (i % 2),
                0,
                0,
                None if i == 0 else 40.0 + i,
                1 + (i % 3),
                0,
                0, 0, 0, 0, 0,
                1 if i == 0 else 0,
            ]
        )
        labels.append(0)
    for i in range(12):
        rows.append(
            [
                7500 + i * 150,
                55 + i * 2,
                30 + i,
                12 + (i % 3),
                0.5 + (i % 2) * 0.2,
                40 + i,
                1,
                0, 0, 0, 0, 0, 0,
            ]
        )
        labels.append(1)
    return rows, labels


def _valid_provenance() -> dict:
    return {
        "dataset_name": "Synthetic Unit Test Dataset",
        "dataset_version": "synthetic-1.0",
        "scenarios": ["fox", "harrison"],
        "scenario_labelled_sha256": {
            "fox": "a" * 64,
            "harrison": "b" * 64,
        },
        "unknown_records_excluded": True,
        "benchmark_outer_test_predictions_excluded": True,
    }


def _save_valid_rf_artifact(tmp_path: Path):
    rows, labels = _synthetic_training_data()
    fitted = train_random_forest_baseline(
        rows,
        labels,
        weighting_regime="class_weighted",
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
    )
    trusted_root = tmp_path / "trusted_models"
    artifact_dir = trusted_root / "rf-synthetic" / "1.0.0"
    manifest_path, manifest = save_model_artifact(
        fitted,
        artifact_dir,
        trusted_root=trusted_root,
        model_id="rf-synthetic",
        model_version="1.0.0",
        decision_threshold=0.5,
        training_dataset_provenance=_valid_provenance(),
        training_config_sha256="c" * 64,
        intended_use="Synthetic unit test inference verification.",
        limitations=["Synthetic test artifact only; not trained on real data."],
        created_at_utc="2026-11-24T00:00:00+00:00",
    )
    return trusted_root, artifact_dir, manifest_path, manifest, rows


def test_save_load_and_deterministic_inference_without_backend_main_import(tmp_path):
    trusted_root, artifact_dir, manifest_path, manifest, rows = _save_valid_rf_artifact(tmp_path)

    assert manifest_path.is_file()
    assert manifest["artifact_schema_version"] == MODEL_ARTIFACT_SCHEMA_VERSION
    assert manifest["score_semantics"]["calibrated_probability"] is False
    # Verify importing backend.ml.model_artifact in a clean process never imports backend.main
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, backend.ml.model_artifact; assert 'backend.main' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0

    loaded = load_model_artifact(
        artifact_dir,
        trusted_root=trusted_root,
        operator_approved=True,
        expected_model_id="rf-synthetic",
        expected_model_version="1.0.0",
    )

    preds_1 = loaded.predict_batch(rows)
    preds_2 = loaded.predict_batch(rows)
    assert preds_1 == preds_2
    assert len(preds_1) == len(rows)

    first_dict = preds_1[0].to_dict()
    assert first_dict["model_id"] == "rf-synthetic"
    assert first_dict["model_version"] == "1.0.0"
    assert first_dict["decision_threshold"] == 0.5
    assert first_dict["is_calibrated_probability"] is False
    assert 0.0 <= first_dict["model_score"] <= 1.0
    assert first_dict["predicted_label"] in (0, 1)

    # Single-event inference via feature mapping matches batch inference identically
    feat_map = dict(zip(EXPECTED_FEATURE_NAMES, rows[0]))
    single_pred = loaded.predict_one(feat_map)
    assert single_pred == preds_1[0]
    assert json.dumps(single_pred.to_dict(), sort_keys=True) == json.dumps(
        first_dict, sort_keys=True
    )


def test_loader_enforces_operator_approval_trusted_root_and_tamper_rejection(tmp_path):
    trusted_root, artifact_dir, manifest_path, manifest, rows = _save_valid_rf_artifact(tmp_path)

    # 1. Missing explicit operator approval
    with pytest.raises(ModelArtifactError, match="operator_approved=True"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=False)

    # 2. Path outside trusted_root
    with pytest.raises(ModelArtifactError, match="trusted root"):
        load_model_artifact(
            artifact_dir, trusted_root=tmp_path / "other_root", operator_approved=True
        )

    # 3. Unknown / mismatched model version
    with pytest.raises(ModelArtifactError, match="Model version mismatch"):
        load_model_artifact(
            artifact_dir,
            trusted_root=trusted_root,
            operator_approved=True,
            expected_model_version="2.0.0",
        )

    # 4. Corrupted / tampered model.joblib rejected before deserialization
    model_path = artifact_dir / "model.joblib"
    original_model_bytes = model_path.read_bytes()
    model_path.write_bytes(original_model_bytes + b"tamper")
    with pytest.raises(ModelArtifactError, match="size mismatch"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=True)
    # Same-size byte flip triggers SHA-256 mismatch
    flipped = bytearray(original_model_bytes)
    flipped[0] ^= 0xFF
    model_path.write_bytes(bytes(flipped))
    with pytest.raises(ModelArtifactError, match="SHA-256 mismatch"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=True)
    model_path.write_bytes(original_model_bytes)

    # 5. Unknown artifact_schema_version, feature-order mismatch, invalid threshold
    original_manifest = manifest_path.read_bytes()

    bad_schema = dict(manifest, artifact_schema_version="9.9.9")
    manifest_path.write_text(json.dumps(bad_schema), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="artifact_schema_version"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=True)

    bad_order = dict(manifest, ordered_feature_names=list(reversed(EXPECTED_FEATURE_NAMES)))
    manifest_path.write_text(json.dumps(bad_order), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="ordered_feature_names"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=True)

    bad_thresh = dict(manifest, decision_threshold=1.5)
    manifest_path.write_text(json.dumps(bad_thresh), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="decision_threshold"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=True)

    bad_family = dict(manifest, model_family="XGBoostClassifier")
    manifest_path.write_text(json.dumps(bad_family), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="model_family"):
        load_model_artifact(artifact_dir, trusted_root=trusted_root, operator_approved=True)

    manifest_path.write_bytes(original_manifest)


def test_inference_rejects_malformed_and_non_finite_inputs(tmp_path):
    trusted_root, artifact_dir, _, _, rows = _save_valid_rf_artifact(tmp_path)
    loaded = load_model_artifact(
        artifact_dir, trusted_root=trusted_root, operator_approved=True
    )

    with pytest.raises(ModelArtifactError, match="expected 13"):
        loaded.predict_one([1, 2, 3])

    bad_inf = list(rows[0])
    bad_inf[4] = float("inf")
    with pytest.raises(ModelArtifactError, match="Non-finite"):
        loaded.predict_one(bad_inf)

    bad_neg = list(rows[0])
    bad_neg[0] = -10
    with pytest.raises(ModelArtifactError, match="Negative values"):
        loaded.predict_one(bad_neg)

    with pytest.raises(ModelArtifactError, match="Feature mapping schema mismatch"):
        loaded.predict_one({"transfer_size_bytes": 100})
