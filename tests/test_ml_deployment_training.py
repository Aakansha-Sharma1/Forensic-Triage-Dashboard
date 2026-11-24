from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backend.ml.benchmark import CHECKPOINT_SCHEMA_VERSION, CompactScenarioDataset
from backend.ml.deployment_training import (
    DEFAULT_DEPLOYMENT_MODEL_ID,
    DEFAULT_DEPLOYMENT_MODEL_VERSION,
    DEPLOYMENT_POLICY_ID,
    DeploymentTrainingError,
    build_synthetic_verification_fixture,
    select_deployment_configuration_from_inner_checkpoints,
    train_and_save_deployment_random_forest,
)
from backend.ml.experiment_runner import build_nested_loso_plan
from backend.ml.model_artifact import ModelArtifactError, load_model_artifact
from backend.ml.training_data import DEFAULT_EXPERIMENT_CONFIG_PATH, ScenarioLoadSummary


def _load_raw_config() -> tuple[dict, str]:
    raw = DEFAULT_EXPERIMENT_CONFIG_PATH.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _canonical_sha256(obj: object) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _expected_provenance(config: dict, config_sha256: str) -> dict:
    prov = config["dataset_provenance"]
    return {
        "experiment_id": config["experiment_id"],
        "experiment_config_sha256": config_sha256,
        "split_manifest_sha256": prov["split_manifest"]["sha256"],
        "quality_report_sha256": prov["quality_report"]["sha256"],
        "scenario_labelled_sha256": {
            k: v["labelled_records_sha256"] for k, v in sorted(prov["scenarios"].items())
        },
    }


def _write_synthetic_inner_checkpoints(
    ckpt_dir: Path,
    config: dict,
    config_sha256: str,
    *,
    score_fn,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plan = build_nested_loso_plan(DEFAULT_EXPERIMENT_CONFIG_PATH)
    rf_spec = next(spec for spec in plan.model_specs if spec.name == "RandomForestClassifier")
    provenance = _expected_provenance(config, config_sha256)

    for fold in plan.outer_folds:
        for inner in fold.inner_candidates:
            for param_idx, hp in enumerate(rf_spec.parameter_combinations):
                for seed in plan.random_seeds:
                    unit_key = (
                        f"inner__RandomForestClassifier__class_weighted__{fold.fold_id}__"
                        f"{inner.candidate_id}__p{param_idx}__s{seed}"
                    )
                    observations = [
                        {
                            "candidate_id": inner.candidate_id,
                            "fold_id": fold.fold_id,
                            "hyperparameters": dict(hp),
                            "macro_f1": float(score_fn(param_idx, float(thr))),
                            "model_name": "RandomForestClassifier",
                            "partition_role": "validation",
                            "seed": int(seed),
                            "threshold": float(thr),
                            "validation_scenario": inner.validation_scenarios[0],
                            "weighting_regime": "class_weighted",
                        }
                        for thr in plan.candidate_thresholds
                    ]
                    payload = {"observations": observations}
                    record = {
                        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "unit_type": "inner_fit",
                        "unit_key": {
                            "candidate_id": inner.candidate_id,
                            "candidate_thresholds": [float(t) for t in plan.candidate_thresholds],
                            "fold_id": fold.fold_id,
                            "hyperparameters": dict(hp),
                            "model_name": "RandomForestClassifier",
                            "param_index": param_idx,
                            "seed": int(seed),
                            "validation_scenario": inner.validation_scenarios[0],
                            "weighting_regime": "class_weighted",
                        },
                        "provenance": provenance,
                        "payload": payload,
                        "payload_sha256": _canonical_sha256(payload),
                    }
                    (ckpt_dir / f"{unit_key}.json").write_text(
                        json.dumps(record, sort_keys=True), encoding="utf-8"
                    )


def _make_synthetic_scenario_datasets(config: dict) -> dict[str, CompactScenarioDataset]:
    scenarios = sorted(config["dataset_provenance"]["scenarios"].keys())
    rng = np.random.default_rng(12345)
    datasets: dict[str, CompactScenarioDataset] = {}

    for idx, scenario_id in enumerate(scenarios):
        n_normal = 40
        n_attack = 60
        normal_rows = np.column_stack(
            [
                rng.uniform(200.0, 900.0, size=n_normal),
                rng.uniform(1.0, 6.0, size=n_normal),
                np.zeros(n_normal),
                rng.uniform(0.0, 1.0, size=n_normal),
                rng.uniform(5.0, 60.0, size=n_normal),
                rng.uniform(1.0, 4.0, size=n_normal),
                np.zeros(n_normal),
                np.zeros((n_normal, 6)),
            ]
        )
        attack_rows = np.column_stack(
            [
                rng.uniform(100.0, 400.0, size=n_attack),
                rng.uniform(150.0, 500.0, size=n_attack),
                rng.uniform(120.0, 480.0, size=n_attack),
                rng.uniform(0.0, 15.0, size=n_attack),
                rng.uniform(0.0, 0.2, size=n_attack),
                rng.uniform(100.0, 450.0, size=n_attack),
                np.ones(n_attack),
                np.zeros((n_attack, 6)),
            ]
        )
        # Include a few deterministic missing values in continuous columns with matching flags.
        normal_rows[0, 0] = np.nan
        normal_rows[0, 11] = 1.0
        attack_rows[0, 4] = np.nan
        attack_rows[0, 12] = 1.0

        features = np.vstack([normal_rows, attack_rows]).astype(np.float64)
        labels = np.array([0] * n_normal + [1] * n_attack, dtype=np.int64)
        categories = tuple(
            () if lbl == 0 else ("dirb",) for lbl in labels.tolist()
        )
        scen_ref = config["dataset_provenance"]["scenarios"][scenario_id]
        summary = ScenarioLoadSummary(
            scenario_id=scenario_id,
            label_manifest_path=scen_ref["label_manifest_path"],
            label_manifest_sha256=scen_ref["label_manifest_sha256"],
            labelled_records_sha256=scen_ref["labelled_records_sha256"],
            prepared_manifest_sha256=scen_ref["prepared_manifest_sha256"],
            prepared_records_sha256=scen_ref["prepared_records_sha256"],
            total_records_read=n_normal + n_attack + 15,
            included_normal_count=n_normal,
            included_attack_count=n_attack,
            excluded_unknown_count=15,
        )
        datasets[scenario_id] = CompactScenarioDataset(
            scenario_id=scenario_id,
            features=features,
            labels=labels,
            attack_categories=categories,
            event_ids_sha256=hashlib.sha256(f"{scenario_id}-{idx}".encode("utf-8")).hexdigest(),
            load_summary=summary,
        )
    return datasets


def test_policy_selection_is_deterministic_and_applies_tie_breaking(tmp_path: Path) -> None:
    config, config_sha256 = _load_raw_config()
    ckpt_dir = tmp_path / "ckpts"

    # Tie between threshold 0.4 and 0.6 (equidistant from 0.5) on param_idx 1 and 2:
    # smaller_threshold (0.4) must beat 0.6, and earlier grid order (param_idx=1) must beat param_idx=2.
    def score_fn(param_idx: int, thr: float) -> float:
        if param_idx in (1, 2) and thr in (0.4, 0.6):
            return 0.995
        return 0.85

    _write_synthetic_inner_checkpoints(ckpt_dir, config, config_sha256, score_fn=score_fn)

    # Plant a decoy outer-test checkpoint to prove outer-test files are never consulted.
    (ckpt_dir / "outer__RandomForestClassifier__class_weighted__outer_test_fox__s42.json").write_text(
        "CORRUPTED_OUTER_FILE_MUST_NOT_BE_READ", encoding="utf-8"
    )

    selection = select_deployment_configuration_from_inner_checkpoints(
        config,
        config_sha256=config_sha256,
        checkpoint_dir=ckpt_dir,
    )
    assert selection.policy_id == DEPLOYMENT_POLICY_ID
    assert selection.inner_checkpoint_count == 240
    assert selection.inner_validation_observation_count == 60
    assert selection.selected_threshold == 0.4
    assert selection.selected_hyperparameters == {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
    }
    assert selection.deployment_random_state == 42
    assert selection.to_dict()["outer_test_used_for_selection"] is False


def test_policy_selection_rejects_outer_test_leakage_and_tampering(tmp_path: Path) -> None:
    config, config_sha256 = _load_raw_config()
    ckpt_dir = tmp_path / "ckpts"
    _write_synthetic_inner_checkpoints(
        ckpt_dir, config, config_sha256, score_fn=lambda p, t: 0.9
    )

    sample_file = sorted(ckpt_dir.glob("inner__*.json"))[0]
    original = json.loads(sample_file.read_text(encoding="utf-8"))

    # Case 1: Tampered payload without updating payload_sha256.
    tampered = json.loads(json.dumps(original))
    tampered["payload"]["observations"][0]["macro_f1"] = 0.9999
    sample_file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DeploymentTrainingError, match="SHA-256 mismatch"):
        select_deployment_configuration_from_inner_checkpoints(
            config, config_sha256=config_sha256, checkpoint_dir=ckpt_dir
        )

    # Case 2: Non-validation partition_role even with valid payload_sha256.
    leaked = json.loads(json.dumps(original))
    leaked["payload"]["observations"][0]["partition_role"] = "test"
    leaked["payload_sha256"] = _canonical_sha256(leaked["payload"])
    sample_file.write_text(json.dumps(leaked), encoding="utf-8")
    with pytest.raises(DeploymentTrainingError, match="non-validation observation"):
        select_deployment_configuration_from_inner_checkpoints(
            config, config_sha256=config_sha256, checkpoint_dir=ckpt_dir
        )

    # Case 3: Missing checkpoint fails closed.
    sample_file.unlink()
    with pytest.raises(DeploymentTrainingError, match="Missing required inner-validation checkpoint"):
        select_deployment_configuration_from_inner_checkpoints(
            config, config_sha256=config_sha256, checkpoint_dir=ckpt_dir
        )


def test_synthetic_deployment_training_creates_verified_reloadable_artifact(tmp_path: Path) -> None:
    config, config_sha256 = _load_raw_config()
    ckpt_dir = tmp_path / "ckpts"
    _write_synthetic_inner_checkpoints(
        ckpt_dir,
        config,
        config_sha256,
        score_fn=lambda p, t: 0.995 if (p == 2 and t == 0.5) else 0.90,
    )
    datasets = _make_synthetic_scenario_datasets(config)
    trusted_root = tmp_path / "models"

    result = train_and_save_deployment_random_forest(
        config_path=DEFAULT_EXPERIMENT_CONFIG_PATH,
        checkpoint_dir=ckpt_dir,
        trusted_root=trusted_root,
        model_id=DEFAULT_DEPLOYMENT_MODEL_ID,
        model_version=DEFAULT_DEPLOYMENT_MODEL_VERSION,
        n_jobs=1,
        scenario_datasets=datasets,
        verified_config=config,
        config_sha256=config_sha256,
        created_at_utc="2026-11-24T00:00:00+00:00",
    )

    assert result.verification_predictions_match is True
    assert result.total_training_records == 400
    assert result.normal_records == 160
    assert result.attack_associated_records == 240
    assert result.excluded_unknown_records == 60
    assert result.policy_selection.selected_hyperparameters == {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    }
    assert result.policy_selection.selected_threshold == 0.5
    assert result.manifest_path.is_file()
    assert result.model_file_path.is_file()

    # Provenance & limitations checks.
    prov = result.manifest["training_dataset_provenance"]
    assert prov["unknown_records_excluded"] is True
    assert prov["benchmark_outer_test_predictions_excluded"] is True
    assert prov["deployment_policy"]["outer_test_used_for_selection"] is False
    limitations_joined = " ".join(result.manifest["limitations"])
    assert "NOT undergone independent external evaluation" in limitations_joined
    assert result.manifest["score_semantics"]["calibrated_probability"] is False

    # Secure loading requires explicit operator_approved=True.
    with pytest.raises(ModelArtifactError, match="operator_approved=True"):
        load_model_artifact(result.artifact_dir, trusted_root=trusted_root)

    reloaded = load_model_artifact(
        result.artifact_dir,
        trusted_root=trusted_root,
        operator_approved=True,
        expected_model_id=DEFAULT_DEPLOYMENT_MODEL_ID,
        expected_model_version=DEFAULT_DEPLOYMENT_MODEL_VERSION,
        expected_model_sha256=result.model_file_sha256,
    )
    preds = reloaded.predict_batch(build_synthetic_verification_fixture())
    assert reloaded.pipeline.named_steps["classifier"].n_jobs == 1
    assert preds == result.verification_predictions
    assert all(p.is_calibrated_probability is False for p in preds)

    # Refuses to overwrite an existing artifact.
    with pytest.raises(FileExistsError):
        train_and_save_deployment_random_forest(
            config_path=DEFAULT_EXPERIMENT_CONFIG_PATH,
            checkpoint_dir=ckpt_dir,
            trusted_root=trusted_root,
            model_id=DEFAULT_DEPLOYMENT_MODEL_ID,
            model_version=DEFAULT_DEPLOYMENT_MODEL_VERSION,
            n_jobs=1,
            scenario_datasets=datasets,
            verified_config=config,
            config_sha256=config_sha256,
        )


def test_deployment_training_does_not_import_fastapi_backend() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, backend.ml.deployment_training; "
                "assert 'backend.main' not in sys.modules; "
                "assert 'fastapi' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
