from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path

import pytest

from backend.ml.benchmark import (
    BenchmarkError,
    BenchmarkIncompleteError,
    VerifiedDatasetScenarioProvider,
    run_full_nested_loso_benchmark,
    write_benchmark_artifact,
)
from backend.ml.features import engineer_features
from backend.ml.schema import FEATURE_SCHEMA, CanonicalEvent
from backend.rules import calculate_rule_fingerprint


SCENARIOS = ("fox", "harrison", "russellmitchell", "santos")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, _sha(raw)


def _build_synthetic_4scenario_workspace(tmp_path: Path, *, tamper_scenario: str | None = None):
    summaries = {}
    scenarios_cfg = {}

    for s_idx, scenario in enumerate(SCENARIOS):
        s_dir = tmp_path / "processed" / scenario
        s_dir.mkdir(parents=True)
        covered_name = "intranet/access.log"
        unknown_name = "web/access.log"
        cov_sha = _sha(f"{scenario}:cov".encode())
        unk_sha = _sha(f"{scenario}:unk".encode())
        gt_sha = _sha(f"{scenario}:gt".encode())
        gt_ref = f"raw/{scenario}/labels/{covered_name}"

        events = []
        # 8 normal, 8 attack (with 403/POST-admin so rules fire), 2 unknown
        for i in range(8):
            events.append(
                CanonicalEvent(
                    case_id=f"case-{scenario}",
                    evidence_id=f"ev-{scenario}-cov",
                    event_id=f"{covered_name}:{i + 1}",
                    timestamp=datetime(2022, 1, 15 + s_idx, 0, i, 0, tzinfo=timezone.utc),
                    source_ip="192.0.2.10",
                    event_type="GET",
                    resource=f"/home/{i}",
                    response_status="200",
                    transfer_size=120 + i * 5,
                    original_evidence_reference=f"raw/{scenario}/gather/{covered_name}",
                )
            )
        for i in range(8):
            events.append(
                CanonicalEvent(
                    case_id=f"case-{scenario}",
                    evidence_id=f"ev-{scenario}-cov",
                    event_id=f"{covered_name}:{9 + i}",
                    timestamp=datetime(2022, 1, 15 + s_idx, 0, 10 + i, 0, tzinfo=timezone.utc),
                    source_ip="198.51.100.77",
                    event_type="POST",
                    resource="/admin/secret" if i < 4 else "/api/upload",
                    response_status="403" if i % 2 == 0 else "200",
                    transfer_size=8000 + i * 200,
                    original_evidence_reference=f"raw/{scenario}/gather/{covered_name}",
                )
            )
        unk_events = [
            CanonicalEvent(
                case_id=f"case-{scenario}",
                evidence_id=f"ev-{scenario}-unk",
                event_id=f"{unknown_name}:{i + 1}",
                timestamp=datetime(2022, 1, 15 + s_idx, 0, 30 + i, 0, tzinfo=timezone.utc),
                source_ip="203.0.113.5",
                event_type="GET",
                resource="/public",
                response_status="200",
                transfer_size=80,
                original_evidence_reference=f"raw/{scenario}/gather/{unknown_name}",
            )
            for i in range(2)
        ]

        cov_feats = engineer_features(events)
        unk_feats = engineer_features(unk_events)

        rows = []
        for idx, fr in enumerate(cov_feats):
            is_attack = idx >= 8
            ev = fr.event
            rows.append(
                {
                    "event": {
                        "case_id": ev.case_id,
                        "evidence_id": ev.evidence_id,
                        "event_id": ev.event_id,
                        "timestamp": ev.timestamp.isoformat(),
                        "source_ip": ev.source_ip,
                        "event_type": ev.event_type,
                        "resource": ev.resource,
                        "response_status": ev.response_status,
                        "transfer_size": ev.transfer_size,
                        "original_evidence_reference": ev.original_evidence_reference,
                    },
                    "provenance": {
                        "scenario_id": scenario,
                        "source_filename": covered_name,
                        "source_line_number": idx + 1,
                        "source_sha256": cov_sha,
                    },
                    "features": dict(fr.features),
                    "ground_truth": (
                        {"labels": ["foothold", "dirb"], "rules": {"foothold": ["r1"]}}
                        if is_attack
                        else None
                    ),
                    "ground_truth_provenance": (
                        {
                            "filename": "access.log",
                            "reference": gt_ref,
                            "sha256": gt_sha,
                            "label_file_line_number": idx - 7,
                        }
                        if is_attack
                        else None
                    ),
                    "research_label": 1 if is_attack else 0,
                    "research_label_name": "attack_associated" if is_attack else "normal",
                    "research_label_basis": (
                        "explicit_ground_truth"
                        if is_attack
                        else "absence_from_verified_ground_truth_file"
                    ),
                    "research_label_provenance": {
                        "policy_id": "ait-lds-v2.1-explicit-ground-truth-v1",
                        "label_schema_version": "1.0.0",
                        "ground_truth_reference": gt_ref,
                        "ground_truth_sha256": gt_sha,
                    },
                }
            )
        for idx, fr in enumerate(unk_feats):
            ev = fr.event
            rows.append(
                {
                    "event": {
                        "case_id": ev.case_id,
                        "evidence_id": ev.evidence_id,
                        "event_id": ev.event_id,
                        "timestamp": ev.timestamp.isoformat(),
                        "source_ip": ev.source_ip,
                        "event_type": ev.event_type,
                        "resource": ev.resource,
                        "response_status": ev.response_status,
                        "transfer_size": ev.transfer_size,
                        "original_evidence_reference": ev.original_evidence_reference,
                    },
                    "provenance": {
                        "scenario_id": scenario,
                        "source_filename": unknown_name,
                        "source_line_number": idx + 1,
                        "source_sha256": unk_sha,
                    },
                    "features": dict(fr.features),
                    "ground_truth": None,
                    "ground_truth_provenance": None,
                    "research_label": None,
                    "research_label_name": "unknown",
                    "research_label_basis": "insufficient_verified_label_coverage",
                    "research_label_provenance": {
                        "policy_id": "ait-lds-v2.1-explicit-ground-truth-v1",
                        "label_schema_version": "1.0.0",
                        "ground_truth_reference": None,
                        "ground_truth_sha256": None,
                    },
                }
            )

        prep_bytes = f"prepared-{scenario}\n".encode("utf-8")
        (s_dir / "prepared_events.jsonl").write_bytes(prep_bytes)
        prep_sha = _sha(prep_bytes)

        lab_bytes = "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
        ).encode("utf-8")
        lab_path = s_dir / "labelled_events.jsonl"
        lab_path.write_bytes(lab_bytes)
        lab_sha = _sha(lab_bytes)

        _, prep_man_sha = _write_json(
            s_dir / "manifest.json",
            {
                "manifest_schema_version": "1.0.0",
                "feature_schema_version": "1.0.0",
                "source_files": [
                    {
                        "source_filename": covered_name,
                        "source_sha256": cov_sha,
                        "ground_truth_filename": "access.log",
                        "ground_truth_reference": gt_ref,
                        "ground_truth_sha256": gt_sha,
                    },
                    {
                        "source_filename": unknown_name,
                        "source_sha256": unk_sha,
                        "ground_truth_filename": None,
                        "ground_truth_reference": None,
                        "ground_truth_sha256": None,
                    },
                ],
                "processed_output": {
                    "filename": "prepared_events.jsonl",
                    "sha256": prep_sha,
                    "record_count": len(rows),
                },
            },
        )
        lab_man_path, lab_man_sha = _write_json(
            s_dir / "labels_manifest.json",
            {
                "manifest_schema_version": "1.0.0",
                "label_schema_version": "1.0.0",
                "scenario_id": scenario,
                "label_policy": {"policy_id": "ait-lds-v2.1-explicit-ground-truth-v1"},
                "input": {
                    "prepared_manifest_sha256": prep_man_sha,
                    "prepared_records_sha256": prep_sha,
                },
                "labelled_output": {
                    "filename": "labelled_events.jsonl",
                    "sha256": lab_sha,
                },
                "record_counts": {"evaluation_records": len(rows)},
                "evaluation_class_distribution": {
                    "normal": 8,
                    "attack_associated": 8,
                    "unknown": 2,
                },
                "per_source_distributions": [
                    {
                        "source_filename": covered_name,
                        "source_sha256": cov_sha,
                        "ground_truth_sha256": gt_sha,
                        "ground_truth_coverage": "verified",
                    },
                    {
                        "source_filename": unknown_name,
                        "source_sha256": unk_sha,
                        "ground_truth_sha256": None,
                        "ground_truth_coverage": "unavailable",
                    },
                ],
            },
        )
        if tamper_scenario == scenario:
            lab_path.write_bytes(lab_bytes + b"{}\n")

        summaries[scenario] = {
            "label_manifest_sha256": lab_man_sha,
            "labelled_records_sha256": lab_sha,
            "prepared_manifest_sha256": prep_man_sha,
            "prepared_records_sha256": prep_sha,
        }
        scenarios_cfg[scenario] = {
            "label_manifest_path": str(lab_man_path.relative_to(tmp_path)),
            **summaries[scenario],
        }

    candidates = []
    outer_folds_map = {s: [] for s in SCENARIOS}
    for val_s, test_s in permutations(SCENARIOS, 2):
        train_s = [s for s in SCENARIOS if s not in (val_s, test_s)]
        assignment = {"train": train_s, "validation": [val_s], "test": [test_s]}
        cid = _sha(json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode())[:16]
        candidates.append(
            {"candidate_id": cid, "assignment": assignment, "scenario_partition_isolation": True}
        )
        outer_folds_map[test_s].append(
            {"candidate_id": cid, "train_scenarios": train_s, "validation_scenarios": [val_s]}
        )

    split_path, split_sha = _write_json(
        tmp_path / "splits" / "four_scenario_split_manifest.json",
        {
            "scenario_ids": list(SCENARIOS),
            "scenario_summaries": summaries,
            "candidate_evaluation": {
                "scientifically_adequate_candidate_count": 0,
                "selected_assignment": None,
            },
            "candidates": candidates,
        },
    )
    qual_path, qual_sha = _write_json(
        tmp_path / "quality" / "four_scenario_quality_report.json",
        {"quality_status": "passed_with_research_limitations"},
    )

    config = {
        "config_schema_version": "1.0.0",
        "experiment_id": "synthetic-complete-benchmark",
        "study_status": "exploratory",
        "dataset_provenance": {
            "feature_schema_version": "1.0.0",
            "label_schema_version": "1.0.0",
            "split_manifest": {
                "path": str(split_path.relative_to(tmp_path)),
                "sha256": split_sha,
                "scientifically_adequate": False,
                "selected_assignment": None,
            },
            "quality_report": {
                "path": str(qual_path.relative_to(tmp_path)),
                "sha256": qual_sha,
                "expected_quality_status": "passed_with_research_limitations",
            },
            "scenarios": scenarios_cfg,
        },
        "label_definition": {
            "positive_class": {"value": 1, "name": "attack_associated"},
            "negative_class": {"value": 0, "name": "normal"},
            "excluded_class": {"value": None, "name": "unknown", "exclude_from_supervised": True},
        },
        "features": {
            "schema_version": "1.0.0",
            "ordered_names": [f["name"] for f in FEATURE_SCHEMA["features"]],
            "log1p_transform_features": [
                "transfer_size_bytes",
                "request_frequency_5m",
                "failed_request_frequency_5m",
                "post_request_frequency_5m",
                "time_since_previous_event_seconds",
                "unique_resources_accessed",
            ],
            "binary_indicator_features": [
                "sensitive_resource_indicator",
                "source_ip_missing",
                "event_type_missing",
                "resource_missing",
                "response_status_missing",
                "transfer_size_missing",
                "time_since_previous_event_missing",
            ],
        },
        "preprocessing": {
            "tree_models": {"apply_log1p": False, "scaler": None, "fit_on": "train_only"}
        },
        "weighting_policy": {
            "mechanism": "sample_weight_only",
            "estimator_class_weight": None,
            "double_weighting_prohibited": True,
            "regimes": [
                {"name": "class_weighted"},
                {"name": "scenario_equalized_class_weighted"},
            ],
        },
        "random_seeds": [42, 137],
        "evaluation_protocol": {
            "strategy": "nested_4fold_loso",
            "outer_test_isolation": True,
            "selection_scope": "inner_validation_only",
            "threshold_selection": {
                "fit_scope": "inner_validation_only",
                "fixed_threshold": 0.5,
                "candidate_thresholds": [0.3, 0.5, 0.7],
            },
            "outer_folds": [
                {
                    "fold_id": f"outer_test_{test_s}",
                    "test_scenarios": [test_s],
                    "outer_train_scenarios": [s for s in SCENARIOS if s != test_s],
                    "inner_candidates": outer_folds_map[test_s],
                }
                for test_s in SCENARIOS
            ],
        },
        "models": [
            {
                "name": "RuleBaseline_R_A_to_D",
                "family": "domain_rule_baseline",
                "probabilistic": False,
                "preprocessing_profile": None,
                "uses_weighting_regimes": False,
                "uses_threshold_tuning": False,
                "stochastic": False,
                "rule_registry_version": "1.0.0",
                "rule_registry_fingerprint": calculate_rule_fingerprint(),
                "rules": ["R-A", "R-B", "R-C", "R-D"],
            },
            {
                "name": "DummyClassifier",
                "family": "reference_baseline",
                "probabilistic": True,
                "preprocessing_profile": "tree_models",
                "uses_weighting_regimes": False,
                "uses_threshold_tuning": False,
                "stochastic": True,
                "hyperparameter_grid": {"strategy": ["most_frequent", "stratified"]},
            },
            {
                "name": "LogisticRegression",
                "family": "linear",
                "probabilistic": True,
                "preprocessing_profile": "linear_models",
                "uses_weighting_regimes": True,
                "uses_threshold_tuning": True,
                "stochastic": False,
                "fixed_parameters": {"class_weight": None},
                "hyperparameter_grid": {
                    "C": [0.1, 1.0],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                    "max_iter": [1000],
                },
            },
            {
                "name": "DecisionTreeClassifier",
                "family": "tree",
                "probabilistic": True,
                "preprocessing_profile": "tree_models",
                "uses_weighting_regimes": True,
                "uses_threshold_tuning": True,
                "stochastic": True,
                "fixed_parameters": {"class_weight": None, "criterion": "gini"},
                "hyperparameter_grid": {"max_depth": [3], "min_samples_leaf": [5]},
            },
            {
                "name": "RandomForestClassifier",
                "family": "ensemble_tree",
                "probabilistic": True,
                "preprocessing_profile": "tree_models",
                "uses_weighting_regimes": True,
                "uses_threshold_tuning": True,
                "stochastic": True,
                "fixed_parameters": {"class_weight": None, "criterion": "gini", "bootstrap": True},
                "hyperparameter_grid": {
                    "n_estimators": [10],
                    "max_depth": [5],
                    "min_samples_leaf": [5],
                    "max_features": ["sqrt"],
                },
            },
        ],
        "metrics": {"primary_selection_metric": "macro_f1"},
        "scientific_limitations": ["Synthetic exploratory test fixture."],
    }
    cfg_path, _ = _write_json(tmp_path / "experiment_config.json", config)
    return cfg_path


def test_verified_provider_and_full_synthetic_benchmark_are_deterministic(tmp_path):
    cfg_path = _build_synthetic_4scenario_workspace(tmp_path)
    provider = VerifiedDatasetScenarioProvider(cfg_path, base_dir=tmp_path)
    fox_recs = provider.load_scenario_records("fox")
    assert len(fox_recs) == 16
    assert provider.summaries["fox"].excluded_unknown_count == 2

    exp_dir = tmp_path / "experiments"
    out_path = exp_dir / "synthetic_benchmark.json"

    rep1, path1, sha1 = run_full_nested_loso_benchmark(
        config_path=cfg_path,
        base_dir=tmp_path,
        output_path=out_path,
        allowed_output_root=exp_dir,
        rf_n_jobs=1,
    )
    rep2, path2, sha2 = run_full_nested_loso_benchmark(
        config_path=cfg_path,
        base_dir=tmp_path,
        output_path=out_path,
        allowed_output_root=exp_dir,
        rf_n_jobs=1,
    )

    assert rep1 == rep2
    assert path1 == path2 == out_path.resolve()
    assert sha1 == sha2
    assert rep1["source_manifest_status"] == {
        "scientifically_adequate": False,
        "selected_assignment": None,
    }
    assert set(rep1["models"]) == {
        "RuleBaseline_R_A_to_D",
        "DummyClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
    }
    assert len(rep1["models"]["RuleBaseline_R_A_to_D"]["outer_folds"]) == 4
    assert len(rep1["models"]["DummyClassifier"]["outer_folds"]) == 4
    for name in ("LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"):
        regimes = rep1["models"][name]["weighting_regimes"]
        assert set(regimes) == {"class_weighted", "scenario_equalized_class_weighted"}
        for reg_data in regimes.values():
            assert len(reg_data["outer_folds"]) == 4


def test_benchmark_rejects_tampered_data_and_unsafe_artifact_paths(tmp_path):
    bad_cfg = _build_synthetic_4scenario_workspace(tmp_path / "tamper", tamper_scenario="santos")
    with pytest.raises(BenchmarkError, match="SHA-256 mismatch"):
        run_full_nested_loso_benchmark(
            config_path=bad_cfg,
            base_dir=tmp_path / "tamper",
            rf_n_jobs=1,
        )

    exp_dir = tmp_path / "experiments"
    with pytest.raises(BenchmarkError, match="must lie inside"):
        write_benchmark_artifact({"a": 1}, tmp_path / "outside.json", allowed_root=exp_dir)

    p, _ = write_benchmark_artifact({"a": 1}, exp_dir / "run.json", allowed_root=exp_dir)
    assert p.is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_benchmark_artifact({"a": 2}, exp_dir / "run.json", allowed_root=exp_dir)


def test_resumable_checkpointing_interruption_resume_and_exact_determinism(tmp_path):
    cfg_path = _build_synthetic_4scenario_workspace(tmp_path)
    exp_dir = tmp_path / "experiments"
    ckpt_dir = exp_dir / "checkpoints" / "synthetic"
    resumed_out = exp_dir / "resumed_benchmark.json"
    direct_out = exp_dir / "direct_benchmark.json"

    # Baseline direct run without checkpoints
    direct_rep, _, direct_sha = run_full_nested_loso_benchmark(
        config_path=cfg_path,
        base_dir=tmp_path,
        output_path=direct_out,
        allowed_output_root=exp_dir,
        rf_n_jobs=1,
    )

    # Interrupted run after 12 units: must raise BenchmarkIncompleteError and NOT write final artifact
    with pytest.raises(BenchmarkIncompleteError, match="max_new_units=12"):
        run_full_nested_loso_benchmark(
            config_path=cfg_path,
            base_dir=tmp_path,
            output_path=resumed_out,
            allowed_output_root=exp_dir,
            checkpoint_dir=ckpt_dir,
            max_new_units=12,
            rf_n_jobs=1,
        )
    assert not resumed_out.exists()
    assert len(list(ckpt_dir.glob("*.json"))) == 12

    # Resume and finish remaining units
    resumed_rep, _, resumed_sha = run_full_nested_loso_benchmark(
        config_path=cfg_path,
        base_dir=tmp_path,
        output_path=resumed_out,
        allowed_output_root=exp_dir,
        checkpoint_dir=ckpt_dir,
        rf_n_jobs=1,
    )
    assert resumed_out.is_file()
    assert resumed_rep == direct_rep
    assert resumed_sha == direct_sha

    # Third invocation with max_new_units=0 proves 100% of units are resumed with zero recomputation
    replay_rep, _, replay_sha = run_full_nested_loso_benchmark(
        config_path=cfg_path,
        base_dir=tmp_path,
        output_path=resumed_out,
        allowed_output_root=exp_dir,
        checkpoint_dir=ckpt_dir,
        max_new_units=0,
        rf_n_jobs=1,
    )
    assert replay_rep == direct_rep
    assert replay_sha == direct_sha


def test_resumable_checkpointing_rejects_corrupted_tampered_and_unsafe_checkpoints(tmp_path):
    cfg_path = _build_synthetic_4scenario_workspace(tmp_path)
    exp_dir = tmp_path / "experiments"
    ckpt_dir = exp_dir / "checkpoints" / "synthetic"

    # Reject checkpoint dir outside allowed root
    with pytest.raises(BenchmarkError, match="must lie inside"):
        run_full_nested_loso_benchmark(
            config_path=cfg_path,
            base_dir=tmp_path,
            allowed_output_root=exp_dir,
            checkpoint_dir=tmp_path / "outside_ckpts",
            max_new_units=1,
            rf_n_jobs=1,
        )

    # Generate 1 valid checkpoint
    with pytest.raises(BenchmarkIncompleteError):
        run_full_nested_loso_benchmark(
            config_path=cfg_path,
            base_dir=tmp_path,
            allowed_output_root=exp_dir,
            checkpoint_dir=ckpt_dir,
            max_new_units=1,
            rf_n_jobs=1,
        )
    ckpt_file = next(ckpt_dir.glob("*.json"))
    valid_bytes = ckpt_file.read_bytes()

    # 1. Payload tampering (payload_sha256 mismatch)
    env = json.loads(valid_bytes.decode("utf-8"))
    env["payload"]["tampered"] = True
    ckpt_file.write_text(json.dumps(env), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="payload SHA-256 integrity mismatch"):
        run_full_nested_loso_benchmark(
            config_path=cfg_path,
            base_dir=tmp_path,
            allowed_output_root=exp_dir,
            checkpoint_dir=ckpt_dir,
            max_new_units=1,
            rf_n_jobs=1,
        )

    # 2. Provenance mismatch
    env = json.loads(valid_bytes.decode("utf-8"))
    env["provenance"]["experiment_config_sha256"] = "0" * 64
    ckpt_file.write_text(json.dumps(env), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="provenance mismatch"):
        run_full_nested_loso_benchmark(
            config_path=cfg_path,
            base_dir=tmp_path,
            allowed_output_root=exp_dir,
            checkpoint_dir=ckpt_dir,
            max_new_units=1,
            rf_n_jobs=1,
        )

    # 3. Corrupted JSON
    ckpt_file.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="Corrupted checkpoint"):
        run_full_nested_loso_benchmark(
            config_path=cfg_path,
            base_dir=tmp_path,
            allowed_output_root=exp_dir,
            checkpoint_dir=ckpt_dir,
            max_new_units=1,
            rf_n_jobs=1,
        )
