import hashlib
from itertools import permutations
import json
from pathlib import Path

import pytest

from backend.ml.experiment_runner import (
    ExperimentRunnerError,
    build_nested_loso_plan,
    run_nested_loso_plan,
)
from backend.ml.schema import FEATURE_SCHEMA


SCENARIOS = ("fox", "harrison", "russellmitchell", "santos")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, _sha(raw)


def _build_synthetic_plan_fixture(
    tmp_path: Path,
    *,
    mutate_config=None,
    tamper_split_hash: bool = False,
):
    summaries = {}
    scenarios_cfg = {}
    for scenario in SCENARIOS:
        lm_sha = _sha(f"{scenario}:lm".encode())
        lr_sha = _sha(f"{scenario}:lr".encode())
        pm_sha = _sha(f"{scenario}:pm".encode())
        pr_sha = _sha(f"{scenario}:pr".encode())
        summaries[scenario] = {
            "label_manifest_sha256": lm_sha,
            "labelled_records_sha256": lr_sha,
            "prepared_manifest_sha256": pm_sha,
            "prepared_records_sha256": pr_sha,
        }
        scenarios_cfg[scenario] = {
            "label_manifest_path": f"processed/{scenario}/labels_manifest.json",
            "label_manifest_sha256": lm_sha,
            "labelled_records_sha256": lr_sha,
            "prepared_manifest_sha256": pm_sha,
            "prepared_records_sha256": pr_sha,
        }

    candidates = []
    outer_folds_map = {s: [] for s in SCENARIOS}
    for val_s, test_s in permutations(SCENARIOS, 2):
        train_s = [s for s in SCENARIOS if s not in (val_s, test_s)]
        assignment = {"train": train_s, "validation": [val_s], "test": [test_s]}
        cid = _sha(json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode())[:16]
        candidates.append(
            {
                "candidate_id": cid,
                "assignment": assignment,
                "scenario_partition_isolation": True,
            }
        )
        outer_folds_map[test_s].append(
            {
                "candidate_id": cid,
                "train_scenarios": train_s,
                "validation_scenarios": [val_s],
            }
        )

    split_manifest = {
        "scenario_ids": list(SCENARIOS),
        "scenario_summaries": summaries,
        "candidate_evaluation": {
            "scientifically_adequate_candidate_count": 0,
            "selected_assignment": None,
        },
        "candidates": candidates,
    }
    split_path, split_sha = _write_json(
        tmp_path / "splits" / "four_scenario_split_manifest.json", split_manifest
    )
    quality_path, quality_sha = _write_json(
        tmp_path / "quality" / "four_scenario_quality_report.json",
        {"quality_status": "passed_with_research_limitations"},
    )

    outer_folds = [
        {
            "fold_id": f"outer_test_{test_s}",
            "test_scenarios": [test_s],
            "outer_train_scenarios": [s for s in SCENARIOS if s != test_s],
            "inner_candidates": outer_folds_map[test_s],
        }
        for test_s in SCENARIOS
    ]

    config = {
        "config_schema_version": "1.0.0",
        "experiment_id": "synthetic-4fold-loso",
        "study_status": "exploratory",
        "dataset_provenance": {
            "feature_schema_version": "1.0.0",
            "label_schema_version": "1.0.0",
            "split_manifest": {
                "path": str(split_path.relative_to(tmp_path)),
                "sha256": ("0" * 64) if tamper_split_hash else split_sha,
                "scientifically_adequate": False,
                "selected_assignment": None,
            },
            "quality_report": {
                "path": str(quality_path.relative_to(tmp_path)),
                "sha256": quality_sha,
                "expected_quality_status": "passed_with_research_limitations",
            },
            "scenarios": scenarios_cfg,
        },
        "label_definition": {
            "positive_class": {"value": 1, "name": "attack_associated"},
            "negative_class": {"value": 0, "name": "normal"},
            "excluded_class": {
                "value": None,
                "name": "unknown",
                "exclude_from_supervised": True,
            },
        },
        "features": {
            "schema_version": "1.0.0",
            "ordered_names": [f["name"] for f in FEATURE_SCHEMA["features"]],
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
                "candidate_thresholds": [0.1, 0.5, 0.9],
            },
            "outer_folds": outer_folds,
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
                "hyperparameter_grid": {"C": [0.1, 1.0], "solver": ["lbfgs"]},
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
                "hyperparameter_grid": {"max_depth": [3, 5], "min_samples_leaf": [10]},
            },
        ],
        "metrics": {"primary_selection_metric": "macro_f1"},
    }
    if mutate_config is not None:
        mutate_config(config)
    config_path, _ = _write_json(tmp_path / "experiment_config.json", config)
    return config_path


def test_build_nested_loso_plan_validates_folds_candidates_and_determinism(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(tmp_path)

    plan_1 = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    plan_2 = build_nested_loso_plan(cfg_path, base_dir=tmp_path)

    assert plan_1 == plan_2
    assert json.dumps(plan_1.to_dict(), sort_keys=True) == json.dumps(
        plan_2.to_dict(), sort_keys=True
    )
    assert len(plan_1.outer_folds) == 4
    all_cids = []
    for fold in plan_1.outer_folds:
        assert len(fold.test_scenarios) == 1
        assert len(fold.outer_train_scenarios) == 3
        assert len(fold.inner_candidates) == 3
        test_set = set(fold.test_scenarios)
        for ic in fold.inner_candidates:
            all_cids.append(ic.candidate_id)
            assert test_set.isdisjoint(ic.train_scenarios)
            assert test_set.isdisjoint(ic.validation_scenarios)
            assert set(ic.train_scenarios).isdisjoint(ic.validation_scenarios)
    assert len(set(all_cids)) == 12

    # Verify deterministic orchestration via a stub FoldEvaluator (no training executed)
    calls = []

    class _StubEvaluator:
        def evaluate_outer_fold(self, plan, fold, model_spec):
            calls.append((model_spec.name, fold.fold_id, fold.test_scenarios))
            return {"inner_count": len(fold.inner_candidates)}

    report = run_nested_loso_plan(
        plan_1, _StubEvaluator(), model_names=["LogisticRegression"]
    )
    assert report["models_evaluated"] == ["LogisticRegression"]
    assert [c[1] for c in calls] == [f"outer_test_{s}" for s in SCENARIOS]


def test_nested_loso_plan_rejects_leakage_duplicate_candidates_and_bad_provenance(tmp_path):
    # 1. Outer test leaked into inner validation
    def _leak_outer_test(cfg):
        cfg["evaluation_protocol"]["outer_folds"][0]["inner_candidates"][0][
            "validation_scenarios"
        ] = ["fox"]

    cfg_leak = _build_synthetic_plan_fixture(tmp_path / "leak", mutate_config=_leak_outer_test)
    with pytest.raises(ExperimentRunnerError, match="leaked into inner candidate"):
        build_nested_loso_plan(cfg_leak, base_dir=tmp_path / "leak")

    # 2. Duplicate candidate_id
    def _dup_candidate(cfg):
        inners = cfg["evaluation_protocol"]["outer_folds"][0]["inner_candidates"]
        inners[1] = dict(inners[0])

    cfg_dup = _build_synthetic_plan_fixture(tmp_path / "dup", mutate_config=_dup_candidate)
    with pytest.raises(ExperimentRunnerError, match="Duplicate inner candidate_id"):
        build_nested_loso_plan(cfg_dup, base_dir=tmp_path / "dup")

    # 3. Split manifest hash tampering
    cfg_tamper = _build_synthetic_plan_fixture(tmp_path / "tamper", tamper_split_hash=True)
    with pytest.raises(ExperimentRunnerError, match="Split manifest SHA-256 mismatch"):
        build_nested_loso_plan(cfg_tamper, base_dir=tmp_path / "tamper")

    # 4. Unsupported model name
    def _bad_model(cfg):
        cfg["models"].append({"name": "XGBoostClassifier", "family": "boosting"})

    cfg_model = _build_synthetic_plan_fixture(tmp_path / "model", mutate_config=_bad_model)
    with pytest.raises(ExperimentRunnerError, match="Unsupported model name"):
        build_nested_loso_plan(cfg_model, base_dir=tmp_path / "model")

    # 5. Double-weighting risk (estimator class_weight != null)
    def _double_weight(cfg):
        cfg["models"][1]["fixed_parameters"]["class_weight"] = "balanced"

    cfg_dw = _build_synthetic_plan_fixture(tmp_path / "dw", mutate_config=_double_weight)
    with pytest.raises(ExperimentRunnerError, match="class_weight=null"):
        build_nested_loso_plan(cfg_dw, base_dir=tmp_path / "dw")
