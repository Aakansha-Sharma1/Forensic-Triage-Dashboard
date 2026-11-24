import hashlib
from itertools import permutations
import json
from pathlib import Path

import pytest

from backend.ml.experiment_runner import (
    ExperimentRunnerError,
    InnerValidationObservation,
    build_nested_loso_plan,
    execute_inner_candidate_validation,
    execute_outer_fold_inner_validation,
    execute_outer_fold_refit_and_evaluation,
    run_nested_loso_plan,
    run_single_outer_fold_experiment,
    select_inner_configurations_by_regime,
    select_inner_validation_configuration,
)
from backend.ml.schema import FEATURE_SCHEMA
from backend.ml.training_data import SupervisedRecord


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


def _make_inner_observations(plan, fold, model_spec, regime, score_fn):
    observations = []
    for ic in fold.inner_candidates:
        for seed in plan.random_seeds:
            for params in model_spec.parameter_combinations:
                for threshold in plan.candidate_thresholds:
                    observations.append(
                        InnerValidationObservation(
                            fold_id=fold.fold_id,
                            candidate_id=ic.candidate_id,
                            validation_scenario=ic.validation_scenarios[0],
                            model_name=model_spec.name,
                            weighting_regime=regime,
                            seed=seed,
                            hyperparameters=params,
                            threshold=threshold,
                            macro_f1=score_fn(params, threshold, ic.candidate_id, seed),
                        )
                    )
    return observations


def test_inner_selection_picks_best_macro_f1_and_applies_deterministic_tie_breaking(tmp_path):
    def _five_seeds_and_thresholds(cfg):
        cfg["random_seeds"] = [42, 137, 256, 512, 1024]
        cfg["evaluation_protocol"]["threshold_selection"]["candidate_thresholds"] = [
            0.3, 0.4, 0.5, 0.6, 0.7
        ]

    cfg_path = _build_synthetic_plan_fixture(tmp_path, mutate_config=_five_seeds_and_thresholds)
    plan = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    fold = plan.outer_folds[0]
    lr_spec = next(s for s in plan.model_specs if s.name == "LogisticRegression")

    # 1. Clear winner: C=1.0, threshold=0.6
    obs_cw = _make_inner_observations(
        plan,
        fold,
        lr_spec,
        "class_weighted",
        lambda p, t, _c, _s: 0.85 if (p["C"] == 1.0 and t == 0.6) else 0.70,
    )
    # 2. Separate regime with exact tie across 0.4 and 0.6 (same distance 0.1 from 0.5)
    #    and across C=0.1 (grid idx 0) and C=1.0 (grid idx 1) -> must pick threshold 0.4, C=0.1
    obs_eq = _make_inner_observations(
        plan,
        fold,
        lr_spec,
        "scenario_equalized_class_weighted",
        lambda p, t, _c, _s: 0.80 if t in (0.4, 0.6) else 0.60,
    )

    by_regime = select_inner_configurations_by_regime(
        plan,
        fold,
        lr_spec,
        {
            "class_weighted": obs_cw,
            "scenario_equalized_class_weighted": obs_eq,
        },
    )

    cw_res = by_regime["class_weighted"]
    assert dict(cw_res.selected_hyperparameters) == {"C": 1.0, "solver": "lbfgs"}
    assert cw_res.selected_threshold == 0.6
    assert abs(cw_res.mean_validation_macro_f1 - 0.85) < 1e-12
    assert cw_res.seeds_used == (42, 137, 256, 512, 1024)
    assert cw_res.excluded_outer_test_scenarios == ("fox",)
    assert cw_res.observations_count == 3 * 5 * 2 * 5

    eq_res = by_regime["scenario_equalized_class_weighted"]
    assert eq_res.selected_threshold == 0.4
    assert dict(eq_res.selected_hyperparameters) == {"C": 0.1, "solver": "lbfgs"}
    assert abs(eq_res.mean_validation_macro_f1 - 0.80) < 1e-12
    # JSON-serializable and deterministic
    assert json.dumps(eq_res.to_dict(), sort_keys=True) == json.dumps(
        select_inner_validation_configuration(
            plan, fold, lr_spec, "scenario_equalized_class_weighted", list(reversed(obs_eq))
        ).to_dict(),
        sort_keys=True,
    )

    # 3. Tie including 0.5 must prefer 0.5 over 0.4 and 0.6
    obs_prefer_half = _make_inner_observations(
        plan,
        fold,
        lr_spec,
        "class_weighted",
        lambda p, t, _c, _s: 0.90 if t in (0.4, 0.5, 0.6) else 0.50,
    )
    half_res = select_inner_validation_configuration(
        plan, fold, lr_spec, "class_weighted", obs_prefer_half
    )
    assert half_res.selected_threshold == 0.5
    assert dict(half_res.selected_hyperparameters) == {"C": 0.1, "solver": "lbfgs"}


def test_inner_selection_fails_closed_on_incomplete_duplicate_leakage_and_invalid_inputs(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(tmp_path)
    plan = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    fold = plan.outer_folds[0]
    lr_spec = next(s for s in plan.model_specs if s.name == "LogisticRegression")
    base_obs = _make_inner_observations(
        plan, fold, lr_spec, "class_weighted", lambda _p, _t, _c, _s: 0.75
    )

    # Incomplete observations
    with pytest.raises(ExperimentRunnerError, match="Incomplete"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", base_obs[:-1])

    # Duplicate entry
    with pytest.raises(ExperimentRunnerError, match="Duplicate"):
        select_inner_validation_configuration(
            plan, fold, lr_spec, "class_weighted", [*base_obs, base_obs[0]]
        )

    # Outer test scenario leakage
    leaked = list(base_obs)
    leaked[0] = InnerValidationObservation(
        **{**leaked[0].__dict__, "validation_scenario": fold.test_scenarios[0]}
    )
    with pytest.raises(ExperimentRunnerError, match="outer test scenario"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", leaked)

    # Non-validation partition_role
    test_role = list(base_obs)
    test_role[0] = InnerValidationObservation(
        **{**test_role[0].__dict__, "partition_role": "test"}
    )
    with pytest.raises(ExperimentRunnerError, match="partition_role"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", test_role)

    # Mixed weighting regimes
    mixed_weight = list(base_obs)
    mixed_weight[0] = InnerValidationObservation(
        **{**mixed_weight[0].__dict__, "weighting_regime": "scenario_equalized_class_weighted"}
    )
    with pytest.raises(ExperimentRunnerError, match="mixed weighting regime"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", mixed_weight)

    # Undefined Macro-F1 (None)
    undef_f1 = list(base_obs)
    undef_f1[0] = InnerValidationObservation(**{**undef_f1[0].__dict__, "macro_f1": None})
    with pytest.raises(ExperimentRunnerError, match="undefined or invalid Macro-F1"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", undef_f1)

    # Unconfigured seed / invalid threshold / unexpected candidate_id
    bad_seed = list(base_obs)
    bad_seed[0] = InnerValidationObservation(**{**bad_seed[0].__dict__, "seed": 999})
    with pytest.raises(ExperimentRunnerError, match="unconfigured seed"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", bad_seed)

    bad_thresh = list(base_obs)
    bad_thresh[0] = InnerValidationObservation(**{**bad_thresh[0].__dict__, "threshold": 0.25})
    with pytest.raises(ExperimentRunnerError, match="invalid threshold"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", bad_thresh)

    bad_cand = list(base_obs)
    bad_cand[0] = InnerValidationObservation(**{**bad_cand[0].__dict__, "candidate_id": "deadbeef"})
    with pytest.raises(ExperimentRunnerError, match="unexpected candidate_id"):
        select_inner_validation_configuration(plan, fold, lr_spec, "class_weighted", bad_cand)


def _make_synthetic_supervised_records(scenario: str) -> list[SupervisedRecord]:
    records: list[SupervisedRecord] = []
    for i in range(10):
        records.append(
            SupervisedRecord(
                scenario_id=scenario,
                event_id=f"intranet/access.log:{i + 1}",
                source_filename="intranet/access.log",
                source_line_number=i + 1,
                source_sha256="a" * 64,
                original_evidence_reference=f"raw/{scenario}/gather/intranet/access.log",
                label=0,
                label_name="normal",
                label_basis="absence_from_verified_ground_truth_file",
                label_provenance={"policy_id": "ait-lds-v2.1-explicit-ground-truth-v1"},
                attack_categories=(),
                features=(
                    200 + i * 10,
                    1 + (i % 2),
                    0,
                    0,
                    None if i == 0 else 50.0 + i,
                    1 + (i % 3),
                    0,
                    0, 0, 0, 0, 0,
                    1 if i == 0 else 0,
                ),
            )
        )
    for i in range(10):
        records.append(
            SupervisedRecord(
                scenario_id=scenario,
                event_id=f"intranet/access.log:{11 + i}",
                source_filename="intranet/access.log",
                source_line_number=11 + i,
                source_sha256="a" * 64,
                original_evidence_reference=f"raw/{scenario}/gather/intranet/access.log",
                label=1,
                label_name="attack_associated",
                label_basis="explicit_ground_truth",
                label_provenance={"policy_id": "ait-lds-v2.1-explicit-ground-truth-v1"},
                attack_categories=("foothold",),
                features=(
                    7000 + i * 200,
                    50 + i * 3,
                    30 + i * 2,
                    12 + (i % 4),
                    0.5 + (i % 3) * 0.2,
                    40 + i,
                    1,
                    0, 0, 0, 0, 0, 0,
                ),
            )
        )
    return records


def _full_baseline_compatible_mutator(cfg):
    cfg["features"]["log1p_transform_features"] = [
        "transfer_size_bytes",
        "request_frequency_5m",
        "failed_request_frequency_5m",
        "post_request_frequency_5m",
        "time_since_previous_event_seconds",
        "unique_resources_accessed",
    ]
    cfg["features"]["binary_indicator_features"] = [
        "sensitive_resource_indicator",
        "source_ip_missing",
        "event_type_missing",
        "resource_missing",
        "response_status_missing",
        "transfer_size_missing",
        "time_since_previous_event_missing",
    ]
    cfg["preprocessing"] = {
        "tree_models": {"apply_log1p": False, "scaler": None, "fit_on": "train_only"}
    }
    lr = next(m for m in cfg["models"] if m["name"] == "LogisticRegression")
    lr["hyperparameter_grid"] = {
        "C": [0.1, 1.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    }
    cfg["models"].append(
        {
            "name": "RandomForestClassifier",
            "family": "ensemble_tree",
            "probabilistic": True,
            "preprocessing_profile": "tree_models",
            "uses_weighting_regimes": True,
            "uses_threshold_tuning": True,
            "stochastic": True,
            "fixed_parameters": {
                "class_weight": None,
                "criterion": "gini",
                "bootstrap": True,
            },
            "hyperparameter_grid": {
                "n_estimators": [10],
                "max_depth": [5],
                "min_samples_leaf": [5],
                "max_features": ["sqrt"],
            },
        }
    )


def test_inner_fold_executor_runs_all_models_deterministically_and_feeds_selector(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(
        tmp_path, mutate_config=_full_baseline_compatible_mutator
    )
    plan = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    fold = plan.outer_folds[0]  # outer_test_fox
    records_by_scenario = {
        scenario: _make_synthetic_supervised_records(scenario)
        for scenario in fold.outer_train_scenarios
    }

    for model_name in ("LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"):
        spec = next(s for s in plan.model_specs if s.name == model_name)
        obs_1 = execute_outer_fold_inner_validation(
            plan,
            fold,
            spec,
            "scenario_equalized_class_weighted",
            records_by_scenario,
            config_path=cfg_path,
        )
        obs_2 = execute_outer_fold_inner_validation(
            plan,
            fold,
            spec,
            "scenario_equalized_class_weighted",
            records_by_scenario,
            config_path=cfg_path,
        )
        assert obs_1 == obs_2
        selection = select_inner_validation_configuration(
            plan, fold, spec, "scenario_equalized_class_weighted", obs_1
        )
        assert selection.excluded_outer_test_scenarios == ("fox",)
        assert selection.mean_validation_macro_f1 >= 0.9


def test_inner_fold_executor_rejects_outer_test_leakage_duplicates_and_missing_scenarios(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(
        tmp_path, mutate_config=_full_baseline_compatible_mutator
    )
    plan = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    fold = plan.outer_folds[0]
    cand = fold.inner_candidates[0]
    lr_spec = next(s for s in plan.model_specs if s.name == "LogisticRegression")

    records_by_scenario = {
        scenario: _make_synthetic_supervised_records(scenario)
        for scenario in fold.outer_train_scenarios
    }
    fox_records = _make_synthetic_supervised_records("fox")

    # 1. Outer test passed to execute_outer_fold_inner_validation
    with pytest.raises(ExperimentRunnerError, match="Outer test scenario"):
        execute_outer_fold_inner_validation(
            plan,
            fold,
            lr_spec,
            "class_weighted",
            {**records_by_scenario, "fox": fox_records},
            config_path=cfg_path,
        )

    train_recs = (
        records_by_scenario[cand.train_scenarios[0]]
        + records_by_scenario[cand.train_scenarios[1]]
    )
    val_recs = records_by_scenario[cand.validation_scenarios[0]]

    # 2. Outer test record injected into train_records
    with pytest.raises(ExperimentRunnerError, match="forbidden scenario"):
        execute_inner_candidate_validation(
            plan, fold, cand, lr_spec, "class_weighted", [*train_recs, fox_records[0]], val_recs, config_path=cfg_path
        )

    # 3. Duplicate record identity in train_records
    with pytest.raises(ExperimentRunnerError, match="duplicate record identity"):
        execute_inner_candidate_validation(
            plan, fold, cand, lr_spec, "class_weighted", [*train_recs, train_recs[0]], val_recs, config_path=cfg_path
        )

    # 4. Missing one of the candidate's training scenarios
    with pytest.raises(ExperimentRunnerError, match="missing expected scenarios"):
        execute_inner_candidate_validation(
            plan, fold, cand, lr_spec, "class_weighted", records_by_scenario[cand.train_scenarios[0]], val_recs, config_path=cfg_path
        )


def test_outer_fold_refit_and_evaluation_is_deterministic_and_reports_all_seeds(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(
        tmp_path, mutate_config=_full_baseline_compatible_mutator
    )
    plan = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    fold = plan.outer_folds[0]  # outer_test_fox
    lr_spec = next(s for s in plan.model_specs if s.name == "LogisticRegression")

    outer_train_by_scenario = {
        scenario: _make_synthetic_supervised_records(scenario)
        for scenario in fold.outer_train_scenarios
    }
    outer_train_records = [
        rec for scenario in fold.outer_train_scenarios for rec in outer_train_by_scenario[scenario]
    ]
    outer_test_records = _make_synthetic_supervised_records("fox")

    # Obtain inner-selected config strictly from outer_train_by_scenario (fox untouched)
    inner_obs = execute_outer_fold_inner_validation(
        plan,
        fold,
        lr_spec,
        "scenario_equalized_class_weighted",
        outer_train_by_scenario,
        config_path=cfg_path,
    )
    selection = select_inner_validation_configuration(
        plan, fold, lr_spec, "scenario_equalized_class_weighted", inner_obs
    )

    res_1 = execute_outer_fold_refit_and_evaluation(
        plan,
        fold,
        lr_spec,
        selection,
        outer_train_records,
        outer_test_records,
        config_path=cfg_path,
    )
    res_2 = execute_outer_fold_refit_and_evaluation(
        plan,
        fold,
        lr_spec,
        selection,
        outer_train_records,
        outer_test_records,
        config_path=cfg_path,
    )

    assert res_1 == res_2
    assert json.dumps(res_1.to_dict(), sort_keys=True) == json.dumps(
        res_2.to_dict(), sort_keys=True
    )
    assert res_1.fold_id == "outer_test_fox"
    assert res_1.test_scenarios == ("fox",)
    assert res_1.seeds_evaluated == plan.random_seeds
    assert len(res_1.per_seed_results) == len(plan.random_seeds)
    assert res_1.outer_train_record_count == 60
    assert res_1.outer_test_record_count == 20
    assert res_1.fixed_threshold == 0.5
    assert res_1.selected_threshold == selection.selected_threshold

    for seed_res in res_1.per_seed_results:
        ev = seed_res.evaluation
        assert "pr_auc_attack" in ev["probabilistic_metrics"]
        assert "pr_auc_normal" in ev["probabilistic_metrics"]
        assert "brier_score" in ev["probabilistic_metrics"]
        for block_key in ("fixed_threshold_metrics", "selected_threshold_metrics"):
            block = ev[block_key]
            assert block["confusion_matrix"]["tn"] + block["confusion_matrix"]["fp"] == 10
            assert block["confusion_matrix"]["tp"] + block["confusion_matrix"]["fn"] == 10
            assert block["macro_f1"] >= 0.9
            assert "normal_false_positive_rate" in block
            assert "foothold" in block["per_attack_category_recall"]


def test_outer_fold_refit_rejects_mismatched_selection_and_scenario_leakage(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(
        tmp_path, mutate_config=_full_baseline_compatible_mutator
    )
    plan = build_nested_loso_plan(cfg_path, base_dir=tmp_path)
    fold_fox = plan.outer_folds[0]
    fold_harrison = plan.outer_folds[1]
    lr_spec = next(s for s in plan.model_specs if s.name == "LogisticRegression")
    dt_spec = next(s for s in plan.model_specs if s.name == "DecisionTreeClassifier")

    outer_train_by_scenario = {
        scenario: _make_synthetic_supervised_records(scenario)
        for scenario in fold_fox.outer_train_scenarios
    }
    outer_train_records = [
        rec for scenario in fold_fox.outer_train_scenarios for rec in outer_train_by_scenario[scenario]
    ]
    outer_test_records = _make_synthetic_supervised_records("fox")

    inner_obs = execute_outer_fold_inner_validation(
        plan, fold_fox, lr_spec, "class_weighted", outer_train_by_scenario, config_path=cfg_path
    )
    selection = select_inner_validation_configuration(
        plan, fold_fox, lr_spec, "class_weighted", inner_obs
    )

    # 1. Selection from fold_fox used on fold_harrison
    with pytest.raises(ExperimentRunnerError, match="fold_id"):
        execute_outer_fold_refit_and_evaluation(
            plan, fold_harrison, lr_spec, selection, outer_train_records, outer_test_records, config_path=cfg_path
        )

    # 2. Selection for LogisticRegression used on DecisionTreeClassifier
    with pytest.raises(ExperimentRunnerError, match="model_name"):
        execute_outer_fold_refit_and_evaluation(
            plan, fold_fox, dt_spec, selection, outer_train_records, outer_test_records, config_path=cfg_path
        )

    # 3. Outer test record leaked into outer_train_records
    with pytest.raises(ExperimentRunnerError, match="forbidden scenario"):
        execute_outer_fold_refit_and_evaluation(
            plan,
            fold_fox,
            lr_spec,
            selection,
            [*outer_train_records, outer_test_records[0]],
            outer_test_records,
            config_path=cfg_path,
        )

    # 4. Duplicate record identity in outer_test_records
    with pytest.raises(ExperimentRunnerError, match="duplicate record identity"):
        execute_outer_fold_refit_and_evaluation(
            plan,
            fold_fox,
            lr_spec,
            selection,
            outer_train_records,
            [*outer_test_records, outer_test_records[0]],
            config_path=cfg_path,
        )


def test_single_outer_fold_orchestrator_is_deterministic_and_isolates_outer_test(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(
        tmp_path, mutate_config=_full_baseline_compatible_mutator
    )
    all_records = {s: _make_synthetic_supervised_records(s) for s in SCENARIOS}

    class _RecordingProvider:
        def __init__(self, data):
            self._data = data
            self.load_order = []

        def load_scenario_records(self, scenario_id: str):
            self.load_order.append(scenario_id)
            return self._data[scenario_id]

    p1 = _RecordingProvider(all_records)
    p2 = _RecordingProvider(all_records)

    res_1 = run_single_outer_fold_experiment(
        fold_id="outer_test_fox",
        model_name="DecisionTreeClassifier",
        weighting_regime="scenario_equalized_class_weighted",
        record_provider=p1,
        config_path=cfg_path,
        base_dir=tmp_path,
    )
    res_2 = run_single_outer_fold_experiment(
        fold_id="outer_test_fox",
        model_name="DecisionTreeClassifier",
        weighting_regime="scenario_equalized_class_weighted",
        record_provider=p2,
        config_path=cfg_path,
        base_dir=tmp_path,
    )

    # Outer test scenario ("fox") must ONLY be loaded AFTER all 3 outer-train scenarios
    assert p1.load_order == ["harrison", "russellmitchell", "santos", "fox"]
    assert p2.load_order == ["harrison", "russellmitchell", "santos", "fox"]
    assert res_1 == res_2
    payload = res_1.to_dict()
    assert json.dumps(payload, sort_keys=True) == json.dumps(res_2.to_dict(), sort_keys=True)
    assert payload["experiment_id"] == "synthetic-4fold-loso"
    assert payload["fold_id"] == "outer_test_fox"
    assert payload["model_name"] == "DecisionTreeClassifier"
    assert payload["weighting_regime"] == "scenario_equalized_class_weighted"
    assert len(payload["inner_selection"]["inner_candidate_ids"]) == 3
    assert payload["inner_selection"]["excluded_outer_test_scenarios"] == ["fox"]
    assert len(payload["outer_evaluation"]["per_seed_results"]) == 2
    assert payload["split_manifest_sha256"]
    assert payload["quality_report_sha256"]


def test_single_outer_fold_orchestrator_fails_closed_on_leaked_or_invalid_provider(tmp_path):
    cfg_path = _build_synthetic_plan_fixture(
        tmp_path, mutate_config=_full_baseline_compatible_mutator
    )
    all_records = {s: _make_synthetic_supervised_records(s) for s in SCENARIOS}

    class _DictProvider:
        def __init__(self, data):
            self._data = data

        def load_scenario_records(self, scenario_id: str):
            return self._data[scenario_id]

    # 1. Unknown fold_id
    with pytest.raises(ExperimentRunnerError, match="Unknown fold_id"):
        run_single_outer_fold_experiment(
            fold_id="outer_test_unknown",
            model_name="LogisticRegression",
            weighting_regime="class_weighted",
            record_provider=_DictProvider(all_records),
            config_path=cfg_path,
            base_dir=tmp_path,
        )

    # 2. Provider leaks outer test ("fox") record when asked for "harrison"
    leaked_data = dict(all_records)
    leaked_data["harrison"] = [*all_records["harrison"], all_records["fox"][0]]
    with pytest.raises(ExperimentRunnerError, match="forbidden scenario"):
        run_single_outer_fold_experiment(
            fold_id="outer_test_fox",
            model_name="LogisticRegression",
            weighting_regime="class_weighted",
            record_provider=_DictProvider(leaked_data),
            config_path=cfg_path,
            base_dir=tmp_path,
        )

    # 3. Provider returns duplicate event identity
    dup_data = dict(all_records)
    dup_data["santos"] = [*all_records["santos"], all_records["santos"][0]]
    with pytest.raises(ExperimentRunnerError, match="duplicate record identity"):
        run_single_outer_fold_experiment(
            fold_id="outer_test_fox",
            model_name="LogisticRegression",
            weighting_regime="class_weighted",
            record_provider=_DictProvider(dup_data),
            config_path=cfg_path,
            base_dir=tmp_path,
        )

    # 4. Provider returns unknown (None) label
    unknown_rec = SupervisedRecord(
        **{**all_records["santos"][0].__dict__, "label": None}  # type: ignore[arg-type]
    )
    unknown_data = dict(all_records)
    unknown_data["santos"] = [unknown_rec, *all_records["santos"][1:]]
    with pytest.raises(ExperimentRunnerError, match="invalid supervised label"):
        run_single_outer_fold_experiment(
            fold_id="outer_test_fox",
            model_name="LogisticRegression",
            weighting_regime="class_weighted",
            record_provider=_DictProvider(unknown_data),
            config_path=cfg_path,
            base_dir=tmp_path,
        )
