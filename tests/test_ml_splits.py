import hashlib
import json

import pytest

from backend.ml.splits import (
    SPLIT_SCHEMA_VERSION,
    SplitPreparationError,
    create_split_manifest,
)


SCENARIOS = ("russellmitchell", "santos", "harrison")


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _source_entry(scenario, filename, *, covered, physical_lines):
    source_hash = _sha256(f"{scenario}:{filename}".encode("utf-8"))
    entry = {
        "source_filename": filename,
        "source_reference": f"raw/{scenario}/gather/{filename}",
        "scenario_id": scenario,
        "source_sha256": source_hash,
        "physical_line_count": physical_lines,
        "parsed_record_count": physical_lines,
        "rejected_log_row_count": 0,
        "rejected_row_count": 0,
        "ground_truth_filename": None,
        "ground_truth_reference": None,
        "ground_truth_sha256": None,
        "ground_truth_physical_line_count": 0,
        "ground_truth_record_count": 0,
        "attached_ground_truth_count": 0,
    }
    if covered:
        entry.update(
            {
                "ground_truth_filename": filename.rsplit("/", 1)[-1],
                "ground_truth_reference": f"raw/{scenario}/labels/{filename}",
                "ground_truth_sha256": _sha256(
                    f"{scenario}:labels:{filename}".encode("utf-8")
                ),
                "ground_truth_physical_line_count": 1,
                "ground_truth_record_count": 1,
                "attached_ground_truth_count": 1,
            }
        )
    return entry


def _record(scenario, source, source_entry, line_number, label_name, *, duplicate=False):
    event_id = f"{source}:{1 if duplicate else line_number}"
    ground_truth = None
    ground_truth_provenance = None
    label_reference = source_entry["ground_truth_reference"]
    label_hash = source_entry["ground_truth_sha256"]
    label_basis = {
        "normal": "absence_from_verified_ground_truth_file",
        "attack_associated": "explicit_ground_truth",
        "unknown": "insufficient_verified_label_coverage",
    }[label_name]
    if label_name == "attack_associated":
        ground_truth = {"labels": ["foothold"], "rules": {"foothold": ["test.rule"]}}
        ground_truth_provenance = {
            "filename": source_entry["ground_truth_filename"],
            "reference": label_reference,
            "sha256": label_hash,
            "label_file_line_number": 1,
        }
    return {
        "event": {
            "case_id": f"case-{scenario}",
            "evidence_id": f"evidence-{scenario}-{source}",
            "event_id": event_id,
            "timestamp": "2022-01-21T00:00:00+00:00",
            "source_ip": "192.0.2.1",
            "event_type": "GET",
            "resource": "/sample",
            "response_status": "200",
            "transfer_size": 12,
            "original_evidence_reference": f"raw/{scenario}/gather/{source}",
        },
        "provenance": {
            "source_filename": source,
            "source_line_number": line_number,
            "scenario_id": scenario,
            "source_sha256": source_entry["source_sha256"],
        },
        "features": {"request_frequency_5m": line_number},
        "ground_truth": ground_truth,
        "ground_truth_provenance": ground_truth_provenance,
        "research_label": {
            "normal": 0,
            "attack_associated": 1,
            "unknown": None,
        }[label_name],
        "research_label_name": label_name,
        "research_label_basis": label_basis,
        "research_label_provenance": {
            "policy_id": "ait-lds-v2.1-explicit-ground-truth-v1",
            "label_schema_version": "1.0.0",
            "ground_truth_reference": label_reference
            if label_name != "unknown"
            else None,
            "ground_truth_sha256": label_hash if label_name != "unknown" else None,
        },
    }


def _write_scenario(tmp_path, scenario, *, duplicate_identity=False):
    directory = tmp_path / scenario
    directory.mkdir()
    covered_name = "intranet/access.log"
    unknown_name = "web/access.log"
    covered = _source_entry(
        scenario, covered_name, covered=True, physical_lines=2
    )
    uncovered = _source_entry(
        scenario, unknown_name, covered=False, physical_lines=1
    )
    rows = [
        _record(scenario, covered_name, covered, 1, "normal"),
        _record(
            scenario,
            covered_name,
            covered,
            2,
            "attack_associated",
            duplicate=duplicate_identity,
        ),
        _record(scenario, unknown_name, uncovered, 1, "unknown"),
    ]
    labelled_content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    labelled_path = directory / "labelled_events.jsonl"
    labelled_path.write_bytes(labelled_content)

    prepared_content = f"prepared-{scenario}\n".encode("utf-8")
    prepared_path = directory / "prepared_events.jsonl"
    prepared_path.write_bytes(prepared_content)
    prepared_manifest = {
        "manifest_schema_version": "1.0.0",
        "dataset_name": "AIT Log Data Set V2.1",
        "source_files": [covered, uncovered],
        "record_counts": {
            "parsed_records": len(rows),
            "attached_ground_truth_records": 1,
            "rejected_rows": 0,
        },
        "processed_output": {
            "filename": prepared_path.name,
            "sha256": _sha256(prepared_content),
            "record_count": len(rows),
        },
    }
    prepared_manifest_content = json.dumps(
        prepared_manifest, sort_keys=True
    ).encode("utf-8")
    (directory / "manifest.json").write_bytes(prepared_manifest_content)
    per_source = [
        {
            "source_filename": covered_name,
            "source_sha256": covered["source_sha256"],
            "ground_truth_sha256": covered["ground_truth_sha256"],
            "ground_truth_coverage": "verified",
            "evaluation_records": 2,
            "classes": {
                "normal": 1,
                "attack_associated": 1,
                "unknown": 0,
            },
        },
        {
            "source_filename": unknown_name,
            "source_sha256": uncovered["source_sha256"],
            "ground_truth_sha256": None,
            "ground_truth_coverage": "unavailable",
            "evaluation_records": 1,
            "classes": {
                "normal": 0,
                "attack_associated": 0,
                "unknown": 1,
            },
        },
    ]
    label_manifest = {
        "manifest_schema_version": "1.0.0",
        "label_schema_version": "1.0.0",
        "scenario_id": scenario,
        "source_url": "https://zenodo.org/records/19483937",
        "version": "v2_1",
        "license": "CC-BY-NC-SA-4.0",
        "input": {
            "prepared_records_sha256": _sha256(prepared_content),
            "prepared_manifest_sha256": _sha256(prepared_manifest_content),
        },
        "labelled_output": {
            "filename": labelled_path.name,
            "sha256": _sha256(labelled_content),
        },
        "record_counts": {
            "input_records": len(rows),
            "evaluation_records": len(rows),
            "excluded_outside_simulation_period": 0,
        },
        "evaluation_class_distribution": {
            "normal": 1,
            "attack_associated": 1,
            "unknown": 1,
        },
        "known_class_percentages": {
            "normal": 50.0,
            "attack_associated": 50.0,
        },
        "per_category_counts": {"foothold": 1},
        "per_source_distributions": per_source,
        "source_bias_assessment": {
            "event_bearing_source_count": 2,
            "sources_with_verified_label_coverage": 1,
            "sources_contributing_known_labels": 1,
            "potential_bias": True,
            "warning": "Synthetic test source concentration.",
        },
    }
    label_manifest_path = directory / "labels_manifest.json"
    label_manifest_path.write_text(
        json.dumps(label_manifest, sort_keys=True), encoding="utf-8"
    )
    return label_manifest_path, labelled_content


def _write_all_scenarios(tmp_path, *, duplicate_identity=False):
    manifest_paths = {}
    originals = {}
    for scenario in SCENARIOS:
        path, content = _write_scenario(
            tmp_path,
            scenario,
            duplicate_identity=duplicate_identity and scenario == "harrison",
        )
        manifest_paths[scenario] = path
        originals[scenario] = content
    return manifest_paths, originals


def test_scenario_partitions_are_isolated_and_unknowns_excluded(tmp_path):
    manifest_paths, original_records = _write_all_scenarios(tmp_path)
    output_path = tmp_path / "splits" / "scenario_split_manifest.json"

    manifest = create_split_manifest(manifest_paths, output_path)

    assert len(manifest["candidates"]) == 6
    assert manifest["candidate_evaluation"]["selected_assignment"] is None
    for candidate in manifest["candidates"]:
        partitions = candidate["partitions"]
        scenarios = [part["scenario_id"] for part in partitions.values()]
        assert len(set(scenarios)) == 3
        assert candidate["scenario_partition_isolation"] is True
        assert candidate["all_partitions_have_both_known_classes"] is True
        for partition in partitions.values():
            assert partition["supervised_record_count"] == 2
            assert partition["unknown_excluded_from_supervised"] == 1
    assert manifest["overall_class_distribution"] == {
        "normal": 3,
        "attack_associated": 3,
        "unknown": 3,
    }
    assert manifest["preprocessing_policy"]["performed_by_this_step"] is False
    assert manifest["preprocessing_policy"]["validation_or_test_fit_allowed"] is False
    assert manifest["preprocessing_policy"]["dataset_records_copied_or_modified"] is False
    assert output_path.is_file()
    for scenario, content in original_records.items():
        assert (
            (manifest_paths[scenario].parent / "labelled_events.jsonl").read_bytes()
            == content
        )


def test_split_manifest_is_deterministic(tmp_path):
    manifest_paths, _ = _write_all_scenarios(tmp_path)
    output_path = tmp_path / "splits" / "scenario_split_manifest.json"

    first = create_split_manifest(manifest_paths, output_path)
    first_bytes = output_path.read_bytes()
    second = create_split_manifest(manifest_paths, output_path)

    assert second == first
    assert output_path.read_bytes() == first_bytes
    assert first["manifest_schema_version"] == SPLIT_SCHEMA_VERSION


def test_cross_scenario_identifier_collisions_and_duplicate_records_are_reported(
    tmp_path,
):
    manifest_paths, _ = _write_all_scenarios(
        tmp_path, duplicate_identity=True
    )
    manifest = create_split_manifest(
        manifest_paths, tmp_path / "splits" / "scenario_split_manifest.json"
    )
    leakage = manifest["cross_scenario_leakage_analysis"]

    assert leakage["duplicate_composite_identities_within_scenario"] == 1
    assert leakage["event_id_strings_shared_across_scenarios"] == 3
    pair = leakage["pairwise"]["russellmitchell:santos"]
    assert pair["shared_event_id_strings"] == 3
    assert pair["shared_exact_event_content_fingerprints"] == 2
    assert pair["shared_exact_feature_vector_fingerprints"] == 2


def test_unknown_records_remain_in_source_and_never_enter_supervised_counts(
    tmp_path,
):
    manifest_paths, original_records = _write_all_scenarios(tmp_path)
    output_path = tmp_path / "splits" / "scenario_split_manifest.json"

    manifest = create_split_manifest(manifest_paths, output_path)

    for candidate in manifest["candidates"]:
        for partition in candidate["partitions"].values():
            assert (
                partition["supervised_record_count"]
                == partition["class_distribution"]["normal"]
                + partition["class_distribution"]["attack_associated"]
            )
            assert partition["unknown_excluded_from_supervised"] == 1
    for scenario, content in original_records.items():
        assert (
            (manifest_paths[scenario].parent / "labelled_events.jsonl").read_bytes()
            == content
        )


def _write_unknown_only_scenario(tmp_path, scenario):
    directory = tmp_path / scenario
    directory.mkdir()
    unknown_name = "web/access.log"
    uncovered = _source_entry(
        scenario, unknown_name, covered=False, physical_lines=1
    )
    rows = [_record(scenario, unknown_name, uncovered, 1, "unknown")]
    labelled_content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    labelled_path = directory / "labelled_events.jsonl"
    labelled_path.write_bytes(labelled_content)

    prepared_content = f"prepared-{scenario}\n".encode("utf-8")
    prepared_path = directory / "prepared_events.jsonl"
    prepared_path.write_bytes(prepared_content)
    prepared_manifest = {
        "manifest_schema_version": "1.0.0",
        "dataset_name": "AIT Log Data Set V2.1",
        "source_files": [uncovered],
        "record_counts": {
            "parsed_records": len(rows),
            "attached_ground_truth_records": 0,
            "rejected_rows": 0,
        },
        "processed_output": {
            "filename": prepared_path.name,
            "sha256": _sha256(prepared_content),
            "record_count": len(rows),
        },
    }
    prepared_manifest_content = json.dumps(
        prepared_manifest, sort_keys=True
    ).encode("utf-8")
    (directory / "manifest.json").write_bytes(prepared_manifest_content)
    label_manifest = {
        "manifest_schema_version": "1.0.0",
        "label_schema_version": "1.0.0",
        "scenario_id": scenario,
        "source_url": "https://zenodo.org/records/19483937",
        "version": "v2_1",
        "license": "CC-BY-NC-SA-4.0",
        "input": {
            "prepared_records_sha256": _sha256(prepared_content),
            "prepared_manifest_sha256": _sha256(prepared_manifest_content),
        },
        "labelled_output": {
            "filename": labelled_path.name,
            "sha256": _sha256(labelled_content),
        },
        "record_counts": {
            "input_records": len(rows),
            "evaluation_records": len(rows),
            "excluded_outside_simulation_period": 0,
        },
        "evaluation_class_distribution": {
            "normal": 0,
            "attack_associated": 0,
            "unknown": 1,
        },
        "known_class_percentages": {"normal": 0.0, "attack_associated": 0.0},
        "per_category_counts": {},
        "per_source_distributions": [
            {
                "source_filename": unknown_name,
                "source_sha256": uncovered["source_sha256"],
                "ground_truth_sha256": None,
                "ground_truth_coverage": "unavailable",
                "evaluation_records": 1,
                "classes": {
                    "normal": 0,
                    "attack_associated": 0,
                    "unknown": 1,
                },
            }
        ],
        "source_bias_assessment": {
            "event_bearing_source_count": 1,
            "sources_with_verified_label_coverage": 0,
            "sources_contributing_known_labels": 0,
            "potential_bias": True,
            "warning": "No verified label coverage.",
        },
    }
    label_manifest_path = directory / "labels_manifest.json"
    label_manifest_path.write_text(
        json.dumps(label_manifest, sort_keys=True), encoding="utf-8"
    )
    return label_manifest_path


def test_four_scenario_split_supports_multi_scenario_training_and_is_deterministic(
    tmp_path,
):
    manifest_paths, _ = _write_all_scenarios(tmp_path)
    wheeler_path, _ = _write_scenario(tmp_path, "wheeler")
    manifest_paths["wheeler"] = wheeler_path
    output_path = tmp_path / "splits" / "scenario_split_manifest.json"

    first = create_split_manifest(manifest_paths, output_path)
    first_bytes = output_path.read_bytes()
    second = create_split_manifest(manifest_paths, output_path)

    assert second == first
    assert output_path.read_bytes() == first_bytes
    assert first["scenario_ids"] == ["harrison", "russellmitchell", "santos", "wheeler"]
    assert len(first["candidates"]) == 12
    assert len(first["cross_scenario_leakage_analysis"]["pairwise"]) == 6
    assert first["candidate_evaluation"]["selected_assignment"] is None
    assert first["candidate_evaluation"]["scientifically_adequate_candidate_count"] == 0
    for candidate in first["candidates"]:
        assignment = candidate["assignment"]
        assert len(assignment["train"]) == 2
        assert len(assignment["validation"]) == 1
        assert len(assignment["test"]) == 1
        assigned = assignment["train"] + assignment["validation"] + assignment["test"]
        assert len(set(assigned)) == 4
        assert candidate["scenario_partition_isolation"] is True
        assert candidate["all_partitions_have_both_known_classes"] is True
        assert candidate["scientifically_adequate"] is False
        assert candidate["partitions"]["train"]["supervised_record_count"] == 4
        assert candidate["partitions"]["validation"]["supervised_record_count"] == 2
        assert candidate["partitions"]["test"]["supervised_record_count"] == 2


def test_four_scenario_split_flags_insufficient_labels_and_rejects_invalid_inputs(
    tmp_path,
):
    manifest_paths, _ = _write_all_scenarios(tmp_path)
    manifest_paths["wheeler"] = _write_unknown_only_scenario(tmp_path, "wheeler")

    manifest = create_split_manifest(
        manifest_paths, tmp_path / "splits" / "scenario_split_manifest.json"
    )

    assert manifest["candidate_evaluation"]["selected_assignment"] is None
    assert (
        manifest["candidate_evaluation"]["all_candidates_have_both_classes_per_partition"]
        is False
    )
    inadequate_candidates = [
        c for c in manifest["candidates"] if not c["all_partitions_have_both_known_classes"]
    ]
    assert len(inadequate_candidates) == 6
    for candidate in inadequate_candidates:
        assert candidate["scientifically_adequate"] is False
        assert any("lacks both known classes" in note for note in candidate["limitations"])

    with pytest.raises(SplitPreparationError):
        create_split_manifest(
            {"russellmitchell": manifest_paths["russellmitchell"], "santos": manifest_paths["santos"]},
            tmp_path / "splits" / "too_few.json",
        )
    with pytest.raises(SplitPreparationError):
        create_split_manifest(
            {
                "russellmitchell": manifest_paths["russellmitchell"],
                "santos": manifest_paths["santos"],
                "harrison": manifest_paths["santos"],
            },
            tmp_path / "splits" / "duplicate_path.json",
        )
