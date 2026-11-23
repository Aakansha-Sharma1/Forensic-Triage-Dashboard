import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.ml.labels import (
    LABELLED_RECORDS_FILENAME,
    LABEL_MANIFEST_FILENAME,
    LabelPreparationError,
    prepare_labelled_dataset,
)


def source_entry(
    source_filename,
    *,
    coverage=True,
    physical_lines=1,
    scenario_id="russellmitchell",
):
    source_hash = hashlib.sha256(source_filename.encode("utf-8")).hexdigest()
    entry = {
        "source_filename": source_filename,
        "source_reference": f"raw/gather/{source_filename}",
        "scenario_id": scenario_id,
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
    if coverage:
        entry.update(
            {
                "ground_truth_filename": Path(source_filename).name,
                "ground_truth_reference": f"raw/labels/{source_filename}",
                "ground_truth_sha256": "b" * 64,
                "ground_truth_physical_line_count": 1,
                "ground_truth_record_count": 1,
                "attached_ground_truth_count": 1,
            }
        )
    return entry


def event_record(
    source,
    source_hash,
    line_number,
    timestamp,
    *,
    event_id=None,
    ground_truth=None,
    ground_truth_provenance=None,
    scenario_id="russellmitchell",
):
    return {
        "event": {
            "event_id": event_id or f"{source}:{line_number}",
            "timestamp": timestamp,
        },
        "provenance": {
            "source_filename": source,
            "source_line_number": line_number,
            "source_sha256": source_hash,
            "scenario_id": scenario_id,
        },
        "features": {"request_frequency_5m": line_number},
        "ground_truth": ground_truth,
        "ground_truth_provenance": ground_truth_provenance,
    }


def labelled_ground_truth(source):
    return (
        {"labels": ["foothold", "attacker_http"], "rules": {"foothold": ["rule.a"]}},
        {
            "filename": Path(source["source_filename"]).name,
            "reference": source["ground_truth_reference"],
            "sha256": source["ground_truth_sha256"],
            "label_file_line_number": 1,
        },
    )


def write_inputs(tmp_path, source_entries, records):
    records_path = tmp_path / "prepared_events.jsonl"
    records_content = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    records_path.write_bytes(records_content)
    source_manifest = {
        "dataset_name": "AIT Log Data Set",
        "source_url": "https://zenodo.org/records/19483937",
        "version": "v2_1",
        "license": "CC-BY-NC-SA-4.0",
        "source_archive": {"sha256": "c" * 64},
        "source_files": source_entries,
        "processed_output": {"sha256": hashlib.sha256(records_content).hexdigest()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="utf-8"
    )
    return records_path, manifest_path, records_content


def read_labelled_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_labels_use_explicit_categories_verified_normal_coverage_and_unknowns(tmp_path):
    covered_name = "intranet/access.log"
    uncovered_name = "webserver/access.log"
    covered = source_entry(covered_name, physical_lines=3)
    covered["ground_truth_physical_line_count"] = 2
    covered["ground_truth_record_count"] = 2
    covered["attached_ground_truth_count"] = 2
    uncovered = source_entry(uncovered_name, coverage=False)
    attack_payload, attack_provenance = labelled_ground_truth(covered)
    records = [
        event_record(
            covered_name,
            covered["source_sha256"],
            1,
            "2022-01-21T00:00:00Z",
            ground_truth=attack_payload,
            ground_truth_provenance=attack_provenance,
        ),
        event_record(
            covered_name, covered["source_sha256"], 2, "2022-01-21T00:01:00Z"
        ),
        event_record(
            covered_name,
            covered["source_sha256"],
            3,
            "2022-01-21T00:02:00Z",
            ground_truth={"labels": [], "rules": {}},
            ground_truth_provenance={
                "filename": covered["ground_truth_filename"],
                "reference": covered["ground_truth_reference"],
                "sha256": covered["ground_truth_sha256"],
                "label_file_line_number": 2,
            },
        ),
        event_record(
            uncovered_name,
            uncovered["source_sha256"],
            1,
            "2022-01-21T00:03:00Z",
        ),
    ]
    records_path, manifest_path, _ = write_inputs(
        tmp_path, [covered, uncovered], records
    )

    manifest = prepare_labelled_dataset(records_path, manifest_path, tmp_path)
    labelled = read_labelled_records(tmp_path / LABELLED_RECORDS_FILENAME)

    assert [row["research_label"] for row in labelled] == [1, 0, None, None]
    assert [row["research_label_name"] for row in labelled] == [
        "attack_associated",
        "normal",
        "unknown",
        "unknown",
    ]
    assert labelled[0]["ground_truth"] == attack_payload
    assert labelled[0]["ground_truth_provenance"] == attack_provenance
    assert labelled[0]["features"] == records[0]["features"]
    assert "research_label" not in labelled[0]["features"]
    assert labelled[1]["research_label_provenance"]["ground_truth_sha256"] == "b" * 64
    assert manifest["evaluation_class_distribution"] == {
        "normal": 1,
        "attack_associated": 1,
        "unknown": 2,
    }
    assert manifest["per_category_counts"] == {
        "attacker_http": 1,
        "foothold": 1,
    }
    assert manifest["source_bias_assessment"]["potential_bias"] is True
    assert manifest["source_bias_assessment"]["sources_contributing_known_labels"] == 1


def test_normal_is_not_inferred_from_incomplete_label_coverage(tmp_path):
    source_name = "host/partially-labelled.log"
    source = source_entry(source_name, physical_lines=2)
    source["ground_truth_physical_line_count"] = 2
    source["ground_truth_record_count"] = 1
    source["attached_ground_truth_count"] = 1
    record = event_record(
        source_name, source["source_sha256"], 1, "2022-01-22T00:00:00Z"
    )
    records_path, manifest_path, _ = write_inputs(tmp_path, [source], [record])

    manifest = prepare_labelled_dataset(records_path, manifest_path, tmp_path)
    labelled = read_labelled_records(tmp_path / LABELLED_RECORDS_FILENAME)

    assert labelled[0]["research_label"] is None
    assert labelled[0]["research_label_basis"] == "insufficient_verified_label_coverage"
    assert manifest["per_source_distributions"][0]["ground_truth_coverage"] == "incomplete"


def test_simulation_window_includes_start_and_excludes_end(tmp_path):
    source_name = "host/access.log"
    source = source_entry(source_name, physical_lines=3)
    records = [
        event_record(
            source_name,
            source["source_sha256"],
            1,
            "2022-01-20T23:59:59.999999Z",
        ),
        event_record(
            source_name, source["source_sha256"], 2, "2022-01-21T00:00:00Z"
        ),
        event_record(
            source_name, source["source_sha256"], 3, "2022-01-25T00:00:00Z"
        ),
    ]
    records_path, manifest_path, _ = write_inputs(tmp_path, [source], records)

    manifest = prepare_labelled_dataset(records_path, manifest_path, tmp_path)
    labelled = read_labelled_records(tmp_path / LABELLED_RECORDS_FILENAME)

    assert [row["event"]["event_id"] for row in labelled] == [
        "host/access.log:2"
    ]
    assert manifest["record_counts"]["excluded_before_start"] == 1
    assert manifest["record_counts"]["excluded_at_or_after_end"] == 1
    assert manifest["record_counts"]["excluded_outside_simulation_period"] == 2


def test_duplicate_event_identifiers_are_counted_without_silent_deduplication(tmp_path):
    source_name = "host/access.log"
    source = source_entry(source_name, physical_lines=2)
    records = [
        event_record(
            source_name,
            source["source_sha256"],
            1,
            "2022-01-22T00:00:00Z",
            event_id="duplicate-id",
        ),
        event_record(
            source_name,
            source["source_sha256"],
            2,
            "2022-01-22T00:01:00Z",
            event_id="duplicate-id",
        ),
    ]
    records_path, manifest_path, _ = write_inputs(tmp_path, [source], records)

    manifest = prepare_labelled_dataset(records_path, manifest_path, tmp_path)
    labelled = read_labelled_records(tmp_path / LABELLED_RECORDS_FILENAME)

    assert len(labelled) == 2
    assert manifest["duplicate_event_identifiers"] == {
        "unique_identifiers_repeated": 1,
        "duplicate_rows_after_first": 1,
        "examples": ["duplicate-id"],
    }


def test_labelled_dataset_and_manifest_are_reproducible_without_replacing_inputs(
    tmp_path,
):
    source_name = "host/access.log"
    source = source_entry(source_name)
    record = event_record(
        source_name, source["source_sha256"], 1, "2022-01-22T00:00:00Z"
    )
    records_path, manifest_path, original_records = write_inputs(
        tmp_path, [source], [record]
    )

    first_manifest = prepare_labelled_dataset(records_path, manifest_path, tmp_path)
    output_path = tmp_path / LABELLED_RECORDS_FILENAME
    label_manifest_path = tmp_path / LABEL_MANIFEST_FILENAME
    first_output = output_path.read_bytes()
    first_manifest_bytes = label_manifest_path.read_bytes()
    second_manifest = prepare_labelled_dataset(records_path, manifest_path, tmp_path)

    assert second_manifest == first_manifest
    assert output_path.read_bytes() == first_output
    assert label_manifest_path.read_bytes() == first_manifest_bytes
    assert records_path.read_bytes() == original_records
    assert manifest_path.exists()


def test_scenario_period_and_identifier_can_be_selected_from_official_metadata(
    tmp_path,
):
    source_name = "intranet/access.log"
    source = source_entry(
        source_name, scenario_id="santos", physical_lines=1
    )
    record = event_record(
        source_name,
        source["source_sha256"],
        1,
        "2022-01-14T00:00:00Z",
        scenario_id="santos",
    )
    records_path, manifest_path, _ = write_inputs(tmp_path, [source], [record])

    manifest = prepare_labelled_dataset(
        records_path,
        manifest_path,
        tmp_path,
        scenario_id="santos",
        simulation_start=datetime(2022, 1, 14, tzinfo=timezone.utc),
        simulation_end=datetime(2022, 1, 18, tzinfo=timezone.utc),
    )
    labelled = read_labelled_records(tmp_path / LABELLED_RECORDS_FILENAME)

    assert manifest["scenario_id"] == "santos"
    assert (
        manifest["simulation_period"]["start_inclusive"]
        == "2022-01-14T00:00:00+00:00"
    )
    assert (
        manifest["simulation_period"]["end_exclusive"]
        == "2022-01-18T00:00:00+00:00"
    )
    assert labelled[0]["research_label"] == 0


def test_scenario_mismatch_is_rejected_before_creating_output(tmp_path):
    source_name = "intranet/access.log"
    source = source_entry(source_name, scenario_id="santos")
    records_path, manifest_path, _ = write_inputs(tmp_path, [source], [])
    output_dir = tmp_path / "not-created"

    with pytest.raises(LabelPreparationError, match="does not match"):
        prepare_labelled_dataset(
            records_path,
            manifest_path,
            output_dir,
            scenario_id="harrison",
        )

    assert not output_dir.exists()
