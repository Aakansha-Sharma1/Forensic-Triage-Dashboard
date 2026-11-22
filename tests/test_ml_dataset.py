import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from backend.ml import (
    AIT_LDS_LICENSE,
    AIT_LDS_SOURCE_URL,
    AIT_LDS_VERSION,
    DatasetImportError,
    create_dataset_manifest,
    extract_selected_archive_members,
    import_ground_truth_labels,
    parse_apache_access_log,
    prepare_apache_source,
    write_processed_dataset,
)
from backend.ml.features import engineer_features


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic_ait_style"
APACHE_FIXTURE = FIXTURE_ROOT / "gather" / "synthetic-host" / "apache_access.txt"
LABEL_FIXTURE = FIXTURE_ROOT / "labels" / "synthetic-host" / "apache_access.txt"


def parse_fixture(path=APACHE_FIXTURE):
    return parse_apache_access_log(
        path,
        case_id="synthetic-case-01",
        evidence_id="synthetic-evidence-01",
        scenario_id="synthetic-scenario-01",
        source_filename="synthetic-host/apache_access.txt",
    )


def test_apache_parser_preserves_canonical_fields_and_provenance():
    parsed = parse_fixture()

    assert parsed.source_filename == "synthetic-host/apache_access.txt"
    assert parsed.scenario_id == "synthetic-scenario-01"
    assert parsed.source_sha256 == hashlib.sha256(APACHE_FIXTURE.read_bytes()).hexdigest()
    assert parsed.physical_line_count == 3
    assert len(parsed.records) == 2
    assert parsed.rejected_rows[0].source_line_number == 3
    assert parsed.rejected_rows[0].reason == "malformed_apache_record"

    first, second = parsed.records
    assert first.event.case_id == "synthetic-case-01"
    assert first.event.evidence_id == "synthetic-evidence-01"
    assert first.event.event_id == "synthetic-host/apache_access.txt:1"
    assert first.event.source_ip == "192.0.2.10"
    assert first.event.event_type == "GET"
    assert first.event.resource == "/home"
    assert first.event.response_status == "200"
    assert first.event.transfer_size == 512
    assert first.event.original_evidence_reference == str(APACHE_FIXTURE)
    assert first.source_line_number == 1
    assert second.source_line_number == 2
    assert second.event.event_type == "POST"
    assert second.event.response_status == "401"
    assert second.event.transfer_size == 128

    features = engineer_features(record.event for record in parsed.records)
    assert features[1].features["failed_request_frequency_5m"] == 1
    assert features[1].features["post_request_frequency_5m"] == 1


def test_apache_parser_does_not_invent_missing_source_values(tmp_path):
    source = tmp_path / "missing_fields.txt"
    source.write_text(
        '- - - [01/Jan/2022:10:00:00 +0000] "-" - -\n',
        encoding="utf-8",
    )
    parsed = parse_apache_access_log(
        source,
        case_id=1,
        evidence_id="explicit-evidence-id",
        scenario_id=None,
    )

    assert len(parsed.records) == 1
    event = parsed.records[0].event
    assert event.source_ip is None
    assert event.event_type is None
    assert event.resource is None
    assert event.response_status is None
    assert event.transfer_size is None
    assert parsed.records[0].scenario_id is None


def test_apache_parser_supports_vhost_combined_records(tmp_path):
    source = tmp_path / "vhost_access.txt"
    source.write_text(
        'cloud.example.test:443 192.0.2.20 - - '
        '[24/Jan/2022:07:36:12 +0000] "GET / HTTP/1.1" 302 4122 "-" '
        '"SyntheticClient/1.0"\n',
        encoding="utf-8",
    )
    parsed = parse_apache_access_log(
        source,
        case_id=1,
        evidence_id="synthetic-vhost",
        scenario_id="synthetic-scenario",
    )

    assert len(parsed.records) == 1
    assert parsed.records[0].event.source_ip == "192.0.2.20"
    assert parsed.records[0].virtual_host == "cloud.example.test:443"


def test_ground_truth_labels_are_attached_verbatim_without_filling_unlabeled_rows():
    parsed = parse_fixture()
    labels = import_ground_truth_labels(
        LABEL_FIXTURE,
        source_filename=parsed.source_filename,
    )
    prepared = prepare_apache_source(parsed, labels)
    expected_label = json.loads(LABEL_FIXTURE.read_text(encoding="utf-8").strip())

    assert labels.label_sha256 == hashlib.sha256(LABEL_FIXTURE.read_bytes()).hexdigest()
    assert len(labels.labels) == 1
    assert prepared.records[0].ground_truth is None
    assert prepared.records[1].ground_truth == expected_label
    assert prepared.records[1].ground_truth["labels"] == ["synthetic_attack_example"]
    assert prepared.records[1].ground_truth_filename == LABEL_FIXTURE.name
    assert prepared.records[1].ground_truth_sha256 == labels.label_sha256
    assert prepared.records[1].ground_truth_label_file_line_number == 1
    assert prepared.records[1].features["post_request_frequency_5m"] == 1
    assert prepared.records[1].features["failed_request_frequency_5m"] == 1


def test_label_rejections_and_unmatched_references_are_reported(tmp_path):
    label_file = tmp_path / APACHE_FIXTURE.name
    label_file.write_text(
        "\n".join(
            [
                '{"line":2,"labels":["first"]}',
                '{"line":2,"labels":["duplicate"]}',
                '{"line":3,"labels":["malformed-log-line"]}',
                '{"line":99,"labels":["outside-file"]}',
                "not json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    labels = import_ground_truth_labels(
        label_file,
        source_filename=parse_fixture().source_filename,
    )
    prepared = prepare_apache_source(parse_fixture(), labels)
    reasons = [row.reason for row in prepared.rejected_rows]

    assert reasons.count("duplicate_event_line_label") == 1
    assert reasons.count("malformed_label_json") == 1
    assert reasons.count("label_references_rejected_log_row") == 1
    assert reasons.count("label_line_reference_out_of_range") == 1
    assert prepared.records[1].ground_truth["labels"] == ["first"]


def test_manifest_and_processed_output_are_reproducible_and_separate(tmp_path):
    prepared = prepare_apache_source(
        parse_fixture(),
        import_ground_truth_labels(
            LABEL_FIXTURE, source_filename=parse_fixture().source_filename
        ),
    )
    first_manifest = create_dataset_manifest([prepared])
    second_manifest = create_dataset_manifest([prepared])
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    archive_metadata = {
        "filename": "synthetic_archive.zip",
        "size_bytes": 64,
        "sha256": "a" * 64,
        "md5": "b" * 32,
    }

    written_manifest = write_processed_dataset(
        [prepared],
        output_dir=processed,
        raw_data_dir=raw,
    )
    records_path = processed / "prepared_events.jsonl"
    manifest_path = processed / "manifest.json"
    first_records = records_path.read_bytes()
    updated_manifest = write_processed_dataset(
        [prepared],
        output_dir=processed,
        raw_data_dir=raw,
        source_archive=archive_metadata,
    )
    repeated_manifest = write_processed_dataset(
        [prepared],
        output_dir=processed,
        raw_data_dir=raw,
        source_archive=archive_metadata,
    )

    assert first_manifest == second_manifest
    assert written_manifest["source_url"] == AIT_LDS_SOURCE_URL
    assert written_manifest["version"] == AIT_LDS_VERSION
    assert written_manifest["license"] == AIT_LDS_LICENSE
    assert "source_archive" not in written_manifest
    assert updated_manifest["source_archive"] == archive_metadata
    assert written_manifest["feature_schema_version"] == "1.0.0"
    assert written_manifest["record_counts"] == {
        "parsed_records": 2,
        "attached_ground_truth_records": 1,
        "rejected_rows": 1,
    }
    assert written_manifest["processed_output"]["sha256"] == hashlib.sha256(first_records).hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == updated_manifest
    assert repeated_manifest == updated_manifest
    assert records_path.read_bytes() == first_records
    saved_records = [json.loads(line) for line in first_records.decode("utf-8").splitlines()]
    assert saved_records[0]["ground_truth"] is None
    assert saved_records[1]["ground_truth"]["labels"] == ["synthetic_attack_example"]

    with pytest.raises(DatasetImportError, match="must be separate"):
        write_processed_dataset([prepared], output_dir=raw, raw_data_dir=raw)


def test_parser_and_label_import_reject_invalid_utf8_and_bad_metadata(tmp_path):
    invalid_log = tmp_path / "invalid.txt"
    invalid_log.write_bytes(b"\xff")
    with pytest.raises(DatasetImportError, match="UTF-8"):
        parse_apache_access_log(
            invalid_log,
            case_id=1,
            evidence_id="evidence",
            scenario_id=None,
        )

    parsed = parse_fixture()
    labels = import_ground_truth_labels(
        LABEL_FIXTURE,
        source_filename="different-host/apache_access.txt",
    )
    with pytest.raises(DatasetImportError, match="does not match"):
        prepare_apache_source(parsed, labels)


def test_selected_zip_extraction_rejects_traversal_and_preserves_member_paths(tmp_path):
    archive_path = tmp_path / "selective.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("gather/host/access.log", "synthetic log")
        archive.writestr("../outside.txt", "must not escape")

    destination = tmp_path / "raw"
    with pytest.raises(DatasetImportError, match="Unsafe archive member path"):
        extract_selected_archive_members(
            archive_path,
            ["gather/host/access.log", "../outside.txt"],
            destination,
        )
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()

    extracted = extract_selected_archive_members(
        archive_path,
        ["gather/host/access.log"],
        destination,
    )
    assert extracted == [destination / "gather" / "host" / "access.log"]
    assert extracted[0].read_text(encoding="utf-8") == "synthetic log"
