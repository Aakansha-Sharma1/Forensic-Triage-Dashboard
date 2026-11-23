from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, permutations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from typing import Mapping

from .dataset import PROCESSED_DATA_DIR


SPLIT_SCHEMA_VERSION = "1.0.0"
SCENARIO_IDS = ("russellmitchell", "santos", "harrison")
SPLIT_OUTPUT_PATH = (
    PROCESSED_DATA_DIR.parent / "splits" / "scenario_split_manifest.json"
)
DEFAULT_LABEL_MANIFEST_PATHS = {
    "russellmitchell": PROCESSED_DATA_DIR / "labels_manifest.json",
    "santos": PROCESSED_DATA_DIR / "santos" / "labels_manifest.json",
    "harrison": PROCESSED_DATA_DIR / "harrison" / "labels_manifest.json",
}

_CLASS_NAMES = {0: "normal", 1: "attack_associated", None: "unknown"}
_BATCH_SIZE = 4096


class SplitPreparationError(ValueError):
    """Raised when scenario datasets or their provenance cannot be verified."""


def _read_json(path: Path, description: str) -> tuple[bytes, dict]:
    try:
        content = path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitPreparationError(f"Unable to read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise SplitPreparationError(f"{description} must be a JSON object: {path}")
    return content, payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SplitPreparationError(f"Unable to hash dataset file: {path}") from exc
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _safe_filename(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or value in (".", "..")
        or "\\" in value
    ):
        raise SplitPreparationError(f"Invalid {description} filename.")
    return value


def _json_fingerprint(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _class_name(label: object, declared_name: object, record_number: int) -> str:
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise SplitPreparationError(
            f"Record {record_number}: research_label must be 0, 1, or null."
        )
    expected_name = _CLASS_NAMES[label]
    if declared_name != expected_name:
        raise SplitPreparationError(
            f"Record {record_number}: research label name does not match its value."
        )
    return expected_name


def _flush_fingerprints(
    connection: sqlite3.Connection,
    buffers: dict[str, list[tuple[str, str]]],
) -> int:
    duplicates = 0
    statements = {
        "identities": "INSERT OR IGNORE INTO identities(scenario, value) VALUES (?, ?)",
        "event_content": "INSERT OR IGNORE INTO event_content(scenario, value) VALUES (?, ?)",
        "feature_vectors": "INSERT OR IGNORE INTO feature_vectors(scenario, value) VALUES (?, ?)",
    }
    for table, rows in buffers.items():
        if not rows:
            continue
        cursor = connection.executemany(statements[table], rows)
        if table == "identities":
            duplicates += len(rows) - cursor.rowcount
        rows.clear()
    return duplicates


def _pairwise_overlap(
    connection: sqlite3.Connection,
    table: str,
) -> dict[str, int]:
    allowed_tables = {
        "identities": "identities",
        "event_content": "event_content",
        "feature_vectors": "feature_vectors",
    }
    table_name = allowed_tables[table]
    query = (
        f"SELECT first.scenario, second.scenario, COUNT(*) "
        f"FROM {table_name} AS first "
        f"JOIN {table_name} AS second "
        "ON first.value = second.value AND first.scenario < second.scenario "
        "GROUP BY first.scenario, second.scenario"
    )
    return {
        f"{first}:{second}": count
        for first, second, count in connection.execute(query)
    }


def _analyse_scenario(
    scenario_id: str,
    label_manifest_path: Path,
    connection: sqlite3.Connection,
) -> dict:
    label_manifest_bytes, label_manifest = _read_json(
        label_manifest_path, "label manifest"
    )
    if label_manifest.get("scenario_id") != scenario_id:
        raise SplitPreparationError(
            f"Manifest scenario_id does not match {scenario_id}: {label_manifest_path}"
        )

    directory = label_manifest_path.parent
    prepared_manifest_path = directory / "manifest.json"
    prepared_manifest_bytes, prepared_manifest = _read_json(
        prepared_manifest_path, "prepared dataset manifest"
    )
    prepared_output = prepared_manifest.get("processed_output")
    labelled_output = label_manifest.get("labelled_output")
    input_metadata = label_manifest.get("input")
    if not isinstance(prepared_output, dict) or not isinstance(labelled_output, dict):
        raise SplitPreparationError(
            f"Missing prepared or labelled output metadata for {scenario_id}."
        )
    if not isinstance(input_metadata, dict):
        raise SplitPreparationError(f"Missing input hashes for {scenario_id}.")

    prepared_filename = _safe_filename(
        prepared_output.get("filename"), "prepared dataset"
    )
    labelled_filename = _safe_filename(
        labelled_output.get("filename"), "labelled dataset"
    )
    prepared_path = directory / prepared_filename
    labelled_path = directory / labelled_filename
    prepared_sha256 = prepared_output.get("sha256")
    labelled_sha256 = labelled_output.get("sha256")
    if not _valid_sha256(prepared_sha256) or not _valid_sha256(labelled_sha256):
        raise SplitPreparationError(f"Invalid output hash in manifest for {scenario_id}.")
    if (
        input_metadata.get("prepared_records_sha256") != prepared_sha256
        or input_metadata.get("prepared_manifest_sha256")
        != hashlib.sha256(prepared_manifest_bytes).hexdigest()
    ):
        raise SplitPreparationError(
            f"Label manifest input hashes do not match prepared manifest for {scenario_id}."
        )
    actual_prepared_sha256 = _sha256_file(prepared_path)
    if actual_prepared_sha256 != prepared_sha256:
        raise SplitPreparationError(
            f"Prepared dataset hash mismatch for {scenario_id}."
        )
    if prepared_manifest.get("record_counts", {}).get("parsed_records") != (
        label_manifest.get("record_counts", {}).get("input_records")
    ):
        raise SplitPreparationError(
            f"Prepared and labelled input record counts differ for {scenario_id}."
        )

    source_files = prepared_manifest.get("source_files")
    per_source_manifest = label_manifest.get("per_source_distributions")
    if not isinstance(source_files, list) or not isinstance(per_source_manifest, list):
        raise SplitPreparationError(f"Missing per-source provenance for {scenario_id}.")
    source_hashes: dict[str, dict] = {}
    for source in source_files:
        if not isinstance(source, dict):
            raise SplitPreparationError(f"Invalid source entry for {scenario_id}.")
        filename = source.get("source_filename")
        source_hash = source.get("source_sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or not _valid_sha256(source_hash)
            or filename in source_hashes
        ):
            raise SplitPreparationError(
                f"Invalid or duplicate source provenance in {scenario_id}."
            )
        source_hashes[filename] = source

    expected_sources = {}
    for item in per_source_manifest:
        if not isinstance(item, dict) or item.get("source_filename") not in source_hashes:
            raise SplitPreparationError(
                f"Label source statistics do not match prepared sources in {scenario_id}."
            )
        filename = item["source_filename"]
        source = source_hashes[filename]
        if (
            item.get("source_sha256") != source.get("source_sha256")
            or item.get("ground_truth_sha256") != source.get("ground_truth_sha256")
        ):
            raise SplitPreparationError(
                f"Source hashes do not match between manifests for {scenario_id}."
            )
        expected_sources[filename] = item
    if set(expected_sources) != set(source_hashes):
        raise SplitPreparationError(
            f"Per-source statistics are incomplete for {scenario_id}."
        )

    classes: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    per_source_classes: dict[str, Counter[str]] = defaultdict(Counter)
    records_digest = hashlib.sha256()
    buffers = {"identities": [], "event_content": [], "feature_vectors": []}
    record_count = 0
    duplicate_identity_count = 0
    try:
        with labelled_path.open("rb") as records:
            for record_number, raw_line in enumerate(records, start=1):
                records_digest.update(raw_line)
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SplitPreparationError(
                        f"Invalid labelled record {record_number} in {scenario_id}."
                    ) from exc
                if not isinstance(record, dict):
                    raise SplitPreparationError(
                        f"Record {record_number} in {scenario_id} must be an object."
                    )
                event = record.get("event")
                provenance = record.get("provenance")
                features = record.get("features")
                if (
                    not isinstance(event, dict)
                    or not isinstance(provenance, dict)
                    or not isinstance(features, dict)
                ):
                    raise SplitPreparationError(
                        f"Record {record_number} in {scenario_id} is missing event data."
                    )
                if any(
                    key in features
                    for key in ("research_label", "research_label_name", "ground_truth")
                ):
                    raise SplitPreparationError(
                        f"Record {record_number} in {scenario_id} has a target in features."
                    )
                if provenance.get("scenario_id") != scenario_id:
                    raise SplitPreparationError(
                        f"Record {record_number} is not from scenario {scenario_id}."
                    )
                filename = provenance.get("source_filename")
                source = source_hashes.get(filename)
                if (
                    source is None
                    or provenance.get("source_sha256") != source["source_sha256"]
                ):
                    raise SplitPreparationError(
                        f"Record {record_number} has unverifiable source provenance."
                    )
                event_id = event.get("event_id")
                timestamp = event.get("timestamp")
                if (
                    not isinstance(event_id, str)
                    or not event_id
                    or not isinstance(timestamp, str)
                    or not timestamp
                ):
                    raise SplitPreparationError(
                        f"Record {record_number} lacks event identity or timestamp."
                    )
                label_name = _class_name(
                    record.get("research_label"),
                    record.get("research_label_name"),
                    record_number,
                )
                ground_truth = record.get("ground_truth")
                ground_truth_provenance = record.get("ground_truth_provenance")
                label_provenance = record.get("research_label_provenance")
                source_metadata = source_hashes[filename]
                source_label_statistics = expected_sources[filename]
                if (
                    not isinstance(label_provenance, dict)
                    or label_provenance.get("label_schema_version")
                    != label_manifest.get("label_schema_version")
                ):
                    raise SplitPreparationError(
                        f"Record {record_number} lacks versioned label provenance."
                    )
                if label_name == "attack_associated":
                    if (
                        not isinstance(ground_truth, dict)
                        or not isinstance(ground_truth.get("labels"), list)
                        or not ground_truth["labels"]
                        or not isinstance(ground_truth_provenance, dict)
                        or ground_truth_provenance.get("reference")
                        != source_metadata.get("ground_truth_reference")
                        or ground_truth_provenance.get("sha256")
                        != source_metadata.get("ground_truth_sha256")
                        or label_provenance.get("ground_truth_reference")
                        != source_metadata.get("ground_truth_reference")
                        or label_provenance.get("ground_truth_sha256")
                        != source_metadata.get("ground_truth_sha256")
                        or record.get("research_label_basis") != "explicit_ground_truth"
                    ):
                        raise SplitPreparationError(
                            f"Attack record {record_number} lacks verified label provenance."
                        )
                    categories.update(ground_truth["labels"])
                elif label_name == "normal":
                    if (
                        ground_truth is not None
                        or source_label_statistics.get("ground_truth_coverage")
                        != "verified"
                        or label_provenance.get("ground_truth_reference")
                        != source_metadata.get("ground_truth_reference")
                        or label_provenance.get("ground_truth_sha256")
                        != source_metadata.get("ground_truth_sha256")
                        or record.get("research_label_basis")
                        != "absence_from_verified_ground_truth_file"
                    ):
                        raise SplitPreparationError(
                            f"Normal record {record_number} lacks verified label coverage."
                        )
                else:
                    if isinstance(ground_truth, dict) and ground_truth.get("labels"):
                        raise SplitPreparationError(
                            f"Unknown record {record_number} has explicit attack categories."
                        )
                    if record.get("research_label_basis") not in (
                        "insufficient_verified_label_coverage",
                        "explicit_ground_truth_without_categories",
                    ):
                        raise SplitPreparationError(
                            f"Unknown record {record_number} has an invalid label basis."
                        )

                classes[label_name] += 1
                per_source_classes[filename][label_name] += 1
                event_content = {
                    "source_filename": filename,
                    **{
                        key: event.get(key)
                        for key in (
                            "timestamp",
                            "source_ip",
                            "event_type",
                            "resource",
                            "response_status",
                            "transfer_size",
                        )
                    },
                }
                buffers["identities"].append((scenario_id, event_id))
                buffers["event_content"].append(
                    (scenario_id, _json_fingerprint(event_content))
                )
                buffers["feature_vectors"].append(
                    (scenario_id, _json_fingerprint(features))
                )
                record_count += 1
                if len(buffers["identities"]) >= _BATCH_SIZE:
                    duplicate_identity_count += _flush_fingerprints(
                        connection, buffers
                    )
        duplicate_identity_count += _flush_fingerprints(connection, buffers)
    except OSError as exc:
        raise SplitPreparationError(
            f"Unable to read labelled records for {scenario_id}."
        ) from exc

    if records_digest.hexdigest() != labelled_sha256:
        raise SplitPreparationError(f"Labelled dataset hash mismatch for {scenario_id}.")

    declared_count = label_manifest.get("record_counts", {}).get("evaluation_records")
    declared_classes = label_manifest.get("evaluation_class_distribution")
    expected_classes = {
        name: classes[name] for name in ("normal", "attack_associated", "unknown")
    }
    if record_count != declared_count or declared_classes != expected_classes:
        raise SplitPreparationError(
            f"Labelled record counts do not match manifest for {scenario_id}."
        )
    if categories != Counter(label_manifest.get("per_category_counts", {})):
        raise SplitPreparationError(
            f"Attack category counts do not match manifest for {scenario_id}."
        )
    for filename, item in expected_sources.items():
        declared = item.get("classes", {})
        actual = {
            name: per_source_classes[filename][name]
            for name in ("normal", "attack_associated", "unknown")
        }
        if declared != actual:
            raise SplitPreparationError(
                f"Per-source class counts do not match manifest for {scenario_id}."
            )

    known_count = expected_classes["normal"] + expected_classes["attack_associated"]
    source_entries_out = [
        {
            "source_filename": filename,
            "source_sha256": source["source_sha256"],
            "ground_truth_sha256": source.get("ground_truth_sha256"),
            "parsed_record_count": source.get("parsed_record_count"),
            "attached_ground_truth_count": source.get("attached_ground_truth_count"),
            "ground_truth_coverage": expected_sources[filename].get(
                "ground_truth_coverage"
            ),
        }
        for filename, source in sorted(source_hashes.items())
    ]
    return {
        "scenario_id": scenario_id,
        "record_count": record_count,
        "class_distribution": expected_classes,
        "supervised_record_count": known_count,
        "unknown_excluded_from_supervised": expected_classes["unknown"],
        "attack_to_normal_ratio": round(
            expected_classes["attack_associated"] / expected_classes["normal"], 6
        )
        if expected_classes["normal"]
        else None,
        "known_class_percentages": label_manifest.get("known_class_percentages"),
        "attack_category_counts": dict(sorted(categories.items())),
        "source_bias_assessment": label_manifest.get("source_bias_assessment"),
        "known_label_sources": [
            {
                "source_filename": filename,
                "class_distribution": {
                    name: per_source_classes[filename][name]
                    for name in ("normal", "attack_associated", "unknown")
                },
            }
            for filename in sorted(per_source_classes)
            if per_source_classes[filename]["normal"]
            or per_source_classes[filename]["attack_associated"]
        ],
        "duplicate_event_identities_within_scenario": duplicate_identity_count,
        "source_hashes": source_entries_out,
        "source_archive": prepared_manifest.get("source_archive"),
        "prepared_manifest_sha256": hashlib.sha256(
            prepared_manifest_bytes
        ).hexdigest(),
        "prepared_records_sha256": prepared_sha256,
        "label_manifest_sha256": hashlib.sha256(label_manifest_bytes).hexdigest(),
        "labelled_records_sha256": labelled_sha256,
    }


def _cross_scenario_findings(
    connection: sqlite3.Connection,
    scenarios: tuple[str, ...],
    scenario_summaries: Mapping[str, dict],
) -> dict:
    id_overlaps = _pairwise_overlap(connection, "identities")
    content_overlaps = _pairwise_overlap(connection, "event_content")
    feature_overlaps = _pairwise_overlap(connection, "feature_vectors")
    pairwise = {}
    for first, second in combinations(scenarios, 2):
        pair_key = f"{first}:{second}"
        shared_ids = id_overlaps.get(pair_key, 0)
        shared_content = content_overlaps.get(pair_key, 0)
        shared_features = feature_overlaps.get(pair_key, 0)
        left = scenario_summaries[first]
        right = scenario_summaries[second]
        left_categories = set(left["attack_category_counts"])
        right_categories = set(right["attack_category_counts"])
        category_union = left_categories | right_categories
        pairwise[pair_key] = {
            "shared_event_id_strings": shared_ids,
            "event_id_collision_note": (
                "event_id omits scenario_id; equal strings across scenarios are "
                "namespace collisions, not proof of duplicate events."
            ),
            "shared_exact_event_content_fingerprints": shared_content,
            "shared_exact_feature_vector_fingerprints": shared_features,
            "exact_event_content_jaccard": round(
                shared_content
                / max(
                    1,
                    connection.execute(
                        "SELECT COUNT(*) FROM event_content WHERE scenario = ?",
                        (first,),
                    ).fetchone()[0]
                    + connection.execute(
                        "SELECT COUNT(*) FROM event_content WHERE scenario = ?",
                        (second,),
                    ).fetchone()[0]
                    - shared_content,
                ),
                6,
            ),
            "feature_vector_jaccard": round(
                shared_features
                / max(
                    1,
                    connection.execute(
                        "SELECT COUNT(*) FROM feature_vectors WHERE scenario = ?",
                        (first,),
                    ).fetchone()[0]
                    + connection.execute(
                        "SELECT COUNT(*) FROM feature_vectors WHERE scenario = ?",
                        (second,),
                    ).fetchone()[0]
                    - shared_features,
                ),
                6,
            ),
            "attack_category_jaccard": round(
                len(left_categories & right_categories) / len(category_union), 6
            )
            if category_union
            else 1.0,
        }
    cross_scenario_identity_values = connection.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT value FROM identities GROUP BY value HAVING COUNT(DISTINCT scenario) > 1"
        ")"
    ).fetchone()[0]
    return {
        "identity_scope": ["scenario_id", "event_id"],
        "duplicate_composite_identities_within_scenario": sum(
            summary["duplicate_event_identities_within_scenario"]
            for summary in scenario_summaries.values()
        ),
        "event_id_strings_shared_across_scenarios": cross_scenario_identity_values,
        "pairwise": pairwise,
        "exact_event_fingerprint_definition": (
            "SHA-256 of source filename, timestamp, source IP, event type, resource, "
            "response status, and transfer size; event identifiers and labels omitted."
        ),
        "feature_fingerprint_definition": (
            "SHA-256 of the numerical feature mapping; this detects repeated feature "
            "patterns, not duplicate source events."
        ),
        "finding": (
            "Exact event-content overlap and within-scenario composite identity "
            "duplicates are checked independently of event_id string collisions."
        ),
    }


def _candidate_assignments(
    scenario_summaries: Mapping[str, dict],
) -> list[dict]:
    scenarios = tuple(sorted(scenario_summaries))
    candidates = []
    for train, validation, test in permutations(scenarios):
        assignment = {
            "train": train,
            "validation": validation,
            "test": test,
        }
        assignment_id = hashlib.sha256(
            json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        partitions = {}
        for partition, scenario in assignment.items():
            summary = scenario_summaries[scenario]
            classes = summary["class_distribution"]
            partitions[partition] = {
                "scenario_id": scenario,
                "record_count": summary["record_count"],
                "supervised_record_count": summary["supervised_record_count"],
                "class_distribution": {
                    "normal": classes["normal"],
                    "attack_associated": classes["attack_associated"],
                },
                "unknown_excluded_from_supervised": summary[
                    "unknown_excluded_from_supervised"
                ],
                "attack_to_normal_ratio": summary["attack_to_normal_ratio"],
                "known_label_source_count": len(summary["known_label_sources"]),
                "labelled_records_sha256": summary["labelled_records_sha256"],
            }
        candidates.append(
            {
                "candidate_id": assignment_id,
                "assignment": assignment,
                "partitions": partitions,
                "scenario_partition_isolation": len(set(assignment.values())) == 3,
                "all_partitions_have_both_known_classes": all(
                    part["class_distribution"]["normal"] > 0
                    and part["class_distribution"]["attack_associated"] > 0
                    for part in partitions.values()
                ),
                "selection_status": "not_selected",
            }
        )
    return candidates


def _write_if_unchanged(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite a different split manifest: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as output:
            created = True
            output.write(content)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def create_split_manifest(
    label_manifest_paths: Mapping[str, str | Path] | None = None,
    output_path: str | Path = SPLIT_OUTPUT_PATH,
) -> dict:
    """Analyze scenario-held-out candidates and write a deterministic manifest."""
    manifest_paths = (
        DEFAULT_LABEL_MANIFEST_PATHS
        if label_manifest_paths is None
        else label_manifest_paths
    )
    if len(manifest_paths) != 3 or len(set(manifest_paths)) != 3:
        raise SplitPreparationError(
            "Exactly three distinct scenario label manifests are required."
        )
    scenarios = tuple(sorted(manifest_paths))
    resolved_paths = {
        scenario: Path(manifest_paths[scenario]).resolve() for scenario in scenarios
    }
    resolved_output = Path(output_path).resolve()
    if resolved_output in set(resolved_paths.values()):
        raise SplitPreparationError("Split output must not overwrite an input manifest.")

    with tempfile.TemporaryDirectory(prefix="forensic-splits-") as temporary_dir:
        database_path = Path(temporary_dir) / "overlap.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute(
                "CREATE TABLE identities (scenario TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (scenario, value)) WITHOUT ROWID"
            )
            connection.execute("CREATE INDEX identities_by_value ON identities(value)")
            connection.execute(
                "CREATE TABLE event_content (scenario TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (scenario, value)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE INDEX event_content_by_value ON event_content(value)"
            )
            connection.execute(
                "CREATE TABLE feature_vectors (scenario TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (scenario, value)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE INDEX feature_vectors_by_value ON feature_vectors(value)"
            )

            scenario_summaries = {
                scenario: _analyse_scenario(
                    scenario, resolved_paths[scenario], connection
                )
                for scenario in scenarios
            }
            connection.commit()
            cross_scenario = _cross_scenario_findings(
                connection, scenarios, scenario_summaries
            )
        finally:
            connection.close()

    totals = {
        name: sum(
            summary["class_distribution"][name]
            for summary in scenario_summaries.values()
        )
        for name in ("normal", "attack_associated", "unknown")
    }
    known_total = totals["normal"] + totals["attack_associated"]
    overall_imbalance = (
        totals["attack_associated"] / totals["normal"]
        if totals["normal"]
        else None
    )
    candidates = _candidate_assignments(scenario_summaries)
    for candidate in candidates:
        candidate["scientifically_adequate"] = False
        candidate["limitations"] = [
            "Exactly one scenario is assigned to training, validation, and testing, leaving no within-partition scenario replication.",
            "All verified binary labels in each scenario come from one intranet-server Apache source.",
            "The merged known labels are attack-majority with a 10:1 threshold for the descriptive severe-imbalance flag.",
        ]

    manifest = {
        "manifest_schema_version": SPLIT_SCHEMA_VERSION,
        "split_method": "scenario_holdout",
        "scenario_ids": list(scenarios),
        "unknown_label_policy": {
            "supervised_inclusion": False,
            "preserved_in_original_labelled_datasets": True,
            "per_partition_counts_recorded": True,
        },
        "preprocessing_policy": {
            "performed_by_this_step": False,
            "future_fit_scope": "known-labelled training partition only",
            "validation_or_test_fit_allowed": False,
            "validation_or_test_use": (
                "Apply transformations fitted exclusively on training data; never "
                "fit scalers, feature selectors, thresholds, or models on validation "
                "or test records."
            ),
            "dataset_records_copied_or_modified": False,
        },
        "scenario_summaries": scenario_summaries,
        "overall_class_distribution": totals,
        "overall_known_records": known_total,
        "overall_attack_to_normal_ratio": round(overall_imbalance, 6)
        if overall_imbalance is not None
        else None,
        "imbalance_assessment": {
            "scenario_attack_to_normal_ratios": {
                scenario: summary["attack_to_normal_ratio"]
                for scenario, summary in scenario_summaries.items()
            },
            "descriptive_severe_imbalance_threshold": "attack:normal >= 10:1",
            "threshold_is_a_reporting_flag_not_a_sampling_or_selection_rule": True,
            "overall_severe_imbalance": bool(
                overall_imbalance is not None and overall_imbalance >= 10
            ),
        },
        "cross_scenario_leakage_analysis": cross_scenario,
        "candidate_evaluation": {
            "candidate_assignment_count": len(candidates),
            "all_scenario_permutations_evaluated": True,
            "all_candidates_have_both_classes_per_partition": all(
                candidate["all_partitions_have_both_known_classes"]
                for candidate in candidates
            ),
            "scientifically_adequate_candidate_count": 0,
            "selected_assignment": None,
            "selection_status": "no_scientifically_adequate_three_way_assignment",
            "limitations": [
                "With exactly three scenarios, every three-way scenario-held-out assignment has only one scenario in each partition and one scenario in training.",
                "Known-label coverage is restricted to one source file per scenario, so source and scenario effects cannot be separated.",
                "Attack-to-normal ratios differ markedly across scenarios, including 76.1459:1 in Harrison; no candidate provides replicated scenario-level estimates.",
                "Shared attack-category vocabularies and repeated feature vectors indicate common simulation patterns, although no exact event-content fingerprints overlap.",
            ],
        },
        "candidates": candidates,
        "preparation": {
            "assignment_is_deterministic": True,
            "candidate_order": "lexicographic scenario ID permutations, train/validation/test",
            "event_level_randomization": False,
            "random_seed": None,
            "oversampling_or_undersampling": False,
            "model_training_or_evaluation": False,
            "unknown_records_deleted": False,
            "source_event_ids_and_provenance_modified": False,
        },
    }
    content = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_if_unchanged(resolved_output, content)
    return manifest
