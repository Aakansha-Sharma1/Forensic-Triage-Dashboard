# Canonical Events and Feature Engineering

This package defines feature schema version `1.0.0`. It prepares explainable numerical inputs and retains the event metadata required to trace each result to a case, evidence item, source row, timestamp, and original evidence reference. It does not train models, assign labels, or consume detection findings.

## Input and canonicalization

`extract_features` accepts event mappings using the validated CSV field names (`timestamp`, `source_ip` or `source`, `event_type` or `event`, `resource` or `description`, `status`, and `bytes`). The same aliases as the existing CSV validator are recognized. Event IDs are the zero-based source-row index, so chronological sorting does not lose source-row traceability. Timestamps are converted to UTC; naive ISO timestamps are interpreted as UTC. Events are ordered by timestamp and then source-row order.

Each `CanonicalEvent` retains `case_id`, `evidence_id`, `event_id`, UTC `timestamp`, `source_ip`, `event_type`, `resource`, `response_status`, `transfer_size`, and `original_evidence_reference`. The identity and source fields remain event metadata rather than model inputs.

Blank source, event type, resource, response status, and transfer size values are represented explicitly as missing. Missing transfer sizes and source-dependent features remain `None`, with numerical missingness indicators provided in the feature vector. Invalid or missing timestamps, non-mapping rows, and non-negative-integer transfer-size violations raise `EventDataError` rather than silently producing features. A non-numeric response status is treated as unavailable for failure classification and indicated by `response_status_missing`.

## Feature calculations

The versioned definitions and meanings are published as `FEATURE_SCHEMA` and `get_feature_schema()`.

- `transfer_size_bytes`: integer bytes parsed from `bytes`.
- `request_frequency_5m`, `failed_request_frequency_5m`, and `post_request_frequency_5m`: counts for the same source in the inclusive five-minute trailing window. The current event is included. Failure means numeric status 400–599; POST means the event type contains `POST`, case-insensitively.
- `time_since_previous_event_seconds`: elapsed UTC seconds since the prior event from the same source; null for its first event.
- `unique_resources_accessed`: cumulative count of distinct, non-empty resource strings for the same source through the current event, after stripping and case-folding.
- `sensitive_resource_indicator`: 1 when the resource contains `secret`, `admin`, `password`, `confidential`, or `.zip`, case-insensitively; otherwise 0.
- Missingness indicators identify absent source, event type, resource, numeric status, transfer size, and prior same-source event.

Temporal state is updated in chronological order only as each event is reached. No later event contributes to an earlier event's rolling counts, resource count, or time delta. Equal timestamps are processed in source-row order. For an absent source IP, source-dependent count and timing features are null.

## Separation and traceability

Each `FeatureRecord` keeps its `CanonicalEvent` separately from `features`. The `model_inputs` property returns only numerical features; case ID, evidence ID, event ID, timestamp, and original evidence reference are not model inputs. The package operates on supplied row data and does not modify or copy evidence files.

Reproducibility requires retaining this package version, `FEATURE_SCHEMA_VERSION`, the original evidence reference and hash recorded by ingestion, and the extraction inputs. Feature schema changes that alter field meaning or calculation require a version change.

## AIT Log Data Set V2.1 import

The official source is [Zenodo record 19483937](https://zenodo.org/records/19483937), version `v2_1`, licensed `CC-BY-NC-SA-4.0`. The record describes eight synthetic testbed scenarios and line-referenced JSONL ground-truth files that mirror the `gather/` tree under `labels/`. Each label entry contains an original event `line`, its `labels`, and labeling `rules`. These values are retained as supplied. A source line without an explicit label remains unlabeled (`ground_truth: null`); this importer does not infer a normal label.

### Download and preparation procedure

1. Review the current [Zenodo record and license](https://zenodo.org/records/19483937). Select one scenario and only the corresponding `*_no-pcaps.zip` archive; do not download the full collection. No dataset archives are included in this repository or downloaded by the preparation code. Zenodo lists its `*_no-pcaps.zip` files at approximately 498 MiB to 1.23 GiB, so obtain approval before downloading one.
2. Manually place the selected archive and extracted `gather/` and `labels/` files under `research_datasets/ait-lds-v2.1/raw/`. Keep these untouched source files separate from `research_datasets/ait-lds-v2.1/processed/`. Both locations are Git-ignored.
3. For each chosen Apache access log, explicitly provide its scenario ID, case ID, and evidence ID to `parse_apache_access_log`. Supply the path relative to the scenario's `gather/` directory as `source_filename`; use the identical relative path under `labels/` when calling `import_ground_truth_labels`. Do not infer missing source fields or identifiers. Call `prepare_apache_source` to attach labels only by the original one-based log line number and calculate feature schema `1.0.0` independently for that source log. Temporal feature histories reset at each source file; rotated logs are not silently merged.
4. Review the returned rejected rows and counts. Call `write_processed_dataset` with the prepared source and processed output directory. It writes deterministic JSONL with canonical events, numerical features, provenance, and only explicit labels, plus `manifest.json`; output cannot be placed in or over the raw-data directory, and different existing processed output is not overwritten.
5. Retain the manifest with the prepared records. It records the record URL, dataset version, license, methodology, optional source archive filename/hash, source and label file SHA-256 values, source line/record/rejection counts, attached label counts, feature schema version, and processed JSONL hash. Zenodo also publishes archive checksums (MD5); the importer independently records SHA-256 of each locally supplied source file.

`parse_apache_access_log` accepts common/combined Apache access-log entries and vhost-combined entries with a leading virtual-host token. It maps the remote-host token to canonical `source_ip` without validating or inventing an IP address, preserves the optional virtual-host token separately, uses `-` as missing only when the source explicitly contains it, converts the timestamp to UTC, and preserves filename, one-based physical line number, scenario ID, source hash, and original path. Unsupported/malformed rows are returned as `RejectedRow` entries; invalid file encoding or file access raises an explicit import error.

The small files under `tests/fixtures/synthetic_ait_style/` are generated synthetic parser fixtures, not AIT data or real ground truth. They are used only by automated tests.

## Russell Mitchell research labels

`backend.ml.labels.prepare_labelled_dataset` creates `labelled_events.jsonl` and `labels_manifest.json` alongside (but never in place of) the prepared dataset. It verifies the prepared-record SHA-256 against its import manifest and verifies each event's source filename and hash. The labelled output contains only events in the Russell Mitchell simulation window; source events before `2022-01-21T00:00:00Z` or at/after `2022-01-25T00:00:00Z` are counted as excluded. The official time table does not state a timezone; boundaries are interpreted as UTC to match the selected Apache log timestamps. This uses the half-open interval `[start, end)` so an event at the end boundary belongs to the following period.

The official [AIT Log Data Set V2.1 record](https://zenodo.org/records/19483937) says that a line absent from its corresponding ground-truth file is normal. This absence rule is applied only when the import manifest verifies a mirrored label file, source and label hashes, complete parsed-line and label-line coverage, and zero import/alignment rejections. An explicit non-empty `ground_truth.labels` list yields `research_label: 1`; absence from a verified file yields `research_label: 0`; insufficient coverage or an explicit empty category list yields JSON `null` (unknown). Events with no corresponding label file are never presumed normal.

The original multi-label `ground_truth` payload and `ground_truth_provenance` are copied unchanged. The derived `research_label`, its display name, decision basis, and policy provenance are top-level fields, never numerical `features`. The label manifest records the input and output hashes, policy and time boundaries, class and category counts, per-source distributions, duplicate event identifiers, and source-coverage bias indicators. Duplicate identifiers are reported, not silently deduplicated.

Only the intranet-server Apache log has a corresponding ground-truth file among the selectively imported Russell Mitchell Apache sources. Consequently, its explicitly labeled attack records and verified unlabeled normal records form the known-label subset; events from other sources remain unknown. This source concentration is recorded as a dataset-bias warning and must be considered when designing later evaluation. This step prepares labels only; it does not train models, balance classes, or use detection-rule findings.
