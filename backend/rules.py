import hashlib
import json


RULE_REGISTRY = {
    "R-A": {
        "rule_id": "R-A",
        "name": "HTTP 403 Forbidden status",
        "version": "1.0.0",
        "description": "Detects events whose status contains HTTP 403.",
        "severity": "MEDIUM",
        "detection_conditions": {
            "status_contains": "403",
        },
    },
    "R-B": {
        "rule_id": "R-B",
        "name": "POST to sensitive resource",
        "version": "1.0.0",
        "description": "Detects POST requests to resources containing configured sensitive keywords.",
        "severity": "HIGH",
        "detection_conditions": {
            "event_type_contains": "POST",
            "resource_keywords": ["secret", "admin", "password", "confidential", ".zip"],
            "match_mode": "any_keyword_substring",
        },
    },
    "R-C": {
        "rule_id": "R-C",
        "name": "Repeated POST requests from one source",
        "version": "1.0.0",
        "description": "Detects three or more POST requests from the same source.",
        "severity": "MEDIUM",
        "detection_conditions": {
            "event_type_contains": "POST",
            "minimum_request_count": 3,
            "group_by": "source",
        },
    },
    "R-D": {
        "rule_id": "R-D",
        "name": "Unusually large data transfer",
        "version": "1.0.0",
        "description": "Detects numeric byte transfers above mean plus two standard deviations.",
        "severity": "MEDIUM",
        "detection_conditions": {
            "minimum_numeric_samples": 3,
            "standard_deviation_type": "sample",
            "threshold_multiplier": 2,
            "comparison": "greater_than_mean_plus_two_sample_standard_deviations",
        },
    },
}


def get_rule(rule_id: str) -> dict:
    import copy

    return copy.deepcopy(RULE_REGISTRY[rule_id])


def get_rule_metadata() -> dict:
    rules = [get_rule(rule_id) for rule_id in sorted(RULE_REGISTRY)]
    return {
        "fingerprint": calculate_rule_fingerprint(),
        "rules": rules,
    }


def calculate_rule_fingerprint(rules: dict | None = None) -> str:
    registry = rules if rules is not None else RULE_REGISTRY
    canonical = json.dumps(registry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
