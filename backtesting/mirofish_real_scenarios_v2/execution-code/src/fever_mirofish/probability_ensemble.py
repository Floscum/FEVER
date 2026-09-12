"""Conservative aggregation of qualitative signals across simulation seeds."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

from .benchmark import validate_forecast_submission
from .contracts import (
    _parse_datetime,
    _require,
    canonical_sha256,
    validate_spec,
)
from .probability_updates import (
    PROMPT_VERSION as SIGNAL_PROMPT_VERSION,
    UPDATE_DIRECTIONS,
    UPDATE_STRENGTHS,
)


AGGREGATION_VERSION = "qualitative-seed-ensemble-v1"
REQUIRED_REPLICATION_COUNT = 3
_DIRECTION_ORDER = ("increase", "decrease", "unchanged", "ambiguous")
_STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}


def _event_targets(spec: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        target
        for target in spec["forecast_targets"]
        if target["kind"] == "event"
        and target["scoring"] in {"brier", "log_loss"}
    ]


def _validate_source_signal(
    signal: Dict[str, Any],
    spec: Dict[str, Any],
    baseline_submission: Dict[str, Any],
) -> None:
    """Validate seed-local signals without needing one merged SimulationResult."""

    _require(
        signal.get("schema_version") == "0.1.0",
        "unsupported source probability signal schema_version",
    )
    _require(
        signal.get("case_id") == spec["case_id"],
        "source probability signal case_id does not match spec",
    )
    _require(
        signal.get("spec_sha256") == canonical_sha256(spec),
        "source probability signal spec hash does not match spec",
    )
    _require(
        signal.get("baseline_submission_sha256")
        == canonical_sha256(baseline_submission),
        "source probability signal baseline hash does not match submission",
    )
    _require(
        isinstance(signal.get("simulation_id"), str)
        and bool(signal["simulation_id"].strip()),
        "source probability signal simulation_id is required",
    )
    _parse_datetime(signal.get("generated_at"), "generated_at")
    _require(
        isinstance(signal.get("model_id"), str) and signal["model_id"],
        "source probability signal model_id is required",
    )
    _require(
        signal.get("prompt_version") == SIGNAL_PROMPT_VERSION,
        "unsupported source probability signal prompt_version",
    )
    _require(
        signal.get("status") == "uncalibrated"
        and signal.get("calibration_version") is None,
        "source probability signal must remain uncalibrated",
    )
    updates = signal.get("updates")
    expected_targets = {target["id"] for target in _event_targets(spec)}
    _require(
        isinstance(updates, list)
        and len(updates) == len(expected_targets)
        and {item.get("target_id") for item in updates} == expected_targets,
        "source probability signal must cover event targets exactly once",
    )
    baseline_by_target = {
        item["target_id"]: item["value"]
        for item in baseline_submission["target_predictions"]
    }
    for update in updates:
        _require(
            update.get("baseline_probability")
            == baseline_by_target[update["target_id"]],
            "source probability signal baseline value does not match",
        )
        _require(
            update.get("update_direction") in UPDATE_DIRECTIONS,
            "source probability signal direction is invalid",
        )
        _require(
            update.get("update_strength") in UPDATE_STRENGTHS,
            "source probability signal strength is invalid",
        )
        support = update.get("support_simulation_refs")
        counter = update.get("counter_simulation_refs")
        _require(
            isinstance(support, list)
            and len(support) == len(set(support)),
            "source support refs must be a unique list",
        )
        _require(
            isinstance(counter, list)
            and len(counter) == len(set(counter)),
            "source counter refs must be a unique list",
        )
        _require(
            not set(support) & set(counter),
            "source support and counter refs must be disjoint",
        )
        _require(
            isinstance(update.get("rationale"), str)
            and bool(update["rationale"].strip()),
            "source probability signal rationale is required",
        )
        _require(
            update.get("adjusted_probability") is None
            and update.get("probability_semantics")
            == "withheld_until_calibrated",
            "source probability signal must withhold probability",
        )


def _aggregate_target(
    target_id: str,
    baseline_probability: float,
    replications: list[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = []
    for replication in replications:
        update = next(
            item
            for item in replication["updates"]
            if item["target_id"] == target_id
        )
        rows.append(
            {
                "replication_id": replication["replication_id"],
                "update_direction": update["update_direction"],
                "update_strength": update["update_strength"],
            }
        )
    counts = Counter(item["update_direction"] for item in rows)
    aggregate_direction = "ambiguous"
    for direction in ("increase", "decrease", "unchanged"):
        if counts[direction] >= 2:
            aggregate_direction = direction
            break

    supporters = (
        [
            item
            for item in rows
            if item["update_direction"] == aggregate_direction
        ]
        if aggregate_direction != "ambiguous"
        else []
    )
    if aggregate_direction in {"increase", "decrease"}:
        aggregate_strength = min(
            (item["update_strength"] for item in supporters),
            key=_STRENGTH_ORDER.__getitem__,
        )
    else:
        aggregate_strength = "weak"

    dominant_count = max(counts.values())
    direction_counts = {
        direction: counts[direction] for direction in _DIRECTION_ORDER
    }
    return {
        "target_id": target_id,
        "baseline_probability": baseline_probability,
        "aggregate_direction": aggregate_direction,
        "aggregate_strength": aggregate_strength,
        "direction_counts": direction_counts,
        "consensus_reached": aggregate_direction != "ambiguous",
        "dominant_direction_fraction": round(
            dominant_count / REQUIRED_REPLICATION_COUNT,
            8,
        ),
        "supporting_replication_ids": [
            item["replication_id"] for item in supporters
        ],
        "dissenting_replication_ids": [
            item["replication_id"]
            for item in rows
            if item["update_direction"] != aggregate_direction
        ],
        "rule_trace": (
            "two_of_three_direction_consensus; "
            "weakest_supporting_strength; "
            "ambiguous_or_unchanged_preserves_baseline"
        ),
        "adjusted_probability": None,
        "probability_semantics": "withheld_until_calibrated",
    }


def build_probability_signal_ensemble(
    spec: Dict[str, Any],
    baseline_submission: Dict[str, Any],
    source_signals: Iterable[Dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Aggregate exactly three seed-local qualitative signal artifacts."""

    validate_spec(spec)
    validate_forecast_submission(baseline_submission, spec)
    _require(
        baseline_submission.get("arm") == "B1",
        "probability signal ensemble baseline must be B1",
    )
    signals = list(source_signals)
    _require(
        len(signals) == REQUIRED_REPLICATION_COUNT,
        "probability signal ensemble requires exactly three replications",
    )
    for signal in signals:
        _validate_source_signal(signal, spec, baseline_submission)
    simulation_ids = [signal["simulation_id"] for signal in signals]
    _require(
        len(simulation_ids) == len(set(simulation_ids)),
        "probability signal ensemble simulation ids must be unique",
    )
    signals.sort(key=lambda item: item["simulation_id"])

    replications = []
    for signal in signals:
        replications.append(
            {
                "replication_id": signal["simulation_id"],
                "model_id": signal["model_id"],
                "signal_sha256": canonical_sha256(signal),
                "updates": [
                    {
                        "target_id": update["target_id"],
                        "update_direction": update["update_direction"],
                        "update_strength": update["update_strength"],
                        "support_simulation_refs": list(
                            update["support_simulation_refs"]
                        ),
                        "counter_simulation_refs": list(
                            update["counter_simulation_refs"]
                        ),
                        "rationale": update["rationale"],
                    }
                    for update in sorted(
                        signal["updates"],
                        key=lambda item: item["target_id"],
                    )
                ],
            }
        )

    baseline_by_target = {
        item["target_id"]: item["value"]
        for item in baseline_submission["target_predictions"]
    }
    target_updates = [
        _aggregate_target(
            target["id"],
            baseline_by_target[target["id"]],
            replications,
        )
        for target in sorted(_event_targets(spec), key=lambda item: item["id"])
    ]
    artifact = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "spec_sha256": canonical_sha256(spec),
        "baseline_submission_sha256": canonical_sha256(
            baseline_submission
        ),
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "aggregation_version": AGGREGATION_VERSION,
        "source_signal_prompt_version": SIGNAL_PROMPT_VERSION,
        "required_replication_count": REQUIRED_REPLICATION_COUNT,
        "observed_replication_count": len(replications),
        "status": "uncalibrated",
        "calibration_version": None,
        "replication_signals": replications,
        "target_updates": target_updates,
        "warnings": [
            "Seed agreement is a stability diagnostic, not an event "
            "probability or an independent sample count.",
            "Adjusted probabilities remain withheld until one frozen "
            "calibration policy is applied to the aggregate signal.",
            "Simulation references remain namespaced inside each "
            "replication and are never flattened across graphs.",
        ],
    }
    validate_probability_signal_ensemble(
        artifact,
        spec,
        baseline_submission,
    )
    return artifact


def validate_probability_signal_ensemble(
    artifact: Dict[str, Any],
    spec: Dict[str, Any],
    baseline_submission: Dict[str, Any],
) -> None:
    """Validate hashes, seed separation, and deterministic consensus output."""

    validate_spec(spec)
    validate_forecast_submission(baseline_submission, spec)
    _require(
        artifact.get("schema_version") == "0.1.0",
        "unsupported probability signal ensemble schema_version",
    )
    _require(
        artifact.get("case_id") == spec["case_id"],
        "probability signal ensemble case_id does not match spec",
    )
    _require(
        artifact.get("spec_sha256") == canonical_sha256(spec),
        "probability signal ensemble spec hash does not match",
    )
    _require(
        artifact.get("baseline_submission_sha256")
        == canonical_sha256(baseline_submission),
        "probability signal ensemble baseline hash does not match",
    )
    _parse_datetime(artifact.get("generated_at"), "generated_at")
    _require(
        artifact.get("aggregation_version") == AGGREGATION_VERSION,
        "unsupported probability signal aggregation_version",
    )
    _require(
        artifact.get("source_signal_prompt_version")
        == SIGNAL_PROMPT_VERSION,
        "unsupported ensemble source signal prompt version",
    )
    _require(
        artifact.get("required_replication_count")
        == REQUIRED_REPLICATION_COUNT
        and artifact.get("observed_replication_count")
        == REQUIRED_REPLICATION_COUNT,
        "probability signal ensemble must contain exactly three seeds",
    )
    _require(
        artifact.get("status") == "uncalibrated"
        and artifact.get("calibration_version") is None,
        "probability signal ensemble must remain uncalibrated",
    )
    replications = artifact.get("replication_signals")
    _require(
        isinstance(replications, list)
        and len(replications) == REQUIRED_REPLICATION_COUNT,
        "probability signal ensemble replications are invalid",
    )
    replication_ids = [item.get("replication_id") for item in replications]
    _require(
        all(isinstance(item, str) and item for item in replication_ids)
        and len(replication_ids) == len(set(replication_ids))
        and replication_ids == sorted(replication_ids),
        "probability signal ensemble replication ids must be unique and sorted",
    )
    event_target_ids = {
        target["id"] for target in _event_targets(spec)
    }
    for replication in replications:
        _require(
            isinstance(replication.get("model_id"), str)
            and bool(replication["model_id"]),
            "ensemble replication model_id is required",
        )
        _require(
            isinstance(replication.get("signal_sha256"), str)
            and len(replication["signal_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in replication["signal_sha256"]
            ),
            "ensemble replication signal_sha256 is invalid",
        )
        updates = replication.get("updates")
        _require(
            isinstance(updates, list)
            and len(updates) == len(event_target_ids)
            and {item.get("target_id") for item in updates}
            == event_target_ids
            and [item.get("target_id") for item in updates]
            == sorted(event_target_ids),
            "ensemble replication must cover event targets exactly once",
        )
        for update in updates:
            _require(
                update.get("update_direction") in UPDATE_DIRECTIONS,
                "ensemble replication direction is invalid",
            )
            _require(
                update.get("update_strength") in UPDATE_STRENGTHS,
                "ensemble replication strength is invalid",
            )
            support = update.get("support_simulation_refs")
            counter = update.get("counter_simulation_refs")
            _require(
                isinstance(support, list)
                and len(support) == len(set(support))
                and isinstance(counter, list)
                and len(counter) == len(set(counter))
                and not set(support) & set(counter),
                "ensemble replication refs must be unique and disjoint",
            )

    target_updates = artifact.get("target_updates")
    _require(
        isinstance(target_updates, list)
        and len(target_updates) == len(event_target_ids)
        and {item.get("target_id") for item in target_updates}
        == event_target_ids
        and [item.get("target_id") for item in target_updates]
        == sorted(event_target_ids),
        "ensemble target updates must cover event targets exactly once",
    )
    baseline_by_target = {
        item["target_id"]: item["value"]
        for item in baseline_submission["target_predictions"]
    }
    expected_by_target = {
        target_id: _aggregate_target(
            target_id,
            baseline_by_target[target_id],
            replications,
        )
        for target_id in event_target_ids
    }
    for update in target_updates:
        _require(
            update == expected_by_target[update["target_id"]],
            "ensemble target update does not match frozen aggregation rule",
        )
    warnings = artifact.get("warnings")
    _require(
        isinstance(warnings, list)
        and all(isinstance(item, str) for item in warnings),
        "probability signal ensemble warnings must be strings",
    )
