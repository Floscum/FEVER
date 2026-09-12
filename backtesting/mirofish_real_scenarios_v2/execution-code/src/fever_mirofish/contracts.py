"""Dependency-free semantic validation for simulation contracts.

The JSON Schema files define the wire format. These checks enforce cross-field
rules that JSON Schema alone does not express cleanly, especially leakage and
reference integrity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set


class ContractError(ValueError):
    """Raised when a contract violates a semantic invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _parse_datetime(value: str, field: str) -> datetime:
    _require(isinstance(value, str) and value, "%s must be a non-empty string" % field)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ContractError("%s is not ISO-8601: %s" % (field, value)) from error
    _require(parsed.tzinfo is not None, "%s must include a timezone: %s" % (field, value))
    return parsed


def _unique_ids(items: Iterable[Dict[str, Any]], label: str) -> Set[str]:
    values: List[str] = []
    for item in items:
        _require(isinstance(item, dict), "%s items must be objects" % label)
        item_id = item.get("id")
        _require(isinstance(item_id, str) and item_id, "%s item is missing id" % label)
        values.append(item_id)
    _require(len(values) == len(set(values)), "%s ids must be unique" % label)
    return set(values)


def _references_exist(refs: Iterable[str], valid: Set[str], field: str) -> None:
    missing = sorted(set(refs) - valid)
    _require(not missing, "%s contains unknown references: %s" % (field, ", ".join(missing)))


def canonical_sha256(value: Dict[str, Any]) -> str:
    """Return a stable SHA-256 for a parsed JSON contract."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_execution_slice(manifest: Dict[str, Any]) -> None:
    """Validate a frozen recovery or expansion slice without outcome access."""

    _require(isinstance(manifest, dict), "execution slice must be an object")
    _require(
        manifest.get("schema_version") == "0.1.0",
        "unsupported execution slice schema_version",
    )
    for field in ("benchmark_id", "parent_benchmark_id"):
        _require(
            isinstance(manifest.get(field), str) and manifest[field],
            f"execution slice {field} must be a non-empty string",
        )
    _require(
        manifest.get("status") == "input_frozen",
        "execution slice status must be input_frozen",
    )
    _parse_datetime(manifest.get("frozen_at"), "frozen_at")

    case_ids = manifest.get("case_ids")
    _require(
        isinstance(case_ids, list)
        and case_ids
        and all(isinstance(case_id, str) and case_id for case_id in case_ids),
        "execution slice case_ids must be a non-empty string list",
    )
    _require(
        len(case_ids) == len(set(case_ids)),
        "execution slice case_ids must be unique",
    )

    artifacts = manifest.get("source_artifacts")
    _require(
        isinstance(artifacts, list) and artifacts,
        "execution slice source_artifacts must be non-empty",
    )
    for artifact in artifacts:
        _require(
            isinstance(artifact, dict),
            "execution slice source_artifacts items must be objects",
        )
        kind = artifact.get("kind")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        _require(
            isinstance(kind, str) and kind,
            "execution slice source artifact kind must be non-empty",
        )
        _require(
            isinstance(path, str) and path,
            "execution slice source artifact path must be non-empty",
        )
        _require(
            "outcome" not in kind.lower()
            and not path.lower().endswith("/outcome.json"),
            "execution slice may not reference outcome artifacts",
        )
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "execution slice source artifact sha256 must be lowercase hex",
        )


def validate_spec(spec: Dict[str, Any]) -> None:
    """Validate semantic invariants for a SimulationSpec."""

    _require(isinstance(spec, dict), "spec must be an object")
    _require(spec.get("schema_version") == "0.1.0", "unsupported spec schema_version")
    as_of = _parse_datetime(spec.get("as_of"), "as_of")
    end_at = _parse_datetime(spec.get("horizon", {}).get("end_at"), "horizon.end_at")
    _require(end_at > as_of, "horizon.end_at must be later than as_of")

    facts = spec.get("facts")
    actors = spec.get("actors")
    relationships = spec.get("relationships")
    interventions = spec.get("interventions")
    targets = spec.get("forecast_targets")
    _require(isinstance(facts, list) and facts, "facts must be a non-empty list")
    _require(isinstance(actors, list) and len(actors) >= 2, "actors must contain at least two actors")
    _require(isinstance(relationships, list), "relationships must be a list")
    _require(isinstance(interventions, list) and interventions, "interventions must be non-empty")
    _require(isinstance(targets, list) and targets, "forecast_targets must be non-empty")

    fact_ids = _unique_ids(facts, "facts")
    actor_ids = _unique_ids(actors, "actors")
    _unique_ids(targets, "forecast_targets")

    for fact in facts:
        observed_at = _parse_datetime(fact.get("observed_at"), "facts[%s].observed_at" % fact["id"])
        _require(
            observed_at <= as_of,
            "future leakage: fact %s is later than as_of" % fact["id"],
        )
        source_url = fact.get("source_url")
        _require(
            isinstance(source_url, str)
            and (
                source_url.startswith("https://")
                or source_url.startswith("fever://artifact/")
            ),
            "fact %s must use an https or fever artifact source_url" % fact["id"],
        )

    for actor in actors:
        if "identity" in actor:
            identity = actor["identity"]
            _require(isinstance(identity, dict) and isinstance(identity.get("name"), str) and bool(identity["name"]), "actor identity requires a name")
            refs = identity.get("evidence_refs")
            _require(isinstance(refs, list) and bool(refs), "actor identity requires admitted evidence")
            _references_exist(refs, fact_ids, "actor identity evidence_refs")
            _require(all(identity["name"] in fact["statement"] for fact in facts if fact["id"] in refs), "actor identity name must occur in every cited fact")
        _references_exist(
            actor.get("observable_fact_ids", []),
            fact_ids,
            "actor %s observable_fact_ids" % actor["id"],
        )

    for relation in relationships:
        _references_exist(
            [relation.get("source_actor_id"), relation.get("target_actor_id")],
            actor_ids,
            "relationship actor ids",
        )
        _require(
            relation.get("source_actor_id") != relation.get("target_actor_id"),
            "relationships may not be self-referential",
        )

    for intervention in interventions:
        _references_exist(
            intervention.get("fact_ids", []),
            fact_ids,
            "intervention %s fact_ids" % intervention.get("id", "?"),
        )

    run_config = spec.get("run_config", {})
    replications = run_config.get("replications")
    seeds = run_config.get("seeds")
    if run_config.get("seed_strategy") == "explicit":
        _require(isinstance(seeds, list), "explicit seed_strategy requires seeds")
        _require(
            len(seeds) == replications,
            "explicit seeds count must equal replications",
        )

    forbidden = {"outcome", "actual", "realized_value", "observed_result"}
    leaked_fields = forbidden.intersection(spec.keys())
    _require(
        not leaked_fields,
        "outcome fields must not be present in a simulation spec: %s"
        % ", ".join(sorted(leaked_fields)),
    )


def validate_result(result: Dict[str, Any], spec: Optional[Dict[str, Any]] = None) -> None:
    """Validate semantic invariants for a SimulationResult."""

    _require(isinstance(result, dict), "result must be an object")
    _require(result.get("schema_version") == "0.1.0", "unsupported result schema_version")
    _parse_datetime(result.get("generated_at"), "generated_at")

    runs = result.get("runs")
    scenarios = result.get("scenarios")
    target_results = result.get("forecast_target_results")
    _require(isinstance(runs, list), "runs must be a list")
    _require(isinstance(scenarios, list), "scenarios must be a list")
    _require(isinstance(target_results, list), "forecast_target_results must be a list")

    run_ids = _unique_ids(
        [{"id": run.get("run_id")} for run in runs],
        "runs",
    )
    _require(len(run_ids) == len(runs), "run ids must be unique")

    if spec is None:
        return

    validate_spec(spec)
    _require(result.get("case_id") == spec.get("case_id"), "result case_id does not match spec")
    _require(
        result.get("spec_sha256") == canonical_sha256(spec),
        "result spec_sha256 does not match canonical spec hash",
    )

    fact_ids = {fact["id"] for fact in spec["facts"]}
    actor_ids = {actor["id"] for actor in spec["actors"]}
    target_ids = {target["id"] for target in spec["forecast_targets"]}

    for run in runs:
        for event in run.get("events", []):
            _references_exist([event.get("actor_id")], actor_ids, "run event actor_id")
            _references_exist(event.get("evidence_refs", []), fact_ids, "run event evidence_refs")

    for scenario in scenarios:
        _require(
            scenario.get("probability_semantics") == "uncalibrated_simulation_frequency",
            "scenario frequency must be marked uncalibrated",
        )
        _references_exist(scenario.get("evidence_refs", []), fact_ids, "scenario evidence_refs")
        _references_exist(
            scenario.get("actor_ids", []),
            actor_ids,
            "scenario actor_ids",
        )
        graph_node_ids = {
            node.get("id")
            for node in result.get("simulation_graph", {}).get("nodes", [])
        }
        _references_exist(
            scenario.get("simulation_refs", []),
            graph_node_ids,
            "scenario simulation_refs",
        )

    for target_result in target_results:
        _references_exist([target_result.get("target_id")], target_ids, "target result id")
        _references_exist(
            target_result.get("evidence_refs", []),
            fact_ids,
            "target result evidence_refs",
        )


def validate_financial_actions(
    artifact: Dict[str, Any],
    spec: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate structured decisions elicited after a multi-agent run."""

    _require(isinstance(artifact, dict), "financial actions must be an object")
    _require(
        artifact.get("schema_version") == "0.1.0",
        "unsupported financial actions schema_version",
    )
    _parse_datetime(artifact.get("generated_at"), "generated_at")
    decisions = artifact.get("decisions")
    failures = artifact.get("failures")
    _require(isinstance(decisions, list), "decisions must be a list")
    _require(isinstance(failures, list), "failures must be a list")
    _unique_ids(decisions, "decisions")

    allowed_action_types = {
        "COMMUNICATE",
        "OPERATE",
        "REGULATE",
        "ALLOCATE",
        "NEGOTIATE",
        "WAIT",
    }
    for decision in decisions:
        _require(
            decision.get("action_type") in allowed_action_types,
            "financial decision uses an unsupported action_type",
        )
        _require(
            decision.get("probability_semantics")
            == "decision_clarity_not_forecast_probability",
            "financial decision confidence must not be labeled as forecast probability",
        )
        for field in ("intensity", "confidence"):
            value = decision.get(field)
            if value is not None:
                _require(
                    isinstance(value, (int, float)) and 0 <= value <= 1,
                    f"financial decision {field} must be between 0 and 1",
                )

    if spec is None:
        return

    validate_spec(spec)
    _require(
        artifact.get("case_id") == spec.get("case_id"),
        "financial actions case_id does not match spec",
    )
    _require(
        artifact.get("spec_sha256") == canonical_sha256(spec),
        "financial actions spec_sha256 does not match canonical spec hash",
    )
    actor_ids = {actor["id"] for actor in spec["actors"]}
    fact_ids = {fact["id"] for fact in spec["facts"]}
    instrument_ids = {
        instrument["symbol"] for instrument in spec["market"]["instruments"]
    }
    constraint_refs = {
        f"{actor['id']}:constraint:{index}"
        for actor in spec["actors"]
        for index, _ in enumerate(actor["constraints"])
    }
    for decision in decisions:
        _references_exist(
            [decision.get("actor_id")],
            actor_ids,
            "financial decision actor_id",
        )
        _references_exist(
            decision.get("evidence_refs", []),
            fact_ids,
            "financial decision evidence_refs",
        )
        _references_exist(
            decision.get("instrument_refs", []),
            instrument_ids,
            "financial decision instrument_refs",
        )
        _references_exist(
            decision.get("constraint_refs", []),
            constraint_refs,
            "financial decision constraint_refs",
        )
    for failure in failures:
        _references_exist(
            [failure.get("actor_id")],
            actor_ids,
            "financial action failure actor_id",
        )


def validate_scenario_branches(
    artifact: Dict[str, Any],
    spec: Optional[Dict[str, Any]] = None,
    simulation_result: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate auditable, falsifiable scenario branches."""

    _require(isinstance(artifact, dict), "scenario branches must be an object")
    _require(
        artifact.get("schema_version") == "0.1.0",
        "unsupported scenario branches schema_version",
    )
    _require(
        artifact.get("prompt_version")
        in {
            "scenario-branch-compiler-v1",
            "scenario-branch-compiler-v2",
            "scenario-branch-compiler-v3",
            "scenario-branch-compiler-v4",
            "scenario-branch-compiler-v5",
            "scenario-branch-compiler-v6",
            "scenario-branch-compiler-v7",
            "scenario-branch-compiler-v8",
            "scenario-branch-compiler-v9",
        },
        "unsupported scenario branch prompt_version",
    )
    _parse_datetime(artifact.get("generated_at"), "generated_at")
    branches = artifact.get("branches")
    _require(isinstance(branches, list), "scenario branches must be a list")
    _require(len(branches) <= 4, "scenario branches may contain at most four items")
    _unique_ids(branches, "scenario branches")
    for branch in branches:
        actor_refs = branch.get("actor_ids")
        actions = branch.get("actions")
        evidence_refs = branch.get("evidence_refs")
        simulation_refs = branch.get("simulation_refs")
        _require(
            isinstance(actor_refs, list)
            and len(actor_refs) >= 2
            and len(actor_refs) == len(set(actor_refs)),
            "scenario branch must contain at least two unique actors",
        )
        _require(
            isinstance(actions, list) and len(actions) >= 2,
            "scenario branch must contain at least two actions",
        )
        _require(
            all(
                isinstance(branch.get(field), list) and branch[field]
                for field in (
                    "trigger_conditions",
                    "consequences",
                    "invalidation_conditions",
                )
            ),
            "scenario branch conditions and consequences must be non-empty",
        )
        _require(
            isinstance(evidence_refs, list) and evidence_refs,
            "scenario branch evidence_refs must be non-empty",
        )
        _require(
            isinstance(simulation_refs, list)
            and len(simulation_refs) >= 2
            and len(simulation_refs) == len(set(simulation_refs)),
            "scenario branch must contain at least two unique simulation refs",
        )
        _require(
            branch.get("confidence_semantics")
            == "branch_coherence_not_forecast_probability",
            "scenario branch confidence must not be labeled as probability",
        )
        confidence = branch.get("confidence")
        _require(
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0 <= confidence <= 1,
            "scenario branch confidence must be between 0 and 1",
        )
        action_actor_ids = [item.get("actor_id") for item in actions]
        _require(
            set(action_actor_ids) <= set(actor_refs),
            "scenario action actor_id must appear in actor_ids",
        )
        _require(
            len(set(action_actor_ids)) >= 2,
            "scenario branch actions must involve at least two actors",
        )
        _require(
            all(
                isinstance(item.get("action_type"), str)
                and item["action_type"]
                and isinstance(item.get("decision_ref"), str)
                and item["decision_ref"]
                for item in actions
            ),
            "scenario actions require action_type and decision_ref",
        )
        if artifact.get("prompt_version") == "scenario-branch-compiler-v8":
            from .product_scenarios import validate_product_fields
            validate_product_fields(branch)

    if spec is None:
        return

    validate_spec(spec)
    _require(
        artifact.get("case_id") == spec.get("case_id"),
        "scenario branches case_id does not match spec",
    )
    _require(
        artifact.get("spec_sha256") == canonical_sha256(spec),
        "scenario branches spec_sha256 does not match canonical spec hash",
    )
    actor_ids = {actor["id"] for actor in spec["actors"]}
    fact_ids = {fact["id"] for fact in spec["facts"]}
    for branch in branches:
        _references_exist(
            branch["actor_ids"],
            actor_ids,
            "scenario branch actor_ids",
        )
        _references_exist(
            branch["evidence_refs"],
            fact_ids,
            "scenario branch evidence_refs",
        )

    if simulation_result is None:
        return
    validate_result(simulation_result, spec)
    simulation_id = (
        simulation_result["runs"][0]["run_id"]
        if simulation_result.get("runs")
        else None
    )
    _require(
        artifact.get("simulation_id") == simulation_id,
        "scenario branches simulation_id does not match result",
    )
    node_ids = {
        node.get("id")
        for node in simulation_result["simulation_graph"]["nodes"]
    }
    nodes_by_id = {
        node.get("id"): node
        for node in simulation_result["simulation_graph"]["nodes"]
    }
    source_actors_by_node: Dict[Any, set[Any]] = {}
    for edge in simulation_result["simulation_graph"]["edges"]:
        source_actors_by_node.setdefault(edge.get("target"), set()).add(
            edge.get("source")
        )
    for branch in branches:
        _references_exist(
            branch["simulation_refs"],
            node_ids,
            "scenario branch simulation_refs",
        )
        for action in branch["actions"]:
            node = nodes_by_id.get(action["decision_ref"]) or {}
            _require(
                node.get("action_type") == action["action_type"],
                "scenario action_type must match its referenced simulation node",
            )
            _require(
                action["actor_id"]
                in source_actors_by_node.get(action["decision_ref"], set()),
                "scenario action actor_id must match its referenced simulation node",
            )


def validate_outcome(outcome: Dict[str, Any], spec: Optional[Dict[str, Any]] = None) -> None:
    """Validate an outcome kept outside the pre-event simulation input."""

    _require(isinstance(outcome, dict), "outcome must be an object")
    _require(outcome.get("schema_version") == "0.1.0", "unsupported outcome schema_version")
    observed_through = _parse_datetime(outcome.get("observed_through"), "observed_through")
    target_results = outcome.get("target_results")
    sources = outcome.get("data_sources")
    _require(isinstance(target_results, list) and target_results, "target_results must be non-empty")
    _require(isinstance(sources, list) and sources, "data_sources must be non-empty")

    source_ids = _unique_ids(sources, "data_sources")
    target_result_ids = _unique_ids(
        [{"id": item.get("target_id")} for item in target_results],
        "outcome target_results",
    )

    for item in target_results:
        observed_at = _parse_datetime(
            item.get("observed_at"),
            "target_results[%s].observed_at" % item.get("target_id", "?"),
        )
        _require(
            observed_at <= observed_through,
            "target result %s is later than observed_through" % item.get("target_id", "?"),
        )
        _references_exist(
            item.get("source_refs", []),
            source_ids,
            "target result %s source_refs" % item.get("target_id", "?"),
        )

    for source in sources:
        _parse_datetime(
            source.get("retrieved_at"),
            "data_sources[%s].retrieved_at" % source.get("id", "?"),
        )

    if spec is None:
        return

    validate_spec(spec)
    _require(outcome.get("case_id") == spec.get("case_id"), "outcome case_id does not match spec")
    _require(
        outcome.get("spec_sha256") == canonical_sha256(spec),
        "outcome spec_sha256 does not match canonical spec hash",
    )

    as_of = _parse_datetime(spec.get("as_of"), "as_of")
    _require(observed_through > as_of, "observed_through must be later than spec as_of")
    expected_target_ids = {item["id"] for item in spec["forecast_targets"]}
    _references_exist(target_result_ids, expected_target_ids, "outcome target ids")
    if outcome.get("status") == "complete":
        _require(
            target_result_ids == expected_target_ids,
            "complete outcome must include every forecast target",
        )
    for item in target_results:
        observed_at = _parse_datetime(item["observed_at"], "target result observed_at")
        _require(
            observed_at > as_of,
            "outcome target %s must occur after spec as_of" % item["target_id"],
        )
