"""Deterministic assessment of a financialized B3 smoke run."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable

from .contracts import validate_financial_actions, validate_result, validate_spec


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _count_records(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(item[key]) for item in records).items()))


def assess_b3_run(
    spec: Dict[str, Any],
    financial_actions: Dict[str, Any],
    sqlite_actions: Dict[str, Any],
    simulation_result: Dict[str, Any],
    *,
    usage: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Summarize engineering gates and research limitations without LLM calls."""

    validate_spec(spec)
    validate_financial_actions(financial_actions, spec)
    validate_result(simulation_result, spec)

    expected_actor_ids = {actor["id"] for actor in spec["actors"]}
    decisions = financial_actions["decisions"]
    decision_actor_ids = {item["actor_id"] for item in decisions}
    action_counts = _count_records(decisions, "action_type")
    status_counts = _count_records(decisions, "decision_status")
    wait_count = action_counts.get("WAIT", 0)

    fact_ids = {fact["id"] for fact in spec["facts"]}
    grounded = sum(bool(item["evidence_refs"]) for item in decisions)
    constrained = sum(bool(item["constraint_refs"]) for item in decisions)
    all_fact_citations = sum(
        set(item["evidence_refs"]) == fact_ids for item in decisions
    )
    instrument_decisions = sum(bool(item["instrument_refs"]) for item in decisions)

    autonomous = [
        item
        for item in sqlite_actions.get("actions", [])
        if item.get("origin") == "autonomous"
    ]
    active_actor_ids = {
        item["actor_id"] for item in autonomous if item.get("actor_id")
    }
    autonomous_text = sum(bool(item.get("has_text_semantics")) for item in autonomous)
    social_signals = len(autonomous) - autonomous_text

    forecast_count = len(simulation_result.get("forecast_target_results", []))
    scenario_count = len(simulation_result.get("scenarios", []))
    decision_coverage = _ratio(len(decision_actor_ids), len(expected_actor_ids))
    wait_share = _ratio(wait_count, len(decisions))
    broad_citation_share = _ratio(all_fact_citations, len(decisions))

    risks = ["single_run_has_no_predictive_validity"]
    if wait_share >= 0.5:
        risks.append("wait_actions_are_concentrated")
    if broad_citation_share >= 0.5:
        risks.append("most_decisions_cite_every_input_fact")
    if not sqlite_actions.get("round_metadata", {}).get(
        "per_action_round_available",
        False,
    ):
        risks.append("social_actions_lack_persisted_round_numbers")
    if not forecast_count and not scenario_count:
        risks.append("no_scenarios_or_forecast_targets_were_produced")
    risks.append("completed_run_used_a_prompt_without_explicit_simulated_time")

    engineering_pass = (
        decision_actor_ids == expected_actor_ids
        and not financial_actions["failures"]
        and not sqlite_actions.get("unresolved_entity_names")
        and len(action_counts) >= 2
    )
    forecast_ready = bool(forecast_count and scenario_count)
    verdict = (
        "forecast_evaluation_ready"
        if engineering_pass and forecast_ready
        else (
            "engineering_pass_research_unproven"
            if engineering_pass
            else "engineering_incomplete"
        )
    )

    summary = {
        "schema_version": "0.1.0",
        "case_id": spec["case_id"],
        "simulation_id": financial_actions["simulation_id"],
        "verdict": verdict,
        "engineering_gates": {
            "expected_actor_count": len(expected_actor_ids),
            "decision_actor_count": len(decision_actor_ids),
            "decision_actor_coverage": decision_coverage,
            "invalid_decision_count": len(financial_actions["failures"]),
            "unresolved_social_actor_count": len(
                sqlite_actions.get("unresolved_entity_names", [])
            ),
            "distinct_financial_action_types": len(action_counts),
            "passed": engineering_pass,
        },
        "financial_decisions": {
            "count": len(decisions),
            "action_type_counts": action_counts,
            "decision_status_counts": status_counts,
            "wait_share": wait_share,
            "evidence_grounded_count": grounded,
            "constraint_grounded_count": constrained,
            "all_input_facts_cited_count": all_fact_citations,
            "all_input_facts_cited_share": broad_citation_share,
            "instrument_referenced_count": instrument_decisions,
        },
        "social_simulation": {
            "autonomous_action_count": len(autonomous),
            "autonomous_text_action_count": autonomous_text,
            "autonomous_social_signal_count": social_signals,
            "active_actor_count": len(active_actor_ids),
            "active_actor_coverage": _ratio(
                len(active_actor_ids),
                len(expected_actor_ids),
            ),
            "active_actor_ids": sorted(active_actor_ids),
        },
        "forecast_readiness": {
            "scenario_count": scenario_count,
            "forecast_target_result_count": forecast_count,
            "passed": forecast_ready,
        },
        "research_risks": risks,
    }
    if usage is not None:
        summary["measured_oasis_usage"] = usage
    return summary
