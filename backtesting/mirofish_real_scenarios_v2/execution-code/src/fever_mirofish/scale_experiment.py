"""Outcome-blind helpers for the FEVER multi-actor scale experiment."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable

from .fever_adapter import compile_evidence_graph


SCHEMA_VERSION = "fever-mirofish-scale-experiment-v1"
ACTOR_BUDGETS = (4, 6, 8)
PREDICTION_CLASSES = ("up", "down", "neutral")
EXPECTED_EVENT_PROFILE = {
    "并购/分拆/再融资": "ma_capital",
    "财报超预期/不及预期": "earnings_guidance",
    "公司指引上调/下调": "earnings_guidance",
    "政策利率调整": "rate_policy",
    "增长/就业数据意外": "macro_data",
    "通胀数据意外": "macro_data",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def event_to_request(
    event: dict[str, Any],
    *,
    experiment_case_id: str,
    actor_budget: int | None,
) -> dict[str, Any]:
    """Compile one public backtest event into the gateway request contract."""

    market = str(event["market"]).upper()
    symbol = str(event["symbol"])
    benchmark = str(event.get("benchmark") or ("sh000300" if market == "CN" else "SPY"))
    title = str(event["title"])
    event_text = str(event.get("event_text") or title)
    event_time = str(event["event_time"])[:10]
    question = (
        f"严格基于该事件时点可得信息，推演未来3个交易日各参与方的行动，"
        f"以及这些行动影响 {symbol} 相对 {benchmark} 表现的条件化机制：{title}"
    )
    source_url = str(event.get("source_url") or "").strip()
    evidence_graph = {
        "question": question,
        "nodes": [
            {
                "id": "E1",
                "kind": "evidence",
                "title": title,
                "body": event_text,
                "source_kind": "official",
                "source_ref": source_url,
                "created_at": f"{event_time}T23:59:59+00:00",
                "confidence": 0.8,
            },
            {
                "id": "C1",
                "kind": "claim",
                "title": (
                    f"{symbol} 在事件后 T+3 相对 {benchmark} 的异常收益方向"
                    "是上涨、下跌还是中性？"
                ),
                "status": "exploring",
            },
        ],
        "edges": [{"source": "E1", "target": "C1", "relation": "informs"}],
    }
    currency = "CNY" if market == "CN" else "USD"
    return {
        "case_id": experiment_case_id,
        "source_graph_artifact_id": f"{SCHEMA_VERSION}:{experiment_case_id}",
        "evidence_graph": evidence_graph,
        "question": question,
        "as_of": f"{event_time}T23:59:59+00:00",
        "horizon_days": 3,
        "mode": "quick",
        "max_actors": actor_budget,
        "market": {
            "region": market,
            "venues": [market],
            "instruments": [
                {
                    "symbol": symbol,
                    "name": symbol,
                    "kind": "equity",
                    "currency": currency,
                }
            ],
        },
    }


def preview_request(request: dict[str, Any]) -> dict[str, Any]:
    spec = compile_evidence_graph(
        request["evidence_graph"],
        case_id=request["case_id"],
        source_graph_artifact_id=request["source_graph_artifact_id"],
        question=request["question"],
        as_of=request["as_of"],
        horizon_days=int(request["horizon_days"]),
        max_actors=request.get("max_actors"),
        market=request["market"],
    )
    return {
        "actor_count": len(spec["actors"]),
        "actor_ids": [item["id"] for item in spec["actors"]],
        "actor_selection": spec["provenance"]["actor_selection"],
        "spec_sha256": canonical_hash(spec),
    }


def build_manifest(
    events: Iterable[dict[str, Any]],
    *,
    seed: str = "fever-mirofish-scale-v1",
    per_market_type: int = 2,
) -> dict[str, Any]:
    """Select a deterministic market×type sample without reading labels."""

    if per_market_type < 1:
        raise ValueError("per_market_type must be positive")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        cells[(str(event["market"]), str(event["event_type_l2"]))].append(event)
    if not cells:
        raise ValueError("event pool is empty")

    previews: dict[str, dict[str, Any]] = {}
    for event in (item for group in cells.values() for item in group):
        event_id = str(event["event_id"])
        auto_request = event_to_request(
            event,
            experiment_case_id=f"preview_{event_id}",
            actor_budget=None,
        )
        previews[event_id] = preview_request(auto_request)

    # Assign constrained cells first within each market.  This keeps 4/6/8
    # balanced in both CN and US instead of confounding actor count with market.
    # A simple event-structure filter also avoids known source-pool category
    # mistakes such as an M&A filing filed under the guidance label.
    used_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    assignments = []
    for market in sorted({key[0] for key in cells}):
        market_cells = [key for key in sorted(cells) if key[0] == market]
        market_slot_count = len(market_cells) * per_market_type
        remaining_budget_counts = Counter(
            ACTOR_BUDGETS[index % len(ACTOR_BUDGETS)]
            for index in range(market_slot_count)
        )
        slots = []
        for _, event_type in market_cells:
            expected_profile = EXPECTED_EVENT_PROFILE.get(event_type)
            eligible = [
                event
                for event in cells[(market, event_type)]
                if expected_profile is None
                or expected_profile
                in {
                    item["id"]
                    for item in previews[str(event["event_id"])][
                        "actor_selection"
                    ]["matched_event_profiles"]
                }
            ]
            if len(eligible) < per_market_type:
                raise ValueError(
                    f"cell {market}/{event_type} has fewer than {per_market_type} "
                    "semantically consistent events"
                )
            maximum = max(
                previews[str(event["event_id"])]["actor_count"]
                for event in eligible
            )
            for cell_slot in range(per_market_type):
                slots.append((maximum, event_type, cell_slot, eligible))
        slots.sort(key=lambda item: (item[0], item[1], item[2]))

        for maximum, event_type, cell_slot, eligible in slots:
            used_ids = used_by_cell[(market, event_type)]
            feasible_budgets = [
                budget
                for budget in reversed(ACTOR_BUDGETS)
                if remaining_budget_counts[budget] > 0
                and any(
                    str(event["event_id"]) not in used_ids
                    and previews[str(event["event_id"])]["actor_count"] >= budget
                    for event in eligible
                )
            ]
            if not feasible_budgets:
                raise ValueError(
                    f"cannot balance actor budgets for cell {market}/{event_type}; "
                    f"maximum supported count is {maximum}"
                )
            actor_budget = feasible_budgets[0]
            candidates = []
            for event in eligible:
                event_id = str(event["event_id"])
                if (
                    event_id in used_ids
                    or previews[event_id]["actor_count"] < actor_budget
                ):
                    continue
                rank = hashlib.sha256(
                    f"{seed}|{market}|{event_type}|{cell_slot}|{event_id}".encode()
                ).hexdigest()
                candidates.append((rank, event, previews[event_id]))
            _, event, auto_preview = min(candidates, key=lambda item: item[0])
            used_ids.add(str(event["event_id"]))
            remaining_budget_counts[actor_budget] -= 1
            assignments.append(
                (market, event_type, cell_slot, actor_budget, event, auto_preview)
            )

        if any(remaining_budget_counts.values()):
            raise AssertionError(
                f"balanced actor budget assignment is incomplete for {market}"
            )

    selected_rows = []
    budget_counts: Counter[int] = Counter()
    for market, event_type, _, actor_budget, event, auto_preview in sorted(assignments):
        event_id = str(event["event_id"])
        experiment_case_id = f"scale_{len(selected_rows) + 1:03d}_{event_id}"
        request = event_to_request(
            event,
            experiment_case_id=experiment_case_id,
            actor_budget=actor_budget,
        )
        frozen_preview = preview_request(request)
        if frozen_preview["actor_count"] != actor_budget:
            raise AssertionError("frozen actor budget did not compile exactly")
        budget_counts[actor_budget] += 1
        selected_rows.append(
            {
                "experiment_case_id": experiment_case_id,
                "event_id": event_id,
                "market": market,
                "event_type_l2": event_type,
                "actor_budget": actor_budget,
                "auto_recommended_count": auto_preview["actor_count"],
                "configured_actor_ids": frozen_preview["actor_ids"],
                "spec_sha256": frozen_preview["spec_sha256"],
                "event": {
                    key: event.get(key)
                    for key in (
                        "event_id", "market", "symbol", "event_time",
                        "event_type_l2", "title", "event_text", "source_url",
                        "sector_etf", "benchmark",
                    )
                },
                "request": request,
            }
        )

    ablation_groups = []
    ablation_types = (
        "并购/分拆/再融资",
        "增长/就业数据意外",
    )
    for market in sorted({row["market"] for row in selected_rows}):
        for event_type in ablation_types:
            base = next(
                row
                for row in selected_rows
                if row["market"] == market
                and row["event_type_l2"] == event_type
                and row["auto_recommended_count"] >= 8
            )
            variants = []
            for actor_budget in ACTOR_BUDGETS:
                if actor_budget == base["actor_budget"]:
                    experiment_case_id = base["experiment_case_id"]
                    request = base["request"]
                    reused_primary_case = True
                else:
                    experiment_case_id = (
                        f"ablate_{market.lower()}_{len(ablation_groups) + 1:02d}"
                        f"_a{actor_budget}_{base['event_id']}"
                    )
                    request = event_to_request(
                        base["event"],
                        experiment_case_id=experiment_case_id,
                        actor_budget=actor_budget,
                    )
                    reused_primary_case = False
                variant_preview = preview_request(request)
                if variant_preview["actor_count"] != actor_budget:
                    raise AssertionError("actor ablation did not compile exactly")
                variants.append(
                    {
                        "experiment_case_id": experiment_case_id,
                        "event_id": base["event_id"],
                        "market": market,
                        "event_type_l2": event_type,
                        "actor_budget": actor_budget,
                        "configured_actor_ids": variant_preview["actor_ids"],
                        "spec_sha256": variant_preview["spec_sha256"],
                        "event": base["event"],
                        "request": request,
                        "reused_primary_case": reused_primary_case,
                    }
                )
            ablation_groups.append(
                {
                    "group_id": f"ablation_{market.lower()}_{len(ablation_groups) + 1:02d}",
                    "market": market,
                    "event_type_l2": event_type,
                    "event_id": base["event_id"],
                    "variants": variants,
                }
            )

    unique_simulation_ids = {
        row["experiment_case_id"] for row in selected_rows
    } | {
        variant["experiment_case_id"]
        for group in ablation_groups
        for variant in group["variants"]
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "seed": seed,
            "method": "sha256_rank_within_market_event_type_without_labels",
            "per_market_event_type": per_market_type,
            "actor_budget_schedule": list(ACTOR_BUDGETS),
            "actor_budget_assignment": (
                "balanced_within_market_constrained_first_using_compiler_capacity"
            ),
            "semantic_consistency_filter": "event_type_to_event_structure_profile_v1",
            "labels_read_during_selection": False,
        },
        "evaluation": {
            "primary_horizon": "t3",
            "secondary_horizon": "avg_all",
            "primary_prediction_metric": "paired_multiclass_brier_delta",
            "engineering_metrics": [
                "completion_rate", "wall_seconds", "active_actor_ratio",
                "valid_decision_ratio", "autonomous_actions_per_actor",
                "scenario_count",
            ],
        },
        "summary": {
            "case_count": len(selected_rows),
            "actor_ablation_group_count": len(ablation_groups),
            "actor_ablation_observation_count": sum(
                len(group["variants"]) for group in ablation_groups
            ),
            "unique_simulation_count": len(unique_simulation_ids),
            "market_count": len({row["market"] for row in selected_rows}),
            "event_type_count": len(
                {row["event_type_l2"] for row in selected_rows}
            ),
            "actor_budget_counts": {
                str(key): budget_counts[key] for key in ACTOR_BUDGETS
            },
        },
        "cases": selected_rows,
        "actor_ablation": {
            "design": "same_event_paired_4_vs_6_vs_8_actors",
            "prediction_scored": False,
            "groups": ablation_groups,
        },
    }


def normalize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    direction = str(payload.get("pred_direction") or "").lower().strip()
    if direction not in PREDICTION_CLASSES:
        raise ValueError("pred_direction must be up, down, or neutral")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as error:
        raise ValueError("prediction confidence must be numeric") from error
    if not math.isfinite(confidence):
        raise ValueError("prediction confidence must be finite")
    confidence = max(1 / 3, min(0.99, confidence))
    remainder = (1.0 - confidence) / 2.0
    probabilities = {item: remainder for item in PREDICTION_CLASSES}
    probabilities[direction] = confidence
    return {
        "pred_direction": direction,
        "confidence": round(confidence, 6),
        "probabilities": probabilities,
        "rationale": str(payload.get("rationale") or "").strip()[:1200],
    }


def _prediction_metrics(rows: list[dict[str, Any]], arm: str, horizon: str) -> dict[str, Any]:
    label_key = f"label_{horizon}"
    valid = [row for row in rows if row["label"].get(label_key) in PREDICTION_CLASSES]
    if not valid:
        return {"n": 0}
    exact = sum(
        row[arm]["pred_direction"] == row["label"][label_key]
        for row in valid
    )
    non_neutral = [
        row for row in valid if row["label"][label_key] in {"up", "down"}
    ]
    non_neutral_exact = sum(
        row[arm]["pred_direction"] == row["label"][label_key]
        for row in non_neutral
    )
    brier_values = []
    for row in valid:
        oracle = row["label"][label_key]
        probabilities = row[arm]["probabilities"]
        brier_values.append(
            sum(
                (float(probabilities[item]) - (1.0 if item == oracle else 0.0)) ** 2
                for item in PREDICTION_CLASSES
            )
        )
    return {
        "n": len(valid),
        "accuracy_3class": exact / len(valid),
        "non_neutral_n": len(non_neutral),
        "non_neutral_accuracy": (
            non_neutral_exact / len(non_neutral) if non_neutral else None
        ),
        "multiclass_brier": sum(brier_values) / len(brier_values),
        "neutral_prediction_rate": sum(
            row[arm]["pred_direction"] == "neutral" for row in valid
        ) / len(valid),
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_effects(
    rows: list[dict[str, Any]],
    *,
    horizon: str,
    bootstrap_seed: int = 20260901,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    label_key = f"label_{horizon}"
    valid = [row for row in rows if row["label"].get(label_key) in PREDICTION_CLASSES]
    if not valid:
        return {"n": 0}

    def effects(sample: list[dict[str, Any]]) -> tuple[float, float]:
        base_correct = []
        assisted_correct = []
        base_brier = []
        assisted_brier = []
        for row in sample:
            oracle = row["label"][label_key]
            base_correct.append(row["baseline"]["pred_direction"] == oracle)
            assisted_correct.append(row["assisted"]["pred_direction"] == oracle)
            for arm, target in (
                ("baseline", base_brier), ("assisted", assisted_brier)
            ):
                target.append(
                    sum(
                        (
                            float(row[arm]["probabilities"][item])
                            - (1.0 if item == oracle else 0.0)
                        ) ** 2
                        for item in PREDICTION_CLASSES
                    )
                )
        accuracy_delta = (
            sum(assisted_correct) - sum(base_correct)
        ) / len(sample)
        brier_delta = (
            sum(assisted_brier) - sum(base_brier)
        ) / len(sample)
        return accuracy_delta, brier_delta

    accuracy_delta, brier_delta = effects(valid)
    beneficial = harmful = changed = 0
    for row in valid:
        oracle = row["label"][label_key]
        before = row["baseline"]["pred_direction"] == oracle
        after = row["assisted"]["pred_direction"] == oracle
        if row["baseline"]["pred_direction"] != row["assisted"]["pred_direction"]:
            changed += 1
        if not before and after:
            beneficial += 1
        elif before and not after:
            harmful += 1

    rng = random.Random(bootstrap_seed)
    accuracy_samples = []
    brier_samples = []
    for _ in range(bootstrap_samples):
        sample = [valid[rng.randrange(len(valid))] for _ in valid]
        sampled_accuracy, sampled_brier = effects(sample)
        accuracy_samples.append(sampled_accuracy)
        brier_samples.append(sampled_brier)
    return {
        "n": len(valid),
        "accuracy_delta_assisted_minus_baseline": accuracy_delta,
        "accuracy_delta_bootstrap_95": [
            _percentile(accuracy_samples, 0.025),
            _percentile(accuracy_samples, 0.975),
        ],
        "brier_delta_assisted_minus_baseline": brier_delta,
        "brier_delta_bootstrap_95": [
            _percentile(brier_samples, 0.025),
            _percentile(brier_samples, 0.975),
        ],
        "prediction_changed_count": changed,
        "beneficial_flip_count": beneficial,
        "harmful_flip_count": harmful,
    }


def score_prediction_pairs(
    predictions: Iterable[dict[str, Any]],
    labels: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    label_by_id = {str(item["event_id"]): item for item in labels}
    rows = []
    for prediction in predictions:
        label = label_by_id.get(str(prediction["event_id"]))
        if label is None or not prediction.get("baseline") or not prediction.get("assisted"):
            continue
        rows.append({**prediction, "label": label})
    result: dict[str, Any] = {
        "paired_case_count": len(rows),
        "horizons": {},
    }
    for horizon in ("t3", "avg_all"):
        result["horizons"][horizon] = {
            "baseline": _prediction_metrics(rows, "baseline", horizon),
            "assisted": _prediction_metrics(rows, "assisted", horizon),
            "paired": _paired_effects(rows, horizon=horizon),
        }
    return result


def parse_duration_seconds(started_at: Any, finished_at: Any) -> float | None:
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (finish - start).total_seconds())


def summarize_gateway_job(case: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    execution = result.get("execution") or {}
    configured = int(execution.get("configured_actor_count") or 0)
    active_counts = execution.get("active_actor_counts") or []
    active = max((int(item or 0) for item in active_counts), default=0)
    decisions = int(execution.get("valid_decision_count") or 0)
    actions = int(execution.get("autonomous_action_count") or 0)
    scenarios = result.get("scenarios") or []
    scenario_actor_ids = {
        str(actor_id)
        for scenario in scenarios
        for actor_id in (scenario.get("actor_ids") or [])
        if actor_id
    }
    scenario_actor_pairs = {
        tuple(sorted(str(actor_id) for actor_id in (scenario.get("actor_ids") or [])))
        for scenario in scenarios
        if len(scenario.get("actor_ids") or []) >= 2
    }
    return {
        "experiment_case_id": case["experiment_case_id"],
        "event_id": case["event_id"],
        "market": case["market"],
        "event_type_l2": case["event_type_l2"],
        "actor_budget": case["actor_budget"],
        "gateway_job_id": job.get("job_id"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "error": job.get("error"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "wall_seconds": parse_duration_seconds(
            job.get("started_at"), job.get("finished_at")
        ),
        "configured_actor_count": configured,
        "active_actor_count": active,
        "active_actor_ratio": active / configured if configured else 0.0,
        "valid_decision_count": decisions,
        "valid_decision_ratio": decisions / configured if configured else 0.0,
        "decision_failure_count": int(execution.get("decision_failure_count") or 0),
        "autonomous_action_count": actions,
        "autonomous_actions_per_actor": actions / configured if configured else 0.0,
        "scenario_count": len(scenarios),
        "scenario_actor_count": len(scenario_actor_ids),
        "scenario_actor_coverage_ratio": (
            len(scenario_actor_ids) / configured if configured else 0.0
        ),
        "unique_scenario_actor_pair_count": len(scenario_actor_pairs),
        "graph_backend": execution.get("graph_backend"),
        "zep_bypassed": bool(execution.get("zep_bypassed")),
        "warning_count": len(result.get("warnings") or []),
    }


def summarize_engineering_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(runs)
    completed = [row for row in rows if row.get("status") in {"completed", "partial"}]

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [row for row in group if row in completed]
        durations = [
            float(row["wall_seconds"])
            for row in successful
            if row.get("wall_seconds") is not None
        ]
        return {
            "n": len(group),
            "completed_n": len(successful),
            "completion_rate": len(successful) / len(group) if group else 0.0,
            "median_wall_seconds": _percentile(durations, 0.5),
            "p90_wall_seconds": _percentile(durations, 0.9),
            "mean_active_actor_ratio": (
                sum(float(row.get("active_actor_ratio") or 0) for row in successful)
                / len(successful) if successful else 0.0
            ),
            "mean_valid_decision_ratio": (
                sum(float(row.get("valid_decision_ratio") or 0) for row in successful)
                / len(successful) if successful else 0.0
            ),
            "mean_actions_per_actor": (
                sum(float(row.get("autonomous_actions_per_actor") or 0) for row in successful)
                / len(successful) if successful else 0.0
            ),
            "mean_scenario_count": (
                sum(int(row.get("scenario_count") or 0) for row in successful)
                / len(successful) if successful else 0.0
            ),
            "mean_scenario_actor_coverage_ratio": (
                sum(
                    float(row.get("scenario_actor_coverage_ratio") or 0)
                    for row in successful
                ) / len(successful) if successful else 0.0
            ),
            "mean_unique_scenario_actor_pair_count": (
                sum(
                    int(row.get("unique_scenario_actor_pair_count") or 0)
                    for row in successful
                ) / len(successful) if successful else 0.0
            ),
        }

    by_budget: dict[str, Any] = {}
    for budget in ACTOR_BUDGETS:
        group = [row for row in rows if int(row.get("actor_budget") or 0) == budget]
        by_budget[str(budget)] = group_summary(group)
    return {"overall": group_summary(rows), "by_actor_budget": by_budget}
