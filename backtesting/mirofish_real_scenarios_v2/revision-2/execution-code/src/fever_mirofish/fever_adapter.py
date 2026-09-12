"""Compile a FEVER Evidence Graph artifact into a bounded SimulationSpec."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from .contracts import validate_spec
from .entity_roles import apply_entity_roles


ACTOR_TEMPLATES = {
    "issuer": ("actor_issuer", "事件主体", "organization"),
    "regulator": ("actor_regulator", "监管机构", "organization"),
    "government": ("actor_government", "政府与司法机构", "organization"),
    "exchange": ("actor_exchange", "交易所", "organization"),
    "institutional_investor": ("actor_institutions", "机构投资者", "cohort"),
    "retail_cohort": ("actor_retail", "个人投资者群体", "cohort"),
    "media": ("actor_media", "财经媒体", "cohort"),
    "competitor": ("actor_competitors", "竞争者", "cohort"),
    "supplier": ("actor_suppliers", "供应商", "cohort"),
    "customer": ("actor_customers", "客户", "cohort"),
    "labor_union": ("actor_labor", "员工与工会", "cohort"),
    "foreign_investor": ("actor_foreign", "境外投资者", "cohort"),
    "broker": ("actor_brokers", "券商与中介机构", "cohort"),
    "analyst": ("actor_analysts", "分析师", "cohort"),
}

ACTOR_KEYWORDS = {
    "regulator": ("监管", "证监", "反垄断", "审查", "执法", "监管机构"),
    "government": ("政府", "法院", "司法", "国会", "部门", "委员会"),
    "exchange": ("交易所", "停牌", "退市"),
    "institutional_investor": ("机构", "基金", "债权人", "银行", "投资者"),
    "retail_cohort": ("散户", "个人投资者", "用户舆情"),
    "media": ("媒体", "新闻", "舆论", "报道"),
    "competitor": ("竞争", "竞品", "同行"),
    "supplier": ("供应商", "供应链", "出租人"),
    "customer": ("客户", "消费者", "用户", "旅客"),
    "labor_union": ("员工", "工会", "劳工"),
    "foreign_investor": ("外资", "境外投资者", "北向资金", "国际资本"),
    "broker": ("券商", "承销商", "做市商", "中介机构"),
    "analyst": ("分析师", "评级", "研报"),
}

ACTOR_FALLBACK_REASONS = {
    "regulator": "基础制衡角色：评估合规、审批与执法响应",
    "institutional_investor": "基础市场角色：评估资本配置与风险敞口",
    "media": "基础信息角色：评估信息传播与预期变化",
}

# Role defaults are modelling assumptions, not claims about an actual institution.
ACTOR_BEHAVIOR = {
    "issuer": ("维持经营与融资能力，控制事件对客户和业务的影响", "行动受现金、执行能力和已披露承诺约束，不假定有无限资源"),
    "regulator": ("核实风险并评估调查、审批或补救措施", "只能在自身权限和程序内行动，不预设审批结果或时间表"),
    "government": ("权衡公共利益、制度目标和政策执行效果", "遵守司法或政策程序，不代替其他机构作出决定"),
    "exchange": ("维护信息披露和交易秩序", "依据已知规则采取措施，不替代公司经营或监管审批"),
    "institutional_investor": ("权衡风险敞口、流动性和资本配置", "受投资授权和流动性约束，不假定能够无限增持或立即退出"),
    "retail_cohort": ("根据公开信息评估个人风险与资金安排", "信息和风险承受能力存在差异，不假设所有个人投资者一致行动"),
    "media": ("核实和传播对事件判断有用的新信息", "区分事实、消息与推测，不凭空增加独家信息"),
    "competitor": ("评估竞争格局变化并调整业务响应", "受自身产能、成本和执行周期限制，不预设可以立即承接全部需求"),
    "supplier": ("保障回款与订单稳定，评估交付和信用风险", "受合同、产能和客户依赖约束，不假定可立即替换所有订单"),
    "customer": ("保障产品或服务连续性，权衡成本和替代方案", "考虑合同、转换成本和替代供给，不假定需求可以即时转移"),
    "labor_union": ("维护就业、安全和劳动条件，评估协商方案", "受劳动协议、组织能力和协商程序约束"),
    "foreign_investor": ("评估跨境风险敞口、汇率与资金安排", "考虑市场准入、资金流动和汇率约束，不假设资本可无成本流转"),
    "broker": ("评估融资、交易与中介服务的连续性", "受授信、资本和合同责任约束，不预设已有未披露融资承诺"),
    "analyst": ("比较事实与原有判断，明确需要更新的假设", "区分已知事实与估计，不将研判当成公司承诺"),
}

# Event structure is a stronger signal than literal stakeholder mentions.  For
# example, an earnings announcement rarely spells out "analysts, brokers and
# competitors" even though those are precisely the actors whose reactions are
# useful in a scenario exercise.  These profiles add simulation roles; they do
# not promote the roles to factual claims.
EVENT_ACTOR_PROFILES = (
    {
        "id": "ma_capital",
        "patterns": (
            "并购", "收购", "重组", "分拆", "再融资", "定增", "配股",
            "merger", "acquisition", "tender offer", "restructuring",
            "spin-off", "spinoff", "refinancing", "secondary offering",
            "rule 425", "securities act",
        ),
        "recommended_count": 8,
        "actor_kinds": (
            "regulator", "exchange", "institutional_investor", "broker",
            "competitor", "analyst", "media", "foreign_investor",
            "government",
        ),
    },
    {
        "id": "earnings_guidance",
        "patterns": (
            "财报", "业绩", "盈利", "亏损", "营收", "净利润", "业绩指引",
            "盈利指引", "经营指引", "指引上调", "指引下调",
            "earnings", "financial results", "revenue", "profit", "loss",
            "guidance", "outlook", "10-q", "10-k",
        ),
        "recommended_count": 6,
        "actor_kinds": (
            "institutional_investor", "analyst", "broker", "exchange",
            "competitor", "media", "customer", "supplier",
        ),
    },
    {
        "id": "rate_policy",
        "patterns": (
            "利率", "降息", "加息", "降准", "准备金率", "lpr", "mlf",
            "央行", "美联储", "联储", "rate decision", "interest rate",
            "rate cut", "rate hike", "federal reserve", "central bank",
            "treasury auction", "fed funds", "fomc",
        ),
        "recommended_count": 8,
        "actor_kinds": (
            "government", "regulator", "institutional_investor",
            "foreign_investor", "broker", "analyst", "media", "customer",
        ),
    },
    {
        "id": "macro_data",
        "patterns": (
            "cpi", "ppi", "pce", "通胀", "物价", "就业", "失业", "非农",
            "pmi", "gdp", "工业增加值", "零售销售", "inflation",
            "employment", "unemployment", "payroll", "consumer price",
            "producer price", "economic growth",
        ),
        "recommended_count": 8,
        "actor_kinds": (
            "government", "institutional_investor", "foreign_investor",
            "broker", "analyst", "media", "customer", "competitor",
        ),
    },
    {
        "id": "operational_disruption",
        "patterns": (
            "服务中断", "系统中断", "系统崩溃", "运营中断", "供应中断",
            "停产", "产品召回", "故障恢复", "service outage",
            "system outage", "operational disruption", "product recall",
        ),
        "recommended_count": 6,
        "actor_kinds": (
            "customer", "supplier", "competitor", "institutional_investor",
            "media", "regulator", "analyst", "exchange",
        ),
    },
    {
        "id": "legal_regulatory",
        "patterns": (
            "调查", "处罚", "诉讼", "法院", "反垄断", "合规", "监管问询",
            "investigation", "lawsuit", "court", "antitrust", "regulatory",
            "enforcement", "compliance",
        ),
        "recommended_count": 8,
        "actor_kinds": (
            "regulator", "government", "exchange", "institutional_investor",
            "broker", "media", "analyst", "customer",
        ),
    },
)

SOURCE_KIND_MAP = {
    "official": "official",
    "filing": "filing",
    "market_data": "market_data",
    "news": "news",
    "research": "research",
    "akshare": "market_data",
    "external": "news",
}

FEVER_FACT_MAX_CHARS = 1200
FEVER_FACT_TOTAL_CHARS = 12000


def _infer_market(text: str) -> dict[str, Any]:
    """Infer explicit A-share symbols instead of using an unusable placeholder."""

    symbols = list(
        dict.fromkeys(
            re.findall(r"(?<!\d)([03468]\d{5})(?!\d)", text)
        )
    )[:8]
    if not symbols:
        return {
            "region": "FEVER_RESEARCH",
            "venues": ["RESEARCH"],
            "instruments": [
                {
                    "symbol": "EVENT_CASE",
                    "name": "事件研究标的",
                    "kind": "equity",
                    "currency": "CNY",
                }
            ],
        }
    return {
        "region": "CN",
        "venues": ["SSE" if symbol.startswith("6") else "SZSE" for symbol in symbols],
        "instruments": [
            {
                "symbol": symbol,
                "name": f"A股 {symbol}",
                "kind": "equity",
                "currency": "CNY",
            }
            for symbol in symbols
        ],
    }


def _parse_datetime(value: Any, fallback: datetime | None = None) -> datetime | None:
    if isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None and fallback is not None:
                parsed = parsed.replace(tzinfo=fallback.tzinfo)
            return parsed
        except ValueError:
            pass
    return fallback


def _source_observed_at(node: dict[str, Any], as_of: datetime) -> tuple[datetime, str]:
    """Preserve source time, including future times, for admission checks.

    Publication time takes precedence over ingestion time. Undated evidence
    remains usable for an interactive graph, but its assumption is exposed.
    """
    source_data = node.get("source_data")
    source_data = source_data if isinstance(source_data, dict) else {}
    candidates = [
        (container.get(key), "source_published_at")
        for container in (source_data, node)
        for key in ("published_at", "publish_time", "datetime", "date", "时间", "日期")
    ] + [
        (source_data.get("observed_at"), "source_observed_at"),
        (node.get("observed_at"), "source_observed_at"),
        (node.get("created_at"), "evidence_created_at"),
    ]
    for candidate, basis in candidates:
        parsed = _parse_datetime(candidate)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=as_of.tzinfo)
            return parsed, basis
    return as_of, "unknown_assumed_as_of"


def _fact_source_url(
    node: dict[str, Any], artifact_id: str, node_id: str
) -> str:
    source_ref = str(node.get("source_ref") or "").strip()
    if source_ref.startswith("https://"):
        return source_ref
    return f"fever://artifact/{artifact_id}/evidence/{node_id}"


def _evidence_statement(node: dict[str, Any]) -> str:
    """Keep explicit source text without serializing arbitrary tool metadata."""
    parts = []
    raw = node.get("source_data")
    raw = raw if isinstance(raw, dict) else {}
    # source_data is a free-form dictionary in FEVER. Read only named text
    # fields; forecasts, reasoning, credentials and nested tool state stay out.
    candidates = [node.get("title"), node.get("body")] + [
        raw.get(key) for key in ("content", "text", "body", "正文", "summary", "摘要")
    ]
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if any(value in part for part in parts):
            continue
        # A source paragraph containing the entire summary replaces that
        # summary, leaving room for the additional clauses.
        parts = [part for part in parts if part not in value]
        parts.append(value)
    return "\n".join(parts)


def _recommended_actor_limit(detected_count: int) -> int:
    """Map grounded stakeholder diversity to a bounded even-number budget."""

    required = max(4, min(10, detected_count + 1))  # +1 for the issuer
    for limit in (4, 6, 8, 10):
        if required <= limit:
            return limit
    return 10


def _event_pattern_matches(pattern: str, text: str) -> bool:
    # Profit margins are not interest-rate policy. A real rate mention still
    # matches when both a margin and an interest rate appear in the same input.
    if pattern == "利率":
        return re.search(r"(?<![毛净])利率", text) is not None
    return pattern.lower() in text


def _select_actor_kinds(
    text: str, max_actors: int | None
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    normalized_text = text.lower()
    selected = ["issuer"]
    reasons = {"issuer": "事件研究主体：始终纳入推演"}
    scores: dict[str, int] = {}
    direct_matches: dict[str, tuple[str, ...]] = {}
    for priority, (kind, keywords) in enumerate(ACTOR_KEYWORDS.items()):
        matched = tuple(
            keyword for keyword in keywords if keyword.lower() in normalized_text
        )
        count = sum(normalized_text.count(keyword.lower()) for keyword in matched)
        if count:
            direct_matches[kind] = matched
            scores[kind] = scores.get(kind, 0) + 100 + count
            reasons[kind] = f"证据或研究问题命中：{'、'.join(matched[:4])}"

    matched_profiles = []
    profile_recommendation = 4
    for profile in EVENT_ACTOR_PROFILES:
        matched = tuple(
            pattern
            for pattern in profile["patterns"]
            if _event_pattern_matches(pattern, normalized_text)
        )
        if not matched:
            continue
        matched_profiles.append(
            {
                "id": profile["id"],
                "matched_patterns": list(matched[:4]),
                "recommended_count": profile["recommended_count"],
            }
        )
        profile_recommendation = max(
            profile_recommendation, int(profile["recommended_count"])
        )
        profile_actors = profile["actor_kinds"]
        for profile_priority, kind in enumerate(profile_actors):
            scores[kind] = scores.get(kind, 0) + (
                20 * (len(profile_actors) - profile_priority) + len(matched)
            )
            if kind not in direct_matches:
                reasons[kind] = (
                    f"事件结构 {profile['id']} 命中：{'、'.join(matched[:3])}"
                )

    recommended = max(
        _recommended_actor_limit(len(direct_matches)), profile_recommendation
    )
    applied_limit = recommended if max_actors is None else max_actors
    actor_priority = {
        kind: index for index, kind in enumerate(ACTOR_KEYWORDS)
    }
    for kind in sorted(
        scores,
        key=lambda item: (-scores[item], actor_priority.get(item, 999)),
    ):
        if kind not in selected:
            selected.append(kind)

    for fallback in ("regulator", "institutional_investor", "media"):
        if fallback not in selected:
            selected.append(fallback)
            reasons[fallback] = ACTOR_FALLBACK_REASONS[fallback]

    # ``max_actors`` remains a cap rather than a requested population size.
    # Event profiles supply enough relevant candidates for controlled 4/6/8
    # ablations without filling unrelated roles into a genuinely simple event.
    target_count = min(applied_limit, recommended)
    selected = selected[:target_count]
    metadata = {
        "mode": "auto" if max_actors is None else "manual_cap",
        "recommended_count": recommended,
        "applied_limit": applied_limit,
        "detected_stakeholder_type_count": len(direct_matches),
        "configured_count": len(selected),
        "matched_event_profiles": matched_profiles,
        "selection_strategy": "literal_mentions_plus_event_structure_v3",
        "rationale": (
            f"从证据与研究问题中识别出 {len(direct_matches)} 类显式利益相关方，"
            f"并命中 {len(matched_profiles)} 类事件结构，按 4/6/8/10 分档"
            f"建议最多 {recommended} 个核心参与方"
        ),
    }
    return selected, metadata, reasons


def _build_relationships(actor_ids: list[str]) -> list[dict[str, str]]:
    issuer = "actor_issuer"
    relationships = []
    for actor_id in actor_ids:
        if actor_id == issuer:
            continue
        relationships.append(
            {
                "source_actor_id": actor_id,
                "target_actor_id": issuer,
                "kind": "constrains_or_influences",
                "visibility": "assumed",
            }
        )
    if len(actor_ids) >= 3:
        relationships.append(
            {
                "source_actor_id": issuer,
                "target_actor_id": actor_ids[-1],
                "kind": "communicates_actions_to",
                "visibility": "assumed",
            }
        )
    return relationships


def compile_evidence_graph(
    graph: dict[str, Any],
    *,
    case_id: str,
    source_graph_artifact_id: str,
    question: str | None = None,
    as_of: str,
    horizon_days: int = 30,
    max_actors: int | None = None,
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated, deterministic, scenario-first SimulationSpec."""

    if not isinstance(graph, dict):
        raise ValueError("evidence graph payload must be an object")
    if max_actors is not None and not 4 <= max_actors <= 10:
        raise ValueError("max_actors must be between 4 and 10")
    if not 1 <= horizon_days <= 365:
        raise ValueError("horizon_days must be between 1 and 365")
    as_of_dt = _parse_datetime(as_of)
    if as_of_dt is None or as_of_dt.tzinfo is None:
        raise ValueError("as_of must include a timezone")

    evidence_nodes = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
        and node.get("kind") == "evidence"
        and str(node.get("source_kind") or "").lower() != "inference"
    ]
    if not evidence_nodes:
        raise ValueError("evidence graph has no source-backed evidence nodes")
    facts = []
    timing = {"excluded_future_ids": [], "unknown_time_ids": []}
    fact_text_chars = 0
    for index, node in enumerate(evidence_nodes, start=1):
        node_id = str(node.get("id") or f"E{index}")
        observed_at, time_basis = _source_observed_at(node, as_of_dt)
        if observed_at > as_of_dt:
            timing["excluded_future_ids"].append(node_id)
            continue
        statement = _evidence_statement(node)
        if not statement:
            continue
        remaining = FEVER_FACT_TOTAL_CHARS - fact_text_chars
        if remaining <= 0 or len(facts) >= 16:
            break
        statement = statement[: min(FEVER_FACT_MAX_CHARS, remaining)]
        source_kind = SOURCE_KIND_MAP.get(
            str(node.get("source_kind") or "").lower(), "research"
        )
        facts.append(
            {
                "id": f"F{len(facts) + 1}",
                "observed_at": observed_at.isoformat(),
                "time_basis": time_basis,
                "statement": statement,
                "source_kind": source_kind,
                "source_url": _fact_source_url(
                    node, source_graph_artifact_id, node_id
                ),
                "evidence_ref": node_id,
                "confidence": max(
                    0.0, min(1.0, float(node.get("confidence") or 0.5))
                ),
            }
        )
        if time_basis == "unknown_assumed_as_of":
            timing["unknown_time_ids"].append(node_id)
        fact_text_chars += len(statement)
    if not facts:
        if timing["excluded_future_ids"]:
            raise ValueError("截止时间前没有可用证据，请补充当时已公开的资料或调整分析截止时间")
        raise ValueError("evidence graph has no usable evidence text")

    graph_question = str(question or graph.get("question") or "").strip()
    if not graph_question:
        raise ValueError("simulation question is required")
    text = "\n".join(
        [graph_question]
        + [fact["statement"] for fact in facts]
        + [
            str(node.get("title") or "")
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("kind") == "claim"
        ]
    )
    actor_kinds, actor_selection, actor_reasons = _select_actor_kinds(
        text, max_actors
    )
    fact_ids = [fact["id"] for fact in facts]
    actors = []
    for kind in actor_kinds:
        actor_id, label, aggregation = ACTOR_TEMPLATES[kind]
        actors.append(
            {
                "id": actor_id,
                "label": label,
                "kind": kind,
                "aggregation": aggregation,
                "goals": [ACTOR_BEHAVIOR[kind][0], "根据对方行动与新增信息调整判断"],
                "constraints": ["只能使用分析截止时间前可观察的信息", ACTOR_BEHAVIOR[kind][1]],
                "observable_fact_ids": fact_ids,
                "assumptions": ["上述目标与资源约束是角色建模假设，具体资源、权限和私有信息仍以证据为准"],
                "selection_reason": actor_reasons[kind],
            }
        )

    actors, identity_selection = apply_entity_roles(actors, facts, graph_question, market)
    actor_selection.update(identity_selection)
    actor_selection["configured_count"] = len(actors)
    actor_selection["selection_strategy"] = "literal_mentions_plus_event_structure_v4_entities"

    claims = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, dict)
        and node.get("kind") == "claim"
        and node.get("status") not in {"rejected", "insufficient"}
    ]
    target_texts = [
        str(node.get("title") or node.get("body") or "").strip()
        for node in claims
    ]
    target_texts = [item for item in target_texts if item][:2]
    if not target_texts:
        target_texts = [graph_question]
    forecast_targets = [
        {
            "id": f"T{index}",
            "kind": "event",
            "definition": definition[:1000],
            "horizon": f"{horizon_days}_calendar_days",
            "scoring": "brier",
        }
        for index, definition in enumerate(target_texts, start=1)
    ]

    safe_case_id = re.sub(r"[^a-z0-9_-]+", "_", case_id.lower()).strip("_")
    safe_case_id = safe_case_id or "fever_case"
    default_market = _infer_market(text)
    spec = {
        "schema_version": "0.1.0",
        "case_id": safe_case_id,
        "title": graph_question[:160],
        "as_of": as_of_dt.isoformat(),
        "question": graph_question,
        "horizon": {
            "kind": "calendar_days",
            "value": horizon_days,
            "end_at": (as_of_dt + timedelta(days=horizon_days)).isoformat(),
        },
        "market": market or default_market,
        "facts": facts,
        "actors": actors,
        "relationships": _build_relationships(
            [actor["id"] for actor in actors]
        ),
        "allowed_actions": [
            {"type": "COMMUNICATE", "description": "公开沟通与预期管理", "visibility": "public"},
            {"type": "OPERATE", "description": "调整业务、产品或治理", "visibility": "either"},
            {"type": "REGULATE", "description": "调查、批准、限制或补救", "visibility": "public"},
            {"type": "ALLOCATE", "description": "调整资本或风险敞口", "visibility": "private"},
            {"type": "NEGOTIATE", "description": "协商交易或治理条件", "visibility": "private"},
            {"type": "WAIT", "description": "等待更多证据或对方行动", "visibility": "either"},
        ],
        "interventions": [
            {
                "id": "I0_FEVER_EVIDENCE",
                "at_round": 0,
                "fact_ids": fact_ids,
                "visibility": "public",
            }
        ],
        "forecast_targets": forecast_targets,
        "run_config": {
            "rounds": 3,
            "replications": 3,
            "seed_strategy": "deterministic_sequence",
            "layers": ["information", "behavior"],
        },
        "provenance": {
            "created_by": "FEVER SimulationSpecBuilder v1",
            "source_graph_artifact_id": source_graph_artifact_id,
            "notes": [
                "Scenario-first integration input; simulation output is not evidence.",
                "Quick mode executes one partial replication from a three-replication spec.",
            ],
            "actor_selection": actor_selection,
            "evidence_timing": timing,
        },
    }
    validate_spec(spec)
    return spec
