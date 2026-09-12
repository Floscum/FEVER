"""Render a validated SimulationSpec into a MiroFish text seed.

MiroFish currently builds its graph from PDF/Markdown/TXT uploads.  This
renderer keeps the wire contract as JSON while producing a deterministic,
human-auditable Markdown document for that ingestion path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .contracts import canonical_sha256, validate_spec


def _items(values: Iterable[str], fallback: str = "无") -> str:
    rendered = [str(value).strip() for value in values if str(value).strip()]
    return "；".join(rendered) if rendered else fallback


def render_seed_markdown(spec: Dict[str, Any]) -> str:
    """Return a deterministic, outcome-free Markdown seed for MiroFish."""

    validate_spec(spec)
    fact_by_id = {fact["id"]: fact for fact in spec["facts"]}

    lines = [
        "# 金融事件多主体推演输入",
        "",
        "## 边界与任务",
        "",
        f"- 案例 ID：`{spec['case_id']}`",
        f"- 输入契约哈希：`{canonical_sha256(spec)}`",
        f"- 信息截止时间（as_of）：`{spec['as_of']}`",
        f"- 推演终点：`{spec['horizon']['end_at']}`",
        f"- 推演问题：{spec['question']}",
        "- 严格约束：只能使用本文列出的截止时点前事实；不得补入真实后验、"
        "未来价格、后续公告或其他结果信息。",
        "- 解释约束：模拟产生的是情景和待验证假设，不是真实证据，也不是已校准概率。",
        "",
        "## 市场范围",
        "",
        f"- 地区：{spec['market']['region']}",
        f"- 场所：{_items(spec['market']['venues'])}",
        "- 标的：",
    ]
    for instrument in spec["market"]["instruments"]:
        lines.append(
            f"  - `{instrument['symbol']}` {instrument['name']}，"
            f"{instrument['kind']}，{instrument['currency']}"
        )

    lines.extend(["", "## 截止时点前可用事实", ""])
    for fact in spec["facts"]:
        confidence = fact.get("confidence")
        confidence_text = "" if confidence is None else f"，置信度 {confidence:.2f}"
        lines.extend(
            [
                f"### {fact['id']}",
                "",
                f"- 观察时间：{fact['observed_at']}",
                f"- 内容：{fact['statement']}",
                f"- 来源类型：{fact['source_kind']}{confidence_text}",
                f"- 来源：{fact['source_url']}",
                "",
            ]
        )

    lines.extend(["## 参与者", ""])
    for actor in spec["actors"]:
        visible_facts = [
            f"{fact_id}: {fact_by_id[fact_id]['statement']}"
            for fact_id in actor["observable_fact_ids"]
        ]
        lines.extend(
            [
                f"### {actor['id']} — {actor['label']}",
                "",
                f"- 类型与粒度：{actor['kind']} / {actor['aggregation']}",
                f"- 目标：{_items(actor['goals'])}",
                f"- 约束：{_items(actor['constraints'])}",
                f"- 假设：{_items(actor['assumptions'])}",
                f"- 可观察事实：{_items(visible_facts)}",
                "",
            ]
        )

    lines.extend(["## 参与者关系", ""])
    if spec["relationships"]:
        for relation in spec["relationships"]:
            lines.append(
                f"- `{relation['source_actor_id']}` --{relation['kind']}--> "
                f"`{relation['target_actor_id']}`（{relation['visibility']}）"
            )
    else:
        lines.append("- 无预设关系。")

    lines.extend(["", "## 允许的金融语义动作", ""])
    for action in spec["allowed_actions"]:
        lines.append(
            f"- **{action['type']}**（{action['visibility']}）："
            f"{action['description']}"
        )

    lines.extend(["", "## 外生干预", ""])
    for intervention in spec["interventions"]:
        lines.append(
            f"- `{intervention['id']}`：第 {intervention['at_round']} 轮，"
            f"向 {intervention['visibility']} 发布事实 "
            f"{', '.join(intervention['fact_ids'])}。"
        )

    lines.extend(["", "## 需要回答的预测目标", ""])
    for target in spec["forecast_targets"]:
        lines.append(
            f"- `{target['id']}`（{target['kind']} / {target['horizon']} / "
            f"{target['scoring']}）：{target['definition']}"
        )

    lines.extend(
        [
            "",
            "## 输出纪律",
            "",
            "每轮记录主体、动作、目标对象、公开或私有可见性、引用的事实 ID、"
            "以及因果理由。若没有充分输入，明确标记信息缺口，不得虚构为事实。",
            "价格方向只能作为行为压力与情景分支的待检验输出；当前信息层模拟不包含"
            "真实订单簿、流动性或价格形成机制。",
            "",
        ]
    )
    return "\n".join(lines)


def build_simulation_requirement(spec: Dict[str, Any], *, rounds: int) -> str:
    """Create a compact, finance-oriented MiroFish simulation requirement."""

    validate_spec(spec)
    target_lines = "；".join(
        f"{target['id']}={target['definition']}" for target in spec["forecast_targets"]
    )
    return (
        f"基于上传材料，在信息截止时间 {spec['as_of']} 下进行 {rounds} 轮多主体"
        "金融事件推演。参与者只代表材料中定义的机构或群体，不得引入截止时间后的"
        "真实信息。重点记录政策传导、公开叙事、风险偏好、配置意图和二阶反馈；"
        "不得把社交互动数量直接当作价格。最终围绕以下目标形成分支、依据和缺口："
        f"{target_lines}。所有出现频率均标记为未校准模拟频率。"
    )
