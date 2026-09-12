"""A compact product compiler that can leave weak candidate paths unused."""
from __future__ import annotations

from .scenario_presentation import HORIZON_UNITS, unsupported_quantities
import re

FOCUSED_PROMPT_VERSION = "scenario-branch-compiler-v9"
STRICT_PROMPT_VERSION = "scenario-branch-compiler-v10"
GUARDED_PROMPT_VERSION = "scenario-branch-compiler-v11"
FOCUSED_VERSIONS = {FOCUSED_PROMPT_VERSION, STRICT_PROMPT_VERSION, GUARDED_PROMPT_VERSION}


def candidate_slots(spec: dict, financial_actions: dict) -> list[dict]:
    decisions = {item["actor_id"]: item for item in financial_actions["decisions"]}
    slots, seen = [], set()
    for relation in spec["relationships"]:
        pair = [relation["source_actor_id"], relation["target_actor_id"]]
        key = frozenset(pair)
        if len(key) != 2 or key in seen or not key <= decisions.keys():
            continue
        seen.add(key)
        slots.append({"slot_id": f"pair-{len(slots) + 1}", "actor_ids": pair,
                      "decision_refs": [decisions[actor]["id"] for actor in pair],
                      "relationship_kind": relation["kind"],
                      "relationship_visibility": relation.get("visibility", "assumed")})
    return slots


def focused_prompt(strict: bool = False) -> str:
    prompt = """你是金融事件研究助手。用截止时点事实和各方模拟决策提出少量值得跟踪的条件路径。
所有模拟动作、对话和关系候选都不是事实。不得引入历史事件的已知后续结果。

从 branch_slots 中选择最有用的 1 至 4 条路径；没有足够依据可返回空数组并说明缺少什么。
不要凑数量或强行覆盖所有角色；同一对角色可以有相反条件的路径，但不能重复改写。
每条用 slot_id 标明候选；系统绑定两方的已有决策。不要输出 actor_ids 或 actions。
每条说明一方怎样改变另一方的选择。WAIT 是合法起点，不可写成已经交易或已经批准。
假设写进 assumptions：交易条件、客户反应、执行能力不明时明确“尚未披露/待核对”。
关系 visibility=assumed 只表示可研究的假设，不是已存在的影响。
summary、触发与失效条件、后果均用条件表述；保留相反分支，避免只列单向乐观路径。
证据中没有的比例、金额、篇数或时间期限不要编造；请改用可核实的事件或明确等待。
同值数字不能跨对象使用，统计占比不能替代财务影响，已重述数据不能再调整。
observations 恰好两项：一个 trigger、一个 invalidation；signal 必须与对应条件逐字相同。
source 写可实际查询的公开渠道，如公司后续公告、交易所披露、官方服务状态更新。
内部会议、匿名传闻、未公开备忘录不能作为随时可查的信号；需要披露时说清楚。
定期持仓报告只说明其报告时点，不证明事后短期交易；未出现公告本身不能证明措施未执行。
观察窗口由系统填入，窗口内可能拿不到的结果应明确待披露，不能虚构发布日。
evidence_refs 只引用输入 F 编号，表示路径的背景依据，不证明假设已经发生。
novelty_claim 写两方之间具体的决策依赖；confidence 只表示内部连贯性，不是概率。
只输出 JSON：
{"branches":[{"slot_id":"pair-1","label":"简短名称","summary":"若…则…",
"trigger_conditions":["若…公开确认…"],"invalidation_conditions":["若…公开确认相反情况…"],
"consequences":["可能…"],"assumptions":["尚未披露…，本路径假设…"],
"observations":[{"kind":"trigger","signal":"若…公开确认…","source":"公司后续公告","evidence_refs":["F1"]},
{"kind":"invalidation","signal":"若…公开确认相反情况…","source":"公司后续公告","evidence_refs":["F1"]}],
"evidence_refs":["F1"],"novelty_claim":"…如何影响…的选择","confidence":0.5}],"warnings":[]}"""
    if strict:
        prompt += """
提交前逐条复核，否则结果会被校验拒绝：
- 本格式只提供事件条件，不提供自行设置的数值阈值。所有带单位数字必须在该分支引用的
  原始事实或用户窗口中逐字有依据；即使写进 assumptions 也不能编造阈值。
- 已按同一口径调整/重述的比较数据不得再次剔除同一影响；可询问经营原因，不能假定再调整会缩小降幅。
- invalidation 必须给出会推翻核心传导的相反公开信息，例如收到解释后仍公开拒绝方案；
  “触发未发生/未发布/没有消息”只表示尚未进入路径，不足以证明路径错误。
- 停牌中的投资者可评估方案或等待复牌，不能把方案披露直接写成已经能交易。
- 只讨论输入所列主体或通用群体，不添加输入外的竞争公司名称、监管机关名称或新技术产品。
- 若短期无法取得数据，使用公开的决定、公告、撤回、恢复进度等事件；不假定下一期财报能在本窗口公布。
优先写两条简短、互有分歧的路径。少而明确；每条 summary、consequences、novelty_claim 分别不超过100字。
"""
    return prompt


def validate_focused_grounding(branch: dict, spec: dict) -> None:
    cited = {**spec, "facts": [fact for fact in spec["facts"] if fact["id"] in branch["evidence_refs"]]}
    quantities = unsupported_quantities(cited, {**branch, "conditional_responses": branch.get("assumptions", [])})
    if quantities:
        raise ValueError("请删除或改写本分支无引用依据的数值条件（标作假设也不能保留）：" + "、".join(quantities))
    facts = "\n".join(fact["statement"] for fact in cited["facts"])
    prose = str([branch.get(key) for key in ("summary", "trigger_conditions", "invalidation_conditions", "consequences")])
    if re.search(r"比较数据.*已.*(?:调整|重述)", facts) and re.search(r"(?:剔除|还原).{0,20}(?:口径|保证费用)|(?:剔除|还原)口径影响|口径调整后.{0,12}降幅", prose):
        raise ValueError("引用事实明确比较数据已调整，不能假定再剔除同一口径影响；请研究经营原因或等待公司解释")


def normalize_focused_fields(raw: dict, decisions: list[dict], spec: dict) -> dict:
    assumptions = raw.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions or not all(isinstance(item, str) and item.strip() for item in assumptions):
        raise ValueError("v9 assumptions must explicitly state the unverified premises")
    observations = raw.get("observations")
    if not isinstance(observations, list) or len(observations) != 2 or not all(isinstance(item, dict) for item in observations):
        raise ValueError("v9 requires one trigger observation and one invalidation observation")
    if sorted(str(item.get("kind")) for item in observations) != ["invalidation", "trigger"]:
        raise ValueError("v9 observations must cover trigger and invalidation")
    refs = set(raw.get("evidence_refs") or [])
    normalized = []
    for item in observations:
        key = "trigger_conditions" if item["kind"] == "trigger" else "invalidation_conditions"
        if raw.get(key) != [item.get("signal")]:
            raise ValueError("v9 observation signal must equal its single branch condition verbatim")
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            raise ValueError("v9 observation source is required")
        cited = item.get("evidence_refs")
        if not isinstance(cited, list) or not cited or not all(isinstance(ref, str) for ref in cited) or not set(cited) <= refs:
            raise ValueError("v9 observation evidence must be a nonempty subset of branch evidence")
        normalized.append({"kind": item["kind"], "signal": item["signal"], "source": item["source"].strip(),
                           "window": f"未来 {spec['horizon']['value']} {HORIZON_UNITS[spec['horizon']['kind']]}",
                           "evidence_refs": list(dict.fromkeys(cited))})
    return {"assumptions": [item.strip() for item in assumptions], "observations": normalized,
            "starting_decisions": [{"actor_id": item["actor_id"], "action_type": item["action_type"],
                                    "decision_status": item["decision_status"], "rationale": item["rationale"],
                                    "decision_ref": item["id"]} for item in decisions]}
