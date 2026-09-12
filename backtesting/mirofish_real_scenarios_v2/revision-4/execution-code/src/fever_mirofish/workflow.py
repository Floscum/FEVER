"""Checkpointed low-cost workflow for one MiroFish compatibility run."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .contracts import ContractError, canonical_sha256, validate_spec
from .financial_actions import (
    append_financial_decisions_to_result,
    build_financial_interview_plan,
    parse_financial_interviews,
)
from .financial_profiles import (
    apply_financial_actor_overlay,
    compile_financial_actor_overlay,
)
from .mirofish_client import MiroFishClient
from .oasis_sqlite import build_smoke_simulation_result, export_oasis_sqlite
from .runtime_config import apply_smoke_runtime_config
from .seed_renderer import build_simulation_requirement, render_seed_markdown
from .smoke_profile import SmokeProfile


TERMINAL_TASK_STATES = {"completed", "failed"}
TERMINAL_RUN_STATES = {"completed", "failed", "stopped"}
GENERIC_ONTOLOGY_TYPES = {"Person", "Organization"}


class WorkflowCancelled(RuntimeError):
    """Raised when the owning asynchronous job requests cancellation."""


def review_ontology_specificity(
    entity_type_names: list[str],
    edge_type_names: list[str],
) -> Dict[str, Any]:
    """Reject the upstream generic fallback before paying for a graph build."""

    entity_types = {
        item for item in entity_type_names if isinstance(item, str) and item
    }
    edge_types = {
        item for item in edge_type_names if isinstance(item, str) and item
    }
    generic_only = bool(
        entity_types
        and entity_types <= GENERIC_ONTOLOGY_TYPES
        and not edge_types
    )
    return {
        "status": "failed" if generic_only else "passed",
        "generic_only": generic_only,
        "entity_type_count": len(entity_types),
        "edge_type_count": len(edge_types),
        "reason": (
            "ontology contains only generic Person/Organization types "
            "and no relationship types"
            if generic_only
            else None
        ),
    }


class SmokeWorkflow:
    """Run and resume expensive upstream stages without repeating them."""

    def __init__(
        self,
        spec: Dict[str, Any],
        run_dir: Path,
        client: MiroFishClient,
        profile: Optional[SmokeProfile] = None,
        simulation_data_dirs: Optional[list[Path]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        validate_spec(spec)
        self.spec = spec
        self.spec_hash = canonical_sha256(spec)
        self.run_dir = Path(run_dir)
        self.client = client
        self.profile = profile or SmokeProfile()
        self.simulation_data_dirs = [
            Path(path) for path in (simulation_data_dirs or [])
        ]
        self.should_cancel = should_cancel or (lambda: False)
        self.seed_path = self.run_dir / "seed.md"
        self.state_path = self.run_dir / "state.json"
        self.actions_path = self.run_dir / "raw-actions.json"
        self.sqlite_actions_path = self.run_dir / "sqlite-actions.json"
        self.result_path = self.run_dir / "simulation-result.json"
        self.financial_interviews_path = (
            self.run_dir / "financial-interviews-raw.json"
        )
        self.financial_actions_path = self.run_dir / "financial-actions.json"

    def _check_cancelled(self) -> None:
        if self.should_cancel():
            raise WorkflowCancelled("simulation job was cancelled")

    def _check_running_cancelled(self, simulation_id: str) -> None:
        if not self.should_cancel():
            return
        try:
            env = self.client.get_env_status(simulation_id)["data"]
            if env.get("env_alive"):
                self.client.close_env(simulation_id, timeout=30)
        except Exception:  # noqa: BLE001 - cancellation must still complete
            pass
        raise WorkflowCancelled("simulation job was cancelled")

    def initialize(self) -> Dict[str, Any]:
        self._check_cancelled()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed_path.write_text(render_seed_markdown(self.spec), encoding="utf-8")

        if self.state_path.exists():
            state = self._read_state()
            if state.get("spec_sha256") != self.spec_hash:
                raise ContractError(
                    "existing smoke state belongs to a different SimulationSpec"
                )
            ontology = state.get("stages", {}).get("ontology")
            if (
                isinstance(ontology, dict)
                and ontology.get("project_id")
                and not ontology.get("status")
            ):
                ontology["status"] = "completed"
                self._write_state(state)
            return state

        state = {
            "schema_version": "0.1.0",
            "case_id": self.spec["case_id"],
            "spec_sha256": self.spec_hash,
            "mirofish_base_url": self.client.base_url,
            "profile": self.profile.to_dict(),
            "stages": {},
        }
        self._write_state(state)
        return state

    def plan(self) -> Dict[str, Any]:
        state = self.initialize()
        stage_statuses = {
            name: self._stage_status(name, data)
            for name, data in state.get("stages", {}).items()
        }
        return {
            "case_id": self.spec["case_id"],
            "spec_sha256": self.spec_hash,
            "seed_path": str(self.seed_path),
            "state_path": str(self.state_path),
            "profile": self.profile.to_dict(),
            "stage_statuses": stage_statuses,
            "completed_stages": sorted(
                name
                for name, status in stage_statuses.items()
                if status in {"ok", "completed"}
            ),
        }

    def reuse_research_stages(self, source_state_path: Path) -> Dict[str, Any]:
        """Reuse a completed ontology/graph in a separate experiment run."""

        state = self.initialize()
        source_state_path = Path(source_state_path)
        source = json.loads(source_state_path.read_text(encoding="utf-8"))
        if source.get("spec_sha256") != self.spec_hash:
            raise ContractError("source checkpoint belongs to a different spec")
        source_stages = source.get("stages") or {}
        ontology = source_stages.get("ontology")
        graph = source_stages.get("graph")
        if not ontology or not ontology.get("project_id"):
            raise ContractError("source checkpoint has no completed ontology")
        if not graph or graph.get("status") != "completed" or not graph.get("graph_id"):
            raise ContractError("source checkpoint has no completed graph")
        if state.get("stages", {}).get("simulation"):
            raise ContractError(
                "cannot replace research stages after a simulation was created"
            )
        state["stages"]["ontology"] = deepcopy(ontology)
        state["stages"]["graph"] = deepcopy(graph)
        state["reused_research_checkpoint"] = {
            "source_state_path": str(source_state_path.resolve()),
            "project_id": ontology["project_id"],
            "graph_id": graph["graph_id"],
        }
        self._write_state(state)
        return state["reused_research_checkpoint"]

    def preflight(self) -> Dict[str, Any]:
        state = self.initialize()
        response = self.client.health()
        state["stages"]["preflight"] = {
            "status": response.get("status", "ok"),
            "service": response.get("service"),
        }
        self._write_state(state)
        return response

    def generate_ontology(self) -> Dict[str, Any]:
        state = self.initialize()
        existing = state["stages"].get("ontology")
        if existing and existing.get("project_id"):
            if (
                (existing.get("specificity_review") or {}).get("status")
                == "failed"
            ):
                raise RuntimeError(
                    "ontology specificity gate failed; graph build is blocked"
                )
            return existing

        response = self.client.generate_ontology(
            self.seed_path,
            simulation_requirement=build_simulation_requirement(
                self.spec,
                rounds=self.profile.rounds,
            ),
            project_name=f"FEVER smoke — {self.spec['title']}",
            additional_context=(
                "这是金融事件信息传播 PoC。实体应优先采用监管者、交易所、券商、"
                "上市公司、境内机构、境外机构、个人投资者和财经媒体等可行动主体。"
            ),
        )
        data = response["data"]
        ontology = data.get("ontology") or {}
        entity_type_names = [
            item.get("name") for item in ontology.get("entity_types", [])
        ]
        edge_type_names = [
            item.get("name") for item in ontology.get("edge_types", [])
        ]
        specificity_review = review_ontology_specificity(
            entity_type_names,
            edge_type_names,
        )
        stage = {
            "status": "completed",
            "project_id": data["project_id"],
            "project_name": data.get("project_name"),
            "total_text_length": data.get("total_text_length"),
            "entity_type_names": entity_type_names,
            "edge_type_names": edge_type_names,
            "analysis_summary": data.get("analysis_summary"),
            "specificity_review": specificity_review,
        }
        state["stages"]["ontology"] = stage
        self._write_state(state)
        if specificity_review["status"] == "failed":
            raise RuntimeError(
                "ontology specificity gate failed; graph build is blocked"
            )
        return stage

    def build_graph(self, *, timeout: float = 900.0, poll_interval: float = 3.0) -> Dict[str, Any]:
        state = self.initialize()
        ontology = state["stages"].get("ontology")
        if not ontology:
            raise RuntimeError("ontology stage must complete before graph build")
        existing = state["stages"].get("graph")
        if existing and existing.get("status") == "completed":
            return existing

        # A previous process may have lost only its polling connection while
        # the durable Zep batch completed. Prefer the authoritative project
        # state over force-rebuilding and duplicating Cloud work.
        project = self.client.get_project(ontology["project_id"])["data"]
        if project.get("status") == "graph_completed" and project.get("graph_id"):
            graph_id = project["graph_id"]
            graph_payload = self.client.get_graph_data(graph_id)["data"]
            stage = {
                "status": "completed",
                "task_id": project.get("graph_build_task_id"),
                "graph_id": graph_id,
                "node_count": len(graph_payload.get("nodes") or []),
                "edge_count": len(graph_payload.get("edges") or []),
                "chunk_count": None,
                "recovered": True,
                "message": "reused durable graph completed outside the HTTP poller",
            }
            state["stages"]["graph"] = stage
            self._write_state(state)
            return stage

        should_submit = (
            not existing
            or not existing.get("task_id")
            or existing.get("status") == "failed"
        )
        if should_submit:
            response = self.client.build_graph(
                ontology["project_id"],
                graph_name=f"FEVER smoke {self.spec['case_id']}",
                force=bool(existing and existing.get("status") == "failed"),
            )
            data = response["data"]
            if data.get("reused") and data.get("graph_id"):
                existing = {
                    "status": "completed",
                    "task_id": data.get("task_id"),
                    "graph_id": data["graph_id"],
                    "reused": True,
                }
            else:
                existing = {
                    "status": "pending",
                    "task_id": data["task_id"],
                }
            state["stages"]["graph"] = existing
            self._write_state(state)

        if existing["status"] == "completed":
            return existing

        task = self._poll_graph_task(
            existing["task_id"],
            timeout=timeout,
            poll_interval=poll_interval,
        )
        result = task.get("result") or {}
        stage = {
            "status": task["status"],
            "task_id": existing["task_id"],
            "graph_id": result.get("graph_id"),
            "node_count": result.get("node_count"),
            "edge_count": result.get("edge_count"),
            "chunk_count": result.get("chunk_count"),
            "message": task.get("message"),
        }
        state = self._read_state()
        state["stages"]["graph"] = stage
        self._write_state(state)
        if task["status"] != "completed":
            message = str(task.get("message") or "unknown upstream error")
            lowered = message.lower()
            if "rate limit" in lowered or "status_code: 429" in lowered:
                message = (
                    "ZEP read quota was temporarily exhausted after the graph batch "
                    "was submitted; wait for the reset window and resume this job"
                )
            else:
                message = message.splitlines()[0][:300]
            raise RuntimeError(f"MiroFish graph build failed: {message}")
        return stage

    def prepare_direct_simulation(self) -> Dict[str, Any]:
        """Compile a ready OASIS run from SimulationSpec without Zep.

        Quick financial runs already replace graph-derived social profiles with
        deterministic actors. This path makes that fact explicit: FEVER's
        evidence graph remains the provenance source, while Zep is optional
        infrastructure rather than a blocking ingestion dependency.
        """

        self._check_cancelled()
        state = self.initialize()
        existing = state["stages"].get("simulation") or {}
        if existing.get("status") in {"ready", "running", "completed"}:
            return existing
        if not self.simulation_data_dirs:
            raise RuntimeError("direct simulation requires a shared data directory")

        simulation_id = existing.get("simulation_id") or f"sim_direct_{uuid.uuid4().hex[:12]}"
        simulation_dir = self.simulation_data_dirs[0] / simulation_id
        simulation_dir.mkdir(parents=True, exist_ok=True)
        actor_ids = self.profile.financial_actor_ids or tuple(
            actor["id"] for actor in self.spec["actors"]
        )
        base_config = {
            "simulation_id": simulation_id,
            "project_id": f"direct_{self.spec_hash[:12]}",
            "graph_id": f"fever_evidence_{self.spec_hash[:16]}",
            "simulation_requirement": build_simulation_requirement(
                self.spec, rounds=self.profile.rounds
            ),
            "time_config": {
                "total_simulation_hours": self.profile.rounds,
                "minutes_per_round": 60,
                "agents_per_hour_min": min(3, len(actor_ids)),
                "agents_per_hour_max": len(actor_ids),
                "peak_hours": list(range(self.profile.rounds)),
                "peak_activity_multiplier": 1.0,
                "off_peak_hours": [],
                "off_peak_activity_multiplier": 1.0,
            },
            "event_config": {
                "initial_posts": [],
                "scheduled_events": [],
                "hot_topics": [
                    item["name"] for item in self.spec["market"]["instruments"]
                ][:8],
                "narrative_direction": self.spec["question"][:500],
            },
            "reddit_config": {
                "platform": "reddit",
                "recency_weight": 0.3,
                "popularity_weight": 0.4,
                "relevance_weight": 0.3,
                "viral_threshold": 15,
                "echo_chamber_strength": 0.6,
            },
            "twitter_config": None,
            "llm_model": os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini"),
            "llm_base_url": os.environ.get(
                "LLM_BASE_URL", "https://api.openai.com/v1"
            ),
            "generated_at": datetime.now().astimezone().isoformat(),
            "generation_reasoning": (
                "Deterministic FEVER SimulationSpec compiler; Zep bypassed for "
                "bounded quick-mode execution"
            ),
            "agent_configs": [],
        }
        compiled, profiles, metadata = compile_financial_actor_overlay(
            base_config,
            self.spec,
            actor_ids=actor_ids,
        )
        actor_to_agent = metadata["actor_to_agent_id"]
        config_by_actor = {
            item["actor_id"]: item for item in compiled["agent_configs"]
        }
        selected_actors = [
            actor for actor in self.spec["actors"] if actor["id"] in actor_to_agent
        ]
        fallback_actor = next(
            (
                actor
                for actor in selected_actors
                if actor.get("kind") in {"media", "issuer", "analyst"}
            ),
            selected_actors[0],
        )
        initial_posts = []
        for fact in self.spec["facts"][: min(6, len(self.spec["facts"]))]:
            actor = next(
                (
                    item
                    for item in selected_actors
                    if fact["id"] in item.get("observable_fact_ids", [])
                ),
                fallback_actor,
            )
            actor_config = config_by_actor[actor["id"]]
            initial_posts.append(
                {
                    "content": fact["statement"][:500],
                    "poster_type": actor_config["entity_type"],
                    "poster_agent_id": actor_to_agent[actor["id"]],
                    "poster_actor_id": actor["id"],
                    "source_fact_id": fact["id"],
                }
            )
        compiled["event_config"]["initial_posts"] = initial_posts
        self._atomic_write_json(simulation_dir / "simulation_config.json", compiled)
        self._atomic_write_json(simulation_dir / "reddit_profiles.json", profiles)
        (simulation_dir / "twitter_profiles.csv").write_text(
            "user_id\n", encoding="utf-8"
        )
        self._atomic_write_json(
            simulation_dir / "state.json",
            {
                "simulation_id": simulation_id,
                "project_id": base_config["project_id"],
                "graph_id": base_config["graph_id"],
                "enable_twitter": False,
                "enable_reddit": True,
                "status": "ready",
                "entities_count": len(profiles),
                "profiles_count": len(profiles),
                "entity_types": sorted(
                    {item["entity_type"] for item in compiled["agent_configs"]}
                ),
                "profiles_generated": True,
                "config_generated": True,
                "config_reasoning": compiled["generation_reasoning"],
                "current_round": 0,
                "twitter_status": "not_started",
                "reddit_status": "not_started",
                "created_at": datetime.now().astimezone().isoformat(),
                "updated_at": datetime.now().astimezone().isoformat(),
                "error": None,
            },
        )
        stage = {
            "simulation_id": simulation_id,
            "status": "ready",
            "prepare_status": "completed",
            "expected_entities_count": len(profiles),
            "graph_backend": "direct",
            "zep_bypassed": True,
        }
        state["stages"]["ontology"] = {
            "status": "skipped",
            "backend": "direct",
        }
        state["stages"]["graph"] = {
            "status": "completed",
            "backend": "direct",
            "graph_id": base_config["graph_id"],
            "node_count": len(self.spec["actors"]) + len(self.spec["facts"]),
            "edge_count": sum(
                len(actor.get("observable_fact_ids", []))
                for actor in self.spec["actors"]
            ),
            "message": "compiled directly from FEVER evidence graph; Zep bypassed",
        }
        state["stages"]["simulation"] = stage
        self._write_state(state)
        return stage

    def run_simulation(
        self,
        *,
        prepare_timeout: float = 900.0,
        run_timeout: float = 900.0,
        poll_interval: float = 3.0,
    ) -> Dict[str, Any]:
        self._check_cancelled()
        state = self.initialize()
        graph = state["stages"].get("graph")
        ontology = state["stages"].get("ontology")
        if not graph or graph.get("status") != "completed" or not ontology:
            raise RuntimeError("graph stage must complete before simulation")

        stage = state["stages"].get("simulation") or {}
        simulation_id = stage.get("simulation_id")
        if not simulation_id:
            self._check_cancelled()
            created = self.client.create_simulation(
                self.profile.create_payload(
                    ontology["project_id"],
                    graph.get("graph_id"),
                )
            )["data"]
            simulation_id = created["simulation_id"]
            stage = {"simulation_id": simulation_id, "status": "created"}
            state["stages"]["simulation"] = stage
            self._write_state(state)

        if stage.get("status") not in {"ready", "running", "completed"}:
            self._check_cancelled()
            prepared = self.client.prepare_simulation(
                self.profile.prepare_payload(simulation_id)
            )["data"]
            task_id = prepared.get("task_id")
            prepare_result = self._poll_prepare(
                simulation_id,
                task_id=task_id,
                timeout=prepare_timeout,
                poll_interval=poll_interval,
            )
            stage.update(
                {
                    "status": "ready",
                    "prepare_task_id": task_id,
                    "expected_entities_count": prepared.get("expected_entities_count"),
                    "prepare_status": prepare_result.get("status"),
                }
            )
            state = self._read_state()
            state["stages"]["simulation"] = stage
            self._write_state(state)

        if stage.get("status") == "ready":
            self._check_cancelled()
            runtime_config_path = self._find_simulation_config(simulation_id)
            if runtime_config_path is None:
                raise RuntimeError(
                    f"cannot find local simulation_config.json for {simulation_id}"
                )
            if self.profile.financial_actor_overlay:
                profiles_path = runtime_config_path.with_name(
                    "reddit_profiles.json"
                )
                overlay = apply_financial_actor_overlay(
                    runtime_config_path,
                    profiles_path,
                    self.spec,
                    actor_ids=self.profile.financial_actor_ids,
                )
                stage["financial_actor_overlay"] = overlay
            runtime_override = apply_smoke_runtime_config(
                runtime_config_path,
                rounds=self.profile.rounds,
                scheduler_seed=self.profile.scheduler_seed,
            )
            stage["runtime_override"] = runtime_override
            started = self.client.start_simulation(
                self.profile.start_payload(simulation_id)
            )["data"]
            stage.update(
                {
                    "status": "running",
                    "runner_status": started.get("runner_status"),
                }
            )
            state = self._read_state()
            state["stages"]["simulation"] = stage
            self._write_state(state)

        final_status = self._poll_run(
            simulation_id,
            timeout=run_timeout,
            poll_interval=poll_interval,
        )
        actions = self.client.get_actions(
            simulation_id,
            platform=self.profile.platform,
            limit=1000,
        )
        self._atomic_write_json(self.actions_path, actions)

        monitor_actions_count = final_status.get("total_actions_count")
        database_path = self._find_simulation_database(simulation_id)
        canonical_export = None
        if database_path is not None:
            config_path = database_path.with_name("simulation_config.json")
            canonical_export = export_oasis_sqlite(
                database_path,
                config_path,
                self.spec,
            )
            self._atomic_write_json(self.sqlite_actions_path, canonical_export)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            result = build_smoke_simulation_result(
                self.spec,
                canonical_export,
                model=str(config.get("llm_model") or "unknown"),
                engine_version="60757b3c825d577a1d7a86ab1ba6d5c21f51c261",
                raw_run_artifacts=[
                    str(database_path.resolve()),
                    str(config_path.resolve()),
                    str(self.sqlite_actions_path.resolve()),
                    str(self.actions_path.resolve()),
                ],
            )
            if self.financial_actions_path.exists():
                financial_actions = json.loads(
                    self.financial_actions_path.read_text(encoding="utf-8")
                )
                if financial_actions.get("decisions"):
                    result = append_financial_decisions_to_result(
                        result,
                        financial_actions,
                        self.spec,
                    )
                    result["provenance"]["raw_run_artifacts"].extend(
                        [
                            str(self.financial_interviews_path.resolve()),
                            str(self.financial_actions_path.resolve()),
                        ]
                    )
            self._atomic_write_json(self.result_path, result)

        # Remove legacy monitor fields that previously looked authoritative.
        # The current upstream monitor does not observe Reddit SQLite actions.
        stage.pop("current_round", None)
        stage.pop("total_actions_count", None)
        stage.update(
            {
                "status": "completed"
                if final_status.get("runner_status") == "completed"
                else final_status.get("runner_status"),
                "runner_status": final_status.get("runner_status"),
                "reported_current_round": final_status.get("current_round"),
                "configured_rounds": self.profile.rounds,
                "monitor_actions_count": monitor_actions_count,
                "actions_path": str(self.actions_path),
            }
        )
        if canonical_export is not None:
            stage.update(
                {
                    "canonical_action_source": "oasis_reddit_sqlite",
                    "sqlite_trace_count": canonical_export["table_counts"]["trace"],
                    "autonomous_actions_count": canonical_export[
                        "autonomous_action_count"
                    ],
                    "autonomous_text_actions_count": canonical_export[
                        "autonomous_text_action_count"
                    ],
                    "active_actor_ids": canonical_export["active_actor_ids"],
                    "canonical_actions_path": str(self.sqlite_actions_path),
                    "simulation_result_path": str(self.result_path),
                }
            )
        financial_summary = final_status.get("_financial_actions")
        if financial_summary:
            stage["financial_actions"] = financial_summary
            stage["financial_interviews_path"] = str(
                self.financial_interviews_path
            )
            stage["financial_actions_path"] = str(self.financial_actions_path)
        state = self._read_state()
        state["stages"]["simulation"] = stage
        self._write_state(state)
        if final_status.get("runner_status") != "completed":
            raise RuntimeError(
                f"MiroFish smoke run ended as {final_status.get('runner_status')}"
            )
        return stage

    def _poll_graph_task(
        self,
        task_id: str,
        *,
        timeout: float,
        poll_interval: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_cancelled()
            data = self.client.get_graph_task(task_id)["data"]
            if data.get("status") in TERMINAL_TASK_STATES:
                return data
            time.sleep(poll_interval)
        raise TimeoutError(f"graph task {task_id} did not finish in {timeout}s")

    def _poll_prepare(
        self,
        simulation_id: str,
        *,
        task_id: Optional[str],
        timeout: float,
        poll_interval: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_cancelled()
            data = self.client.get_prepare_status(
                simulation_id,
                task_id=task_id,
            )["data"]
            status = data.get("status")
            if status in {"ready", "completed"}:
                return data
            if status == "failed":
                raise RuntimeError("MiroFish simulation preparation failed")
            time.sleep(poll_interval)
        raise TimeoutError(f"simulation preparation did not finish in {timeout}s")

    def _poll_run(
        self,
        simulation_id: str,
        *,
        timeout: float,
        poll_interval: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        financial_summary = None
        financial_attempted = False
        while time.monotonic() < deadline:
            self._check_running_cancelled(simulation_id)
            data = self.client.get_run_status(simulation_id)["data"]
            if data.get("runner_status") in TERMINAL_RUN_STATES:
                if financial_summary:
                    data["_financial_actions"] = financial_summary
                return data
            if data.get("runner_status") == "running":
                env = self.client.get_env_status(simulation_id)["data"]
                if env.get("env_alive"):
                    if (
                        self.profile.collect_financial_decisions
                        and not financial_attempted
                    ):
                        financial_attempted = True
                        try:
                            financial_summary = (
                                self._collect_financial_decisions(simulation_id)
                            )
                        except Exception as error:
                            financial_summary = {
                                "status": "failed",
                                "error": str(error)[:300],
                            }
                    self.client.close_env(simulation_id, timeout=30)
            time.sleep(poll_interval)
        raise TimeoutError(f"simulation {simulation_id} did not finish in {timeout}s")

    def _collect_financial_decisions(
        self,
        simulation_id: str,
    ) -> Dict[str, Any]:
        config_path = self._find_simulation_config(simulation_id)
        if config_path is None:
            raise RuntimeError("cannot find simulation config for financial interview")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        overlay = config.get("fever_financial_actor_overlay") or {}
        actor_to_agent_id = overlay.get("actor_to_agent_id")
        if not isinstance(actor_to_agent_id, dict):
            raise RuntimeError("financial actor overlay is missing")
        plan = build_financial_interview_plan(
            self.spec,
            {key: int(value) for key, value in actor_to_agent_id.items()},
            platform=self.profile.platform,
            decision_round=self.profile.rounds,
        )
        api_interviews = [
            {
                "agent_id": item["agent_id"],
                "prompt": item["prompt"],
                "platform": item["platform"],
            }
            for item in plan
        ]
        raw = self.client.batch_interview(
            simulation_id,
            api_interviews,
            platform=self.profile.platform,
            timeout=300,
        )
        self._atomic_write_json(self.financial_interviews_path, raw)
        artifact = parse_financial_interviews(
            raw,
            plan,
            self.spec,
            simulation_id=simulation_id,
            round_num=self.profile.rounds,
        )
        self._atomic_write_json(self.financial_actions_path, artifact)
        return {
            "status": artifact["status"],
            "decision_count": len(artifact["decisions"]),
            "failure_count": len(artifact["failures"]),
        }

    def _find_simulation_config(self, simulation_id: str) -> Optional[Path]:
        for directory in self.simulation_data_dirs:
            candidate = directory / simulation_id / "simulation_config.json"
            if candidate.exists():
                return candidate
        return None

    def _find_simulation_database(self, simulation_id: str) -> Optional[Path]:
        for directory in self.simulation_data_dirs:
            candidate = directory / simulation_id / "reddit_simulation.db"
            if candidate.exists():
                return candidate
        return None

    def _read_state(self) -> Dict[str, Any]:
        with self.state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _stage_status(name: str, data: Dict[str, Any]) -> str:
        status = data.get("status")
        if isinstance(status, str):
            return status
        if name == "ontology" and data.get("project_id"):
            return "completed"
        return "unknown"

    def _write_state(self, state: Dict[str, Any]) -> None:
        self._atomic_write_json(self.state_path, state)

    @staticmethod
    def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
