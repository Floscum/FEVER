"""Persistent asynchronous gateway for FEVER-triggered MiroFish simulations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import canonical_sha256
from .fever_adapter import compile_evidence_graph
from .trace_digest import compact_interactions
from .scenario_presentation import prepare_scenarios


TERMINAL_STATUSES = {"completed", "partial", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class SimulationGateway:
    """Own durable job state and execute one bounded simulation per worker."""

    def __init__(
        self,
        root: Path,
        *,
        mode: str = "live",
        max_workers: int = 1,
        mirofish_base_url: str = "http://127.0.0.1:5001",
    ) -> None:
        if mode not in {"live", "stub"}:
            raise ValueError("gateway mode must be live or stub")
        self.root = Path(root).resolve()
        self.mode = mode
        self.product_version = os.environ.get("FEVER_SCENARIO_PRODUCT_VERSION", "v7")
        if self.product_version not in {"v7", "v8", "v9", "v10"}:
            raise ValueError("unsupported scenario product version")
        self.mirofish_base_url = mirofish_base_url.rstrip("/")
        self.jobs_root = self.root / ".data" / "gateway" / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="fever-simulation"
        )
        self._lock = threading.RLock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        """Fail orphaned work honestly after a gateway process restart."""
        for path in self.jobs_root.glob("simjob_*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") not in {"queued", "running", "cancelling"}:
                continue
            job.update(
                {
                    "status": "failed",
                    "stage": "interrupted",
                    "error": "gateway restarted before the job reached a terminal checkpoint",
                    "finished_at": _now(),
                    "updated_at": _now(),
                }
            )
            self._write_job(job)

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        spec = self._compile_request(request)
        mode = str(request.get("mode") or "quick")
        if mode not in {"quick", "calibrated"}:
            raise ValueError("simulation mode must be quick or calibrated")
        if mode == "calibrated":
            raise ValueError(
                "calibrated mode requires a structured B1 adapter and is not enabled in MVP-1"
            )
        spec_sha = canonical_sha256(spec)
        reuse_key = f"{spec_sha}:{mode}:{self.mode}:product-{self.product_version}:presentation-v3:trace-v2"
        with self._lock:
            reusable = self._find_reusable(reuse_key, completed=not request.get("rerun", False))
            if reusable:
                reusable = dict(reusable)
                reusable["reused"] = True
                return reusable
            job_id = f"simjob_{uuid.uuid4().hex[:16]}"
            job = {
                "schema_version": "0.1.0",
                "job_id": job_id,
                "case_id": str(request.get("case_id") or spec["case_id"]),
                "source_graph_artifact_id": str(
                    request.get("source_graph_artifact_id") or "unknown"
                ),
                "spec_sha256": spec_sha,
                "reuse_key": reuse_key,
                "mode": mode,
                "gateway_mode": self.mode,
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "created_at": _now(),
                "updated_at": _now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "result": None,
                "reused": False,
            }
            job_dir = self.jobs_root / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(job_dir / "spec.json", spec)
            self._write_job(job)
            cancel_event = threading.Event()
            self._cancel_events[job_id] = cancel_event
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
            return job

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        """Compile actor selection without starting a model-backed run."""

        spec = self._compile_request(request)
        return {
            "actor_selection": spec["provenance"]["actor_selection"],
            "actors": [
                {
                    "id": actor["id"],
                    "label": actor["label"],
                    "kind": actor["kind"],
                    "selection_reason": actor["selection_reason"],
                    "focus": actor["goals"][0],
                    "constraints": actor["constraints"],
                }
                for actor in spec["actors"]
            ],
            "evidence_count": len(spec["facts"]),
            "notices": self._input_notices(spec),
            "as_of": spec["as_of"],
        }

    @staticmethod
    def _compile_request(request: dict[str, Any]) -> dict[str, Any]:
        graph = request.get("evidence_graph")
        raw_max_actors = request.get("max_actors")
        return compile_evidence_graph(
            graph,
            case_id=str(request.get("case_id") or "fever_case"),
            source_graph_artifact_id=str(
                request.get("source_graph_artifact_id") or "unknown"
            ),
            question=request.get("question"),
            as_of=str(request.get("as_of") or _now()),
            horizon_days=int(request.get("horizon_days") or 30),
            max_actors=(
                int(raw_max_actors) if raw_max_actors is not None else None
            ),
            market=request.get("market"),
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self.jobs_root / job_id / "job.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self.get(job_id)
            if job is None or job.get("status") in TERMINAL_STATUSES:
                return job
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                return self._update(
                    job_id,
                    status="cancelled",
                    stage="cancelled",
                    finished_at=_now(),
                )
            return self._update(
                job_id,
                status="cancelling",
                stage="cancel_requested",
            )

    def resume(self, job_id: str) -> dict[str, Any] | None:
        """Resume a failed live job from its durable workflow checkpoints."""

        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            if job.get("status") != "failed":
                raise ValueError("only failed simulation jobs can be resumed")
            job_dir = self.jobs_root / job_id
            if not (job_dir / "spec.json").exists():
                raise ValueError("simulation job has no durable specification")
            if self.mode == "live" and not (job_dir / "run" / "state.json").exists():
                raise ValueError("simulation job has no durable workflow checkpoint")
            resumed = self._update(
                job_id,
                status="queued",
                stage="resuming",
                error=None,
                result=None,
                finished_at=None,
            )
            cancel_event = threading.Event()
            self._cancel_events[job_id] = cancel_event
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
            return resumed

    def _is_cancelled(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        return bool(event and event.is_set())

    def _find_reusable(self, reuse_key: str, *, completed: bool = True) -> dict[str, Any] | None:
        reusable_statuses = {"queued", "running"}
        if completed:
            reusable_statuses.update({"completed", "partial"})
        for path in sorted(self.jobs_root.glob("simjob_*/job.json"), reverse=True):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("reuse_key") == reuse_key and job.get("status") in reusable_statuses:
                return job
        return None

    def _execute(self, job_id: str) -> None:
        from .workflow import WorkflowCancelled

        try:
            self._update(
                job_id,
                status="running",
                stage="compiling_spec",
                progress=0.05,
                started_at=_now(),
            )
            spec = self._read_json(self.jobs_root / job_id / "spec.json")
            if self.mode == "stub":
                self._update(job_id, stage="simulating", progress=0.55)
                result = self._stub_result(job_id, spec)
            else:
                result = self._live_result(job_id, spec)
            terminal_status = str(
                result.get("job", {}).get("status") or "completed"
            )
            if terminal_status not in {"completed", "partial"}:
                terminal_status = "completed"
            self._update(
                job_id,
                status=terminal_status,
                stage=terminal_status,
                progress=1.0,
                result=result,
                finished_at=_now(),
            )
        except WorkflowCancelled:
            self._update(
                job_id,
                status="cancelled",
                stage="cancelled",
                error=None,
                finished_at=_now(),
            )
        except Exception as error:  # noqa: BLE001
            self._update(
                job_id,
                status="failed",
                stage="failed",
                error=f"{type(error).__name__}: {str(error)[:500]}",
                finished_at=_now(),
            )

    def _live_result(self, job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        from .mirofish_client import MiroFishClient
        from .smoke_profile import SmokeProfile
        from .workflow import SmokeWorkflow

        job_dir = self.jobs_root / job_id
        run_dir = job_dir / "run"
        profile = SmokeProfile(
            rounds=2,
            financial_actor_overlay=True,
            collect_financial_decisions=True,
            financial_actor_ids=tuple(actor["id"] for actor in spec["actors"]),
        )
        workflow = SmokeWorkflow(
            spec,
            run_dir,
            MiroFishClient(self.mirofish_base_url, timeout=180.0),
            profile=profile,
            simulation_data_dirs=[
                Path(
                    os.environ.get(
                        "MIROFISH_LOCAL_DATA_DIR",
                        self.root / ".data" / "local-stack" / "data" / "mirofish",
                    )
                ).expanduser().resolve() / "simulations",
                self.root / ".data" / "mirofish" / "simulations",
                self.root
                / "upstreams"
                / "MiroFish"
                / "backend"
                / "uploads"
                / "simulations",
            ],
            should_cancel=lambda: self._is_cancelled(job_id),
        )
        self._update(job_id, stage="validating", progress=0.1)
        workflow.preflight()
        graph_backend = os.environ.get(
            "FEVER_SIMULATION_GRAPH_BACKEND", "direct"
        ).strip().lower()
        if graph_backend not in {"direct", "zep"}:
            raise ValueError(
                "FEVER_SIMULATION_GRAPH_BACKEND must be direct or zep"
            )
        if graph_backend == "direct":
            self._update(job_id, stage="preparing_direct", progress=0.2)
            workflow.prepare_direct_simulation()
        else:
            self._update(job_id, stage="building_graph", progress=0.2)
            self._generate_ontology_with_retry(job_id, workflow)
            workflow.build_graph(timeout=1800.0)
        self._update(job_id, stage="simulating", progress=0.5)
        simulation_stage = workflow.run_simulation()
        self._update(job_id, stage="compiling_scenarios", progress=0.85)
        financial_actions = self._read_json(
            run_dir / "financial-actions.json"
        )
        decision_actor_ids = {
            str(item.get("actor_id"))
            for item in financial_actions.get("decisions", [])
            if item.get("actor_id")
        }
        if len(decision_actor_ids) < 2:
            simulation_result = self._read_json(
                run_dir / "simulation-result.json"
            )
            simulation_result["status"] = "partial"
            warnings = simulation_result.setdefault("warnings", [])
            warnings.append(
                "情景编译已跳过：至少需要两个参与方产出有效结构化决策；"
                f"本次仅有 {len(decision_actor_ids)} 个。已保留模拟过程供排查，"
                "未生成或伪造多主体情景。"
            )
            simulation_stage["scenario_compilation"] = {
                "status": "skipped_insufficient_decisions",
                "required_actor_count": 2,
                "valid_actor_count": len(decision_actor_ids),
            }
            return self._artifact_payload(
                job_id,
                spec,
                simulation_result,
                simulation_stage,
                job_status="partial",
            )
        scenario_path = job_dir / "scenario-branches.json"
        updated_result_path = job_dir / "simulation-result.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root / "src")
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.root / "scripts" / "generate_scenario_branches.py"),
                str(job_dir / "spec.json"),
                "--simulation-result",
                str(run_dir / "simulation-result.json"),
                "--financial-actions",
                str(run_dir / "financial-actions.json"),
                "--output",
                str(scenario_path),
                "--updated-result",
                str(updated_result_path),
                "--benchmark-id",
                f"fever-integration-{job_id}",
                "--generation-audit",
                str(job_dir / "generation-audit.jsonl"),
                "--max-semantic-retries",
                "1",
                "--allow-billable",
                "--product",
                "--product-version",
                self.product_version,
            ],
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 360
        while process.poll() is None:
            if self._is_cancelled(job_id):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                from .workflow import WorkflowCancelled

                raise WorkflowCancelled("simulation job was cancelled")
            if time.monotonic() >= deadline:
                process.terminate()
                raise TimeoutError("scenario compiler did not finish in 360s")
            time.sleep(0.5)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "scenario compiler failed: "
                + (stderr or stdout)[-500:]
            )
        simulation_result = self._read_json(updated_result_path)
        scenario_status = "completed" if simulation_result.get("scenarios") else "partial_no_paths"
        simulation_stage["scenario_compilation"] = {"status": scenario_status, "compiler_version": self.product_version}
        return self._artifact_payload(
            job_id, spec, simulation_result, simulation_stage,
            job_status="completed" if scenario_status == "completed" else "partial",
        )

    def _generate_ontology_with_retry(
        self,
        job_id: str,
        workflow: Any,
        *,
        max_attempts: int = 2,
        retry_delay: float = 2.0,
    ) -> Any:
        """Retry one transient upstream ontology failure without rerunning later stages."""

        from .mirofish_client import MiroFishApiError
        from .workflow import WorkflowCancelled

        for attempt in range(1, max_attempts + 1):
            try:
                return workflow.generate_ontology()
            except MiroFishApiError as error:
                retryable = error.status is not None and 500 <= error.status < 600
                if not retryable or attempt >= max_attempts:
                    raise
                self._update(
                    job_id,
                    stage="retrying_ontology",
                    progress=0.2,
                )
                deadline = time.monotonic() + retry_delay
                while time.monotonic() < deadline:
                    if self._is_cancelled(job_id):
                        raise WorkflowCancelled("simulation job was cancelled")
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                self._update(job_id, stage="building_graph", progress=0.2)

        raise RuntimeError("ontology retry loop exited unexpectedly")

    def _stub_result(self, job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        actors = spec["actors"]
        facts = spec["facts"]
        actor_ids = [actor["id"] for actor in actors]
        scenario = {
            "id": "branch-1",
            "label": "关键参与方响应分支",
            "summary": (
                f"{actors[0]['label']}在新信息下调整公开行动，"
                f"{actors[1]['label']}随后采取约束或回应措施。"
            ),
            "run_count": 1,
            "frequency": 1.0,
            "probability_semantics": "uncalibrated_simulation_frequency",
            "triggers": [facts[0]["statement"][:200]],
            "consequences": ["相关参与方重新评估行动与风险敞口"],
            "invalidation_conditions": ["关键事实被后续可靠证据否定"],
            "actor_ids": actor_ids[:2],
            "evidence_refs": [facts[0]["id"]],
            "simulation_refs": ["stub-decision-1", "stub-decision-2"],
            "novelty_claim": "开发模式产物，仅用于验证FEVER接入与展示。",
            "confidence": 0.5,
            "confidence_semantics": "branch_coherence_not_forecast_probability",
        }
        simulation_result = {
            "schema_version": "0.1.0",
            "case_id": spec["case_id"],
            "spec_sha256": canonical_sha256(spec),
            "generated_at": _now(),
            "status": "partial",
            "runs": [
                {
                    "run_id": f"stub_{job_id}",
                    "seed": 0,
                    "status": "completed",
                    "events": [],
                    "warnings": ["stub gateway mode"],
                }
            ],
            "scenarios": [scenario],
            "forecast_target_results": [],
            "simulation_graph": {"nodes": [], "edges": []},
            "warnings": ["开发模式：未调用模型，不可用于金融判断。"],
            "provenance": {
                "engine": "FEVER-MiroFish gateway stub",
                "engine_version": "0.1.0",
                "model": "none",
                "raw_run_artifacts": [],
            },
        }
        stage = {
            "expected_entities_count": len(actors),
            "autonomous_actions_count": 0,
            "active_actor_ids": actor_ids[:2],
            "financial_actions": {
                "decision_count": 2,
                "failure_count": 0,
            },
        }
        return self._artifact_payload(job_id, spec, simulation_result, stage)

    def _artifact_payload(
        self,
        job_id: str,
        spec: dict[str, Any],
        simulation_result: dict[str, Any],
        simulation_stage: dict[str, Any],
        *,
        job_status: str = "completed",
    ) -> dict[str, Any]:
        financial = simulation_stage.get("financial_actions") or {}
        scenarios = prepare_scenarios(spec, simulation_result)
        covered = {actor_id for scenario in scenarios for actor_id in scenario.get("actor_ids", [])}
        omitted = [actor["label"] for actor in spec["actors"] if actor["id"] not in covered]
        _, interaction_summary = compact_interactions(simulation_result)
        return {
            "schema_version": "0.1.0",
            "job": {
                "id": job_id,
                "status": job_status,
                "mode": "quick",
            },
            "source": {
                "case_id": spec["case_id"],
                "evidence_graph_artifact_id": spec["provenance"][
                    "source_graph_artifact_id"
                ],
                "simulation_spec_sha256": canonical_sha256(spec),
                "as_of": spec["as_of"],
                "question": spec["question"],
                "horizon_days": spec["horizon"]["value"],
            },
            "execution": {
                "configured_actor_count": len(spec["actors"]),
                "actor_selection": spec["provenance"].get(
                    "actor_selection", {}
                ),
                "configured_actors": [
                    {
                        "id": actor["id"],
                        "label": actor["label"],
                        "kind": actor["kind"],
                        "selection_reason": actor.get("selection_reason", ""),
                        "focus": actor["goals"][0],
                    }
                    for actor in spec["actors"]
                ],
                "active_actor_counts": [
                    len(simulation_stage.get("active_actor_ids") or [])
                ],
                "prepared_entity_counts": [
                    int(simulation_stage.get("expected_entities_count") or 0)
                ],
                "replication_count": 1,
                "valid_decision_count": int(
                    financial.get("decision_count") or 0
                ),
                "decision_failure_count": int(
                    financial.get("failure_count") or 0
                ),
                "autonomous_action_count": int(
                    simulation_stage.get("autonomous_actions_count") or 0
                ),
                "scenario_actor_count": len(covered),
                "interaction_input": interaction_summary,
                "scenario_compilation": simulation_stage.get(
                    "scenario_compilation", {"status": "completed"}
                ),
                "graph_backend": simulation_stage.get("graph_backend", "zep"),
                "zep_bypassed": bool(simulation_stage.get("zep_bypassed")),
            },
            "scenarios": scenarios,
            "evidence": spec["facts"],
            "notices": self._input_notices(spec) + ([f"本次情景尚未包含这些参与方：{'、'.join(omitted)}。复核时请留意是否遗漏其影响。"] if omitted and scenarios else []),
            "probability_calibration": {
                "available": False,
                "reason": "quick mode is scenario-only; B3 requires three replications and structured B1",
            },
            "simulation_graph": simulation_result.get("simulation_graph")
            or {"nodes": [], "edges": []},
            "warnings": simulation_result.get("warnings") or [],
            "audit": {
                "model": simulation_result.get("provenance", {}).get("model"),
                "gateway_mode": self.mode,
            },
        }

    @staticmethod
    def _input_notices(spec: dict[str, Any]) -> list[str]:
        timing = spec.get("provenance", {}).get("evidence_timing", {})
        notices = []
        if timing.get("excluded_future_ids"):
            notices.append(f"已排除 {len(timing['excluded_future_ids'])} 条晚于分析截止时间的证据。")
        if timing.get("unknown_time_ids"):
            notices.append(f"有 {len(timing['unknown_time_ids'])} 条证据缺少时间记录，暂按本次截止时间使用；历史回放前请核实。")
        facts = spec.get("facts", [])
        if len(facts) == 1 and len(str(facts[0].get("statement", ""))) < 200:
            notices.append("当前只有一条简短证据。建议补充公告正文、关键条款或数值，再复核情景是否适用。")
        return notices

    def _update(self, job_id: str, **patch: Any) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.update(patch)
            job["updated_at"] = _now()
            self._write_job(job)
            return job

    def _write_job(self, job: dict[str, Any]) -> None:
        self._write_json(
            self.jobs_root / job["job_id"] / "job.json", job
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
