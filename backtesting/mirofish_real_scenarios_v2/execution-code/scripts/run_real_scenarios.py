#!/usr/bin/env python3
"""Run named public cases through the real quick gateway with one shared cap.

The loopback relay bounds and records every provider request, including child
OASIS and compiler processes. It neither modifies scenario prompts nor scores
them. Results and code snapshots are kept separately for each run label.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".data/real-scenarios-v1"
ARTIFACTS = ROOT / "artifacts/real-scenarios-v1"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
sys.path.insert(0, str(ROOT / "src"))
from fever_mirofish.gateway import SimulationGateway


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Relay:
    def __init__(self, key, manifest):
        self.key = key
        self.local_key = uuid.uuid4().hex
        self.manifest = manifest
        self.lock = threading.RLock()
        self.case = None
        self.run_label = None
        self.path = DATA / "budget.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"attempts": 0, "settled_usd": 0, "pending": {}}
        if self.state["pending"]:
            raise RuntimeError("unresolved provider reservations; inspect before resuming")

    def reserve(self, payload):
        if payload.get("model") != MODEL or payload.get("stream"):
            raise ValueError("only the reviewed non-streaming model is allowed")
        output_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens") or 5000
        if not isinstance(output_tokens, int) or not 1 <= output_tokens <= 5000:
            raise ValueError("output token limit exceeds the reviewed scope")
        payload["max_tokens"] = output_tokens
        payload.pop("max_completion_tokens", None)
        raw = json.dumps(payload, ensure_ascii=False).encode()
        if len(raw) > self.manifest["max_request_bytes"]:
            raise ValueError("request exceeds the reviewed byte limit")
        reserved = ((len(raw) + 4096) * .135 + output_tokens * .27) / 1_000_000
        with self.lock:
            if self.state.get("halted") or self.state["attempts"] >= self.manifest["max_calls"]:
                raise ValueError("experiment call budget exhausted or halted")
            if self.state["settled_usd"] + sum(self.state["pending"].values()) + reserved > self.manifest["max_cost_usd"]:
                raise ValueError("experiment dollar budget exhausted")
            request_id = uuid.uuid4().hex
            self.state["attempts"] += 1
            self.state["pending"][request_id] = reserved
            write(self.path, self.state)
            write(DATA / "requests" / (request_id + ".json"), {"case_id": self.case, "run_label": self.run_label, "payload": payload})
            return request_id, raw, reserved

    def complete(self, request_id, reserved, code, body, elapsed):
        try:
            response = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            response = {"non_json_response": True}
        usage = response.get("usage") or {}
        known = all(isinstance(usage.get(key), int) for key in ("prompt_tokens", "completion_tokens"))
        estimate = usage.get("estimated_cost")
        charged = max((usage["prompt_tokens"] * .135 + usage["completion_tokens"] * .27) / 1_000_000, estimate if isinstance(estimate, (int, float)) else 0) if known else reserved
        record = {"request_id": request_id, "case_id": self.case, "run_label": self.run_label, "http_status": code, "elapsed_seconds": elapsed, "usage": usage, "usage_known": known, "reserved_usd": reserved, "charged_budget_usd": charged}
        with self.lock:
            write(DATA / "responses" / (request_id + ".json"), response)
            with (DATA / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.state["pending"].pop(request_id)
            self.state["settled_usd"] += charged
            if charged > reserved + 1e-6:
                self.state["halted"] = True
            write(self.path, self.state)

    def handler(self):
        relay = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions" or self.headers.get("Authorization") != "Bearer " + relay.local_key:
                    self.send_error(403)
                    return
                request_id = None
                started = time.monotonic()
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= relay.manifest["max_request_bytes"]:
                        raise ValueError("invalid request length")
                    payload = json.loads(self.rfile.read(length))
                    request_id, raw, reserved = relay.reserve(payload)
                    request = Request(ENDPOINT, raw, {"Authorization": "Bearer " + relay.key, "Content-Type": "application/json"})
                    # No relay retries; SDK retries must obtain a fresh reservation.
                    with urlopen(request, timeout=175) as response:
                        body, code = response.read(), response.status
                except HTTPError as error:
                    body, code = error.read(), error.code
                except Exception as error:
                    body = json.dumps({"error": {"message": type(error).__name__ + ": request failed or exceeded experiment bounds", "type": "experiment_error"}}).encode()
                    code = 400 if request_id is None else 502
                if request_id:
                    relay.complete(request_id, reserved, code, body, time.monotonic() - started)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        return Handler


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def execute(args):
    from dotenv import dotenv_values
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    credentials = dotenv_values(ROOT / ".env")
    if (credentials.get("LLM_BASE_URL") or "").rstrip("/") != "https://api.deepinfra.com/v1/openai":
        raise RuntimeError("provider differs from the reviewed DeepInfra endpoint")
    key = credentials.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("existing provider key unavailable")
    relay = Relay(key, manifest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), relay.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    workspace = DATA / "workspaces" / args.label
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("scripts", "src", "upstreams"):
        path = workspace / name
        if not path.exists():
            path.symlink_to(ROOT / name, target_is_directory=True)
    port = unused_port()
    simulation_data = DATA / "mirofish" / args.label
    proxy_base = f"http://127.0.0.1:{server.server_port}/v1"
    env = os.environ.copy()
    env.update({"LLM_BASE_URL": proxy_base, "ARK_API_URL": proxy_base, "LLM_API_KEY": relay.local_key, "ARK_API_KEY": relay.local_key, "LLM_MODEL_NAME": MODEL, "ARK_MODEL": MODEL, "MIROFISH_LOCAL_HOST": "127.0.0.1", "MIROFISH_LOCAL_PORT": str(port), "MIROFISH_LOCAL_DATA_DIR": str(simulation_data), "FEVER_MIROFISH_HOST_LEDGER_PATH": str(DATA / f"host-{args.label}-usage.jsonl"), "FEVER_MIROFISH_MAX_LLM_CALLS": "192", "FEVER_MIROFISH_MAX_TOTAL_TOKENS": "1000000", "FEVER_MIROFISH_OASIS_MAX_LLM_CALLS": "48", "FEVER_MIROFISH_OASIS_MAX_TOTAL_TOKENS": "250000", "FEVER_SIMULATION_GRAPH_BACKEND": "direct", "PYTHONPYCACHEPREFIX": "/tmp/fever-real-scenarios-pycache"})
    # Upstream Config otherwise reloads .env with override=True, bypassing the
    # relay. The installed python-dotenv supports this explicit switch.
    env.update({"PYTHON_DOTENV_DISABLED": "1", "ZEP_API_KEY": "unused-direct-mode", "OPENAI_API_KEY": relay.local_key, "OPENAI_API_BASE_URL": proxy_base})
    # The gateway compiler inherits these explicit local relay settings.
    os.environ.update({key: value for key, value in env.items() if key.startswith(("LLM_", "ARK_", "MIROFISH_", "FEVER_"))})
    snapshot = DATA / "execution-code" / args.label
    for source in [*sorted((ROOT / "src/fever_mirofish").glob("*.py")), ROOT / "scripts/generate_scenario_branches.py", ROOT / "scripts/run_real_scenarios.py"]:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError("run label already has different code; preserve it and use another label")
        target.write_bytes(source.read_bytes())
    log = (DATA / f"mirofish-{args.label}.log").open("a")
    backend = subprocess.Popen([str(ROOT / "upstreams/MiroFish/backend/.venv/bin/python"), str(ROOT / "scripts/run_mirofish_local.py")], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    gateway = None
    try:
        for _ in range(60):
            if backend.poll() is not None:
                raise RuntimeError("isolated MiroFish did not start; inspect its local log")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except OSError:
                time.sleep(.5)
        else:
            raise RuntimeError("isolated MiroFish startup timed out")
        gateway = SimulationGateway(workspace, mode="live", mirofish_base_url=f"http://127.0.0.1:{port}")
        for case in manifest["cases"]:
            if args.case and case["id"] not in args.case:
                continue
            directory = ARTIFACTS / "cases" / case["id"]
            request_path = directory / "request.json"
            if hashlib.sha256(request_path.read_bytes()).hexdigest() != case["request_sha256"]:
                raise RuntimeError("registered public case input changed")
            destination = directory / args.label
            if (destination / "job.json").exists():
                print(f"sealed {case['id']} {args.label}: reuse", flush=True)
                continue
            relay.case, relay.run_label = case["id"], args.label
            write(destination / "preview.json", gateway.preview(json.loads(request_path.read_text())))
            before = time.monotonic()
            job = gateway.create(json.loads(request_path.read_text()))
            print(f"started {case['id']} {args.label}: {job['job_id']}", flush=True)
            write(DATA / "runner-state.json", {"case_id": case["id"], "run_label": args.label, "job_id": job["job_id"], "mirofish_pid": backend.pid})
            last_stage = None
            while True:
                current = gateway.get(job["job_id"])
                if current["stage"] != last_stage:
                    last_stage = current["stage"]
                    print(f"{case['id']}: {last_stage}", flush=True)
                if current["status"] in {"completed", "partial", "failed", "cancelled"}:
                    break
                if time.monotonic() - before > 1500:
                    gateway.cancel(job["job_id"])
                    raise RuntimeError("case exceeded the 25-minute wall-time limit")
                time.sleep(2)
            write(destination / "job.json", current)
            write(destination / "timing.json", {"end_to_end_seconds": time.monotonic() - before, "job_dir": str(gateway.jobs_root / job["job_id"]), "model": MODEL, "actual_simulation_rerun": True})
            print(f"finished {case['id']}: {current['status']}; budget=${relay.state['settled_usd']:.6f}", flush=True)
    finally:
        if gateway:
            for job_id, future in gateway._futures.items():
                if not future.done():
                    gateway.cancel(job_id)
            gateway._executor.shutdown(wait=True)
        backend.terminate()
        try:
            backend.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(backend.pid, signal.SIGTERM)
            backend.wait(timeout=10)
        server.shutdown()
        server.server_close()
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--case", action="append", choices=("byd-h1-2025", "guotai-haitong-2024", "crowdstrike-outage-2024"))
    args = parser.parse_args()
    if not args.label.replace("-", "").isalnum():
        raise ValueError("invalid run label")
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == "__main__":
    main()
