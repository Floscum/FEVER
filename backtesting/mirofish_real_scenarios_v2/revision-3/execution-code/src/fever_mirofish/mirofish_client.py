"""Small stdlib client for the upstream MiroFish HTTP API."""

from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import OpenerDirector, Request, build_opener


class MiroFishApiError(RuntimeError):
    """A bounded error that does not expose request bodies or credentials."""

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class MiroFishClient:
    """HTTP adapter that keeps upstream-specific routes out of FEVER code."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5001",
        *,
        timeout: float = 30.0,
        opener: Optional[OpenerDirector] = None,
    ):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = opener or build_opener()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Mapping[str, Any]] = None,
        raw_body: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        body = raw_body
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = f"MiroFish API returned HTTP {error.code}"
            try:
                error_payload = json.loads(error.read().decode("utf-8"))
                public_error = error_payload.get("error")
                if isinstance(public_error, str) and public_error:
                    detail += f": {public_error[:300]}"
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
            raise MiroFishApiError(detail, status=error.code) from error
        except URLError as error:
            raise MiroFishApiError(
                f"cannot reach MiroFish API at {self.base_url}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MiroFishApiError("MiroFish API returned invalid JSON") from error

        if not isinstance(payload, dict):
            raise MiroFishApiError("MiroFish API returned a non-object JSON payload")
        if payload.get("success") is False:
            message = payload.get("error") or "MiroFish API reported failure"
            raise MiroFishApiError(str(message)[:300])
        return payload

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def generate_ontology(
        self,
        seed_path: Path,
        *,
        simulation_requirement: str,
        project_name: str,
        additional_context: str = "",
    ) -> Dict[str, Any]:
        seed_path = Path(seed_path)
        if seed_path.suffix.lower() not in {".md", ".markdown", ".txt", ".pdf"}:
            raise ValueError("MiroFish seed must be Markdown, TXT, or PDF")
        file_bytes = seed_path.read_bytes()
        boundary = f"----fever-mirofish-{uuid.uuid4().hex}"
        fields = {
            "simulation_requirement": simulation_requirement,
            "project_name": project_name,
            "additional_context": additional_context,
        }
        chunks = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    (
                        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    ).encode(),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        content_type = mimetypes.guess_type(seed_path.name)[0] or "text/plain"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="files"; '
                    f'filename="{seed_path.name}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                file_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        return self._request(
            "POST",
            "/api/graph/ontology/generate",
            raw_body=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def build_graph(
        self,
        project_id: str,
        *,
        graph_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        force: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "force": force,
        }
        if graph_name:
            payload["graph_name"] = graph_name
        return self._request("POST", "/api/graph/build", json_body=payload)

    def get_graph_task(self, task_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/graph/task/{task_id}")

    def get_project(self, project_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/graph/project/{project_id}")

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/graph/data/{graph_id}")

    def create_simulation(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/simulation/create", json_body=payload)

    def prepare_simulation(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/simulation/prepare", json_body=payload)

    def get_prepare_status(
        self,
        simulation_id: str,
        *,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {"simulation_id": simulation_id}
        if task_id:
            payload["task_id"] = task_id
        return self._request(
            "POST",
            "/api/simulation/prepare/status",
            json_body=payload,
        )

    def start_simulation(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/simulation/start", json_body=payload)

    def get_run_status(self, simulation_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/simulation/{simulation_id}/run-status")

    def get_env_status(self, simulation_id: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/simulation/env-status",
            json_body={"simulation_id": simulation_id},
        )

    def close_env(self, simulation_id: str, *, timeout: int = 30) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/simulation/close-env",
            json_body={"simulation_id": simulation_id, "timeout": timeout},
        )

    def batch_interview(
        self,
        simulation_id: str,
        interviews: list[Mapping[str, Any]],
        *,
        platform: str = "reddit",
        timeout: int = 180,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/simulation/interview/batch",
            json_body={
                "simulation_id": simulation_id,
                "interviews": interviews,
                "platform": platform,
                "timeout": timeout,
            },
        )

    def get_actions(
        self,
        simulation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        query: Dict[str, Any] = {"limit": limit, "offset": offset}
        if platform:
            query["platform"] = platform
        return self._request(
            "GET",
            f"/api/simulation/{simulation_id}/actions?{urlencode(query)}",
        )
