"""Credential-isolating OpenAI Responses proxy for hardened CORAL agents."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_REQUEST_BYTES = 32 * 1024 * 1024
ALLOWED_PATHS = {
    "/responses": "/api/v1/responses",
    "/v1/responses": "/api/v1/responses",
    "/responses/compact": "/api/v1/responses/compact",
    "/v1/responses/compact": "/api/v1/responses/compact",
}
_AUDIT_LOCK = threading.Lock()


def agent_token(master_token: str, agent_id: str) -> str:
    digest = hmac.new(
        master_token.encode(),
        agent_id.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"sk-frontieror-{agent_id}-{digest}"


class ModelProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    master_token: str
    upstream_api_key: str
    allowed_model: str
    agent_count: int
    audit_path: Path

    def log_message(self, _format: str, *_args) -> None:
        return

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _agent_id(self) -> str | None:
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return None
        supplied = authorization[len(prefix):]
        for index in range(1, self.agent_count + 1):
            candidate = f"agent-{index}"
            if hmac.compare_digest(
                supplied,
                agent_token(self.master_token, candidate),
            ):
                return candidate
        return None

    def _audit(self, payload: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with _AUDIT_LOCK:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok"})
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        agent_id = self._agent_id()
        if agent_id is None:
            self._write_json(401, {"error": "invalid ephemeral model token"})
            return
        upstream_path = ALLOWED_PATHS.get(self.path)
        if upstream_path is None:
            self._write_json(404, {"error": "unsupported model endpoint"})
            return
        if self.headers.get("Transfer-Encoding"):
            self._write_json(400, {"error": "chunked requests are not supported"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            content_length = -1
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            self._write_json(413, {"error": "invalid model request size"})
            return
        request_body = self.rfile.read(content_length)
        request_hash = hashlib.sha256(request_body).hexdigest()
        try:
            request_payload = json.loads(request_body)
        except (TypeError, ValueError):
            self._write_json(400, {"error": "model request must be JSON"})
            return
        if not isinstance(request_payload, dict):
            self._write_json(400, {"error": "model request must be an object"})
            return
        request_payload["model"] = self.allowed_model
        upstream_body = json.dumps(
            request_payload,
            separators=(",", ":"),
        ).encode()

        status = 502
        response_bytes = 0
        response_hash = hashlib.sha256()
        response_started = False
        connection: http.client.HTTPSConnection | None = None
        try:
            connection = http.client.HTTPSConnection(
                "openrouter.ai",
                443,
                timeout=600,
            )
            connection.request(
                "POST",
                upstream_path,
                body=upstream_body,
                headers={
                    "Authorization": f"Bearer {self.upstream_api_key}",
                    "Content-Type": "application/json",
                    "Accept": self.headers.get("Accept", "text/event-stream"),
                    "User-Agent": "FrontierOR-Infra/secure-model-proxy",
                },
            )
            response = connection.getresponse()
            status = response.status
            self.send_response(status)
            content_type = response.getheader("Content-Type")
            if content_type:
                self.send_header("Content-Type", content_type)
            request_id = response.getheader("X-Request-Id")
            if request_id:
                self.send_header("X-Request-Id", request_id)
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            while chunk := response.read1(64 * 1024):
                response_bytes += len(chunk)
                response_hash.update(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionError, OSError, http.client.HTTPException):
            if not response_started and not self.wfile.closed:
                try:
                    self._write_json(502, {"error": "upstream model request failed"})
                except (BrokenPipeError, ConnectionError, OSError):
                    pass
        finally:
            if connection is not None:
                connection.close()
            self._audit(
                {
                    "agent_id": agent_id,
                    "endpoint": self.path,
                    "forced_model": self.allowed_model,
                    "request_sha256": request_hash,
                    "request_bytes": len(request_body),
                    "response_sha256": response_hash.hexdigest(),
                    "response_bytes": response_bytes,
                    "status_code": status,
                }
            )


def main() -> None:
    upstream_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    master_token = os.environ.get("FRONTIER_OR_PROXY_MASTER_TOKEN", "")
    allowed_model = os.environ.get("FRONTIER_OR_ALLOWED_MODEL", "")
    audit_path = Path(
        os.environ.get(
            "FRONTIER_OR_MODEL_AUDIT_PATH",
            "/frontieror/model-audit/requests.jsonl",
        )
    )
    try:
        agent_count = int(os.environ.get("FRONTIER_OR_AGENT_COUNT", "0"))
    except ValueError as exc:
        raise RuntimeError("FRONTIER_OR_AGENT_COUNT must be an integer") from exc
    if not upstream_api_key or not master_token or not allowed_model:
        raise RuntimeError("secure model proxy is missing trusted configuration")
    if not 1 <= agent_count <= 8:
        raise RuntimeError("secure model proxy received an invalid agent count")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    ModelProxyHandler.master_token = master_token
    ModelProxyHandler.upstream_api_key = upstream_api_key
    ModelProxyHandler.allowed_model = allowed_model
    ModelProxyHandler.agent_count = agent_count
    ModelProxyHandler.audit_path = audit_path
    server = ThreadingHTTPServer(("0.0.0.0", 8080), ModelProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
