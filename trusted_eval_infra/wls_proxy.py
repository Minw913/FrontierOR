"""Minimal CONNECT proxy with an exact-host allowlist.

This process runs in a trusted sidecar container. Candidate containers are
attached only to an internal Docker network and must use this proxy for
outbound TLS, so they cannot route directly to the internet.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
from collections.abc import Collection


MAX_REQUEST_LINE = 8192
MAX_HEADER_BYTES = 64 * 1024
MAX_HEADER_LINES = 64


def host_allowed(host: str, allowed_hosts: Collection[str]) -> bool:
    normalized = host.rstrip(".").lower()
    return normalized in {item.rstrip(".").lower() for item in allowed_hosts}


def make_handler(allowed_hosts: Collection[str]):
    exact_hosts = frozenset(item.rstrip(".").lower() for item in allowed_hosts)

    class RestrictedConnectHandler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            raw_request = self.rfile.readline(MAX_REQUEST_LINE + 1)
            if len(raw_request) > MAX_REQUEST_LINE:
                self.wfile.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                return
            request = raw_request.decode("latin1", errors="replace").strip()
            parts = request.split()
            if len(parts) != 3 or parts[0].upper() != "CONNECT":
                self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                return

            authority = parts[1]
            if authority.count(":") != 1:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            host, port_text = authority.rsplit(":", 1)
            try:
                port = int(port_text)
            except ValueError:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            if port != 443 or not host_allowed(host, exact_hosts):
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return

            header_bytes = 0
            for _ in range(MAX_HEADER_LINES):
                line = self.rfile.readline(MAX_REQUEST_LINE + 1)
                header_bytes += len(line)
                if len(line) > MAX_REQUEST_LINE or header_bytes > MAX_HEADER_BYTES:
                    self.wfile.write(
                        b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n"
                    )
                    return
                if line in (b"\r\n", b"\n", b""):
                    break
            else:
                self.wfile.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                return

            try:
                upstream = socket.create_connection((host, port), timeout=15)
            except OSError:
                self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.wfile.flush()

            sockets = [self.connection, upstream]
            try:
                while True:
                    readable, _, _ = select.select(sockets, [], [], 120)
                    if not readable:
                        return
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = (
                            upstream if source is self.connection else self.connection
                        )
                        destination.sendall(data)
            finally:
                upstream.close()

    return RestrictedConnectHandler


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-host", action="append", required=True)
    parser.add_argument("--port", type=int, default=3128)
    args = parser.parse_args()
    handler = make_handler(args.allow_host)
    with Server(("0.0.0.0", args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
