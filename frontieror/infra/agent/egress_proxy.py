"""Small CONNECT proxy with a fixed model-service allowlist."""

from __future__ import annotations

import select
import socket
import socketserver


ALLOWED_SUFFIXES = (".openai.com", ".chatgpt.com")
ALLOWED_EXACT = {"openai.com", "chatgpt.com"}


def _allowed(host: str) -> bool:
    host = host.rstrip(".").lower()
    return host in ALLOWED_EXACT or any(host.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request = self.rfile.readline(8192).decode("latin1", errors="replace").strip()
        parts = request.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        authority = parts[1].rsplit(":", 1)
        if len(authority) != 2 or not _allowed(authority[0]):
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            port = int(authority[1])
        except ValueError:
            return
        if port != 443:
            self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        while self.rfile.readline(8192) not in (b"\r\n", b"\n", b""):
            pass
        try:
            upstream = socket.create_connection((authority[0], port), timeout=15)
        except OSError:
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.wfile.flush()
        sockets = [self.connection, upstream]
        try:
            while True:
                readable, _, _ = select.select(sockets, [], [], 60)
                if not readable:
                    break
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.connection else self.connection).sendall(data)
        finally:
            upstream.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server(("0.0.0.0", 3128), Handler) as server:
        server.serve_forever()
