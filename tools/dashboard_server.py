#!/usr/bin/env python3
"""
dashboard_server.py — tiny static HTTP server for the live_scanner dashboard.

Serves ~/video-pipeline/logs/live_scanner/ on 127.0.0.1:8765.
Bound to loopback; cloudflared tunnels it to a public trycloudflare.com URL.
"""

from __future__ import annotations

import http.server
import os
import socketserver
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8765
SERVE_DIR = Path.home() / "video-pipeline" / "logs" / "live_scanner"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # No-cache so iPhone Safari always fetches the freshest dashboard
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(str(SERVE_DIR))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
        print(f"Serving {SERVE_DIR} on http://{HOST}:{PORT}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
