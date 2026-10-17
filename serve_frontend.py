"""Serve the FINDETECT frontend with correct JavaScript MIME types.

On Windows, ``python -m http.server`` can serve ``.js`` as ``text/plain`` because
it consults the registry. Browsers then refuse to execute ES module scripts.
This small server overrides the MIME map and serves the ``frontend/`` folder.

Usage:
    python serve_frontend.py --port 5600
"""

from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[frontend] {self.address_string()} {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the FINDETECT investigator UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5600)
    args = parser.parse_args()

    frontend = Path(__file__).resolve().parent / "frontend"
    if not frontend.is_dir():
        raise SystemExit(f"Frontend directory not found: {frontend}")
    os.chdir(frontend)
    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    print(f"FINDETECT UI: http://{args.host}:{args.port}/")
    print("API expected at http://127.0.0.1:8000 (configurable in Settings).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
