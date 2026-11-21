"""Serve one local read-only occupancy dashboard."""

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .state import (
    HISTORY_LIMIT,
    SCHEMA_VERSION,
    Geometry,
    StateError,
    current_status,
    read_history,
    read_snapshot,
)

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
    "/freshness.js": ("freshness.js", "text/javascript; charset=utf-8"),
}


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, port, directory, geometry=None, *, clock=None):
        self.directory = directory
        self.geometry = geometry
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.assets = {
            route: ((Path(__file__).parent / "static" / filename).read_bytes(), media)
            for route, (filename, media) in ASSETS.items()
        }
        super().__init__(("127.0.0.1", port), DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ParkVisionDashboard"
    sys_version = ""

    def log_message(self, *_):
        pass

    def respond(self, status, body, media):
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        if status == 405:
            self.send_header("Allow", "GET, HEAD")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json_result(self, status, result):
        self.respond(
            status,
            json.dumps(result, ensure_ascii=False, allow_nan=False).encode(),
            "application/json; charset=utf-8",
        )

    def do_GET(self):
        try:
            port = self.server.server_port
            accepted = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if port == 80:
                accepted.update(("127.0.0.1", "localhost"))
            if self.headers.get("Host") not in accepted:
                raise StateError("invalid_host", 403)
            url = urlsplit(self.path)
            if url.netloc or url.scheme or url.fragment:
                raise StateError("invalid_request", 400)
            if url.path in self.server.assets and not url.query:
                body, media = self.server.assets[url.path]
                self.respond(200, body, media)
                return
            self.json_result(200, self.resource(url))
        except ValueError as exc:
            error = (
                exc
                if isinstance(exc, StateError)
                else StateError("invalid_request", 400)
            )
            self.json_result(
                error.status, {"schema_version": SCHEMA_VERSION, "error": error.code}
            )

    do_HEAD = do_GET

    def method_not_allowed(self):
        self.json_result(
            405, {"schema_version": SCHEMA_VERSION, "error": "method_not_allowed"}
        )

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = method_not_allowed

    def resource(self, url):
        if url.path == "/api/v1/status" and not url.query:
            data = read_snapshot(self.server.directory / "snapshot.json")
            result = current_status(data, self.server.clock())
            geometry = self.server.geometry
            result["geometry"] = geometry.for_snapshot(data) if geometry else None
            return result
        if url.path == "/api/v1/health" and not url.query:
            return {"schema_version": SCHEMA_VERSION, "status": "ok"}
        if url.path == "/api/v1/history":
            query = parse_qs(url.query, keep_blank_values=True, max_num_fields=1)
            if not query:
                limit = 24
            elif (
                set(query) != {"limit"}
                or not query["limit"][0].isascii()
                or not query["limit"][0].isdigit()
                or len(query["limit"][0]) > 2
                or not 1 <= (limit := int(query["limit"][0])) <= HISTORY_LIMIT
            ):
                raise StateError("invalid_history_limit", 400)
            return read_history(self.server.directory / "events.jsonl", limit)
        raise StateError("not_found", 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Runner output directory"
    )
    parser.add_argument(
        "--configuration", type=Path, help="Optional approved geometry for the diagram"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1–65535")
    try:
        geometry = Geometry(args.configuration) if args.configuration else None
        server = DashboardServer(args.port, args.output_dir, geometry)
    except ImportError:
        parser.exit(1, "Use the engine environment to supply a configuration.\n")
    except (OSError, ValueError):
        parser.exit(
            1,
            "Cannot start. Check the port and approved configuration.\n",
        )
    with server:
        print(
            f"ParkVision Dashboard: http://127.0.0.1:{server.server_port}/", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
