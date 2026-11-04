"""Serve the web app locally without browser caching, so a refresh always picks up changes.

    python scripts/serve.py          # http://localhost:8000
    python scripts/serve.py 8765
"""

import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("localhost", port), functools.partial(NoCacheHandler, directory=str(WEB)))
    print(f"Beyond Words running at http://localhost:{port}")
    server.serve_forever()
