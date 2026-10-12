#!/usr/bin/env python3
"""Serve dist-test over HTTP and drive the real page with Playwright, printing failures verbatim."""
import os, socketserver, sys, threading, time, http.server, functools, json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / os.environ.get("SMOKE_DIST", "dist-test")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".js": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".json": "application/json", ".data": "application/octet-stream"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache"); super().end_headers()
    def log_message(self, *a): pass


def main():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(DIST)))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page()
        logs = []
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"[:400]))
        pg.on("pageerror", lambda e: logs.append("[pageerror] " + str(e)[:400]))
        pg.on("requestfailed", lambda r: logs.append(f"[reqfail] {r.url[-120:]} {r.failure}"))
        pg.goto(f"http://127.0.0.1:{port}/index.html")
        for _ in range(30):
            time.sleep(1)
            try:
                if pg.evaluate("self.crossOriginIsolated"): break
            except Exception:
                pass
        print("isolated:", pg.evaluate("self.crossOriginIsolated"))
        pg.wait_for_function("document.querySelectorAll('#example option').length > 0", timeout=30000)
        print("examples:", pg.evaluate("document.querySelectorAll('#example option').length"))

        # Intercept: check whether the graph load itself succeeds by watching network.
        pg.select_option("#backend", "wasm")
        pg.click("#loadBtn")
        st = ""
        for _ in range(600):
            time.sleep(1)
            try:
                st = pg.evaluate("document.querySelector('#status').textContent") or ""
            except Exception:
                continue
            if st.startswith("Ready.") or st.startswith("Could not load") or st.startswith("No model"):
                break
        print("STATUS:", st[:400])
        print("--- console ---")
        for l in logs[-30:]:
            print("  " + l)
        if st.startswith("Ready."):
            pg.click("#runBtn")
            for _ in range(120):
                time.sleep(1)
                if pg.evaluate("document.querySelector('#status').textContent") == "Done.":
                    break
            data = json.loads(pg.evaluate("document.querySelector('#raw').textContent"))
            print("violations:", len(data["violations"]), "latency:", round(data["latency_ms"], 1))
            for v in data["violations"][:5]:
                print(f"  {v['criterion']} ({v['level']}) {v['description']}: {v['probability']:.3f}")
            print("table rows:", pg.evaluate("document.querySelectorAll('#violRows tr').length"))
            print("SMOKE PASSED")
        else:
            print("SMOKE FAILED")


if __name__ == "__main__":
    main()
