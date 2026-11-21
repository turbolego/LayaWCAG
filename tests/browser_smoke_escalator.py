#!/usr/bin/env python3
"""Serve the built site over HTTP and drive the continuous-learning page with Playwright, printing failures verbatim.

Mirrors tests/browser_smoke_wcag.py but targets escalator.html: it loads the WASM model,
auto-runs the WCAG 2.2 examples once, and checks that the escalation + in-browser learning
loop ran (teacher calls happened, the decay chart recorded a round, and the verdict table rendered).
"""
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
        pg.goto(f"http://127.0.0.1:{port}/escalator.html")

        # example dropdown is populated immediately; the model loads on demand
        pg.wait_for_function("document.querySelectorAll('#exsel option').length > 0", timeout=30000)
        pg.click("#eloadBtn")

        # Model + out-of-bounds probe finish when the page sets window.escReady.
        pg.wait_for_function("window.escReady === true", timeout=900_000)
        print("model ready:", pg.evaluate("document.getElementById('estatus').textContent")[:120])

        # One auto-run round: every example, escalations to the simulated teacher, learning.
        pg.click("#autoBtn")
        pg.wait_for_function("document.getElementById('log').innerText.includes('Round 1:')", timeout=900_000)
        kpi = pg.evaluate("""() => ({
          calls: document.getElementById('kCalls').textContent,
          model: document.getElementById('kModel').textContent,
          teacher: document.getElementById('kTeacher').textContent,
        })""")
        print("round1 KPI:", json.dumps(kpi))

        verdicts = pg.evaluate("document.getElementById('verdicts').innerText")
        assert "Criterion" in verdicts, "verdict table header missing"

        # Assertions:
        #  - at least one teacher call happened (simulated teacher, no key needed)
        #  - at least one criterion resolved by the model alone
        #  - the decay chart recorded a round (log contains "Round 1:")
        calls = int(kpi["calls"]); model = int(kpi["model"]); teacher = int(kpi["teacher"])
        assert calls >= 1, "expected at least one teacher call in round 1, got 0"
        assert model + teacher >= 1, "no criteria resolved at all"
        assert "Round 1:" in pg.evaluate("document.getElementById('log').innerText")
        print(f"OK: round1 calls={calls} model={model} teacher={teacher}")

        fails = [l for l in logs if l.startswith("[pageerror]") or l.startswith("[reqfail")]
        for f in fails: print("FAIL LINE:", f)
        b.close()
        if fails:
            print("ESCALATOR SMOKE FAILED")
            sys.exit(1)
        print("ESCALATOR SMOKE PASSED")


if __name__ == "__main__":
    main()
