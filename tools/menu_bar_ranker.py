#!/usr/bin/env python3
"""
menu_bar_ranker.py — macOS menu bar app for live_scanner rankings.

Reads the latest scan cycle from ~/video-pipeline/logs/live_scanner/*.json
and shows the top 5 tickers (by urgency) in a dropdown. Clicking a ticker
opens its TradingView chart. Also serves the scanner output dir over
127.0.0.1:8765 so the dashboard.html renders with relative assets.
"""

from __future__ import annotations

import glob
import http.server
import json
import os
import socketserver
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import rumps

SCAN_DIR = Path.home() / "video-pipeline" / "logs" / "live_scanner"
SERVER_PORT = 8765
SERVER_HOST = "127.0.0.1"
REFRESH_SECONDS = 60
TOP_N = 5


# ---------- local HTTP server so dashboard.html loads cleanly ---------- #

class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        return  # keep logs quiet


def _start_local_server() -> None:
    os.chdir(str(SCAN_DIR))
    httpd = socketserver.TCPServer((SERVER_HOST, SERVER_PORT), _SilentHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()


# ---------- scan file helpers ---------- #

def _latest_json_path() -> Path | None:
    files = sorted(glob.glob(str(SCAN_DIR / "*.json")))
    files = [f for f in files if "_diag" not in os.path.basename(f)
             and "_session" not in os.path.basename(f)]
    return Path(files[-1]) if files else None


def _load_latest_cycle() -> tuple[list[dict], str | None]:
    """Return (results_list, timestamp_iso) for the most recent cycle."""
    path = _latest_json_path()
    if path is None:
        return [], None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return [], None
    if not isinstance(data, list) or not data:
        return [], None
    last = data[-1]
    results = last.get("results", []) or []
    results = sorted(results, key=lambda r: -float(r.get("urgency", 0) or 0))
    return results, last.get("timestamp")


def _format_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%H:%M ET")
    except Exception:
        return ts[-8:]


# ---------- menu bar app ---------- #

class RankerApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("🎯 —", quit_button=None)
        self._rows: list[rumps.MenuItem] = []
        self._last_ts: str | None = None
        self._build_static_menu()
        self.refresh(None)
        # rumps.Timer auto-starts
        self.timer = rumps.Timer(self.refresh, REFRESH_SECONDS)
        self.timer.start()

    def _build_static_menu(self) -> None:
        self.refresh_item = rumps.MenuItem("🔄 Refresh now", callback=self.refresh)
        self.dashboard_item = rumps.MenuItem(
            "📊 Open full dashboard", callback=self.open_dashboard
        )
        self.updated_item = rumps.MenuItem("Last updated: —")
        self.updated_item.set_callback(None)  # disabled/info row
        self.quit_item = rumps.MenuItem("❌ Quit", callback=rumps.quit_application)
        self._rebuild_menu([])

    def _rebuild_menu(self, results: list[dict]) -> None:
        self.menu.clear()
        self._rows = []
        if not results:
            empty = rumps.MenuItem("(no picks in latest cycle)")
            empty.set_callback(None)
            self.menu.add(empty)
        else:
            for r in results[:TOP_N]:
                ticker = str(r.get("ticker", "?"))
                urgency = float(r.get("urgency", 0) or 0)
                signal = str(r.get("signal", "")).strip()
                label = f"{ticker} · {urgency:.1f}"
                if signal and signal.upper() not in {"WAIT", ""}:
                    label += f"  [{signal}]"
                item = rumps.MenuItem(label, callback=self._make_open_chart(ticker))
                self._rows.append(item)
                self.menu.add(item)
        self.menu.add(rumps.separator)
        self.menu.add(self.refresh_item)
        self.menu.add(self.dashboard_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.updated_item)
        self.menu.add(rumps.separator)
        self.menu.add(self.quit_item)

    def _make_open_chart(self, ticker: str):
        def _cb(_sender) -> None:
            url = f"https://www.tradingview.com/chart/?symbol=NASDAQ:{ticker}"
            webbrowser.open(url)
        return _cb

    # ---- menu callbacks ---- #

    def refresh(self, _sender) -> None:
        results, ts = _load_latest_cycle()
        self._last_ts = ts
        count = len(results)
        self.title = f"🎯 {count} picks" if count else "🎯 —"
        self._rebuild_menu(results)
        self.updated_item.title = f"Last updated: {_format_ts(ts)}"

    def open_dashboard(self, _sender) -> None:
        url = f"http://{SERVER_HOST}:{SERVER_PORT}/dashboard.html"
        webbrowser.open(url)


def main() -> None:
    try:
        _start_local_server()
    except OSError:
        # port already in use — another instance probably running; keep going
        pass
    RankerApp().run()


if __name__ == "__main__":
    main()
