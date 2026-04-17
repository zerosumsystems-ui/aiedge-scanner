"""Out-of-band alert + notification outputs for the live scanner.

Three sinks:
  - Apple Notes via AppleScript (osascript) — overwrites a named note
    every cycle with the plain-text leaderboard.
  - Console alert — ANSI-colorized per-ticker box printed when a high
    urgency + actionable signal fires.
  - Discord webhook — optional, triggered on actionable alerts when
    `DISCORD_WEBHOOK_URL` is set in the environment.

Extracted from live_scanner.py (Phase 4i).
"""

from __future__ import annotations

import logging
import os
import subprocess

import requests

from aiedge.dashboard.console import _BOLD, _COLORS, _RST

logger = logging.getLogger(__name__)


def update_apple_note(content: str, note_name: str = "Live Scanner") -> None:
    """Write leaderboard to an Apple Note via AppleScript (overwrites each cycle)."""
    escaped = (
        content
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )
    script = f'''
tell application "Notes"
    set targetNote to missing value
    repeat with n in notes of default account
        if name of n is "{note_name}" then
            set targetNote to n
            exit repeat
        end if
    end repeat
    if targetNote is missing value then
        make new note at default account with properties {{name:"{note_name}", body:"{escaped}"}}
    else
        set body of targetNote to "{escaped}"
    end if
end tell
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(f"Apple Notes update failed (rc={result.returncode}): {err}")
        else:
            logger.debug("Apple Notes updated.")
    except subprocess.TimeoutExpired:
        logger.warning("Apple Notes update timed out (10s).")
    except Exception as e:
        logger.warning(f"Apple Notes update error: {e}")


def fire_alert(result: dict) -> None:
    """Print a console alert box for one scored result."""
    sig = result.get("signal", "?")
    ticker = result.get("ticker", "?")
    urg = result.get("urgency", 0.0)
    unc = result.get("uncertainty", 0.0)
    day = result.get("day_type", "?")
    summary = result.get("summary", "")
    warn = result.get("day_type_warning", "")

    color = _COLORS.get(sig, "")
    print(f"\n{_BOLD}{'─'*60}{_RST}")
    print(f"{_BOLD}ALERT: {color}{sig}{_RST}{_BOLD} → {ticker}{_RST}")
    print(f"  Urgency: {urg}  |  Uncertainty: {unc}  |  Day type: {day}")
    print(f"  {summary}")
    if warn and warn.lower() not in ("", "none"):
        print(f"  ⚠  {warn}")
    print(f"{_BOLD}{'─'*60}{_RST}\n")

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return
    try:
        emoji = "🟢" if "BUY" in sig else "🔴"
        payload = {
            "content": (
                f"{emoji} **{sig}** → **{ticker}**\n"
                f"Urgency: `{urg}` | Uncertainty: `{unc}` | Day type: `{day}`\n"
                f"{summary}"
                + (f"\n⚠️ {warn}" if warn and warn.lower() not in ("", "none") else "")
            )
        }
        resp = requests.post(webhook, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            logger.warning(f"Discord webhook returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Discord alert failed: {e}")
