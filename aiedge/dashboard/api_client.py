"""HTTP client for the aiedge.trade /api/scan endpoint.

Fire-and-forget POST of each scan cycle's ScanPayload JSON. The public
site renders the interactive scanner dashboard from these payloads.

Requires `SYNC_SECRET` in the environment for bearer auth; skipped with
a warning if unset so local-dev scanners don't 401.

Extracted from live_scanner.py (Phase 4h).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime
from typing import Optional

import pandas as pd

from aiedge.dashboard.serializers import _serialize_scan_payload

logger = logging.getLogger(__name__)

# apex redirects 307 → www and urllib drops POST body on redirect
AIEDGE_SCAN_URL = "https://www.aiedge.trade/api/scan"


def _post_to_aiedge(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    intraday_levels: dict,
    first_scan_hour: int,
    first_scan_min: int,
    df5m_map: Optional[dict[str, pd.DataFrame]] = None,
) -> None:
    """POST scan results to aiedge.trade/api/scan. Fire-and-forget in bg thread."""
    try:
        sync_secret = os.environ.get("SYNC_SECRET")
        if not sync_secret:
            logger.warning(
                "aiedge.trade POST skipped: SYNC_SECRET env var not set. "
                "Export it so the scanner can authenticate with /api/scan."
            )
            return
        payload = _serialize_scan_payload(
            results, now_et, total_symbols, passed, elapsed, interval_min,
            intraday_levels, first_scan_hour, first_scan_min, df5m_map,
        )
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            AIEDGE_SCAN_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sync_secret}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info(f"aiedge.trade POST → {resp.status} ({len(data)//1024}KB)")
    except Exception as e:
        logger.warning(f"aiedge.trade POST failed: {e}")
