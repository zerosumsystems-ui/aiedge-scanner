"""Base64 candlestick chart rendering for dashboard cards.

Wraps `shared.chart_renderer.render_chart` — writes to a temp PNG, reads
it back as base64, deletes the file, and returns the string. On any
failure returns None so callers can skip a card cleanly.

Extracted from live_scanner.py (Phase 4d).
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import Optional

import pandas as pd

from shared.chart_renderer import render_chart

logger = logging.getLogger(__name__)


def render_chart_base64(df_5m: pd.DataFrame, ticker: str, prior_close: float,
                        adr_multiple: Optional[float] = None,
                        levels: Optional[dict] = None) -> Optional[str]:
    """Render a 5-min candlestick chart and return base64-encoded PNG.

    `levels` is an optional per-ticker dict of intraday reference levels:
        {"pdh":..., "pdl":..., "onh":..., "onl":..., "pmh":..., "pml":...}
    Any combination may be missing. The renderer handles in-range vs
    off-screen (corner badges) placement itself — axis is NEVER stretched.
    """
    path: Optional[str] = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        path = tmp.name

        # Build key_levels dict in the shape chart_renderer expects.
        # Periods: prior_day (PDO/PDH/PDL), overnight (ONH/ONL), premarket (PMH/PML).
        key_levels: dict = {}
        if prior_close and prior_close > 0:
            key_levels.setdefault("prior_day", {})["open"] = prior_close
        if levels:
            pd_e = key_levels.setdefault("prior_day", {})
            if levels.get("pdh") is not None: pd_e["high"] = levels["pdh"]
            if levels.get("pdl") is not None: pd_e["low"] = levels["pdl"]
            on_e = {}
            if levels.get("onh") is not None: on_e["high"] = levels["onh"]
            if levels.get("onl") is not None: on_e["low"] = levels["onl"]
            if on_e:
                key_levels["overnight"] = on_e
            pm_e = {}
            if levels.get("pmh") is not None: pm_e["high"] = levels["pmh"]
            if levels.get("pml") is not None: pm_e["low"] = levels["pml"]
            if pm_e:
                key_levels["premarket"] = pm_e

        render_chart(
            ticker=ticker,
            timeframe="5min",
            df=df_5m,
            output_path=path,
            key_levels=key_levels,
            theme="dark_color",
            show_volume=True,
            adr_multiple=adr_multiple,
        )

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(path)
        return b64
    except Exception as e:
        logger.debug(f"Chart render failed for {ticker}: {e}")
        if path is not None:
            try:
                os.unlink(path)
            except Exception:
                pass
        return None
