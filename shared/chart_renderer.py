"""
chart_renderer.py — Production chart renderer
---------------------------------------------
Style: Iteration C Bold Hierarchy header · 1920×1080 at 120 DPI

Themes:
    dark        — white/gray B&W candles on charcoal #1A1A1A (default)
    light       — hollow/solid black candles on white #FAFAFA
    dark_color  — teal/red colored candles on charcoal #1A1A1A
    light_color — teal/red colored candles on white #FAFAFA

Interface:
    render_chart(ticker, timeframe, lookback, annotations, overlays,
                 title, output_path, df=None, company=None, theme="dark")
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# ── Layout constants ───────────────────────────────────────────────────────────
_FIG_W, _FIG_H = 16.0, 9.0
_DPI        = 120
_FIG_W_PX   = _FIG_W * _DPI    # 1920
_FIG_H_PX   = _FIG_H * _DPI    # 1080

_BRAND_PX   = 40
_BRAND_FRAC = _BRAND_PX / _FIG_H_PX

_STRIP_PX   = 120
_STRIP_FRAC = _STRIP_PX / _FIG_H_PX
_SEP_FRAC   = 1 / _FIG_H_PX
_GAP_BELOW  = 40 / _FIG_H_PX

_BAR_W = 0.68

_AX_L = 0.05
_AX_R    = 1.0 - 30  / _FIG_W_PX   # default right margin (30px)
_AX_R_KL = 1.0 - 100 / _FIG_W_PX   # wider right margin when key_levels labels are drawn
_AX_T = 1.0 - _STRIP_FRAC - _SEP_FRAC - _GAP_BELOW
_AX_B = _BRAND_FRAC + 0.03

_FUTURES = frozenset({
    "ES", "NQ", "YM", "RTY", "GC", "CL",
    "MES", "MNQ", "MYM", "M2K", "MGC",
})

# Key-level periods that get DRAWN on the chart. Monthly levels (prior_month:
# PMH/PML/PMO) are intentionally excluded — they're usually far from intraday
# price and compress the chart. They remain in the key_levels dict that's
# passed to the script/Gemini writer so the voiceover can still reference them.
_CHART_DRAWN_PERIODS = frozenset({"overnight", "prior_day", "prior_week", "premarket"})

# Timeframes that represent daily-or-longer candles. On these charts we
# suppress key-level drawing entirely — overnight/prior-day/prior-week levels
# are meaningless clutter on a multi-month daily chart (best_stocks,
# new_highs, new_lows, industry_groups pipelines). Intraday charts
# (premarket, gap_up, gap_down) still draw the 9 levels as before.
_NO_LEVEL_TIMEFRAMES = frozenset({"daily", "1d", "weekly", "1w", "week", "monthly", "1mo"})

# ── Theme palettes ─────────────────────────────────────────────────────────────
# Each palette defines every visual property so render_chart() is theme-agnostic.
# Volume keys: vol_bull_face/edge/lw/a and vol_bear_face/edge/lw/a
# Wick keys:   wick_bull, wick_bear (colored themes differ; B&W themes share one color)
# Header keys: hdr_ticker, hdr_sub, hdr_price  (text colors in the strip)
# pos/neg:     change-line color (green/red) — theme-appropriate shade

_THEMES = {

    "dark": {
        "bg":           "#1A1A1A",
        "brand":        "#141414",
        "strip":        "#141414",
        "sep":          "#333333",
        "tick_color":   "#888888",
        # candles
        "bull_face":    "#FFFFFF",
        "bull_edge":    "#FFFFFF",
        "bear_face":    "#444444",
        "bear_edge":    "#FFFFFF",
        "wick_bull":    "#FFFFFF",
        "wick_bear":    "#FFFFFF",
        # volume
        "vol_bull_face": "#FFFFFF",
        "vol_bull_edge": "none",
        "vol_bull_lw":   0,
        "vol_bull_a":    0.25,
        "vol_bear_face": "#444444",
        "vol_bear_edge": "none",
        "vol_bear_lw":   0,
        "vol_bear_a":    0.25,
        # header text
        "hdr_ticker":   "#FFFFFF",
        "hdr_sub":      "#888888",
        "hdr_price":    "#FFFFFF",
        "pos":          "#00FF88",
        "neg":          "#FF4444",
    },

    "light": {
        "bg":           "#FAFAFA",
        "brand":        "#EEEEEE",
        "strip":        "#F0F0F0",
        "sep":          "#E0E0E0",
        "tick_color":   "#444444",
        # candles — hollow bull, solid bear
        "bull_face":    "#FFFFFF",
        "bull_edge":    "#000000",
        "bear_face":    "#1F1F1F",
        "bear_edge":    "#000000",
        "wick_bull":    "#1F1F1F",
        "wick_bear":    "#1F1F1F",
        # volume — light gray (bull) / dark (bear), solid, outlined bull
        "vol_bull_face": "#BBBBBB",
        "vol_bull_edge": "#1F1F1F",
        "vol_bull_lw":   0.6,
        "vol_bull_a":    1.0,
        "vol_bear_face": "#444444",
        "vol_bear_edge": "none",
        "vol_bear_lw":   0,
        "vol_bear_a":    1.0,
        # header text
        "hdr_ticker":   "#000000",
        "hdr_sub":      "#666666",
        "hdr_price":    "#000000",
        "pos":          "#2E7D32",
        "neg":          "#D32F2F",
    },

    "dark_color": {
        "bg":           "#1A1A1A",
        "brand":        "#141414",
        "strip":        "#141414",
        "sep":          "#333333",
        "tick_color":   "#888888",
        # candles — teal bull, coral-red bear
        "bull_face":    "#26A69A",
        "bull_edge":    "#26A69A",
        "bear_face":    "#EF5350",
        "bear_edge":    "#EF5350",
        "wick_bull":    "#26A69A",
        "wick_bear":    "#EF5350",
        # volume — matching candle color at 40% opacity
        "vol_bull_face": "#26A69A",
        "vol_bull_edge": "none",
        "vol_bull_lw":   0,
        "vol_bull_a":    0.40,
        "vol_bear_face": "#EF5350",
        "vol_bear_edge": "none",
        "vol_bear_lw":   0,
        "vol_bear_a":    0.40,
        # header text
        "hdr_ticker":   "#FFFFFF",
        "hdr_sub":      "#888888",
        "hdr_price":    "#FFFFFF",
        "pos":          "#26A69A",
        "neg":          "#EF5350",
    },

    "light_color": {
        "bg":           "#FAFAFA",
        "brand":        "#EEEEEE",
        "strip":        "#F0F0F0",
        "sep":          "#E0E0E0",
        "tick_color":   "#444444",
        # candles — vivid green/red on white background
        "bull_face":    "#00C853",
        "bull_edge":    "#00C853",
        "bear_face":    "#D50000",
        "bear_edge":    "#D50000",
        "wick_bull":    "#00C853",
        "wick_bear":    "#D50000",
        # volume — matching candle color at 40% opacity
        "vol_bull_face": "#00C853",
        "vol_bull_edge": "none",
        "vol_bull_lw":   0,
        "vol_bull_a":    0.40,
        "vol_bear_face": "#D50000",
        "vol_bear_edge": "none",
        "vol_bear_lw":   0,
        "vol_bear_a":    0.40,
        # header text
        "hdr_ticker":   "#000000",
        "hdr_sub":      "#666666",
        "hdr_price":    "#000000",
        "pos":          "#00C853",
        "neg":          "#D50000",
    },
}


# ── Public entry point ────────────────────────────────────────────────────────


def _filter_session_window(
    df: pd.DataFrame,
    session_open_et: str = "09:30",
    session_close_et: str = "16:00",
    expand_premarket_to: str | None = None,
) -> pd.DataFrame:
    """
    Filter OHLCV DataFrame to trading session window (default RTH 09:30-16:00 ET).
    Supports optional pre-market expansion.
    
    This REPLACES the problematic `iloc[-lookback:]` slicing which would show
    only the last N bars (tail) without regard to what time of day it is.
    Timestamp-based filtering ensures morning breakouts are shown even if
    bars arrive after 15:00 ET.
    
    Args:
        df: OHLCV DataFrame with datetime column or index (timezone-aware ET preferred)
        session_open_et: Start time HH:MM ET (default "09:30")
        session_close_et: End time HH:MM ET (default "16:00")
        expand_premarket_to: Optional start time HH:MM ET (e.g., "04:00" for pre-market)
        
    Returns:
        Filtered DataFrame containing only bars in the session window.
    """
    ET = pytz.timezone("America/New_York")
    
    if df.empty:
        return df
    
    df = df.copy()
    
    # Normalize datetime to index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(df.index)
    
    # Ensure timezone-aware (ET)
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    elif df.index.tz != ET:
        df.index = df.index.tz_convert(ET)
    
    if len(df) == 0:
        return df.reset_index() if "datetime" not in df.columns else df
    
    # Trading day = date of first bar
    trading_day = df.index[0].normalize()
    
    # Determine session bounds
    open_hm = expand_premarket_to or session_open_et
    close_hm = session_close_et
    
    # Parse HH:MM strings to time objects
    open_time = datetime.strptime(open_hm, "%H:%M").time()
    close_time = datetime.strptime(close_hm, "%H:%M").time()
    
    session_start = ET.localize(datetime.combine(trading_day.date(), open_time))
    session_end = ET.localize(datetime.combine(trading_day.date(), close_time))
    
    # Filter: keep bars where timestamp is in [session_start, session_end)
    mask = (df.index >= session_start) & (df.index < session_end)
    result = df[mask]
    
    # Reset index to restore datetime column if it was removed
    return result.reset_index() if "datetime" not in result.columns else result


def render_chart(
    ticker: str,
    timeframe: str = "daily",
    lookback: int = 60,
    annotations: dict = None,
    overlays: list = None,
    title: str = None,
    output_path: str = "chart.png",
    df: pd.DataFrame = None,
    company: str = None,
    theme: str = "dark_color",
    key_levels: dict = None,
    show_volume: bool = False,
    adr_multiple: float | None = None,
    session_open_et: str = "09:30",
    session_close_et: str = "16:00",
    expand_premarket_to: str | None = None,
) -> str:
    """
    Render a broadcast-quality trading chart.

    Args:
        ticker:       Symbol (e.g. "AAPL", "ES")
        timeframe:    "daily" | "5min" | "60min" | "1min"
        lookback:     Deprecated (ignored for intraday). For daily+ kept for compat.
        annotations:  Optional dict of Brooks structural overlays drawn on the
                      price axis AFTER candles/key-levels, BEFORE the header.
                      Keys (all optional): signal_bar, trendline, stop, target,
                      phase_label, always_in_arrow, verdict_badge,
                      agreement_watermark. See _draw_brooks_annotations() for
                      the full schema. Passing None or {} leaves the chart
                      unannotated (legacy behavior preserved).
        overlays:     Accepted for API compat, currently ignored
        title:        Accepted for API compat, ignored
        output_path:  Destination PNG path
        df:           Real OHLCV DataFrame (optional). Uses synthetic data if None.
        company:      Company name shown in header (optional, defaults to ticker)
        theme:        "dark" | "light" | "dark_color" | "light_color"
                      (default "dark_color" — approved B&W/teal-red theme)
        key_levels:   dict of {period: {high/low/open: price}}. On intraday
                      charts only periods in _CHART_DRAWN_PERIODS (overnight
                      / prior_day / prior_week) are drawn — monthly levels
                      are ignored by the renderer. On daily/weekly charts
                      (see _NO_LEVEL_TIMEFRAMES) NO levels are drawn at all.
                      The full dict remains available upstream in the
                      screener output for the script/Gemini voiceover.
        show_volume:  If True, render the volume pane below the price pane.
                      Default False — charts are price-only unless a caller
                      explicitly opts in.
        session_open_et:    Start time HH:MM ET for intraday timestamp window.
                            Default "09:30" (RTH open).
        session_close_et:   End time HH:MM ET for intraday timestamp window.
                            Default "16:00" (RTH close).
        expand_premarket_to: Optional start time HH:MM ET (e.g., "04:00").
                             If provided, extends session window back to this time.
                             Useful for pre-market analysis.

    Returns:
        Absolute path to the saved PNG.
    """
    p = _THEMES.get(theme, _THEMES["dark_color"])

    # Daily / weekly charts never draw key levels, regardless of what the
    # caller passed in. The full key_levels dict still flows through the
    # screener output → script/Gemini writer pipeline untouched; only the
    # visual rendering is suppressed here.
    _draw_levels = bool(key_levels) and timeframe.lower() not in _NO_LEVEL_TIMEFRAMES

    if df is None or (hasattr(df, "empty") and df.empty):
        df = _synthetic_df(ticker, lookback)
    else:
        df = _normalize_columns(df)
        # For intraday charts (1min, 5min), filter by session window (timestamp-based).
        # For daily/weekly, apply lookback if specified (backward compat).
        tf_lower = timeframe.lower()
        if tf_lower in ("1min", "5min", "15min", "15m", "60min", "1h"):
            df = _filter_session_window(
                df,
                session_open_et=session_open_et,
                session_close_et=session_close_et,
                expand_premarket_to=expand_premarket_to,
            )
            # Resample to the target timeframe. Input is assumed 1-min bars.
            # Fixes the bug where timeframe only set the title label without
            # actually aggregating bars (dashboard was showing 1-min bars
            # labeled as 5-min).
            _RESAMPLE_RULES = {
                "1min": None,      # no resample needed
                "5min": "5min",
                "15min": "15min",
                "15m": "15min",
                "60min": "60min",
                "1h": "60min",
            }
            _rule = _RESAMPLE_RULES.get(tf_lower, "5min")  # default to 5min
            if _rule is not None and len(df) > 0:
                _agg = {"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"}
                # df here has a 'datetime' column (reset_index from filter)
                _dt_col = "datetime" if "datetime" in df.columns else df.columns[0]
                df = (df.set_index(_dt_col)
                        .resample(_rule, label="left", closed="left")
                        .agg(_agg)
                        .dropna()
                        .reset_index())
        elif lookback and lookback < len(df):
            df = df.iloc[-lookback:]

    n      = len(df)
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    vols   = df["volume"].values if "volume" in df.columns else None

    dchg    = closes[-1] - closes[0]
    pct     = dchg / closes[0] * 100
    sign    = "+" if dchg >= 0 else "\u2212"
    dollar_str = f"{sign}${abs(dchg):,.2f}"
    pct_str    = f"({sign}{abs(pct):.2f}%)"
    pct_col    = p["pos"] if pct >= 0 else p["neg"]
    price_str  = f"${closes[-1]:,.2f}"

    company_name = company or ticker
    # Compute actual session window shown for subtitle
    session_start_str = expand_premarket_to or session_open_et
    session_end_str = session_close_et
    subtitle = _subtitle(ticker, timeframe, n, session_start_str, session_end_str)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(_FIG_W, _FIG_H), dpi=_DPI, facecolor=p["bg"])

    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0), 1, _BRAND_FRAC, boxstyle="square,pad=0",
        facecolor=p["brand"], edgecolor="none",
        transform=fig.transFigure, zorder=10, clip_on=False,
    ))
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 1.0 - _STRIP_FRAC), 1, _STRIP_FRAC, boxstyle="square,pad=0",
        facecolor=p["strip"], edgecolor="none",
        transform=fig.transFigure, zorder=10, clip_on=False,
    ))
    fig.add_artist(mpatches.Rectangle(
        (0, 1.0 - _STRIP_FRAC - _SEP_FRAC), 1, _SEP_FRAC,
        facecolor=p["sep"], edgecolor="none",
        transform=fig.transFigure, zorder=11, clip_on=False,
    ))

    ax_r = _AX_R_KL if _draw_levels else _AX_R
    if show_volume:
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[3, 1], hspace=0,
            left=_AX_L, right=ax_r, top=_AX_T, bottom=_AX_B, figure=fig,
        )
        ax_p = fig.add_subplot(gs[0])
        ax_v = fig.add_subplot(gs[1], sharex=ax_p)
    else:
        # Volume off → price pane takes the entire plotting area.
        gs = gridspec.GridSpec(
            1, 1, hspace=0,
            left=_AX_L, right=ax_r, top=_AX_T, bottom=_AX_B, figure=fig,
        )
        ax_p = fig.add_subplot(gs[0])
        ax_v = None

    _styled_axes = (ax_p, ax_v) if ax_v is not None else (ax_p,)
    for ax in _styled_axes:
        ax.set_facecolor(p["bg"])
        ax.tick_params(colors=p["tick_color"], labelsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)

    ax_p.yaxis.tick_left()
    ax_p.yaxis.set_label_position("left")
    ax_p.tick_params(axis="y", left=True, right=False, labelright=False)
    plt.setp(ax_p.get_xticklabels(), visible=False)
    ax_p.tick_params(bottom=False)

    # ── Candles ───────────────────────────────────────────────────────────────
    half = _BAR_W / 2
    for i in range(n):
        bull    = closes[i] >= opens[i]
        face    = p["bull_face"] if bull else p["bear_face"]
        edge    = p["bull_edge"] if bull else p["bear_edge"]
        wick_c  = p["wick_bull"] if bull else p["wick_bear"]
        body_lo = min(opens[i], closes[i])
        body_h  = max(abs(closes[i] - opens[i]), 0.01)
        ax_p.plot([i, i], [lows[i], highs[i]], color=wick_c, lw=0.8, zorder=2)
        ax_p.add_patch(mpatches.Rectangle(
            (i - half, body_lo), _BAR_W, body_h,
            facecolor=face, edgecolor=edge, linewidth=0.6, zorder=3,
        ))

    pad = (highs.max() - lows.min()) * 0.04
    y_lo = lows.min() - pad
    y_hi = highs.max() + pad
    ax_p.set_xlim(-1, n)
    ax_p.set_ylim(y_lo, y_hi)

    if _draw_levels:
        _draw_key_levels(ax_p, key_levels, p)

    if annotations:
        _draw_brooks_annotations(ax_p, annotations, p, n, highs, lows)

    # ── Volume ────────────────────────────────────────────────────────────────
    if show_volume and ax_v is not None:
        if vols is not None:
            vol_max = vols.max()
            if vol_max > 0:
                for i in range(n):
                    bull = closes[i] >= opens[i]
                    fc   = p["vol_bull_face"] if bull else p["vol_bear_face"]
                    ec   = p["vol_bull_edge"] if bull else p["vol_bear_edge"]
                    lw   = p["vol_bull_lw"]   if bull else p["vol_bear_lw"]
                    a    = p["vol_bull_a"]    if bull else p["vol_bear_a"]
                    ax_v.add_patch(mpatches.Rectangle(
                        (i - half, 0), _BAR_W, vols[i],
                        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=a, zorder=2,
                    ))
                ax_v.set_ylim(0, vol_max * 1.2)

        ax_v.set_xlim(-1, n)
        ax_v.yaxis.set_visible(False)
        ax_v.xaxis.set_visible(False)

    # ── Header ────────────────────────────────────────────────────────────────
    _draw_header_c(fig, ticker, company_name, price_str, dollar_str, pct_str,
                   pct_col, subtitle, p, adr_multiple=adr_multiple)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=_DPI, facecolor=p["bg"])
    plt.close(fig)

    logger.info(f"Chart saved: {out} ({ticker}, {theme}, {timeframe}, {n} bars)")
    return str(out)


# ── Header renderer ───────────────────────────────────────────────────────────

def _adr_mult_color(mult: float, p: dict) -> tuple[str, str]:
    """Return (fg_color, face_color_or_None) for an ADR-multiple badge.

    Thresholds (Minervini-style):
      < 1.0×   — neutral/muted (normal range day)
      1.0–1.5× — teal, mild expansion
      1.5–2.0× — teal, strong expansion (filled pill)
      ≥ 2.0×   — teal, extreme expansion (filled pill + white text)
    Colors pull from the active theme's palette so light/dark themes Just Work.
    """
    teal   = p.get("pos", "#26A69A")
    muted  = p.get("hdr_sub", "#888888")
    white  = p.get("hdr_ticker", "#FFFFFF")
    if mult < 1.0:
        return (muted, None)
    if mult < 1.5:
        return (teal, None)
    if mult < 2.0:
        return (white, teal + "55")  # 33% alpha (CSS hex alpha doesn't apply in mpl; we blend below)
    return (white, teal)


def _draw_header_c(fig, ticker, company, price_str, dollar_str, pct_str,
                   pct_col, subtitle, p, adr_multiple=None):
    """Iteration C Bold Hierarchy header — theme-aware."""
    tr = fig.transFigure

    def fpt(pt):
        return pt * _DPI / 72.0 / _FIG_H_PX

    INNER_GAP   = 3 / _FIG_H_PX
    PAD         = 30 / _FIG_W_PX
    STRIP_CY    = 1.0 - _STRIP_FRAC / 2

    # Top anchor — 28px from the top edge of the figure, giving visible breathing room.
    # (Previously used STRIP_CY + block_h/2, which overflowed the figure top boundary.)
    TOP_ANCHOR  = 1.0 - 28 / _FIG_H_PX

    # LEFT: ticker (28pt) top-anchored, company (11pt italic) below
    fig.text(PAD, TOP_ANCHOR, ticker,
             transform=tr, color=p["hdr_ticker"],
             fontsize=28, fontweight="bold",
             va="top", ha="left", zorder=20)
    fig.text(PAD, TOP_ANCHOR - fpt(28 * 1.2) - INNER_GAP, company,
             transform=tr, color=p["hdr_sub"],
             fontsize=11, style="italic",
             va="top", ha="left", zorder=20)

    # CENTER: price (22pt) top-anchored, change (14pt colored) below
    fig.text(0.50, TOP_ANCHOR, price_str,
             transform=tr, color=p["hdr_price"],
             fontsize=22, fontweight="bold",
             va="top", ha="center", zorder=20)
    fig.text(0.50, TOP_ANCHOR - fpt(22 * 1.2) - INNER_GAP,
             f"{dollar_str}  {pct_str}",
             transform=tr, color=pct_col,
             fontsize=14, fontweight="medium",
             va="top", ha="center", zorder=20)

    # RIGHT: ADR multiple badge (top line) + subtitle (bottom line)
    if adr_multiple is not None and adr_multiple > 0:
        fg, bg = _adr_mult_color(adr_multiple, p)
        adr_text = f"{adr_multiple:.2f}\u00d7 ADR"
        badge_y = TOP_ANCHOR - fpt(11 * 0.3)  # sit near top of strip
        if bg is None:
            fig.text(1.0 - PAD, badge_y, adr_text,
                     transform=tr, color=fg,
                     fontsize=12, fontweight="bold",
                     va="top", ha="right", zorder=20)
        else:
            # Filled pill — use bbox for background
            # bg may be a hex+alpha suffix; strip alpha for matplotlib (handles via alpha kwarg on bbox)
            bg_clean = bg[:7] if len(bg) == 9 else bg
            alpha = int(bg[7:9], 16) / 255.0 if len(bg) == 9 else 1.0
            fig.text(1.0 - PAD, badge_y, adr_text,
                     transform=tr, color=fg,
                     fontsize=12, fontweight="bold",
                     va="top", ha="right", zorder=20,
                     bbox=dict(boxstyle="round,pad=0.35",
                               facecolor=bg_clean, edgecolor="none",
                               alpha=alpha))
        # subtitle sits below the badge
        sub_y = TOP_ANCHOR - fpt(12 * 1.6)
        fig.text(1.0 - PAD, sub_y, subtitle,
                 transform=tr, color=p["hdr_sub"],
                 fontsize=11,
                 va="top", ha="right", zorder=20)
    else:
        # Original behavior: subtitle vertically centered in strip
        fig.text(1.0 - PAD, STRIP_CY, subtitle,
                 transform=tr, color=p["hdr_sub"],
                 fontsize=11,
                 va="center", ha="right", zorder=20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _subtitle(ticker: str, timeframe: str, n_bars: int, session_open_et: str = "09:30", session_close_et: str = "16:00") -> str:
    tf = timeframe.lower()
    if tf == "daily":
        return f"Daily \u00b7 {n_bars} bars"
    if tf in ("15min", "15m"):
        is_futures = ticker.upper() in _FUTURES or ".c." in ticker
        return "15min \u00b7 Globex Overnight" if is_futures else f"15min \u00b7 {n_bars} bars"
    if tf == "5min":
        is_futures = ticker.upper() in _FUTURES or ".c." in ticker
        if is_futures:
            return f"5min · {session_open_et}–{session_close_et} ({n_bars} bars)"
        # For stocks, show the session window
        return f"5min · {session_open_et}–{session_close_et} ({n_bars} bars)"
    if tf in ("60min", "1h"):
        return f"60min \u00b7 {n_bars} bars"
    if tf in ("weekly", "1w", "week"):
        return f"Weekly \u00b7 {n_bars} bars"
    return f"{timeframe} \u00b7 {n_bars} bars"


def _draw_key_levels(ax_p, key_levels: dict, p: dict):
    """
    Draw up to 9 horizontal reference levels on the price axis with right-edge
    labels.

    Level groups DRAWN (see _CHART_DRAWN_PERIODS):
        overnight   (#00D4FF cyan)   — ONH, ONL, ONO
        prior_day   (#FFEB3B yellow) — PDH, PDL, PDO
        prior_week  (#FF9800 orange) — PWH, PWL, PWO

    Level groups SKIPPED (kept in the dict for script/voiceover only):
        prior_month                  — PMH, PML, PMO

    H/L drawn solid, O drawn dotted.
    Labels placed just right of the axes, de-overlapped vertically.
    """
    from matplotlib.transforms import blended_transform_factory

    _COLOR = {
        "overnight":   "#00D4FF",
        "prior_day":   "#FFEB3B",
        "prior_week":  "#FF9800",
        "prior_month": "#E91E63",
        "premarket":   "#B388FF",   # lavender — distinct from cyan/yellow/orange
    }
    # All three kinds are dotted — high/low used to be solid at lw 0.9 but
    # felt too prominent relative to the candles. Dotted at lw 1.0 keeps the
    # dots clearly visible on a 1920px render while letting the lines recede
    # into the background. Opens stay at a lower alpha so H/L still read as
    # the "primary" levels by hierarchy.
    _LINE = {
        "high": (":", 1.0, 0.85),
        "low":  (":", 1.0, 0.85),
        "open": (":", 1.0, 0.70),
    }
    _ABBR = {
        "overnight":   {"high": "ONH", "low": "ONL", "open": "ONO"},
        "prior_day":   {"high": "PDH", "low": "PDL", "open": "PDO"},
        "prior_week":  {"high": "PWH", "low": "PWL", "open": "PWO"},
        "prior_month": {"high": "PMH", "low": "PML", "open": "PMO"},
        # Pre-market narrow window (08:00–09:30 ET) — labels distinct from prior-month
        # (which isn't drawn on intraday charts anyway). "P" here = pre-market, not prior.
        "premarket":   {"high": "PMH", "low": "PML", "open": "PMO"},
    }

    entries = []
    for period, levels in key_levels.items():
        if period not in _CHART_DRAWN_PERIODS:
            continue  # monthly levels stay in the dict but aren't drawn
        if not isinstance(levels, dict):
            continue
        color = _COLOR.get(period, "#AAAAAA")
        for kind, price in levels.items():
            if price is None:
                continue
            ls, lw, alpha = _LINE.get(kind, _LINE["high"])
            label = _ABBR.get(period, {}).get(kind, "")
            entries.append((float(price), label, color, ls, lw, alpha))

    if not entries:
        return

    entries.sort(key=lambda x: x[0], reverse=True)

    label_tr = blended_transform_factory(ax_p.transAxes, ax_p.transData)
    y_lo, y_hi = ax_p.get_ylim()
    y_range = y_hi - y_lo
    min_gap = y_range * 0.018   # 1.8% of visible range

    # Reference price for "distance" callouts on edge markers. Use the last
    # close (same series ax_p was built from) when available; fall back to
    # the midpoint of the visible range.
    ref_price = None
    for ln in ax_p.get_lines():
        pass  # lines are wicks; we can't recover closes from them
    # Pull last close from the rightmost candle body: body_lo..body_h stored
    # as Rectangles in ax_p.patches. The last patch is the final candle body.
    try:
        body_patches = [pch for pch in ax_p.patches if hasattr(pch, "get_xy")]
        if body_patches:
            last = body_patches[-1]
            x, y0 = last.get_xy()
            ref_price = y0 + last.get_height() / 2
    except Exception:
        pass
    if ref_price is None:
        ref_price = (y_lo + y_hi) / 2

    placed_y = []   # de-overlap tracking for in-range labels
    above_entries = []  # (price, label, color) — off-screen above
    below_entries = []  # (price, label, color) — off-screen below

    for price, label, color, ls, lw, alpha in entries:
        if price > y_hi:
            above_entries.append((price, label, color))
            continue
        if price < y_lo:
            below_entries.append((price, label, color))
            continue
        ax_p.axhline(y=price, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=1)

        # Nudge label down if it would overlap a previously placed label
        y_label = price
        for prev in placed_y:
            if abs(y_label - prev) < min_gap:
                y_label = prev - min_gap
        placed_y.append(y_label)

        ax_p.text(
            1.008, y_label, label,
            transform=label_tr,
            color=color, fontsize=7, fontweight="bold",
            va="center", ha="left", clip_on=False,
        )

    # ── Edge markers for off-screen levels ───────────────────────────────
    # For levels above/below the visible window, draw a small arrow/triangle
    # pinned to the top or bottom edge with "LABEL +N.NN%" so traders can
    # see direction + distance without expanding the y-axis.
    def _edge_markers(items, at_top: bool):
        if not items:
            return
        # Sort by proximity to the visible window so closest is first.
        items = sorted(items, key=lambda t: (t[0] - y_hi) if at_top else (y_lo - t[0]))
        # x in axes coords (right side, stacked leftward if multiple)
        for i, (price, label, color) in enumerate(items[:3]):
            dist_pct = (price - ref_price) / ref_price * 100 if ref_price else 0.0
            sign = "+" if dist_pct >= 0 else "\u2212"
            arrow = "\u25B2" if at_top else "\u25BC"  # ▲ / ▼
            txt = f"{arrow} {label} {sign}{abs(dist_pct):.2f}%"
            # Stack vertically near the top/bottom inside edge
            if at_top:
                y_ax = 0.985 - i * 0.045
                va = "top"
            else:
                y_ax = 0.015 + i * 0.045
                va = "bottom"
            ax_p.text(
                0.992, y_ax, txt,
                transform=ax_p.transAxes,
                color=color, fontsize=8, fontweight="bold",
                va=va, ha="right",
                bbox=dict(facecolor=p["bg"], edgecolor=color, boxstyle="round,pad=0.25",
                          linewidth=0.6, alpha=0.85),
                zorder=12, clip_on=False,
            )

    _edge_markers(above_entries, at_top=True)
    _edge_markers(below_entries, at_top=False)


def _draw_brooks_annotations(
    ax_p,
    annotations: dict,
    p: dict,
    n: int,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
) -> None:
    """Draw Brooks structural overlays on the price axis.

    Schema (all keys optional — only present keys render):
      signal_bar:          {"bar_index": int, "color": str?}
                           Draws a hollow rectangle around a single bar with
                           a "SIG" label above. If highs/lows are provided,
                           the box tightly wraps the bar's actual range;
                           otherwise falls back to a vertical accent line.
      trendline:           [[x1, y1], [x2, y2]]  or
                           [[x1, y1], [x2, y2], {"color": str}]
                           Dashed line in data coordinates (bar_index, price).
      stop:                {"price": float, "label": str?}
                           Horizontal dashed red line with right-edge label.
      target:              {"price": float, "label": str?}
                           Horizontal dashed green line with right-edge label.
      phase_label:         str
                           Top-left axis overlay, boxed for legibility.
      always_in_arrow:     "up" | "down" | "neutral"
                           Top-right glyph colored by direction.
      verdict_badge:       {"text": str, "color": str?}
                           Bottom-right pill; color defaults to muted gray.
      agreement_watermark: str
                           Bottom-left small italic text (eg "vs scanner: AGREE").

    No return value; mutates ax_p in place.
    """
    y_lo, y_hi = ax_p.get_ylim()

    sb = annotations.get("signal_bar")
    if sb and "bar_index" in sb:
        bi = int(sb["bar_index"])
        color = sb.get("color", "#FFD700")
        if (highs is not None and lows is not None and 0 <= bi < n):
            y0, y1 = float(lows[bi]), float(highs[bi])
            pad = max((y1 - y0) * 0.18, (y_hi - y_lo) * 0.004)
            ax_p.add_patch(mpatches.Rectangle(
                (bi - 0.6, y0 - pad), 1.2, (y1 - y0) + 2 * pad,
                facecolor="none", edgecolor=color, linewidth=1.8, zorder=5,
            ))
            ax_p.text(bi, y1 + pad, "SIG", color=color, fontsize=7.5,
                      ha="center", va="bottom", zorder=6, fontweight="bold")
        else:
            ax_p.axvline(bi, color=color, linewidth=1.2, alpha=0.6, zorder=4)

    tl = annotations.get("trendline")
    if tl and len(tl) >= 2:
        (x1, y1), (x2, y2) = tl[0], tl[1]
        color = (tl[2].get("color") if (len(tl) > 2 and isinstance(tl[2], dict))
                 else p.get("hdr_sub", "#AAAAAA"))
        ax_p.plot([x1, x2], [y1, y2], color=color, lw=1.2,
                  linestyle="--", zorder=4)

    s = annotations.get("stop")
    if s and "price" in s:
        px = float(s["price"])
        ax_p.axhline(px, color=p.get("neg", "#EF5350"),
                     linestyle="--", linewidth=1.0, zorder=4)
        label = s.get("label", f"STOP {px:.2f}")
        ax_p.text(n - 0.5, px, f"  {label}",
                  color=p.get("neg", "#EF5350"), fontsize=8,
                  ha="left", va="center", zorder=5, fontweight="bold",
                  clip_on=False)

    t = annotations.get("target")
    if t and "price" in t:
        px = float(t["price"])
        ax_p.axhline(px, color=p.get("pos", "#26A69A"),
                     linestyle="--", linewidth=1.0, zorder=4)
        label = t.get("label", f"TGT {px:.2f}")
        ax_p.text(n - 0.5, px, f"  {label}",
                  color=p.get("pos", "#26A69A"), fontsize=8,
                  ha="left", va="center", zorder=5, fontweight="bold",
                  clip_on=False)

    phase = annotations.get("phase_label")
    if phase:
        ax_p.text(0.01, 0.98, phase, transform=ax_p.transAxes,
                  color=p.get("hdr_sub", "#CCCCCC"),
                  fontsize=8.5, ha="left", va="top", zorder=5,
                  bbox=dict(facecolor=p.get("bg", "#000000"),
                            edgecolor="none",
                            boxstyle="round,pad=0.3", alpha=0.75))

    aia = annotations.get("always_in_arrow")
    if aia:
        # Text labels rather than Unicode glyphs: the system sans-serif fallback
        # chain doesn't reliably fire for arrow/triangle code points on all
        # platforms, which produces tofu boxes.
        label = {"up": "AI \u2191 LONG", "down": "AI \u2193 SHORT",
                 "neutral": "AI \u2014 UNCLEAR"}.get(aia, "AI UNCLEAR")
        # ASCII fallback if Unicode arrow isn't in the chosen font family
        label = {"up": "LONG", "down": "SHORT",
                 "neutral": "UNCLEAR"}.get(aia, "UNCLEAR")
        color = {"up": p.get("pos", "#26A69A"),
                 "down": p.get("neg", "#EF5350"),
                 "neutral": p.get("hdr_sub", "#888888")}.get(aia, "#888888")
        ax_p.text(0.99, 0.98, label, transform=ax_p.transAxes,
                  color=color, fontsize=9, ha="right", va="top", zorder=5,
                  fontweight="bold",
                  bbox=dict(facecolor=p.get("bg", "#000000"),
                            edgecolor=color, boxstyle="round,pad=0.3",
                            linewidth=0.8, alpha=0.9))

    v = annotations.get("verdict_badge")
    if v and "text" in v:
        bg = v.get("color", "#888888")
        ax_p.text(0.99, 0.02, v["text"], transform=ax_p.transAxes,
                  color="#FFFFFF", fontsize=10, ha="right", va="bottom",
                  zorder=6, fontweight="bold",
                  bbox=dict(facecolor=bg, edgecolor="none",
                            boxstyle="round,pad=0.5", alpha=0.95))

    wm = annotations.get("agreement_watermark")
    if wm:
        ax_p.text(0.01, 0.02, wm, transform=ax_p.transAxes,
                  color=p.get("hdr_sub", "#888888"),
                  fontsize=7, ha="left", va="bottom", zorder=5, alpha=0.7,
                  fontstyle="italic")


def _synthetic_df(ticker: str, n: int = 60) -> pd.DataFrame:
    seed = sum(ord(c) for c in ticker.upper()) % (2 ** 31)
    rng  = np.random.default_rng(seed)
    close0  = float(rng.uniform(50, 400))
    returns = rng.normal(0.0003, 0.015, n)
    closes  = close0 * np.cumprod(1 + returns)
    opens   = np.roll(closes, 1); opens[0] = close0 * 0.998
    highs   = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lows    = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vols    = rng.integers(2_000_000, 20_000_000, n).astype(float)
    idx     = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": vols}, index=idx,
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_map = {
        c: c.lower() for c in df.columns
        if c.lower() in ("open", "high", "low", "close", "volume")
    }
    if col_map:
        df = df.rename(columns=col_map)
    required = ["open", "high", "low", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Have: {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check for datetime column first (common from live_scanner.py output)
        for cand in ("datetime", "timestamp", "date", "ts_event"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    return df
