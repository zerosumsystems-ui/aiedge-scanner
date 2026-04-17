"""HTML dashboard + card rendering for the live scanner.

Generates the CSS/JS-heavy dashboard.html that the scanner writes every cycle.
Also exposes per-card builders that scratch/test_dashboard.py pulls in for
snapshot testing.

Extracted from live_scanner.py (Phase 4g-2). `_generate_dashboard` takes
`intraday_levels`, `dashboard_path`, `first_scan_hour`, and `first_scan_min`
as explicit parameters instead of reading live_scanner globals. The live
scanner keeps a thin wrapper that binds these.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from aiedge.dashboard.charts import render_chart_base64
from aiedge.dashboard.console import _next_scan_time_str

logger = logging.getLogger(__name__)


# ── HTML Dashboard ────────────────────────────────────────────────────────────

# Signal badge colors (CSS)
_SIG_CSS = {
    "BUY_PULLBACK":  ("bg-buy",   "BUY PULLBACK"),
    "BUY_SPIKE":     ("bg-buy",   "BUY SPIKE"),
    "SELL_PULLBACK": ("bg-sell",  "SELL PULLBACK"),
    "SELL_SPIKE":    ("bg-sell",  "SELL SPIKE"),
    "SELL_PULLBACK_INTRADAY": ("bg-sell", "SELL FLIP"),
    "BUY_PULLBACK_INTRADAY":  ("bg-buy",  "BUY FLIP"),
    "WAIT":          ("bg-wait",  "WAIT"),
    "FOG":           ("bg-fog",   "FOG"),
    "AVOID":         ("bg-avoid", "AVOID"),
    "PASS":          ("bg-avoid", "PASS"),
}

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta http-equiv="refresh" content="300">
<title>Live Scanner</title>
<style>
  :root {
    --bg:       #1a1a1a;
    --surface:  #242424;
    --border:   #333;
    --text:     #e0e0e0;
    --sub:      #888;
    --teal:     #00c896;
    --red:      #e05555;
    --yellow:   #f5c842;
    --orange:   #f5a623;
    --gray:     #555;
    --radius:   8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 14px;
    padding: 12px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
  header .meta { font-size: 12px; color: var(--sub); text-align: right; }
  .stats {
    display: flex; gap: 16px;
    font-size: 12px; color: var(--sub);
    margin-bottom: 12px;
  }
  .stats span { white-space: nowrap; }

  /* Stock card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 8px;
    overflow: hidden;
  }
  .card summary {
    list-style: none;
    cursor: pointer;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: 28px 62px 1fr 86px auto;
    align-items: center;
    gap: 8px;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  .card summary::-webkit-details-marker { display: none; }
  .card[open] summary { border-bottom: 1px solid var(--border); }

  .rank {
    font-size: 12px; color: var(--sub);
    text-align: center; line-height: 1;
  }
  .rank .arrow { font-size: 10px; display: block; }
  .ticker-block { min-width: 0; }
  .ticker { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
  .day-type { font-size: 11px; color: var(--sub); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cycle-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 9.5px;
    letter-spacing: 0.3px;
    color: var(--teal);
    background: rgba(0,200,150,.06);
  }

  /* ADR block (fills the 1fr spacer column in the card grid) */
  .adr-block { min-width: 0; padding-left: 4px; margin-right: 12px; }
  .adr-val { font-size: 12px; color: var(--sub); white-space: nowrap; }
  .adr-ratio { font-size: 11px; color: var(--sub); opacity: 0.7; white-space: nowrap; }

  /* ADR-multiple column — Minervini-style range-expansion tile */
  .adr-mult {
    text-align: center;
    padding: 6px 8px;
    border-radius: 6px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    line-height: 1.15;
    min-width: 50px;
  }
  .adr-mult .mult-val { font-size: 15px; font-weight: 700; letter-spacing: -0.2px; }
  .adr-mult .mult-lbl { font-size: 9px; opacity: .65; letter-spacing: 0.6px; }
  .adr-mult.tier-cold { color: var(--sub); background: transparent; }
  .adr-mult.tier-warm { color: var(--teal); background: rgba(0,200,150,.10); border: 1px solid rgba(0,200,150,.25); }
  .adr-mult.tier-hot  {
    color: #fff; background: rgba(0,200,150,.35);
    border: 1px solid var(--teal);
    box-shadow: 0 0 0 1px rgba(0,200,150,.20);
  }
  .adr-mult.tier-extreme {
    color: #fff; background: var(--teal);
    border: 1px solid var(--teal);
    box-shadow: 0 0 12px rgba(0,200,150,.40);
  }

  /* Sort toolbar */
  .sort-bar {
    display: flex; gap: 6px; align-items: center;
    margin: 4px 0 10px;
    font-size: 11px; color: var(--sub);
  }
  .sort-bar button {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 4px 10px; border-radius: 4px;
    font: inherit; cursor: pointer;
  }
  .sort-bar button.active { background: rgba(0,200,150,.18); border-color: var(--teal); color: var(--teal); }
  .sort-bar button:hover { border-color: var(--teal); }

  /* Scoring legend */
  .legend { margin: 0 0 12px; font-size: 12px; color: var(--sub); }
  .legend summary {
    cursor: pointer; color: var(--teal); font-size: 11px; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase; padding: 4px 0;
  }
  .legend summary:hover { text-decoration: underline; }
  .legend-body {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px; padding: 12px 0 8px;
  }
  .legend-section h3 {
    font-size: 11px; font-weight: 700; color: var(--text);
    text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 6px;
    border-bottom: 1px solid var(--border); padding-bottom: 4px;
  }
  .legend-section dl { margin: 0; }
  .legend-section dt {
    font-weight: 600; color: var(--text); font-size: 11px; margin-top: 6px;
    font-family: 'SF Mono', 'Menlo', monospace;
  }
  .legend-section dd { margin: 2px 0 0 0; font-size: 11px; line-height: 1.4; color: var(--sub); }

  .scores {
    display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
  }
  .score-row { display: flex; gap: 6px; align-items: center; }
  .score-label { font-size: 11px; color: var(--sub); width: 24px; text-align: right; }
  .bar-track {
    width: 80px; height: 5px;
    background: var(--border); border-radius: 3px; overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-urg  { background: var(--teal); }
  .bar-unc  { background: var(--red); }
  .score-val { font-size: 12px; font-weight: 600; width: 28px; text-align: right; }

  /* Component subscore strip (under URG/UNC bars) */
  .comp-strip {
    display: flex; flex-wrap: wrap; gap: 4px;
    padding: 4px 12px 8px; font-variant-numeric: tabular-nums;
  }
  .comp-tile {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 2px 6px; min-width: 36px; border-radius: 4px;
    background: rgba(255,255,255,.03); border: 1px solid var(--border);
    color: var(--text); line-height: 1.1;
  }
  .comp-tile .comp-lbl { font-size: 9px; color: var(--sub); letter-spacing: 0.3px; }
  .comp-tile .comp-val { font-size: 11px; font-weight: 600; }
  .comp-tile.dim { opacity: 0.35; }
  .comp-tile.spt { border-color: rgba(0,200,150,.45); background: rgba(0,200,150,.08); }
  .comp-tile.spt .comp-lbl { color: var(--teal); }
  .comp-tile.bpa { border-color: rgba(220,170,50,.45); background: rgba(220,170,50,.08); }
  .comp-tile.bpa .comp-lbl { color: #dcaa32; }
  .fill-badge { display:inline-block; font-size:10px; padding:1px 5px; border-radius:3px; margin-left:4px; }
  .fill-held { background:rgba(0,200,150,.15); color:var(--teal); }
  .fill-partial { background:rgba(255,200,50,.15); color:#cca000; }
  .fill-recovered { background:rgba(100,180,255,.15); color:#5ba8e6; }
  .fill-failed { background:rgba(255,80,80,.15); color:#e05050; }

  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }
  .bg-buy   { background: rgba(0,200,150,.18); color: #00c896; border: 1px solid rgba(0,200,150,.35); }
  .bg-sell  { background: rgba(224,85,85,.18);  color: #e05555; border: 1px solid rgba(224,85,85,.35); }
  .bg-wait  { background: rgba(245,200,66,.15); color: #f5c842; border: 1px solid rgba(245,200,66,.30); }
  .bg-fog   { background: rgba(245,166,35,.13); color: #f5a623; border: 1px solid rgba(245,166,35,.28); }
  .bg-avoid { background: rgba(85,85,85,.25);   color: #888;    border: 1px solid rgba(85,85,85,.45); }

  /* Movement */
  .movement { font-size: 11px; color: var(--sub); margin-top: 2px; }
  .up   { color: var(--teal); }
  .down { color: var(--red); }
  .delta-up   { color: var(--teal); font-weight: 600; }
  .delta-down { color: var(--red);  font-weight: 600; }

  /* Expanded chart area */
  .chart-wrap {
    position: relative;
    padding: 10px 12px 12px;
    background: #111;
  }
  .chart-wrap img {
    width: 100%; height: auto;
    display: block; border-radius: 4px;
  }
  .chart-overlay {
    position: absolute;
    top: 16px; right: 18px;
    background: rgba(0,0,0,.55);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
    line-height: 1.6;
    color: rgba(255,255,255,.75);
    pointer-events: none;
  }
  .no-chart {
    padding: 18px 12px;
    font-size: 12px; color: var(--sub);
    text-align: center;
  }
  .warn {
    margin: 0 12px 8px;
    padding: 6px 10px;
    background: rgba(245,166,35,.1);
    border-left: 3px solid var(--orange);
    border-radius: 3px;
    font-size: 12px; color: var(--orange);
  }
  .summary-text {
    padding: 8px 12px 10px;
    font-size: 12px; color: var(--sub); line-height: 1.5;
  }
  footer {
    margin-top: 14px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 11px; color: var(--sub);
    text-align: center;
  }

  /* Mobile layout: compact 3-row grid (not fully stacked)
     Row 1: rank | ticker (+family tag) | signal badge (right)
     Row 2: $ADR  move×  | ADR× pill (right)
     Row 3: URG bar+val  |  UNC bar+val (right) */
  @media (max-width: 500px) {
    .card { margin-bottom: 6px; }
    .card summary {
      padding: 8px 10px;
      grid-template-columns: auto minmax(0,1fr) auto;
      grid-template-areas:
        "rank  ticker  badge"
        "adr   adr     adrmult"
        "urg   urg     unc";
      row-gap: 6px;
      column-gap: 8px;
      align-items: center;
    }
    .rank { grid-area: rank; font-size: 11px; margin: 0; }
    .ticker-block { grid-area: ticker; min-width: 0; margin: 0; }
    .ticker { font-size: 15px; }
    .day-type { font-size: 10px; }

    .adr-block {
      grid-area: adr; margin: 0; padding: 0;
      display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;
    }
    .adr-val { font-size: 11px; }
    .adr-ratio { font-size: 11px; }

    .adr-mult {
      grid-area: adrmult; padding: 3px 8px; min-width: 0; margin: 0;
      justify-self: end;
    }
    .adr-mult .mult-val { font-size: 12px; }
    .adr-mult .mult-lbl { display: none; }

    /* Break .scores into its 3 children and place each in its own grid cell */
    .scores { display: contents; }
    .scores > :nth-child(1) { grid-area: badge; justify-self: end; }
    .scores > :nth-child(2) { grid-area: urg; }
    .scores > :nth-child(3) { grid-area: unc; justify-self: end; }

    .score-row { gap: 4px; }
    .score-label { width: auto; font-size: 10px; }
    .bar-track { width: 60px; }
    .score-val { width: auto; min-width: 22px; font-size: 11px; }
    .badge { font-size: 10px; padding: 2px 6px; }
  }
</style>
</head>
<body>
"""

_HTML_FOOT = """\
<footer>Auto-refreshes every 5 min &nbsp;·&nbsp; Brooks Price Action Scanner</footer>
<script>
(function(){
  var container = document.getElementById('cards');
  if (!container) return;
  var buttons = document.querySelectorAll('.sort-bar button[data-sort]');
  var originals = Array.prototype.slice.call(container.querySelectorAll('details.card'));

  var KEY_MAP = {
    'rank': 'rank',
    'urgency': 'urgency',
    'uncertainty': 'uncertainty',
    'adr-mult': 'adrMult'
  };
  function sortBy(key){
    var dkey = KEY_MAP[key] || key;
    var cards = Array.prototype.slice.call(container.querySelectorAll('details.card'));
    if (key === 'rank'){
      cards.sort(function(a,b){
        return (parseFloat(a.dataset.rank)||0) - (parseFloat(b.dataset.rank)||0);
      });
    } else {
      cards.sort(function(a,b){
        return (parseFloat(b.dataset[dkey])||0) - (parseFloat(a.dataset[dkey])||0);
      });
    }
    cards.forEach(function(c){ container.appendChild(c); });
  }

  buttons.forEach(function(btn){
    btn.addEventListener('click', function(){
      buttons.forEach(function(b){ b.classList.remove('active'); });
      btn.classList.add('active');
      sortBy(btn.dataset.sort);
    });
  });
})();
</script>
</body>
</html>
"""


def _signal_badge(sig: str) -> str:
    css_cls, label = _SIG_CSS.get(sig, ("bg-avoid", sig))
    return f'<span class="badge {css_cls}">{label}</span>'


def _rank_arrow_html(r: dict) -> str:
    rc = r.get("rank_change")
    if r.get("_first_scan") or rc is None:
        return ""
    if rc > 0:
        return f'<span class="arrow up">▲</span>'
    if rc < 0:
        return f'<span class="arrow down">▼</span>'
    return '<span class="arrow" style="color:#555">—</span>'


def _movement_html(r: dict) -> str:
    if r.get("_first_scan"):
        return '<span class="movement">first scan</span>'
    rc = r.get("rank_change")
    ud = r.get("urgency_delta")
    parts = []
    if rc is None:
        parts.append('<span class="movement up">NEW</span>')
    else:
        pr = r["prev_rank"]
        sign = "+" if rc > 0 else ""
        css = "up" if rc > 0 else ("down" if rc < 0 else "")
        parts.append(f'<span class="movement {css}">was #{pr} ({sign}{rc})</span>')
    if ud is not None:
        sign = "+" if ud >= 0 else ""
        css = "delta-up" if ud >= 0 else "delta-down"
        parts.append(f'<span class="{css}" style="font-size:11px;margin-left:6px">{sign}{ud:.1f} U</span>')
    return " ".join(parts)


def _bar_html(value: float, cls: str) -> str:
    pct = min(100, max(0, value * 10))
    return (
        f'<div class="bar-track">'
        f'<div class="bar-fill {cls}" style="width:{pct:.0f}%"></div>'
        f'</div>'
    )


def _adr_mult_tier(mult: float) -> str:
    """Return CSS tier class for an ADR multiple (matches chart header thresholds)."""
    if mult <= 0:
        return "tier-cold"
    if mult < 1.0:
        return "tier-cold"
    if mult < 1.5:
        return "tier-warm"
    if mult < 2.0:
        return "tier-hot"
    return "tier-extreme"


def _build_component_strip(details: dict) -> str:
    """Mini per-component subscore strip for the card.

    Order: spike · gap · pull · FT · MA · vol · tail · SPT (new)
    Each tile shows label + rounded value. Dimmed when value == 0.
    SPT tile gets a teal accent so it reads as the newest signal.
    """
    if not details:
        return ""
    _TILE_TIPS = {
        "spike": "Spike Quality (0–4): Strength of opening move — strong body ratio, closes near extreme, small tails",
        "gap":   "Gap Integrity (-2–+2): Did the gap hold? +2=intact, +1=partial fill, 0=filled but recovered, -2=filled and failed",
        "pull":  "Pullback Quality (-1–+2): Depth and character of first pullback — shallow + tight = bullish",
        "FT":    "Follow Through (-1.5–+2): Do bars after the spike continue in direction? Consecutive trend bars = strong",
        "MA":    "MA Separation (0–1): Price distance from 20-EMA — wider spread = stronger trend",
        "vol":   "Volume Confirmation (0–1): Volume expanding on trend bars, contracting on pullback bars",
        "tail":  "Tail Quality (-0.5–+1): Wicks on the right side (rejection tails) — tails favoring direction = bullish",
        "SPT":   "Small Pullback Trend (0–3): Calm drifting trend with tiny pullbacks — the strongest continuation pattern",
        "BPA":   "BPA Alignment (-1–+2): Brooks pattern confirmation — H2/L2=+2, H1/L1=+1.5, opposing=-1. Overlays the signal, not the urgency score",
    }
    tiles = [
        ("spike", details.get("spike_quality", 0.0), ""),
        ("gap",   details.get("gap_integrity", 0.0), ""),
        ("pull",  details.get("pullback_quality", 0.0), ""),
        ("FT",    details.get("follow_through", 0.0), ""),
        ("MA",    details.get("ma_separation", 0.0), ""),
        ("vol",   details.get("volume_conf", 0.0), ""),
        ("tail",  details.get("tail_quality", 0.0), ""),
        ("SPT",   details.get("small_pullback_trend", 0.0), "spt"),
        ("BPA",   details.get("bpa_alignment", 0.0), "bpa"),
    ]
    parts = ['<div class="comp-strip">']
    for label, val, extra_cls in tiles:
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = 0.0
        dim = " dim" if abs(v) < 0.005 else ""
        cls = f"comp-tile {extra_cls}{dim}".strip()
        sign = "" if v >= 0 else "−"
        av = abs(v)
        val_str = f"{sign}{av:.1f}" if av < 10 else f"{sign}{av:.0f}"
        tip = _TILE_TIPS.get(label, label)
        parts.append(
            f'<span class="{cls}" title="{tip}">'
            f'<span class="comp-lbl">{label}</span>'
            f'<span class="comp-val">{val_str}</span>'
            f'</span>'
        )
    parts.append('</div>')
    return "".join(parts)


def _build_card_html(r: dict, chart_b64: str | None) -> str:
    ticker   = r.get("ticker", "?")
    sig      = r.get("signal", "PASS")
    urg      = r.get("urgency", 0.0)
    unc      = r.get("uncertainty", 0.0)
    day_type = r.get("day_type", "")
    summary  = r.get("summary", "")
    warn     = r.get("day_type_warning", "")
    pc       = r.get("_prior_close", 0.0)
    rank     = r.get("rank", 0)

    # ADR / move_ratio
    adr_val   = r.get("daily_atr", 0.0) or 0.0
    move_rat  = r.get("move_ratio", 0.0) or 0.0
    adr_mult  = r.get("adr_multiple", 0.0) or 0.0
    adr_str   = f"${adr_val:.2f} ADR" if adr_val > 0 else ""
    ratio_str = f"{move_rat:.1f}× move" if adr_val > 0 else ""

    # Rank block
    rank_html = (
        f'<div class="rank">'
        f'{_rank_arrow_html(r)}'
        f'<span style="font-size:13px;font-weight:600;color:{"#e0e0e0"}">{rank}</span>'
        f'</div>'
    )

    # Ticker + day type (+ family-dedup badge when this row represents a family)
    fam      = r.get("family")
    fam_sibs = r.get("family_siblings") or []
    fam_n    = len(fam_sibs)
    if fam and fam_n:
        sib_tip   = ", ".join(fam_sibs)
        fam_badge = (
            f'<span class="fam-badge" '
            f'title="represents {fam} family ({fam_n + 1} tickers): {ticker}, {sib_tip}" '
            f'style="display:inline-block;margin-left:6px;padding:1px 6px;'
            f'font-size:10px;font-weight:600;letter-spacing:0.3px;'
            f'background:#2b3b52;color:#8ab4f8;border:1px solid #3a5272;'
            f'border-radius:10px;vertical-align:middle;">'
            f'{fam}&nbsp;+{fam_n}'
            f'</span>'
        )
    else:
        fam_badge = ""

    # Cycle-phase badge (Layer 1 classifier) — shown inline with day_type
    details_r = r.get("details") or {}
    cp = details_r.get("cycle_phase") or {}
    cp_label_map = {
        "bull_spike": "↑ SPIKE",
        "bear_spike": "↓ SPIKE",
        "bull_channel": "↑ channel",
        "bear_channel": "↓ channel",
        "trading_range": "↔ range",
    }
    cp_top = cp.get("top") if isinstance(cp, dict) else None
    cp_conf = cp.get("confidence", 0.0) if isinstance(cp, dict) else 0.0
    if cp_top and cp_conf >= 0.30:
        cp_html = (
            f'<span class="cycle-badge" '
            f'title="Cycle phase (Layer 1) — top-1 with confidence. Higher is cleaner.">'
            f'{cp_label_map.get(cp_top, cp_top)} {cp_conf:.2f}'
            f'</span>'
        )
    else:
        cp_html = ""

    # Gap fill-status badge
    gfs = details_r.get("gap_fill_status", "")
    _fill_cls = {"held": "fill-held", "partial_fill": "fill-partial",
                 "filled_recovered": "fill-recovered", "filled_failed": "fill-failed"}
    _fill_label = {"held": "gap held", "partial_fill": "partial fill",
                   "filled_recovered": "fill recovered", "filled_failed": "fill failed"}
    if gfs and gfs != "held":
        fill_badge = f'<span class="fill-badge {_fill_cls.get(gfs, "")}">{_fill_label.get(gfs, gfs)}</span>'
    else:
        fill_badge = ""

    ticker_html = (
        f'<div class="ticker-block">'
        f'<div class="ticker">{ticker}{fam_badge}</div>'
        f'<div class="day-type">{day_type} {cp_html} {fill_badge}</div>'
        f'</div>'
    )

    # ADR block (fills the 1fr spacer column)
    adr_html = (
        f'<div class="adr-block">'
        f'<div class="adr-val">{adr_str}</div>'
        f'<div class="adr-ratio">{ratio_str}</div>'
        f'</div>'
    )

    # ADR-multiple tile (Minervini-style expansion gauge)
    tier = _adr_mult_tier(adr_mult)
    if adr_mult > 0:
        mult_tile = (
            f'<div class="adr-mult {tier}">'
            f'<div class="mult-val">{adr_mult:.2f}\u00d7</div>'
            f'<div class="mult-lbl">ADR</div>'
            f'</div>'
        )
    else:
        mult_tile = (
            f'<div class="adr-mult tier-cold">'
            f'<div class="mult-val" style="font-size:12px">—</div>'
            f'<div class="mult-lbl">ADR</div>'
            f'</div>'
        )

    # Scores + badge
    scores_html = (
        f'<div class="scores">'
        f'<div>'
        f'{_signal_badge(sig)}'
        f'</div>'
        f'<div class="score-row">'
        f'<span class="score-label">URG</span>'
        f'{_bar_html(urg, "bar-urg")}'
        f'<span class="score-val" style="color:#00c896">{urg:.1f}</span>'
        f'</div>'
        f'<div class="score-row">'
        f'<span class="score-label">UNC</span>'
        f'{_bar_html(unc, "bar-unc")}'
        f'<span class="score-val" style="color:#e05555">{unc:.1f}</span>'
        f'</div>'
        f'</div>'
    )

    # Movement
    movement_html = f'<div style="padding:0 12px 8px;display:flex;gap:8px;align-items:center">{_movement_html(r)}</div>'

    # Per-component subscore strip (spike · gap · pull · FT · MA · vol · tail · SPT)
    comp_strip_html = _build_component_strip(r.get("details") or {})

    # Chart area
    if chart_b64:
        overlay = f'URG {urg:.1f} / UNC {unc:.1f}'
        if adr_mult > 0:
            overlay += f' · {adr_mult:.2f}× ADR'
        elif adr_val > 0:
            overlay += f' · {ratio_str}'
        chart_html = (
            f'<div class="chart-wrap">'
            f'<img src="data:image/png;base64,{chart_b64}" alt="{ticker} 5-min chart">'
            f'<div class="chart-overlay">{overlay}</div>'
            f'</div>'
        )
    else:
        chart_html = '<div class="no-chart">Chart not available</div>'

    # Warning
    warn_html = ""
    if warn and warn.lower() not in ("", "none"):
        warn_html = f'<div class="warn">⚠ {warn}</div>'

    # Summary
    summary_html = f'<div class="summary-text">{summary}</div>' if summary else ""

    # Sortable data attrs — consumed by the dashboard sort toolbar JS
    data_attrs = (
        f'data-urgency="{urg:.3f}" '
        f'data-uncertainty="{unc:.3f}" '
        f'data-adr-mult="{adr_mult:.3f}" '
        f'data-rank="{rank}" '
        f'data-ticker="{ticker}"'
    )

    return (
        f'<details class="card" {data_attrs}>'
        f'<summary>'
        f'{rank_html}'
        f'{ticker_html}'
        f'{adr_html}'
        f'{mult_tile}'
        f'{scores_html}'
        f'</summary>'
        f'{movement_html}'
        f'{comp_strip_html}'
        f'{warn_html}'
        f'{summary_html}'
        f'{chart_html}'
        f'</details>\n'
    )


# ── aiedge.trade Integration ──────────────────────────────────────────────────

AIEDGE_SCAN_URL = "https://www.aiedge.trade/api/scan"  # apex redirects 307 → www and urllib drops POST body on redirect

def _serialize_bars(df_5m: pd.DataFrame, last_n: int = 80) -> list[dict]:
    """5-min OHLCV DataFrame → compact bars list for ScanResult.chart.bars."""
    if df_5m is None or len(df_5m) == 0:
        return []
    df = df_5m.copy()
    # chart_renderer expects a "datetime" column, but some callers pass an index.
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        return []
    df = df.sort_index().tail(last_n)
    bars: list[dict] = []
    for ts, row in df.iterrows():
        try:
            bar = {
                "t": int(pd.Timestamp(ts).timestamp()),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        vol = row.get("volume") if hasattr(row, "get") else None
        if vol is not None:
            try:
                bar["v"] = float(vol)
            except (TypeError, ValueError):
                pass
        bars.append(bar)
    return bars


def _serialize_key_levels(
    prior_close: float | None,
    levels: dict | None,
) -> dict | None:
    """Map internal levels dict → KeyLevels shape the site expects, or None."""
    out: dict[str, float] = {}
    if prior_close and prior_close > 0:
        out["priorClose"] = float(prior_close)
    if levels:
        def _maybe(src_key: str, dst_key: str) -> None:
            v = levels.get(src_key)
            if v is not None:
                try:
                    out[dst_key] = float(v)
                except (TypeError, ValueError):
                    pass
        _maybe("pdh", "priorDayHigh")
        _maybe("pdl", "priorDayLow")
        _maybe("onh", "overnightHigh")
        _maybe("onl", "overnightLow")
        _maybe("pmh", "premarketHigh")
        _maybe("pml", "premarketLow")
    return out or None


def _serialize_scan_payload(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    df5m_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Convert internal result dicts to the ScanPayload JSON format for aiedge.trade."""
    time_str = now_et.strftime("%I:%M %p").lstrip("0") + " ET"
    date_str = now_et.strftime("%Y-%m-%d")
    next_str = _next_scan_time_str(now_et, interval_min, first_scan_hour, first_scan_min)

    def _adr_tier(mult: float) -> str:
        if mult >= 2.0: return "extreme"
        if mult >= 1.5: return "hot"
        if mult >= 1.0: return "warm"
        return "cold"

    def _map_fill_status(gfs: str) -> str | None:
        m = {"held": "held", "partial_fill": "partial", "filled_recovered": "recovered", "filled_failed": "failed"}
        return m.get(gfs)

    def _map_signal(sig: str) -> str:
        sig_up = sig.upper()
        if "BUY" in sig_up: return "BUY"
        if "SELL" in sig_up: return "SELL"
        if sig_up == "WAIT": return "WAIT"
        if sig_up == "FOG": return "FOG"
        if sig_up in ("AVOID", "PASS"): return "AVOID"
        return "AVOID"

    serialized = []
    for r in results:
        details = r.get("details") or {}
        cp = details.get("cycle_phase") or {}
        cp_top = cp.get("top") if isinstance(cp, dict) else None
        cp_conf = cp.get("confidence", 0.0) if isinstance(cp, dict) else 0.0
        cp_label_map = {
            "bull_spike": "↑ SPIKE",
            "bear_spike": "↓ SPIKE",
            "bull_channel": "↑ channel",
            "bear_channel": "↓ channel",
            "trading_range": "↔ range",
        }
        cycle_phase = None
        if cp_top and cp_conf >= 0.30:
            cycle_phase = f"{cp_label_map.get(cp_top, cp_top)} {cp_conf:.2f}"

        gfs = details.get("gap_fill_status", "")
        fill_status = _map_fill_status(gfs) if gfs and gfs != "held" else None

        adr_mult = r.get("adr_multiple", 0.0) or 0.0

        ticker = r.get("ticker", "?")
        chart_obj: dict | None = None
        if df5m_map is not None:
            df5 = df5m_map.get(ticker)
            bars = _serialize_bars(df5) if df5 is not None else []
            if bars:
                chart_obj = {"bars": bars, "timeframe": "5min"}
                kl = _serialize_key_levels(
                    r.get("_prior_close"),
                    intraday_levels.get(ticker),
                )
                if kl:
                    chart_obj["keyLevels"] = kl

        entry = {
            "ticker": ticker,
            "rank": r.get("rank", 0),
            "urgency": round(r.get("urgency", 0.0), 1),
            "uncertainty": round(r.get("uncertainty", 0.0), 1),
            "signal": _map_signal(r.get("signal", "PASS")),
            "dayType": (r.get("day_type", "") or "").replace(" ", "_"),
            "adr": round(r.get("daily_atr", 0.0) or 0.0, 2),
            "adrRatio": round(r.get("move_ratio", 0.0) or 0.0, 1),
            "adrMult": round(adr_mult, 2),
            "adrTier": _adr_tier(adr_mult),
            "movement": _fmt_movement(r),
            "components": {
                "spike": round(float(details.get("spike_quality", 0.0)), 1),
                "gap": round(float(details.get("gap_integrity", 0.0)), 1),
                "pull": round(float(details.get("pullback_quality", 0.0)), 1),
                "ft": round(float(details.get("follow_through", 0.0)), 1),
                "ma": round(float(details.get("ma_separation", 0.0)), 1),
                "vol": round(float(details.get("volume_conf", 0.0)), 1),
                "tail": round(float(details.get("tail_quality", 0.0)), 1),
                "spt": round(float(details.get("small_pullback_trend", 0.0)), 1),
                "bpa": round(float(details.get("bpa_alignment", 0.0)), 1),
            },
            "summary": r.get("summary", ""),
        }
        if cycle_phase:
            entry["cyclePhase"] = cycle_phase
        if fill_status:
            entry["fillStatus"] = fill_status
        if r.get("day_type_warning"):
            entry["warning"] = r["day_type_warning"]
        if chart_obj:
            entry["chart"] = chart_obj

        serialized.append(entry)

    return {
        "timestamp": time_str,
        "date": date_str,
        "symbolsScanned": total_symbols,
        "passedFilters": passed,
        "scanTime": f"{elapsed:.2f}s",
        "nextScan": next_str,
        "results": serialized,
    }


def _post_to_aiedge(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    df5m_map: dict[str, pd.DataFrame] | None = None,
) -> None:
    """POST scan results to aiedge.trade/api/scan. Fire-and-forget in bg thread."""
    try:
        import os
        import urllib.request
        sync_secret = os.environ.get("SYNC_SECRET")
        if not sync_secret:
            logger.warning(
                "aiedge.trade POST skipped: SYNC_SECRET env var not set. "
                "Export it so the scanner can authenticate with /api/scan."
            )
            return
        payload = _serialize_scan_payload(
            results, now_et, total_symbols, passed, elapsed, interval_min, df5m_map
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


def _generate_dashboard(
    results: list[dict],
    df5m_map: dict[str, pd.DataFrame],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    intraday_levels: dict,
    dashboard_path: Path,
    first_scan_hour: int,
    first_scan_min: int,
) -> None:
    """Render charts + build HTML dashboard. Runs in background thread."""
    t0 = time.monotonic()
    logger.info(f"Dashboard: rendering charts for {len(results)} stocks…")

    # Render charts
    chart_b64_map: dict[str, str | None] = {}
    for r in results:
        ticker = r["ticker"]
        df5 = df5m_map.get(ticker)
        if df5 is not None and len(df5) >= 2:
            chart_b64_map[ticker] = render_chart_base64(
                df5, ticker, r.get("_prior_close", 0),
                adr_multiple=r.get("adr_multiple"),
                levels=intraday_levels.get(ticker),
            )
        else:
            chart_b64_map[ticker] = None

    charts_ok = sum(1 for v in chart_b64_map.values() if v is not None)
    logger.info(f"Dashboard: {charts_ok}/{len(results)} charts rendered in {time.monotonic()-t0:.1f}s")

    # Build HTML
    time_str = now_et.strftime("%I:%M %p").lstrip("0")
    next_str = _next_scan_time_str(now_et, interval_min, first_scan_hour, first_scan_min)

    body_parts = [_HTML_HEAD]
    body_parts.append(
        f'<header>'
        f'<h1>Live Scanner</h1>'
        f'<div class="meta">{time_str} ET &nbsp;·&nbsp; {now_et.strftime("%Y-%m-%d")}</div>'
        f'</header>\n'
    )
    body_parts.append(
        f'<div class="stats">'
        f'<span>📡 {total_symbols:,} symbols</span>'
        f'<span>✅ {passed} passed filters</span>'
        f'<span>⏱ {elapsed:.2f}s</span>'
        f'<span>Next: {next_str}</span>'
        f'</div>\n'
    )

    # Sort toolbar — JS in footer wires the clicks to reorder cards in place.
    body_parts.append(
        '<div class="sort-bar">'
        '<span>Sort:</span>'
        '<button data-sort="rank" class="active">Rank</button>'
        '<button data-sort="urgency">Urgency</button>'
        '<button data-sort="adr-mult">ADR ×</button>'
        '<button data-sort="uncertainty">Uncertainty</button>'
        '</div>\n'
    )

    # Scoring legend — collapsible
    body_parts.append(
        '<details class="legend">'
        '<summary>How Scoring Works</summary>'
        '<div class="legend-body">'

        '<div class="legend-section">'
        '<h3>Main Scores</h3>'
        '<dl>'
        '<dt>URG (Urgency 0–10)</dt>'
        '<dd>How strongly the chart is pulling in one direction. Weighted sum of 16 components below, normalized by day type. Higher = clearer trend.</dd>'
        '<dt>UNC (Uncertainty 0–10)</dt>'
        '<dd>How confused or two-sided the chart is. Overlapping bars, dojis, alternating colors. Higher = harder to read.</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>Signals</h3>'
        '<dl>'
        '<dt>BUY PULLBACK / SELL PULLBACK</dt>'
        '<dd>High urgency + low uncertainty + good R:R. The highest-confidence entry — trend is strong, pullback gives you a stop.</dd>'
        '<dt>BUY SPIKE / SELL SPIKE</dt>'
        '<dd>Strong directional move with no pullback yet. Market is leaving — consider market order with wide stop.</dd>'
        '<dt>SELL FLIP / BUY FLIP</dt>'
        '<dd>Intraday direction reversal. Always-in direction flipped against the gap, confirmed by BPA pattern (L2/H2).</dd>'
        '<dt>WAIT</dt>'
        '<dd>Promising direction but needs more bars. Urgency decent but not actionable yet.</dd>'
        '<dt>FOG</dt>'
        '<dd>Can\'t read the chart. High uncertainty regardless of urgency. Sit on hands.</dd>'
        '<dt>AVOID</dt>'
        '<dd>Gap failed (price through prior close) or trap state (high urgency + high uncertainty — both sides showing strength).</dd>'
        '<dt>PASS</dt>'
        '<dd>Readable but weak. No urgency, no edge.</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>Urgency Components (hover tiles for details)</h3>'
        '<dl>'
        '<dt>spike (0–4)</dt><dd>Opening spike strength — body ratio, close location, tail size on first 1–4 trend bars</dd>'
        '<dt>gap (-2–+2)</dt><dd>Gap integrity — did price hold the gap? +2 intact, -2 filled. Post-fill recovery analysis: filled_recovered avoids the -2 penalty</dd>'
        '<dt>pull (-1–+2)</dt><dd>First pullback quality — shallow (good) vs deep (bad), tight vs loose</dd>'
        '<dt>FT (-1.5–+2)</dt><dd>Follow through — are bars after the spike continuing in direction?</dd>'
        '<dt>MA (0–1)</dt><dd>Moving average separation — price distance from 20-EMA</dd>'
        '<dt>vol (0–1)</dt><dd>Volume confirmation — volume expanding on trend bars, contracting on pullbacks</dd>'
        '<dt>tail (-0.5–+1)</dt><dd>Tail quality — wicks rejecting the wrong direction</dd>'
        '<dt>SPT (0–3)</dt><dd>Small pullback trend — calm drifting trend with tiny pullbacks, strongest continuation</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>BPA Overlay (not part of urgency sum)</h3>'
        '<dl>'
        '<dt>BPA (-1–+2)</dt><dd>Brooks pattern alignment — runs 8 pattern detectors (H1, H2, L1, L2, FL1, FL2, spike&amp;channel, failed breakout). '
        'Modifies the signal after urgency/uncertainty are computed: strong in-direction pattern can upgrade WAIT→BUY, opposing pattern can downgrade BUY→WAIT</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>Other Indicators</h3>'
        '<dl>'
        '<dt>ADR ×</dt><dd>Today\'s range as a multiple of the 20-day average daily range. &gt;1.0 = expanding beyond normal, &lt;0.5 = quiet day</dd>'
        '<dt>Fill Status</dt><dd>Gap fill analysis: held (gap intact), partial fill (&lt;50% filled), fill recovered (filled but price recovered), fill failed (filled, no recovery)</dd>'
        '</dl>'
        '</div>'

        '</div>'
        '</details>\n'
    )

    body_parts.append('<div id="cards">\n')
    if not results:
        body_parts.append('<p style="color:var(--sub);padding:20px 0">No liquid stocks passed filters this cycle.</p>\n')
    else:
        for r in results:
            ticker = r["ticker"]
            body_parts.append(_build_card_html(r, chart_b64_map.get(ticker)))
    body_parts.append('</div>\n')

    body_parts.append(_HTML_FOOT)
    html = "".join(body_parts)

    try:
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Dashboard saved → {dashboard_path} ({len(html)//1024}KB)")
    except Exception as e:
        logger.warning(f"Dashboard write failed: {e}")
