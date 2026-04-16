#!/usr/bin/env python3
"""Generate a test dashboard HTML with sample data."""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from live_scanner import (
    _HTML_HEAD, _HTML_FOOT, _build_card_html, DASHBOARD_PATH
)

# Sample data for testing
sample_results = [
    {
        "rank": 1,
        "ticker": "SPY",
        "signal": "BUY_PULLBACK",
        "urgency": 7.5,
        "uncertainty": 2.1,
        "daily_atr": 3.45,
        "move_ratio": 0.8,
        "adr_multiple": 1.25,
        "day_type": "Normal distribution",
        "summary": "Strong uptrend with healthy pullback.",
        "_first_scan": False,
        "rank_change": 2,
        "prev_rank": 3,
        "urgency_delta": 0.5,
    },
    {
        "rank": 2,
        "ticker": "QQQ",
        "signal": "BUY_SPIKE",
        "urgency": 8.2,
        "uncertainty": 1.8,
        "daily_atr": 4.12,
        "move_ratio": 1.3,
        "adr_multiple": 1.75,
        "day_type": "Breakout",
        "summary": "Gap up with momentum continuation.",
        "_first_scan": False,
        "rank_change": -1,
        "prev_rank": 1,
        "urgency_delta": 0.3,
    },
    {
        "rank": 3,
        "ticker": "TQQQ",
        "signal": "WAIT",
        "urgency": 5.1,
        "uncertainty": 4.5,
        "daily_atr": 12.50,
        "move_ratio": 2.1,
        "adr_multiple": 2.15,
        "day_type": "High volatility",
        "summary": "Too uncertain at current levels.",
        "_first_scan": False,
        "rank_change": 0,
        "prev_rank": 3,
        "urgency_delta": -0.1,
    },
    {
        "rank": 4,
        "ticker": "AMZN",
        "signal": "SELL_PULLBACK",
        "urgency": 6.3,
        "uncertainty": 3.2,
        "daily_atr": 2.78,
        "move_ratio": 0.6,
        "adr_multiple": 0.95,
        "day_type": "Distribution",
        "summary": "Weakness in sector.",
        "_first_scan": False,
        "rank_change": 1,
        "prev_rank": 5,
        "urgency_delta": 0.2,
    },
    {
        "rank": 5,
        "ticker": "SQQQ",
        "signal": "FOG",
        "urgency": 3.7,
        "uncertainty": 6.1,
        "daily_atr": 0.85,
        "move_ratio": 0.3,
        "adr_multiple": 0.42,
        "day_type": "Consolidation",
        "summary": "Waiting for clarity.",
        "_first_scan": False,
        "rank_change": None,
        "prev_rank": None,
        "urgency_delta": None,
    },
]

# Create output directory
DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

# Build HTML
now_et = datetime.now(timezone.utc)
body_parts = [_HTML_HEAD]

# Header
time_str = now_et.strftime("%I:%M %p").lstrip("0")
body_parts.append(
    f'<header>'
    f'<h1>Live Scanner</h1>'
    f'<div class="meta">{time_str} ET &nbsp;·&nbsp; {now_et.strftime("%Y-%m-%d")}</div>'
    f'</header>\n'
)

# Stats
body_parts.append(
    f'<div class="stats">'
    f'<span>📡 2,500 symbols</span>'
    f'<span>✅ 127 passed filters</span>'
    f'<span>⏱ 1.23s</span>'
    f'<span>Next: 2:30 PM</span>'
    f'</div>\n'
)

# Sort toolbar
body_parts.append(
    '<div class="sort-bar">'
    '<span>Sort:</span>'
    '<button data-sort="rank" class="active">Rank</button>'
    '<button data-sort="urgency">Urgency</button>'
    '<button data-sort="adr-mult">ADR ×</button>'
    '<button data-sort="uncertainty">Uncertainty</button>'
    '</div>\n'
)

# Cards
body_parts.append('<div id="cards">\n')
for r in sample_results:
    body_parts.append(_build_card_html(r, None))
body_parts.append('</div>\n')

# Footer
body_parts.append(_HTML_FOOT)

html = "".join(body_parts)

# Write to file
with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard generated: {DASHBOARD_PATH}")
print(f"File size: {len(html) // 1024} KB")
