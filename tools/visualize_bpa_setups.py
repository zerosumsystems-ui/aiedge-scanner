"""
Render a candlestick diagram for every BPA setup using the canonical fixtures
from tests/test_bpa_detector.py. Each diagram shows:
  - OHLC bars (green bull, red bear, grey doji)
  - Signal bar highlighted in gold
  - Entry / stop / target as horizontal lines (green / red / blue)
  - Structural annotations (leg top, pullback, H1, L1, etc.)
  - R:R and entry mode in the corner

Outputs individual PNGs to assets/bpa_setups/ plus a composite grid.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_bpa_detector import (  # noqa: E402
    _h1_canonical, _h2_canonical, _l1_canonical, _l2_canonical,
    _failed_bo_up_canonical, _spike_channel_canonical,
    _fl1_canonical, _fl2_canonical, _fh1_canonical,
)
from shared.bpa_detector import (  # noqa: E402
    _detect_h1, _detect_h2, _detect_l1, _detect_l2,
    _detect_fl1, _detect_fl2, _detect_fh1, _detect_fh2,
    _detect_spike_and_channel, _detect_failed_breakout,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "bpa_setups"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _df(bars):
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"]).assign(volume=1000)


def _fh2_fixture():
    return _df([
        (100.0, 101.0, 99.5, 100.9),
        (100.9, 102.0, 100.5, 101.9),
        (101.9, 103.5, 101.5, 103.4),
        (103.4, 105.0, 103.3, 104.9),
        (104.9, 104.95, 104.0, 104.1),
        (104.1, 104.2, 103.5, 103.6),
        (103.6, 104.7, 103.55, 104.6),
        (104.6, 104.65, 103.3, 103.4),
        (103.4, 103.6, 103.0, 103.1),
        (103.1, 103.3, 102.8, 103.0),
        (103.0, 104.5, 102.9, 104.4),
        (104.4, 104.45, 102.5, 102.6),
    ])


BODY_RATIO_DOJI = 0.30


def _bar_color(row):
    body = abs(row["close"] - row["open"])
    rng = row["high"] - row["low"]
    if rng <= 0 or (body / rng) < BODY_RATIO_DOJI:
        return "#888"  # doji
    return "#1b9e3f" if row["close"] > row["open"] else "#d62728"


def _draw_candles(ax, df, signal_idx, highlight_bars=None):
    highlight_bars = highlight_bars or {}
    width = 0.6
    for i, r in df.iterrows():
        color = _bar_color(r)
        body_low = min(r["open"], r["close"])
        body_high = max(r["open"], r["close"])
        body_h = max(body_high - body_low, (r["high"] - r["low"]) * 0.02)

        # Wick
        ax.plot([i, i], [r["low"], r["high"]], color=color, linewidth=1.2, zorder=2)
        # Body
        rect = patches.Rectangle(
            (i - width / 2, body_low), width, body_h,
            facecolor=color, edgecolor=color, zorder=3,
        )
        ax.add_patch(rect)

        # Highlight (gold ring around signal bar / labeled bar)
        if i == signal_idx:
            hi = r["high"] + (r["high"] - r["low"]) * 0.1
            lo = r["low"] - (r["high"] - r["low"]) * 0.1
            ax.add_patch(patches.Rectangle(
                (i - width / 2 - 0.1, lo), width + 0.2, hi - lo,
                fill=False, edgecolor="#f0b400", linewidth=2.5, zorder=4,
            ))

        label = highlight_bars.get(i)
        if label:
            ax.annotate(
                label, xy=(i, r["high"]), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color="#333",
            )


def _draw_level(ax, price, label, color, xmin, xmax):
    ax.axhline(price, color=color, linewidth=1.3, linestyle="--", alpha=0.8, zorder=1)
    ax.text(
        xmax + 0.2, price, f" {label} {price:.2f}",
        va="center", ha="left", fontsize=9, color=color, fontweight="bold",
    )


def render_setup(ax, title, df, result, highlight_bars, subtitle=""):
    # Price range
    y_min = df["low"].min()
    y_max = df["high"].max()
    # Include entry/stop/target in y range
    if result:
        y_min = min(y_min, result.stop, result.target)
        y_max = max(y_max, result.stop, result.target)
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    x_min = -0.8
    x_max = len(df) - 0.2

    signal_idx = result.bar_index if result else -1
    _draw_candles(ax, df, signal_idx, highlight_bars)

    if result:
        _draw_level(ax, result.entry, "Entry", "#1b9e3f", x_min, x_max)
        _draw_level(ax, result.stop, "Stop", "#d62728", x_min, x_max)
        _draw_level(ax, result.target, "Target", "#1f77b4", x_min, x_max)

    ax.set_xlim(x_min, x_max + 3.2)  # extra right margin for level labels
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{title}", fontsize=11, fontweight="bold", loc="left")
    if subtitle:
        ax.text(
            x_min, y_max - (y_max - y_min) * 0.03, subtitle,
            fontsize=8, color="#555", va="top",
        )
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    # R:R badge — bottom-left inside axes, won't collide with price labels
    if result:
        risk = abs(result.entry - result.stop)
        reward = abs(result.target - result.entry)
        rr = reward / risk if risk else 0
        direction = "LONG" if result.entry > result.stop else "SHORT"
        ax.text(
            0.02, 0.02,
            f"{direction}   {result.entry_mode.upper()}   R:R {rr:.2f}x",
            transform=ax.transAxes, fontsize=9, ha="left", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f6", ec="#bbb"),
        )


# ── Per-setup annotations ──────────────────────────────────────────────────

SETUPS = [
    {
        "name": "H1",
        "title": "H1 — First Pullback Long  (STOP entry)",
        "subtitle": "up-leg → pullback → bull signal bar breaks prior high",
        "df": _h1_canonical(),
        "detector": _detect_h1,
        "annotations": {3: "leg top", 4: "pb", 5: "pb", 6: "H1"},
    },
    {
        "name": "H2",
        "title": "H2 — Second Buy Inside Extended Pullback  (STOP entry)",
        "subtitle": "H1 triggers, price drops below H1 low, H2 triggers on recovery",
        "df": _h2_canonical(),
        "detector": _detect_h2,
        "annotations": {3: "leg top", 6: "H1 (failed)", 7: "below H1 low", 10: "H2"},
    },
    {
        "name": "L1",
        "title": "L1 — First Rally Short  (STOP entry)",
        "subtitle": "down-leg → rally → bear signal bar breaks prior low",
        "df": _l1_canonical(),
        "detector": _detect_l1,
        "annotations": {3: "leg bot", 4: "rally", 5: "rally", 6: "L1"},
    },
    {
        "name": "L2",
        "title": "L2 — Second Sell Inside Extended Rally  (STOP entry)",
        "subtitle": "L1 triggers, price rallies above L1 high, L2 triggers on rejection",
        "df": _l2_canonical(),
        "detector": _detect_l2,
        "annotations": {3: "leg bot", 6: "L1 (failed)", 7: "above L1 high", 10: "L2"},
    },
    {
        "name": "failed_bo",
        "title": "failed_bo — Failed Breakout  (LIMIT entry at boundary)",
        "subtitle": "15+ bar range with boundary tests, breakout fails, reverse at the level",
        "df": _failed_bo_up_canonical(),
        "detector": _detect_failed_breakout,
        "annotations": {18: "breakout", 19: "failure"},
    },
    {
        "name": "spike_channel",
        "title": "spike_channel — With-Trend Continuation  (STOP entry)",
        "subtitle": "strong spike → shallow channel → pullback → continuation signal",
        "df": _spike_channel_canonical(),
        "detector": _detect_spike_and_channel,
        "annotations": {0: "spike", 2: "spike end", 7: "channel top", 8: "pb", 9: "signal"},
    },
    {
        "name": "FL1",
        "title": "FL1 — Failed L1 (Bull Trap Reversal)  (LIMIT at failed L1 low)",
        "subtitle": "L1 fires, fails to continue down, limit-buy at the L1 low",
        "df": _fl1_canonical(),
        "detector": _detect_fl1,
        "annotations": {6: "L1 (failed)", 7: "reversal"},
    },
    {
        "name": "FL2",
        "title": "FL2 — Failed L2 (Two Bear Traps)  (LIMIT at failed L2 low)",
        "subtitle": "L1 fails, price rallies, L2 fires and also fails",
        "df": _fl2_canonical(),
        "detector": _detect_fl2,
        "annotations": {6: "L1 (failed)", 10: "L2 (failed)", 11: "reversal"},
    },
    {
        "name": "FH1",
        "title": "FH1 — Failed H1 (Bull Trap Reversal)  (LIMIT at failed H1 high)",
        "subtitle": "H1 fires, fails to continue up, limit-sell at the H1 high",
        "df": _fh1_canonical(),
        "detector": _detect_fh1,
        "annotations": {6: "H1 (failed)", 7: "reversal"},
    },
    {
        "name": "FH2",
        "title": "FH2 — Failed H2 (Two Bull Traps)  (LIMIT at failed H2 high)",
        "subtitle": "H1 fails, price drops, H2 fires and also fails",
        "df": _fh2_fixture(),
        "detector": _detect_fh2,
        "annotations": {6: "H1 (failed)", 10: "H2 (failed)", 11: "reversal"},
    },
]


def main():
    # Individual PNGs
    for s in SETUPS:
        result = s["detector"](s["df"])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        render_setup(ax, s["title"], s["df"], result, s["annotations"], subtitle=s["subtitle"])
        fig.tight_layout()
        out_path = OUT_DIR / f"{s['name']}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out_path.relative_to(OUT_DIR.parent.parent)}")

    # Composite 2x5 grid
    fig, axes = plt.subplots(5, 2, figsize=(15, 22))
    for ax, s in zip(axes.flat, SETUPS):
        result = s["detector"](s["df"])
        render_setup(ax, s["title"], s["df"], result, s["annotations"], subtitle=s["subtitle"])
    fig.suptitle("Brooks Price Action — all 10 setups (canonical fixtures)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    composite = OUT_DIR / "all_setups.png"
    fig.savefig(composite, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {composite.relative_to(OUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
