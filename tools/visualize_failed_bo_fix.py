"""
Side-by-side proof that the failed_bo stop fix matters.

Renders two scenarios with a wild breakout bar, showing:
  - Where the OLD stop would have landed (breakout bar's high — hindsight)
  - Where the NEW stop lands  (RH + buffer — pre-breakout-knowable)

The canonical fixture has a tame breakout; on real charts a breakout can
spike 5-10× the typical bar size, which is where the old hindsight stop
would have set an absurd stop-loss level.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_bpa_detector import _failed_bo_up_canonical  # noqa: E402
from shared.bpa_detector import _detect_failed_breakout  # noqa: E402

OUT = ROOT / "assets" / "bpa_setups" / "failed_bo_stop_fix.png"


def _bar_color(row):
    body = abs(row["close"] - row["open"])
    rng = row["high"] - row["low"]
    if rng <= 0 or (body / rng) < 0.30:
        return "#888"
    return "#1b9e3f" if row["close"] > row["open"] else "#d62728"


def render_candles(ax, df, signal_idx):
    width = 0.6
    for i, r in df.iterrows():
        color = _bar_color(r)
        body_low = min(r["open"], r["close"])
        body_high = max(r["open"], r["close"])
        body_h = max(body_high - body_low, (r["high"] - r["low"]) * 0.02)
        ax.plot([i, i], [r["low"], r["high"]], color=color, linewidth=1.2, zorder=2)
        rect = patches.Rectangle(
            (i - width / 2, body_low), width, body_h,
            facecolor=color, edgecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        if i == signal_idx:
            hi = r["high"] + (r["high"] - r["low"]) * 0.1
            lo = r["low"] - (r["high"] - r["low"]) * 0.1
            ax.add_patch(patches.Rectangle(
                (i - width / 2 - 0.1, lo), width + 0.2, hi - lo,
                fill=False, edgecolor="#f0b400", linewidth=2.5, zorder=4,
            ))


def render_panel(ax, df, title):
    # Fire the detector
    result = _detect_failed_breakout(df.copy())

    # Compute what the OLD (hindsight) stop would have been
    breakout_bar_high = float(df.iloc[-2]["high"])
    old_stop = breakout_bar_high  # the bug
    new_stop = result.stop if result else None

    signal_idx = result.bar_index if result else len(df) - 1
    render_candles(ax, df, signal_idx)

    x_min = -0.8
    x_max = len(df) - 0.2
    y_min = df["low"].min() - 1
    y_max = df["high"].max() + 1

    # Draw the OLD stop level in orange dashed
    ax.axhline(old_stop, color="#ff6b35", linewidth=1.8, linestyle=":",
               alpha=0.85, zorder=1)
    ax.text(x_max + 0.2, old_stop,
            f"  OLD stop (hindsight) {old_stop:.2f}",
            va="center", ha="left", fontsize=9, color="#ff6b35", fontweight="bold")

    if result:
        # Entry (green dashed)
        ax.axhline(result.entry, color="#1b9e3f", linewidth=1.3,
                   linestyle="--", alpha=0.8)
        ax.text(x_max + 0.2, result.entry, f"  Entry {result.entry:.2f}",
                va="center", ha="left", fontsize=9, color="#1b9e3f", fontweight="bold")
        # NEW stop (red dashed)
        ax.axhline(new_stop, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9)
        ax.text(x_max + 0.2, new_stop, f"  NEW stop {new_stop:.2f}",
                va="center", ha="left", fontsize=9, color="#d62728", fontweight="bold")
        # Target
        ax.axhline(result.target, color="#1f77b4", linewidth=1.3,
                   linestyle="--", alpha=0.8)
        ax.text(x_max + 0.2, result.target, f"  Target {result.target:.2f}",
                va="center", ha="left", fontsize=9, color="#1f77b4", fontweight="bold")

    # Widen y-range to fit old_stop
    y_min = min(y_min, old_stop - 1, (new_stop or y_min) - 1)
    y_max = max(y_max, old_stop + 1, (new_stop or y_max) + 1)
    if result:
        y_min = min(y_min, result.target - 1)

    ax.set_xlim(x_min, x_max + 4.0)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    # Summary badge
    if result and new_stop and old_stop:
        risk_new = new_stop - result.entry
        risk_old = old_stop - result.entry
        ax.text(
            0.02, 0.97,
            f"New R: {risk_new:.2f}   Old R (hindsight): {risk_old:.2f}\n"
            f"Ratio old/new: {risk_old / risk_new:.2f}×",
            transform=ax.transAxes, fontsize=9, ha="left", va="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8ed", ec="#ff6b35"),
        )


def _mk_wild(df_base: pd.DataFrame, spike_amount: float) -> pd.DataFrame:
    """Replace the breakout bar with a violent spike `spike_amount` above RH."""
    df = df_base.copy()
    idx = len(df) - 2  # breakout bar
    orig = df.iloc[idx]
    new_high = float(orig["high"]) + spike_amount
    # Keep close below RH? No — breakout requires close > RH. Push close up too.
    df.iloc[idx, df.columns.get_loc("high")] = new_high
    df.iloc[idx, df.columns.get_loc("close")] = max(
        float(orig["close"]), new_high - 0.2
    )
    return df


def main():
    df_tame = _failed_bo_up_canonical()
    df_wild = _mk_wild(df_tame, spike_amount=3.5)     # big spike
    df_insane = _mk_wild(df_tame, spike_amount=8.0)   # gap-spike fakeout

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    render_panel(axes[0], df_tame,
                 "TAME breakout — canonical fixture. Old vs new ≈ same.")
    render_panel(axes[1], df_wild,
                 "WILD breakout — big spike bar. Old stop balloons; new stop stays put.")
    render_panel(axes[2], df_insane,
                 "GAP-SPIKE fakeout — 8pt violence. Old stop is absurd; new stop is the same.")
    fig.suptitle(
        "failed_bo stop: hindsight vs pre-breakout-knowable",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
