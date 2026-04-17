"""Console leaderboard + Apple Note text formatters.

Everything that produces plain-text / ANSI-colorized output for the
scanner's terminal and Apple Note views. Extracted from live_scanner.py
(Phase 4g-1).

`_next_scan_after` takes the first-scan wall-clock as explicit
parameters rather than reading live_scanner's FIRST_SCAN_HOUR /
FIRST_SCAN_MIN globals, so the formatter stays pure.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aiedge.signals.postprocess import _fmt_delta, _fmt_movement


# ── ANSI colors ─────────────────────────────────────────────────────
_COLORS = {
    "BUY_PULLBACK":  "\033[92m",
    "BUY_SPIKE":     "\033[32m",
    "SELL_PULLBACK": "\033[91m",
    "SELL_SPIKE":    "\033[31m",
    "WAIT":          "\033[93m",
    "FOG":           "\033[33m",
    "AVOID":         "\033[90m",
    "PASS":          "\033[90m",
}
_RST = "\033[0m"
_BOLD = "\033[1m"
_W = 100
_UP = "\033[92m▲\033[0m"
_DOWN = "\033[91m▼\033[0m"


def _movement_arrow(r: dict) -> str:
    if r.get("_first_scan") or r.get("rank_change") is None:
        return " "
    rc = r["rank_change"]
    if rc > 0:
        return _UP
    if rc < 0:
        return _DOWN
    return "="


def _next_scan_after(now: datetime, interval_min: int,
                     first_scan_hour: int, first_scan_min: int) -> datetime:
    first = now.replace(
        hour=first_scan_hour, minute=first_scan_min, second=0, microsecond=0
    )
    if now < first:
        return first
    elapsed_s = (now - first).total_seconds()
    periods_done = int(elapsed_s / (interval_min * 60))
    return first + timedelta(minutes=interval_min * (periods_done + 1))


def _next_scan_time_str(now_et: datetime, interval_min: int,
                        first_scan_hour: int, first_scan_min: int) -> str:
    nxt = _next_scan_after(now_et, interval_min, first_scan_hour, first_scan_min)
    return nxt.strftime("%I:%M %p").lstrip("0")


def print_leaderboard(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    first_scan_hour: int,
    first_scan_min: int,
) -> None:
    time_str = now_et.strftime("%I:%M %p").lstrip("0")
    title = f" LIVE SCAN — {time_str} ET | {now_et.strftime('%Y-%m-%d')} | {total_symbols:,} symbols "
    print()
    print(_BOLD + "━" * _W + _RST)
    print(_BOLD + title.center(_W) + _RST)
    print(_BOLD + "━" * _W + _RST)

    if not results:
        print("  (no liquid stocks passed filters this cycle)")
    else:
        hdr = (
            f"{'':1}{'#':>3}  {'TICKER':<8}  {'URGENCY':>7}  "
            f"{'ADR':>7}  {'MOVE':>5}  {'UNCERT':>6}  "
            f"{'SIGNAL':<16}  {'MOVEMENT':<18}  DELTA"
        )
        print(_BOLD + hdr + _RST)
        print("─" * _W)

        for r in results:
            sig = r.get("signal", "?")
            color = _COLORS.get(sig, "")
            urg = r.get("urgency", 0.0)
            unc = r.get("uncertainty", 0.0)
            ticker = r.get("ticker", "?")
            warn = r.get("day_type_warning", "")
            arrow = _movement_arrow(r)
            mov = _fmt_movement(r)
            dlt = _fmt_delta(r)
            adr_v = r.get("daily_atr") or 0.0
            rat_v = r.get("move_ratio") or 0.0
            adr_col = f"${adr_v:.2f}" if adr_v > 0 else "  —  "
            rat_col = f"{rat_v:.1f}x" if adr_v > 0 else "  — "

            print(
                f"{arrow:1} {r['rank']:>3}  {ticker:<8}  {urg:>7.1f}  "
                f"{adr_col:>7}  {rat_col:>5}  {unc:>6.1f}  "
                f"{color}{sig:<16}{_RST}  {mov:<18}  {dlt}"
            )
            if warn and warn.lower() not in ("", "none") and r["rank"] <= 10:
                print(f"  {'':>3}  {'':8}  {'':7}  {'':7}  {'':5}  {'':6}  {'':16}  ⚠  {warn[:50]}")

    next_str = _next_scan_time_str(now_et, interval_min, first_scan_hour, first_scan_min)
    print("─" * _W)
    print(f"  {passed} stocks passed | scan took {elapsed:.2f}s | next scan {next_str}")
    print()


def _format_note_text(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    first_scan_hour: int,
    first_scan_min: int,
) -> str:
    time_str = now_et.strftime("%I:%M %p").lstrip("0")
    lines = [
        f"LIVE SCAN — {time_str} ET",
        f"Last updated: {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET",
        f"{total_symbols:,} symbols streaming",
        "━" * 72,
        "",
    ]

    if not results:
        lines.append("(no liquid stocks passed filters this cycle)")
    else:
        hdr = (
            f"{'#':>3}  {'TICKER':<8}  {'URGENCY':>7}  {'UNCERT':>6}  "
            f"{'SIGNAL':<16}  {'MOVEMENT':<18}  DELTA"
        )
        lines.append(hdr)
        lines.append("─" * 72)
        for r in results:
            sig = r.get("signal", "?")
            urg = r.get("urgency", 0.0)
            unc = r.get("uncertainty", 0.0)
            ticker = r.get("ticker", "?")
            warn = r.get("day_type_warning", "")
            mov = _fmt_movement(r)
            dlt = _fmt_delta(r)
            lines.append(
                f"{r['rank']:>3}  {ticker:<8}  {urg:>7.1f}  {unc:>6.1f}  "
                f"{sig:<16}  {mov:<18}  {dlt}"
            )
            if warn and warn.lower() not in ("", "none") and r["rank"] <= 10:
                lines.append(f"     {'':8}  {'':7}  {'':6}  {'':16}  ⚠ {warn[:58]}")

    next_str = _next_scan_time_str(now_et, interval_min, first_scan_hour, first_scan_min)
    lines += [
        "─" * 72,
        f"{passed} stocks passed | scan took {elapsed:.2f}s | next scan {next_str}",
    ]
    return "\n".join(lines)
