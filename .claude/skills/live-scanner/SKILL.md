---
name: live-scanner
description: Real-time US equities live scanner — Databento Live API, Brooks price action scoring every 5 min, urgency-ranked HTML dashboard. Use when Will says "run the scanner", "start the live scan", "urgency ranking", "show me the dashboard", "what's scanning", "open the scanner", "live scan results", "what are the top stocks right now", "check the urgency", "what's hot", "replay the session", or any variant asking about real-time gap / trend-from-open scores during or after market hours.
---

# Live Scanner — Brooks Price Action

Real-time US equities scanner. Streams ALL US equities via Databento Live websocket (EQUS.MINI, ohlcv-1m), accumulates 1-min bars, scores every 5 minutes using Brooks gap and trend-from-open methodology, outputs an auto-refreshing HTML urgency-ranking dashboard.

## Quick Reference

| Thing | Path |
|-------|------|
| Main script | `~/video-pipeline/live_scanner.py` |
| Dashboard | `~/video-pipeline/logs/live_scanner/dashboard.html` |
| Log (daily) | `~/video-pipeline/logs/live_scanner/YYYY-MM-DD.log` |
| Results JSON | `~/video-pipeline/logs/live_scanner/YYYY-MM-DD.json` |
| Session pickle | `~/video-pipeline/logs/live_scanner/YYYY-MM-DD_session.pkl` |
| Scoring module | `~/video-pipeline/shared/brooks_score.py` |
| Launchd plist | `~/video-pipeline/launchd/com.videopipeline.live_scanner.plist` |

## Starting the Scanner

**Automatic (launchd):** Fires Mon–Fri at **6:25 AM PDT / 9:25 AM ET**
```
~/video-pipeline/launchd/com.videopipeline.live_scanner.plist
```

**Manual start (production):**
```bash
cd ~/video-pipeline && python3 live_scanner.py
```

**Test mode** (2-min accumulate, one scan, exit — safe for testing):
```bash
python3 live_scanner.py --test
```

**Replay a past session:**
```bash
python3 live_scanner.py --replay 2026-04-13
```
Replay loads the session pickle, re-scores all symbols with fresh ADR data, regenerates the dashboard, and opens it in the browser. Use this after market close to review the day.

**Other flags:**
```bash
python3 live_scanner.py --scan-interval 3   # Scan every 3 min instead of 5
python3 live_scanner.py --top 30            # Show top 30 on leaderboard
python3 live_scanner.py --min-urgency 7     # Alert only on urgency >= 7
```

## Scan Cycle

- **9:25 AM ET** — Connect to Databento Live, fetch prior closes + 20-day ADR for all ~12,000 symbols
- **9:35 AM ET** — First scan cycle
- **Every 5 min** — Re-score all symbols with ≥2 bars, update dashboard + Apple Notes + Discord alert
- **4:05 PM ET** — Clean shutdown, save session pickle to `YYYY-MM-DD_session.pkl`

## Scoring Pipeline

1. **Data source**: Databento Live EQUS.MINI `ohlcv-1m` → 1-min bars accumulate in memory per symbol
2. **Filters** (live mode):
   - Price ≥ $5.00
   - Avg dollar volume ≥ $350K/bar (EQUS.MINI is a mini feed; volumes are ~15–25× lower than consolidated tape)
   - Minimum 2 bars received
3. **Resample**: 1-min → 5-min bars, aligned to 9:30 ET market open
4. **Score** via `shared/brooks_score.score_gap()`:
   - `prior_close` — prior day's close from Databento Historical
   - `daily_atr` — 20-day average daily range (high − low) per symbol in dollars
   - `gap_direction` — "up" if open > prior_close, "down" otherwise
5. **Day classification**: trend_from_open / spike_and_channel / trading_range / trending_tr / tight_tr
   - Uses first 70% of bars to prevent late chop from reclassifying a morning trend
   - Chop ratio guard: if late-session range < 25% of day's range, overrides trading_range → trend_from_open
6. **Magnitude gate**: move must be ≥ 0.5× ADR to appear on leaderboard; ≥ 0.7× for urgency > 9.0; ≥ 1.0× for urgency > 9.5
7. **Perfect 10 rule**: urgency reaches 10.0 only if zero bars closed on the wrong side of the 20-bar EMA
8. **Rank**: sorted descending by urgency (0.0–10.0 scale)

## Reading the Dashboard

Open `~/video-pipeline/logs/live_scanner/dashboard.html` in Safari — auto-refreshes every 5 minutes.

Each card (collapsible) shows:

```
#   TICKER                    [ADR block]      URG ████  7.4
    trend_from_open           $12.40 ADR       UNC ██    2.1
                              1.6× move        [BUY PULLBACK]
```

- **#** — rank by urgency; ▲/▼ arrows show rank change vs prior scan
- **TICKER** — symbol + day_type classification
- **ADR** — 20-day average daily range in dollars
- **Move** — today's open-to-current move as a multiple of ADR (e.g., 1.6× means moved 1.6× a typical full day)
- **URG** — urgency 0–10 (higher = cleaner trend, more tradeable)
- **UNC** — uncertainty 0–10 (higher = choppier, riskier)
- **SIGNAL** — actionable signal for this cycle

Clicking a card opens the 5-min chart + warning + summary.

## Urgency Interpretation

| URG | What it means | Action |
|-----|---------------|--------|
| 9.5–10.0 | Perfect trend-from-open. Zero EMA violations. Move ≥ 1× ADR. | Buy/sell all pullbacks aggressively |
| 8.0–9.4 | Strong trend. Occasional chop but structure intact. | WAIT signal → enter on first clean pullback |
| 5.0–7.9 | Moderate trend. Wider stops needed. | Caution — size down |
| 2.0–4.9 | Weak trend or very early in day. | Watch only |
| 0.0–1.9 | Trading range or no structure. | AVOID / FOG |

## Signals

| Signal | Meaning |
|--------|---------|
| BUY PULLBACK | Uptrend, pullback entry conditions met — go long |
| SELL PULLBACK | Downtrend, pullback entry conditions met — go short |
| WAIT | Strong trend but no clean pullback entry yet; wait for it |
| FOG | Contradictory signals; skip this stock |
| AVOID | High uncertainty or trading range; don't trade |
| PASS | Below threshold or filtered |

## Day Types

| Day Type | What it means |
|----------|---------------|
| trend_from_open | Trended from the open bar — all pullbacks are entries |
| spike_and_channel | Opening spike then broad channel — pullbacks ok, wider stops |
| trending_tr | Range with directional bias — wait for breakout confirmation |
| trading_range | Balanced two-sided — buy low / sell high, not breakouts |
| tight_tr | Very tight range — avoid entirely |

## Reporting to Will

When Will asks "what's the scanner showing?" or "urgency ranking":

1. **During market hours** — read the current leaderboard from `YYYY-MM-DD.json`:
   ```bash
   python3 -c "import json; data=json.load(open('logs/live_scanner/$(date +%Y-%m-%d).json')); [print(f\"{r['rank']:>2}. {r['ticker']:<6} URG={r['urgency']:.1f} {r['signal']}\") for r in data[-1]['results'][:10]]"
   ```

2. **After market close** — replay the session:
   ```bash
   python3 live_scanner.py --replay $(date +%Y-%m-%d)
   ```

3. **Present top stocks** in this format:
   ```
    #  TICKER  URG   ADR    MOVE    SIGNAL          DAY_TYPE
    1  T       10.0  $0.73  1.2×    WAIT            trend_from_open
    2  NEE      6.6  $1.12  0.9×    PASS            trend_from_open
    3  CRWD     1.6  $12.4  0.09×   PASS            trend_from_open
   ```

4. Highlight URG ≥ 8.0 stocks — those are the tradeable ones for Will
5. Note any large urgency_delta between scan cycles (stock accelerating or collapsing)

## Constraints

- **Databento only** — never use yfinance or any other data source
- **Never fake data** — if scanner isn't running or has no bars, say so explicitly
- **Never start the scanner yourself** — launchd or Will starts it; your role is to read and interpret
- **ADR is in dollars** (not percent) — CRWD ADR $12.40 means a typical full day moves $12.40
- **EQUS.MINI volumes** are 15–25× lower than consolidated tape — don't compare to normal dollar-volume thresholds
