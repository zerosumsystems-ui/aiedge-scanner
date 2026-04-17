"""Batch scoring runners — score_multiple + scan_universe.

Extracted from shared/brooks_score.py (ROADMAP C1). These are the N-symbol
entry points:

  - score_multiple(gaps, df_dict) — score a pre-built dict of DataFrames
  - scan_universe(tickers, ...)   — two-query Databento scan + score

Both delegate to the single-symbol orchestrator in
`aiedge.signals.pipeline.score_gap`.
"""

import logging

import pandas as pd

from aiedge.data.normalize import _normalize_databento_df
from aiedge.data.resample import SCAN_BAR_SCHEMA, _resample_to_5min
from aiedge.data.universe import _get_default_universe
from aiedge.features.volatility import _compute_daily_atr
from aiedge.signals.components import LIQUIDITY_MIN_DOLLAR_VOL, _check_liquidity
from aiedge.signals.pipeline import score_gap

logger = logging.getLogger(__name__)

# Magnitude filter floor — used by scan_universe to drop tiny-move noise
# from the leaderboard. MAGNITUDE_CAP_9/10 live in aiedge.signals.pipeline.
MAGNITUDE_FLOOR = 0.5  # Min move/ATR to appear on leaderboard


def score_multiple(
    gaps: list[dict],
    df_dict: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Score and rank a list of gap stocks.

    Parameters
    ----------
    gaps : list[dict]
        Each dict must have: ticker, prior_close, gap_direction.
    df_dict : dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame of 5-min bars.

    Returns
    -------
    list[dict] sorted by urgency desc, uncertainty asc. Best setups first.
    """
    results = []
    for gap in gaps:
        ticker = gap["ticker"]
        if ticker not in df_dict:
            logger.warning(f"No bar data for {ticker}, skipping.")
            continue
        result = score_gap(
            df=df_dict[ticker],
            prior_close=gap["prior_close"],
            gap_direction=gap["gap_direction"],
            ticker=ticker,
        )
        results.append(result)

    # Sort: highest urgency first, then lowest uncertainty
    results.sort(key=lambda r: (-r["urgency"], r["uncertainty"]))
    return results


def scan_universe(
    tickers: list[str] = None,
    min_urgency: float = 3.0,
    max_uncertainty: float = 7.0,
    min_dollar_vol: float = LIQUIDITY_MIN_DOLLAR_VOL,
    databento_key: str = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Score every stock in the universe and return a ranked leaderboard.

    Two Databento API calls total (NOT per-symbol):
      1. ohlcv-1d for all symbols → prior close
      2. ohlcv-1m for all symbols → today's 5-min bars (resampled internally)

    Parameters
    ----------
    tickers : list[str], optional
        Symbols to scan. Defaults to the full 498-symbol universe.
    min_urgency : float
        Filter out anything below this urgency score (default 3.0).
    max_uncertainty : float
        Filter out anything above this uncertainty score (default 7.0).
    databento_key : str, optional
        Databento API key. Falls back to DATABENTO_API_KEY env var.
    verbose : bool
        If True, log progress per symbol.

    Returns
    -------
    list[dict]
        Ranked list of score_gap() outputs, sorted by urgency desc.
        Only includes stocks passing the urgency/uncertainty filters.
    """
    import os
    import time as _time
    from datetime import datetime, timedelta, timezone

    # Lazy import — DatabentClient lives in the same shared/ package
    from shared.databento_client import DatabentClient, _safe_end

    if tickers is None:
        tickers = _get_default_universe()

    api_key = databento_key or os.environ.get("DATABENTO_API_KEY")
    client = DatabentClient(api_key=api_key)

    now_utc = datetime.now(timezone.utc)
    today_midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    # Detect ET offset for market open time
    is_dst = bool(_time.localtime().tm_isdst)
    utc_offset_hours = 4 if is_dst else 5
    market_open_utc = today_midnight_utc.replace(
        hour=13 + (0 if utc_offset_hours == 4 else 1), minute=30
    )

    safe_end = _safe_end()

    # ── QUERY 1: Prior day's close (ohlcv-1d, all symbols, last 5 days) ──
    logger.info(f"Scanning {len(tickers)} symbols — fetching daily bars (30 days for ADR)...")
    try:
        daily_df = client.query_ohlcv(
            dataset="EQUS.MINI",
            symbols=tickers,
            schema="ohlcv-1d",
            start=today_midnight_utc - timedelta(days=30),  # ~20 trading days
            end=today_midnight_utc,
        )
    except Exception as e:
        logger.error(f"Daily data query failed: {e}")
        return []

    if daily_df.empty:
        logger.error("No daily data returned")
        return []

    daily_df = _normalize_databento_df(daily_df)

    # Compute 20-day Average Daily Range per symbol for magnitude scoring
    daily_atrs: dict[str, float] = {}
    if "symbol" in daily_df.columns:
        for sym in tickers:
            sym_d = daily_df[daily_df["symbol"] == sym]
            daily_atrs[sym] = _compute_daily_atr(sym_d)
    else:
        # Single-symbol query
        if len(tickers) == 1:
            daily_atrs[tickers[0]] = _compute_daily_atr(daily_df)

    # ── QUERY 2: Today's 1-min bars from market open (all symbols) ──
    if safe_end <= market_open_utc:
        logger.warning("Market hasn't opened yet or data not available")
        return []

    logger.info(f"Fetching intraday 1-min bars ({market_open_utc.strftime('%H:%M')} → {safe_end.strftime('%H:%M')} UTC)...")
    try:
        intra_df = client.query_ohlcv(
            dataset="EQUS.MINI",
            symbols=tickers,
            schema=SCAN_BAR_SCHEMA,
            start=market_open_utc,
            end=safe_end,
        )
    except Exception as e:
        logger.error(f"Intraday data query failed: {e}")
        return []

    if intra_df.empty:
        logger.error("No intraday data returned")
        return []

    intra_df = _normalize_databento_df(intra_df)

    # ── PARSE & SCORE each symbol ──
    results = []
    skipped = 0

    for ticker in tickers:
        try:
            # Extract this symbol's daily data → prior close
            if "symbol" in daily_df.columns:
                sym_daily = daily_df[daily_df["symbol"] == ticker]
            else:
                # Single-symbol query won't have a symbol column
                sym_daily = daily_df if len(tickers) == 1 else pd.DataFrame()

            if sym_daily.empty:
                skipped += 1
                continue

            prior_close = float(sym_daily["close"].iloc[-1])
            if prior_close <= 0:
                skipped += 1
                continue

            # Extract this symbol's intraday data
            if "symbol" in intra_df.columns:
                sym_intra = intra_df[intra_df["symbol"] == ticker].copy()
            else:
                sym_intra = intra_df.copy() if len(tickers) == 1 else pd.DataFrame()

            if sym_intra.empty or len(sym_intra) < 5:
                skipped += 1
                continue

            # Drop the symbol column before resampling (keep only OHLCV)
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"]
                          if c in sym_intra.columns]
            sym_intra = sym_intra[ohlcv_cols]

            # Resample 1-min → 5-min
            df_5m = _resample_to_5min(sym_intra)
            if len(df_5m) < 3:
                skipped += 1
                continue

            # Hard liquidity filter: skip illiquid stocks
            liq = _check_liquidity(df_5m)
            if not liq["passed"] and min_dollar_vol > 0:
                if verbose:
                    logger.info(f"  {ticker:6s}  SKIP — avg $vol/bar ${liq['avg_dollar_vol']:,.0f} < ${min_dollar_vol:,.0f}")
                skipped += 1
                continue

            # Infer gap direction: today's open vs prior close
            today_open = float(df_5m.iloc[0]["open"])
            gap_direction = "up" if today_open > prior_close else "down"
            gap_pct = (today_open - prior_close) / prior_close * 100

            # Score it
            result = score_gap(
                df=df_5m.reset_index(drop=True),  # score_gap expects integer index
                prior_close=prior_close,
                gap_direction=gap_direction,
                ticker=ticker,
                daily_atr=daily_atrs.get(ticker),
            )

            # Attach gap % and liquidity info for the leaderboard display
            result["gap_pct"] = round(gap_pct, 2)
            result["today_open"] = round(today_open, 2)
            result["prior_close_price"] = round(prior_close, 2)
            result["bars_scored"] = len(df_5m)
            result["liquidity"] = liq

            # Apply filters
            if result["urgency"] < min_urgency:
                continue
            if result["uncertainty"] > max_uncertainty:
                continue
            if result.get("move_ratio", 0) < MAGNITUDE_FLOOR:
                continue

            results.append(result)

            if verbose:
                logger.info(
                    f"  {ticker:6s}  U={result['urgency']:4.1f}  "
                    f"Unc={result['uncertainty']:4.1f}  {result['signal']:16s}  "
                    f"Gap={gap_pct:+.1f}%"
                )

        except Exception as e:
            logger.debug(f"Scoring failed for {ticker}: {e}")
            skipped += 1

    # Sort: urgency desc, uncertainty asc
    results.sort(key=lambda r: (-r["urgency"], r["uncertainty"]))

    logger.info(
        f"Scan complete: {len(results)} stocks passed filters, "
        f"{skipped} skipped (no data or filtered out)"
    )

    return results
