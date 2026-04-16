"""
Standalone Databento client for market data queries.
No dependency on AI Edge — this is a self-contained connection.
Includes retry logic, caching, and stale data fallback.
"""

import os
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import databento as db
import pandas as pd

from shared.config_loader import get_project_root

logger = logging.getLogger(__name__)

CACHE_DIR = get_project_root() / "cache" / "databento"
DATABENTO_LAG_MINUTES = 30  # Historical API lags real-time by ~15-30 min


def _safe_end() -> datetime:
    """Return a safe end timestamp that Databento's historical API can serve."""
    return datetime.now(timezone.utc) - timedelta(minutes=DATABENTO_LAG_MINUTES)


def _prior_trading_day(ref_dt: datetime = None):
    """Return the most recent trading day (Mon–Fri) strictly before ref_dt's date."""
    d = (ref_dt or datetime.now(timezone.utc)).date()
    offset = 1
    while True:
        candidate = d - timedelta(days=offset)
        if candidate.weekday() < 5:   # Mon=0 … Fri=4
            return candidate
        offset += 1


def _last_completed_friday(ref_dt: datetime = None):
    """Return the most recent completed Friday (never today, even if today is Friday)."""
    d = (ref_dt or datetime.now(timezone.utc)).date()
    days_back = (d.weekday() - 4) % 7 or 7
    return d - timedelta(days=days_back)


class DatabentClientError(Exception):
    pass


class DatabentClient:
    """Standalone Databento client with retry, caching, and fallback."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        if not self.api_key:
            raise DatabentClientError("DATABENTO_API_KEY not set")
        self.client = db.Historical(self.api_key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def query_ohlcv(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: datetime,
        end: datetime = None,
        retry_count: int = 3,
    ) -> pd.DataFrame:
        """
        Query OHLCV data from Databento with retry and caching.

        Args:
            dataset: e.g. "GLBX.MDP3" for CME futures
            symbols: e.g. ["ES.c.0", "NQ.c.0"]
            schema: e.g. "ohlcv-1h", "ohlcv-1m", "ohlcv-1d"
            start: start datetime (UTC)
            end: end datetime (UTC), defaults to now
            retry_count: max retries on failure

        Returns:
            DataFrame with OHLCV columns
        """
        if end is None:
            end = _safe_end()

        cache_key = self._cache_key(dataset, symbols, schema, start, end)
        cached = self._load_cache(cache_key, schema)
        if cached is not None:
            logger.info(f"Using cached data for {symbols} ({schema})")
            return cached

        # Auto-detect continuous futures symbols (ES.c.0, NQ.c.0, etc.)
        # GLBX.MDP3 requires stype_in='continuous' for .c.0 / .n.0 symbols
        is_continuous = dataset == "GLBX.MDP3" and any(
            ".c." in s or ".n." in s for s in symbols
        )
        extra_kwargs = {"stype_in": "continuous"} if is_continuous else {}

        last_error = None
        for attempt in range(retry_count):
            try:
                logger.info(
                    f"Databento query: {dataset} {symbols} {schema} "
                    f"[{start.isoformat()} → {end.isoformat()}] (attempt {attempt + 1})"
                )
                data = self.client.timeseries.get_range(
                    dataset=dataset,
                    symbols=symbols,
                    schema=schema,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    **extra_kwargs,
                )
                df = data.to_df()
                if df.empty:
                    logger.warning(f"Empty result for {symbols}")
                    return df

                self._save_cache(cache_key, df)
                return df

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Schema not fully available: the requested end is beyond what's indexed.
                # Fetch the actual available end from metadata and cap to it, then retry.
                if "data_schema_not_fully_available" in error_str:
                    try:
                        range_info = self.client.metadata.get_dataset_range(dataset=dataset)
                        schema_ranges = range_info.get("schema", {})
                        schema_end_str = (
                            schema_ranges.get(schema, {}).get("end")
                            or range_info.get("end")
                        )
                        if schema_end_str:
                            available_end = datetime.fromisoformat(
                                schema_end_str.replace("Z", "+00:00")
                            )
                            if end > available_end:
                                logger.info(
                                    f"Schema {schema} end capped: {end.isoformat()} → "
                                    f"{available_end.isoformat()} (dataset limit)"
                                )
                                end = available_end
                                cache_key = self._cache_key(dataset, symbols, schema, start, end)
                                cached = self._load_cache(cache_key, schema)
                                if cached is not None:
                                    return cached
                                continue  # retry with capped end, no sleep
                    except Exception as meta_err:
                        logger.debug(f"Metadata range fetch failed: {meta_err}")

                    # Fallback for ohlcv-1d: cap to today midnight
                    if "ohlcv-1d" in schema:
                        today_midnight = datetime.now(timezone.utc).replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                        if end > today_midnight:
                            logger.info(
                                f"Daily schema: today's bar not yet available, "
                                f"capping end to {today_midnight.date()} and retrying"
                            )
                            end = today_midnight
                            cache_key = self._cache_key(dataset, symbols, schema, start, end)
                            cached = self._load_cache(cache_key, schema)
                            if cached is not None:
                                return cached
                            continue  # retry with new end, no sleep

                # dataset_unavailable_range: end is beyond what this license can serve.
                # Error message contains "Try again with an end time before <ISO>".
                # Parse that timestamp and cap end to it, then retry immediately.
                if "dataset_unavailable_range" in error_str:
                    import re as _re
                    # Capture full ISO timestamp including fractional seconds and Z
                    m = _re.search(
                        r"before (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)",
                        error_str,
                    )
                    if m:
                        try:
                            raw = m.group(1)
                            # Normalise to offset-aware: replace trailing Z, add UTC if bare
                            if raw.endswith("Z"):
                                raw = raw[:-1] + "+00:00"
                            elif "+" not in raw and raw.count("-") <= 2:
                                raw += "+00:00"
                            # Strip nanosecond precision to microseconds (fromisoformat limit)
                            if "." in raw:
                                base, frac_tz = raw.split(".", 1)
                                # frac_tz may be "973654000+00:00" — split off tz
                                if "+" in frac_tz:
                                    frac, tz = frac_tz.split("+", 1)
                                    raw = f"{base}.{frac[:6]}+{tz}"
                                elif "-" in frac_tz[1:]:
                                    idx = frac_tz.index("-", 1)
                                    frac, tz = frac_tz[:idx], frac_tz[idx:]
                                    raw = f"{base}.{frac[:6]}{tz}"
                            cap_end = datetime.fromisoformat(raw)
                            if end > cap_end:
                                logger.info(
                                    f"dataset_unavailable_range: capping end "
                                    f"{end.isoformat()} → {cap_end.isoformat()}"
                                )
                                end = cap_end
                                cache_key = self._cache_key(dataset, symbols, schema, start, end)
                                cached = self._load_cache(cache_key, schema)
                                if cached is not None:
                                    return cached
                                continue  # retry with capped end, no sleep
                        except Exception as parse_err:
                            logger.debug(f"Could not parse cap timestamp: {parse_err}")

                # License/auth errors won't be fixed by retrying
                if "license_not_found_unauthorized" in str(e) or "403" in str(e):
                    logger.warning(
                        f"Databento query failed (no license/auth): {e}. Not retrying."
                    )
                    break

                wait = 5 * (3 ** attempt)  # 5s, 15s, 45s
                logger.warning(
                    f"Databento query failed (attempt {attempt + 1}/{retry_count}): {e}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < retry_count - 1:
                    time.sleep(wait)

        # All retries failed — try stale cache
        stale = self._load_cache(cache_key, schema, allow_stale=True)
        if stale is not None:
            logger.warning(f"Using STALE cached data for {symbols} after all retries failed")
            return stale

        raise DatabentClientError(
            f"Databento query failed after {retry_count} attempts and no cache available: {last_error}"
        )

    def query_overnight_futures(
        self, symbols: list[str], lookback_hours: int = 18
    ) -> pd.DataFrame:
        """
        Query overnight futures data (convenience method for pre-market brief).

        Uses a session-aware end time: the most recently completed Globex hour
        (capped to avoid the not-yet-indexed window). GLBX.MDP3 ohlcv-1h data
        lags real-time by ~1-2 hours, so _safe_end() (now-30min) overshoots
        the available range and produces data_schema_not_fully_available errors.
        """
        import time as _time
        now_utc = datetime.now(timezone.utc)

        # Determine ET offset (EDT=UTC-4 Mar-Nov, EST=UTC-5 Nov-Mar)
        is_dst = bool(_time.localtime().tm_isdst)
        et_offset_hours = 4 if is_dst else 5

        # Globex pre-market end: NY market open (9:30 AM ET) → cap to that
        market_open_utc = now_utc.replace(
            hour=13 + (0 if et_offset_hours == 4 else 1),
            minute=30, second=0, microsecond=0,
        )
        # Use the lesser of: (now - 2h) or market open, to stay within indexed range
        safe_cme_end = min(now_utc - timedelta(hours=2), market_open_utc)

        end   = safe_cme_end
        start = end - timedelta(hours=lookback_hours)
        return self.query_ohlcv(
            dataset="GLBX.MDP3",
            symbols=symbols,
            schema="ohlcv-1h",
            start=start,
            end=end,
        )

    def query_globex_session(
        self,
        symbol: str = "ES.c.0",
        schema: str = "ohlcv-1h",
        end_dt: datetime = None,
    ) -> pd.DataFrame:
        """
        Query the current overnight Globex session.
        Window: 18:00 ET prior calendar day → safe CME end (≤ 9:30 AM ET or now−2h).

        Args:
            symbol: CME continuous symbol e.g. "ES.c.0"
            schema: "ohlcv-1h" (default) or "ohlcv-1m"
            end_dt: override end datetime (UTC). Defaults to safe CME end.
        """
        import time as _time
        now_utc = datetime.now(timezone.utc)
        is_dst = bool(_time.localtime().tm_isdst)
        et_offset_hours = 4 if is_dst else 5

        market_open_utc = now_utc.replace(
            hour=13 + (0 if is_dst else 1), minute=30, second=0, microsecond=0
        )
        safe_end = end_dt or min(now_utc - timedelta(hours=2), market_open_utc)

        prior_cal = now_utc.date() - timedelta(days=1)
        globex_open_utc = datetime(
            prior_cal.year, prior_cal.month, prior_cal.day,
            18 + et_offset_hours, 0, 0, tzinfo=timezone.utc,
        )

        return self.query_ohlcv(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            schema=schema,
            start=globex_open_utc,
            end=safe_end,
        )

    def compute_key_levels(
        self,
        symbol: str = "ES.c.0",
        current_dt: datetime = None,
    ) -> dict:
        """
        Compute 12 key horizontal reference levels for a CME continuous futures symbol.

        Returns dict with four groups, each containing 'high', 'low', 'open':
            overnight   — current Globex session (18:00 ET prior day → now)
            prior_day   — last RTH session (approx 9:00–16:00 ET on most recent trading day)
            prior_week  — last complete Mon–Fri RTH week
            prior_month — last complete calendar month RTH bars
        """
        import time as _time
        now_utc = current_dt or datetime.now(timezone.utc)
        is_dst = bool(_time.localtime().tm_isdst)
        et_offset_hours = 4 if is_dst else 5

        market_open_utc = now_utc.replace(
            hour=13 + (0 if is_dst else 1), minute=30, second=0, microsecond=0
        )
        safe_end = min(now_utc - timedelta(hours=2), market_open_utc)

        # ── Overnight ────────────────────────────────────────────────────────
        prior_cal = now_utc.date() - timedelta(days=1)
        globex_open_utc = datetime(
            prior_cal.year, prior_cal.month, prior_cal.day,
            18 + et_offset_hours, 0, 0, tzinfo=timezone.utc,
        )
        overnight = {"high": None, "low": None, "open": None}
        try:
            on_df = self.query_ohlcv(
                dataset="GLBX.MDP3", symbols=[symbol], schema="ohlcv-1h",
                start=globex_open_utc, end=safe_end,
            )
            if not on_df.empty:
                on_df.columns = [c.lower() for c in on_df.columns]
                overnight = {
                    "high": float(on_df["high"].max()),
                    "low":  float(on_df["low"].min()),
                    "open": float(on_df["open"].iloc[0]),
                }
        except Exception as e:
            logger.error(f"compute_key_levels overnight query failed ({symbol}): {e}")

        # ── Historical RTH (prior day / week / month) ─────────────────────────
        prior_day   = {"high": None, "low": None, "open": None}
        prior_week  = {"high": None, "low": None, "open": None}
        prior_month = {"high": None, "low": None, "open": None}
        try:
            hist_df = self.query_ohlcv(
                dataset="GLBX.MDP3", symbols=[symbol], schema="ohlcv-1h",
                start=now_utc - timedelta(days=40), end=safe_end,
            )
            if not hist_df.empty:
                hist_df.columns = [c.lower() for c in hist_df.columns]
                if not isinstance(hist_df.index, pd.DatetimeIndex):
                    for cand in ("ts_event", "timestamp", "date"):
                        if cand in hist_df.columns:
                            hist_df = hist_df.set_index(cand)
                            break
                if hist_df.index.tz is None:
                    hist_df.index = hist_df.index.tz_localize("UTC")
                else:
                    hist_df.index = hist_df.index.tz_convert("UTC")

                # Convert to ET and filter to approximate RTH hours (9–16 ET)
                hist_et = hist_df.copy()
                hist_et.index = hist_df.index.tz_convert("America/New_York")
                rth_df = hist_et[
                    (hist_et.index.hour >= 9) & (hist_et.index.hour <= 16)
                ]

                # Prior trading day
                ptd = _prior_trading_day(now_utc)
                pd_rows = rth_df[rth_df.index.date == ptd]
                if not pd_rows.empty:
                    prior_day = {
                        "high": float(pd_rows["high"].max()),
                        "low":  float(pd_rows["low"].min()),
                        "open": float(pd_rows["open"].iloc[0]),
                    }

                # Prior week (Mon–Fri ending on last completed Friday)
                last_fri   = _last_completed_friday(now_utc)
                week_start = last_fri - timedelta(days=4)
                pw_rows = rth_df[
                    (rth_df.index.date >= week_start) & (rth_df.index.date <= last_fri)
                ]
                if not pw_rows.empty:
                    prior_week = {
                        "high": float(pw_rows["high"].max()),
                        "low":  float(pw_rows["low"].min()),
                        "open": float(pw_rows["open"].iloc[0]),
                    }

                # Prior calendar month
                pm_year = now_utc.year if now_utc.month > 1 else now_utc.year - 1
                pm_num  = now_utc.month - 1 if now_utc.month > 1 else 12
                pm_rows = rth_df[
                    [d.month == pm_num and d.year == pm_year for d in rth_df.index.date]
                ]
                if not pm_rows.empty:
                    prior_month = {
                        "high": float(pm_rows["high"].max()),
                        "low":  float(pm_rows["low"].min()),
                        "open": float(pm_rows["open"].iloc[0]),
                    }

        except Exception as e:
            logger.error(f"compute_key_levels historical query failed ({symbol}): {e}")

        return {
            "overnight":   overnight,
            "prior_day":   prior_day,
            "prior_week":  prior_week,
            "prior_month": prior_month,
        }

    def query_premarket_15min(
        self,
        symbol: str = "ES.c.0",
        end_dt: datetime = None,
    ) -> pd.DataFrame:
        """
        Fetch the overnight Globex session resampled to 15-minute OHLCV bars.
        Window: 18:00 ET prior calendar day → safe CME end.
        Fetches ohlcv-1m and resamples internally.
        """
        import time as _time
        now_utc = datetime.now(timezone.utc)
        is_dst = bool(_time.localtime().tm_isdst)
        et_offset_hours = 4 if is_dst else 5

        market_open_utc = now_utc.replace(
            hour=13 + (0 if is_dst else 1), minute=30, second=0, microsecond=0
        )
        safe_end = end_dt or min(now_utc - timedelta(hours=2), market_open_utc)

        prior_cal = now_utc.date() - timedelta(days=1)
        globex_open_utc = datetime(
            prior_cal.year, prior_cal.month, prior_cal.day,
            18 + et_offset_hours, 0, 0, tzinfo=timezone.utc,
        )

        df = self.query_ohlcv(
            dataset="GLBX.MDP3", symbols=[symbol], schema="ohlcv-1m",
            start=globex_open_utc, end=safe_end,
        )
        if df.empty:
            return df

        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            for cand in ("ts_event", "timestamp", "date"):
                if cand in df.columns:
                    df = df.set_index(cand)
                    break
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[ohlcv_cols]
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in ohlcv_cols:
            agg["volume"] = "sum"
        df = df.resample("15min").agg(agg).dropna(subset=["open", "close"])

        return df

    def query_daily(
        self, symbols: list[str], lookback_days: int = 60, dataset: str = "EQUS.MINI"
    ) -> pd.DataFrame:
        """Query daily OHLCV for equities."""
        end = _safe_end()
        start = end - timedelta(days=lookback_days)
        return self.query_ohlcv(
            dataset=dataset,
            symbols=symbols,
            schema="ohlcv-1d",
            start=start,
            end=end,
        )

    def query_intraday(
        self,
        symbols: list[str],
        schema: str = "ohlcv-5m",
        lookback_hours: int = 8,
        dataset: str = "EQUS.MINI",
    ) -> pd.DataFrame:
        """Query intraday OHLCV for equities."""
        end = _safe_end()
        start = end - timedelta(hours=lookback_hours)
        return self.query_ohlcv(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
        )

    def query_gap_candidates(
        self,
        symbols: list[str],
        direction: str = "up",
        min_gap_pct: float = 3.0,
        min_volume: int = 500000,
        lookback_minutes: int = 30,
        dataset: str = "EQUS.MINI",
    ) -> list[dict]:
        """
        Find stocks gapping up or down today.

        Two-step query:
          1. ohlcv-1d with end=today_midnight_utc → yesterday's close
          2. ohlcv-1m from market open to now → today's open/price action

        Gap % = (today_open - prior_close) / prior_close * 100
        """
        now_utc = datetime.now(timezone.utc)
        today_midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

        # NYSE/NASDAQ open is 9:30 AM ET.
        # EDT = UTC-4 → 13:30 UTC; EST = UTC-5 → 14:30 UTC.
        # Detect offset: March 2nd Sunday → November 1st Sunday is EDT.
        import time as _time
        is_dst = bool(_time.localtime().tm_isdst)
        utc_offset_hours = 4 if is_dst else 5
        market_open_utc = today_midnight_utc.replace(
            hour=13 + (0 if utc_offset_hours == 4 else 1), minute=30
        )

        # 1. Prior day close — must end at today midnight to stay in ohlcv-1d availability window
        try:
            prior_df = self.query_ohlcv(
                dataset=dataset,
                symbols=symbols,
                schema="ohlcv-1d",
                start=today_midnight_utc - timedelta(days=5),
                end=today_midnight_utc,
            )
        except Exception as e:
            logger.error(f"Gap scanner: prior-close query failed: {e}")
            return []

        if prior_df.empty:
            logger.warning("Gap scanner: no prior close data")
            return []

        # 2. Today's intraday — from market open to now (capped at safe end)
        intraday_end = _safe_end()
        if intraday_end <= market_open_utc:
            logger.warning("Gap scanner: market hasn't opened yet or data not available")
            return []

        try:
            intra_df = self.query_ohlcv(
                dataset=dataset,
                symbols=symbols,
                schema="ohlcv-1m",
                start=market_open_utc,
                end=intraday_end,
            )
        except Exception as e:
            logger.error(f"Gap scanner: intraday query failed: {e}")
            return []

        if intra_df.empty:
            logger.warning("Gap scanner: no intraday data (market may not be open yet)")
            return []

        # 3. Calculate gaps per symbol
        gaps = []
        for symbol in symbols:
            try:
                if "symbol" in prior_df.columns:
                    sym_daily = prior_df[prior_df["symbol"] == symbol]
                else:
                    sym_daily = prior_df

                if sym_daily.empty:
                    continue
                prior_close = float(sym_daily["close"].iloc[-1])

                if "symbol" in intra_df.columns:
                    sym_intra = intra_df[intra_df["symbol"] == symbol]
                else:
                    sym_intra = intra_df

                if sym_intra.empty:
                    continue

                today_open = float(sym_intra["open"].iloc[0])
                today_high = float(sym_intra["high"].max())
                today_low = float(sym_intra["low"].min())
                today_close = float(sym_intra["close"].iloc[-1])
                today_volume = int(sym_intra["volume"].sum())

                if prior_close == 0:
                    continue
                gap_pct = (today_open - prior_close) / prior_close * 100

                # Direction filter
                if direction == "up" and gap_pct < min_gap_pct:
                    continue
                if direction == "down" and gap_pct > -min_gap_pct:
                    continue

                # Volume filter
                if today_volume < min_volume:
                    continue

                # Did gap hold through the lookback window?
                gap_held = (
                    (direction == "up" and today_close > prior_close) or
                    (direction == "down" and today_close < prior_close)
                )

                # BPA context from intraday bars
                key_levels = _key_levels_from_df(sym_intra, prior_close)

                gaps.append({
                    "ticker": symbol,
                    "gap_pct": round(gap_pct, 2),
                    "prior_close": round(prior_close, 2),
                    "today_open": round(today_open, 2),
                    "today_high": round(today_high, 2),
                    "today_low": round(today_low, 2),
                    "today_close": round(today_close, 2),
                    "volume": today_volume,
                    "gap_held": gap_held,
                    "direction": direction,
                    "key_levels": key_levels,
                    "bars_scanned": len(sym_intra),
                })
            except Exception as e:
                logger.debug(f"Gap calculation failed for {symbol}: {e}")

        gaps.sort(key=lambda x: abs(x.get("gap_pct", 0)), reverse=True)
        return gaps

    def _cache_key(self, dataset, symbols, schema, start, end) -> str:
        """Generate a deterministic cache key."""
        raw = f"{dataset}|{'|'.join(sorted(symbols))}|{schema}|{start.isoformat()}|{end.isoformat()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        d = CACHE_DIR / today
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{cache_key}.parquet"

    def _save_cache(self, cache_key: str, df: pd.DataFrame):
        try:
            path = self._cache_path(cache_key)
            df.to_parquet(path)
            logger.debug(f"Cached data to {path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    def _load_cache(
        self, cache_key: str, schema: str, allow_stale: bool = False
    ) -> Optional[pd.DataFrame]:
        """Load from cache. Check today first, then stale if allowed."""
        path = self._cache_path(cache_key)
        if path.exists():
            ttl = self._cache_ttl(schema)
            age = time.time() - path.stat().st_mtime
            if age < ttl:
                try:
                    return pd.read_parquet(path)
                except Exception:
                    return None

        if allow_stale:
            # Search recent dates for any cached version
            for days_back in range(1, 8):
                d = datetime.now() - timedelta(days=days_back)
                old_path = CACHE_DIR / d.strftime("%Y-%m-%d") / f"{cache_key}.parquet"
                if old_path.exists():
                    try:
                        logger.warning(f"Loading stale cache from {old_path}")
                        return pd.read_parquet(old_path)
                    except Exception:
                        continue

        return None

    def _cache_ttl(self, schema: str) -> float:
        """Cache TTL in seconds based on data type."""
        if "1d" in schema or "daily" in schema:
            return 86400  # 24 hours
        elif "1h" in schema or "60" in schema:
            return 600  # 10 minutes
        else:
            return 300  # 5 minutes for intraday


def _key_levels_from_df(df: pd.DataFrame, prior_close: float) -> list[float]:
    """Extract key price levels from intraday data."""
    levels = set()
    levels.add(round(float(df["high"].max()), 2))
    levels.add(round(float(df["low"].min()), 2))
    levels.add(round(prior_close, 2))
    if "volume" in df.columns and df["volume"].sum() > 0:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = float((tp * df["volume"]).sum() / df["volume"].sum())
        levels.add(round(vwap, 2))
    return sorted(levels)


def get_economic_calendar() -> list[dict]:
    """
    Get today's economic calendar events.
    Uses a simple approach — in production, you'd query an API.
    For now, returns a placeholder that the screener can populate.
    """
    # TODO: Integrate with a real economic calendar API (e.g., Investing.com, TradingEconomics)
    # For now, return an empty list — the script writer will handle missing calendar gracefully
    logger.info("Economic calendar: returning empty (no calendar API configured yet)")
    return []
