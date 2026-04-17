"""
Screener stage: runs the YAML-defined data queries and returns structured results.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

from shared.databento_client import DatabentClient, get_economic_calendar
from shared.bpa_detector import detect_all

logger = logging.getLogger(__name__)


def run(config: dict, run_id: str, run_dir: str) -> dict:
    """
    Execute the screener stage.

    Reads screener config from YAML, queries Databento,
    runs BPA detection, and returns structured results.

    Returns dict with all screener data for the script writer.
    """
    screener_config = config["screener"]
    screener_type = screener_config.get("type", "generic")
    pipeline_name = config["pipeline_name"]

    logger.info(f"Running screener: {screener_type} for {pipeline_name}")

    client = DatabentClient()
    results = {"type": screener_type, "timestamp": datetime.now(timezone.utc).isoformat(), "data": {}}

    if screener_type == "premarket_brief":
        results["data"] = _run_premarket(client, screener_config)
    elif screener_type in ("gap_up", "gap_down"):
        results["data"] = _run_gap_scanner(client, screener_config)
    elif screener_type in ("new_highs", "new_lows"):
        results["data"] = _run_new_extremes(client, screener_config)
    elif screener_type == "best_stocks":
        results["data"] = _run_top_gainers(client, screener_config)
    elif screener_type == "industry_groups":
        results["data"] = _run_industry_groups(client, screener_config)
    else:
        results["data"] = _run_generic(client, screener_config)

    # Save screener output
    from pathlib import Path
    output_path = Path(run_dir) / "screener_output.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    result_count = _count_results(results)
    logger.info(f"Screener complete: {result_count} items")

    return results


def _run_premarket(client: DatabentClient, screener_config: dict) -> dict:
    """Pre-market brief: overnight futures + pre-market movers + calendar."""
    data = {"futures": {}, "premarket_movers": [], "economic_calendar": []}

    queries = screener_config.get("databento_queries", [])

    for query in queries:
        dataset = query.get("dataset", "")

        if dataset == "GLBX.MDP3":
            # Overnight futures — 12 key levels + 15-min Globex session bars
            symbols = query.get("symbols", ["ES.c.0", "NQ.c.0"])
            for symbol in symbols:
                ticker = symbol.replace(".c.0", "").replace(".c.", "").upper()
                try:
                    key_levels = client.compute_key_levels(symbol=symbol)
                    df_15 = client.query_premarket_15min(symbol=symbol)
                    entry = {"key_levels": key_levels}
                    if not df_15.empty:
                        df_15.columns = [c.lower() for c in df_15.columns]
                        entry.update({
                            "last_price": float(df_15["close"].iloc[-1]),
                            "open":       float(df_15["open"].iloc[0]),
                            "high":       float(df_15["high"].max()),
                            "low":        float(df_15["low"].min()),
                            "change_pct": float(
                                (df_15["close"].iloc[-1] - df_15["open"].iloc[0])
                                / df_15["open"].iloc[0] * 100
                            ),
                            "bars": len(df_15),
                        })
                    data["futures"][ticker] = entry
                    logger.info(f"Key levels computed for {ticker}")
                except Exception as e:
                    logger.error(f"Futures query failed for {symbol}: {e}")
                    data["futures"][ticker] = {"error": str(e)}

        elif query.get("type") == "premarket_movers":
            # Pre-market movers (equities gapping)
            try:
                # For pre-market movers, we query the most liquid stocks
                # In production, you'd use a universe of stocks
                # For now, query a watchlist
                watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
                             "AMD", "SPY", "QQQ"]
                min_pct = query.get("min_pct_change", 3.0)

                df = client.query_daily(watchlist, lookback_days=5, dataset=query.get("dataset", "EQUS.MINI"))
                if not df.empty:
                    # Calculate gap from prior close
                    for ticker in watchlist:
                        try:
                            ticker_df = df[df.get("symbol", df.index) == ticker] if "symbol" in df.columns else df
                            if len(ticker_df) >= 2:
                                prev_close = float(ticker_df["close"].iloc[-2])
                                curr_open = float(ticker_df["open"].iloc[-1])
                                gap_pct = (curr_open - prev_close) / prev_close * 100

                                if abs(gap_pct) >= min_pct:
                                    data["premarket_movers"].append({
                                        "ticker": ticker,
                                        "gap_pct": round(gap_pct, 2),
                                        "prev_close": round(prev_close, 2),
                                        "open": round(curr_open, 2),
                                        "direction": "up" if gap_pct > 0 else "down",
                                    })
                        except Exception:
                            continue

                data["premarket_movers"].sort(key=lambda x: abs(x.get("gap_pct", 0)), reverse=True)

            except Exception as e:
                logger.error(f"Pre-market movers query failed: {e}")

    # Economic calendar
    if screener_config.get("include_economic_calendar", True):
        data["economic_calendar"] = get_economic_calendar()

    return data


def _default_gap_universe() -> list[str]:
    """
    Default liquid equity universe for gap scanning (~500 names).

    Covers: S&P 500 components, high-beta / momentum names, popular ETFs,
    recent IPOs (2023–2025), and biotech gap movers. Sorted alphabetically,
    deduplicated, and validated against EQUS.MINI ohlcv-1d (2026-04-11).

    Expanded from ~141 → 498 on 2026-04-13. Symbols that failed Databento
    resolution were removed: ANSS, BGNE, CFLT, DARDEN, DFS, EXAS, FI, HES,
    IPG, JNPR, K, MMC, MRO, MRTX, PARA, SGEN.

    Configurable per-pipeline by adding a 'symbols' list to the databento_query.
    """
    return [
        # ── S&P 500 + major US equities (alphabetical) ──────────────
        "A",    "AAL",  "AAPL", "ABBV", "ABNB", "ABT",  "ACAD", "ACN",  "ADBE", "ADI",
        "ADM",  "ADP",  "AEE",  "AEHR", "AEP",  "AES",  "AFL",  "AFRM", "AIG",  "AJG",
        "AKAM", "ALB",  "ALGN", "ALL",  "ALNY", "AMAT", "AMC",  "AMD",  "AME",  "AMGN",
        "AMP",  "AMT",  "AMZN", "AON",  "APA",  "APD",  "APLS", "APP",  "APTV", "ARE",
        "ARKF", "ARKG", "ARKK", "ARM",  "ARWR", "ASTS", "ATO",  "AVB",  "AVGO", "AXON",
        "AXP",  "AZO",
        "BA",   "BAC",  "BALL", "BAX",  "BBY",  "BDX",  "BEN",  "BG",   "BIIB", "BIRK",
        "BITO", "BK",   "BKNG", "BKR",  "BLK",  "BMRN", "BR",   "BRK.B","BSX",  "BWA",
        "BXP",
        "C",    "CAG",  "CARR", "CART", "CASY", "CAT",  "CAVA", "CB",   "CBOE", "CCI",
        "CCL",  "CDNS", "CDW",  "CE",   "CEG",  "CF",   "CFG",  "CHD",  "CHTR", "CINF",
        "CL",   "CLOV", "CLX",  "CMCSA","CME",  "CMG",  "CMI",  "CMS",  "COF",  "COIN",
        "COP",  "COST", "CPB",  "CPRT", "CPT",  "CRCL", "CRL",  "CRM",  "CRNX", "CRWD",
        "CSCO", "CSX",  "CTAS", "CTRA", "CTVA", "CVX",  "CYTK", "CZR",
        "D",    "DAL",  "DD",   "DDOG", "DE",   "DECK", "DELL", "DG",   "DHI",  "DHR",
        "DIA",  "DIS",  "DLR",  "DLTR", "DOCS", "DOV",  "DOW",  "DPZ",  "DUK",  "DUOL",
        "DVN",
        "DXCM",
        "EA",   "EBAY", "ECL",  "ED",   "EEM",  "EL",   "EMN",  "EMR",  "EOG",  "EPAM",
        "EQIX",
        "EQR",  "ES",   "ESS",  "ET",   "ETN",  "ETSY", "EVRG", "EW",   "EXC",  "EXPE",
        "F",    "FANG", "FAST", "FCX",  "FDX",  "FE",   "FFIV", "FIS",  "FISV", "FITB",
        "FIVN", "FMC",  "FOX",  "FOXA", "FRT",  "FTNT", "FXI",
        "GD",   "GE",   "GEHC", "GEN",  "GILD", "GIS",  "GL",   "GLBE", "GLD",  "GLW",
        "GM",   "GME",  "GOOG", "GOOGL","GPC",  "GPN",  "GRMN", "GS",   "GWW",
        "H",    "HAL",  "HALO", "HAS",  "HBAN", "HD",   "HII",  "HLT",  "HOLX", "HON",
        "HOOD", "HPE",  "HPQ",  "HRL",  "HST",  "HSY",  "HUBS", "HYG",  "HYLN",
        "IBB",  "IBIT", "IBKR", "IBM",  "ICE",  "IDXX", "IFF",  "ILMN", "INCY", "INSM",
        "INTA",
        "INTC", "INTU", "INVH", "IONQ", "IONS", "IOT",  "IP",   "IQV",  "IR",   "IRM",
        "ISRG", "IT",   "ITW",  "IVZ",  "IWM",
        "JBHT", "JNJ",  "JOBY", "JPM",
        "KDP",  "KEY",  "KEYS", "KHC",  "KIM",  "KLAC", "KMB",  "KMI",  "KMX",  "KO",
        "KRE",  "KRYS", "KVYO", "KWEB",
        "LCID", "LEN",  "LI",   "LIN",  "LKQ",  "LLY",  "LMT",  "LNT",  "LOW",  "LQD",
        "LRCX", "LULU", "LUNR", "LUV",  "LVS",  "LYV",
        "MA",   "MAA",  "MAR",  "MCD",  "MCHP", "MCO",  "MDB",  "MDLZ", "MDT",  "MET",
        "META", "MGM",  "MKC",  "MLM",  "MNDY", "MNST", "MO",   "MOS",  "MPC",  "MPWR",
        "MRK",  "MRNA", "MRVL", "MS",   "MSCI", "MSFT", "MSTR", "MTB",  "MTCH", "MTD",
        "MU",
        "NBIX", "NCLH", "NDAQ", "NEE",  "NEM",  "NET",  "NFLX", "NI",   "NIO",  "NKE",
        "NOC",  "NOW",  "NRG",  "NSC",  "NTRS", "NUE",  "NVDA", "NVR",  "NWS",  "NWSA",
        "NXPI",
        "O",    "ODFL", "OKE",  "OKTA", "OMC",  "ON",   "ONON", "OPEN", "ORCL", "ORLY",
        "OXY",
        "PANW", "PAYC", "PAYX", "PCAR", "PCVX", "PEG",  "PEP",  "PFE",  "PG",   "PH",
        "PHM",  "PINS", "PKG",  "PLD",  "PLTR", "PM",   "PNC",  "PODD", "POOL", "PPG",
        "PRU",  "PSA",  "PSX",  "PYPL",
        "QCOM", "QQQ",  "QUBT",
        "RARE", "RBLX", "RCL",  "RDDT", "REG",  "REGN", "RF",   "RGTI", "RIVN", "RJF",
        "RKLB", "RL",   "ROK",  "ROP",  "ROST", "RSG",  "RTX",  "RVMD", "RXRX",
        "SBUX", "SCHW", "SEE",  "SHW",  "SJM",  "SLB",  "SLV",  "SMCI", "SMH",  "SNA",
        "SNAP", "SNOW", "SNPS", "SO",   "SOFI", "SOXL", "SPG",  "SPGI", "SPOT", "SPY",
        "SRE",  "SRPT", "STT",  "STX",  "STZ",  "SWK",  "SWKS", "SYF",  "SYK",  "SYY",
        "T",    "TAP",  "TDY",  "TEAM", "TFC",  "TGT",  "TJX",  "TLT",  "TMO",  "TMUS",
        "TOST", "TPR",  "TQQQ", "TRGP", "TRMB", "TROW", "TRV",  "TSCO", "TSLA", "TSN",
        "TT",   "TTD",  "TTWO", "TXN",  "TXT",
        "UAL",  "UDR",  "ULTA", "UNH",  "UNP",  "UPS",  "UPST", "URI",  "USB",  "UTHR",
        "V",    "VEEV", "VICI", "VLO",  "VMC",  "VRSK", "VRT",  "VRTX", "VST",  "VTR",
        "VZ",
        "WAT",  "WBD",  "WDAY", "WDC",  "WEC",  "WELL", "WFC",  "WM",   "WMB",  "WMT",
        "WTRG", "WYNN",
        "XBI",  "XEL",  "XENE", "XHB",  "XLB",  "XLE",  "XLF",  "XLI",  "XLK",  "XLP",
        "XLRE", "XLU",  "XLV",  "XLY",  "XOM",  "XPEV", "XYL",
        "YUM",
        "ZBH",  "ZBRA", "ZION", "ZS",
        # ── ETFs (duplicated above in alpha order, listed here for findability) ──
        # SPY, QQQ, IWM, DIA, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLB, XLRE, XLU,
        # GLD, SLV, TLT, HYG, LQD, ARKK, ARKG, ARKF, SOXL, TQQQ, SMH, XBI, IBB,
        # KRE, XHB, KWEB, EEM, FXI, BITO, IBIT
    ]


def _run_gap_scanner(client: DatabentClient, screener_config: dict) -> dict:
    """
    Gap scanner for gap_up and gap_down pipelines.

    Uses ohlcv-1d (end=today midnight UTC) for yesterday's close and
    ohlcv-1m (from market open) for today's intraday action.
    """
    data = {
        "gaps": [],
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "universe_size": 0,
    }
    queries = screener_config.get("databento_queries", [])

    for query in queries:
        if query.get("type") != "gap_scanner":
            continue

        direction = query.get("direction", "up")
        min_gap_pct = abs(query.get("min_gap_pct", 3.0))
        min_volume = query.get("min_volume", 500000)
        lookback_minutes = query.get("lookback_minutes", 30)
        dataset = query.get("dataset", "EQUS.MINI")

        # Symbol universe: use config symbols or default liquid universe
        symbols = query.get("symbols") or _default_gap_universe()
        # Deduplicate while preserving order
        seen = set()
        symbols = [s for s in symbols if not (s in seen or seen.add(s))]
        data["universe_size"] = len(symbols)

        logger.info(
            f"Gap scanner: direction={direction}, min_gap={min_gap_pct}%, "
            f"min_vol={min_volume:,}, universe={len(symbols)} symbols"
        )

        try:
            gaps = client.query_gap_candidates(
                symbols=symbols,
                direction=direction,
                min_gap_pct=min_gap_pct,
                min_volume=min_volume,
                lookback_minutes=lookback_minutes,
                dataset=dataset,
            )

            # Enrich with BPA setups where we have intraday bars
            # (query_gap_candidates already returns key_levels)
            data["gaps"].extend(gaps)
            logger.info(f"Gap scanner found {len(gaps)} qualifying gaps (direction={direction})")

        except Exception as e:
            logger.error(f"Gap scanner query failed: {e}")
            data["error"] = str(e)

    return data


_SECTOR_MAP = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLC":  "Communication Services",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    # Sub-sector / industry group proxies
    "SOXX": "Semiconductors",
    "IGV":  "Software",
    "HACK": "Cybersecurity",
    "IBB":  "Biotech",
    "XPH":  "Pharmaceuticals",
    "XOP":  "Oil & Gas E&P",
    "OIH":  "Oil Services",
    "KBE":  "Banks",
    "KRE":  "Regional Banks",
    "ITA":  "Aerospace & Defense",
    "JETS": "Airlines",
    "ARKK": "Disruptive Innovation",
    "ARKG": "Genomics",
    "ARKW": "Next Gen Internet",
}

_SECTOR_ETFS = list(_SECTOR_MAP.keys())

_EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "AMD",
    "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI", "CRWD", "PANW",
    "FTNT", "SNOW", "PLTR", "DDOG", "ZS", "NET", "WDAY", "NOW", "CRM", "ADBE",
    "INTU", "COIN", "HOOD", "MSTR", "TTD", "RBLX",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "V", "MA", "PYPL", "SQ",
    "LLY", "UNH", "JNJ", "PFE", "ABBV", "MRK", "AMGN", "GILD", "BIIB", "REGN",
    "MRNA", "VRTX", "IDXX", "ISRG", "DXCM",
    "COST", "WMT", "TGT", "HD", "NKE", "SBUX", "MCD", "CMG", "BKNG", "ABNB",
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "DVN", "MPC", "VLO",
    "CAT", "DE", "BA", "LMT", "RTX", "GE", "HON", "UPS", "FDX", "DAL", "UAL",
    "NFLX", "DIS", "SPOT", "SNAP",
    "GME", "AMC", "LCID", "RIVN", "NIO", "SOFI", "UPST", "AFRM",
    "SPY", "QQQ", "IWM",
]


def _run_new_extremes(client: DatabentClient, screener_config: dict) -> dict:
    """
    New highs / new lows screener.

    Queries ohlcv-1d over the max lookback period (252 days).
    For each symbol, checks whether today's close sets a 20/50/252-day high or low.
    Ranks by relative strength vs 50-day MA.
    """
    queries = screener_config.get("databento_queries", [])
    screener_type = screener_config.get("type", "new_highs")

    direction = "high" if screener_type == "new_highs" else "low"
    periods = [20, 50, 252]
    dataset = "EQUS.MINI"
    symbols = _EQUITY_UNIVERSE

    for q in queries:
        direction = q.get("direction", direction)
        periods = q.get("periods", periods)
        dataset = q.get("dataset", dataset)
        symbols = q.get("symbols") or symbols

    lookback_days = max(periods) + 10  # a few extra days buffer

    data = {
        "extremes": [],
        "direction": direction,
        "periods_checked": periods,
        "scan_time": datetime.now(timezone.utc).isoformat(),
    }

    try:
        df = client.query_daily(symbols, lookback_days=lookback_days, dataset=dataset)
    except Exception as e:
        logger.error(f"New extremes query failed: {e}")
        data["error"] = str(e)
        return data

    if df.empty:
        logger.warning("New extremes: no data returned")
        return data

    # Normalise symbol column
    sym_col = "symbol" if "symbol" in df.columns else None

    for symbol in symbols:
        try:
            sym_df = df[df[sym_col] == symbol] if sym_col else df
            if sym_df.empty or len(sym_df) < 5:
                continue

            sym_df = sym_df.sort_index()
            today_close = float(sym_df["close"].iloc[-1])
            today_vol = int(sym_df["volume"].iloc[-1]) if "volume" in sym_df.columns else 0
            prev_close = float(sym_df["close"].iloc[-2])
            change_pct = (today_close - prev_close) / prev_close * 100

            # Which period extremes does today's close satisfy?
            matched_periods = []
            for p in periods:
                window = sym_df["close"].iloc[-p:] if len(sym_df) >= p else sym_df["close"]
                if direction == "high" and today_close >= float(window.max()):
                    matched_periods.append(p)
                elif direction == "low" and today_close <= float(window.min()):
                    matched_periods.append(p)

            if not matched_periods:
                continue

            # Relative strength proxy: today_close vs 50-day MA
            ma50 = float(sym_df["close"].tail(50).mean()) if len(sym_df) >= 50 else float(sym_df["close"].mean())
            rs_pct = (today_close - ma50) / ma50 * 100

            # Min RS filter
            min_rs = screener_config.get("min_relative_strength", 0)
            if direction == "high" and rs_pct < (min_rs - 50):  # above MA
                continue
            if direction == "low" and rs_pct > -(min_rs - 50):  # below MA
                continue

            key_levels = _identify_key_levels(sym_df)
            extreme_label = "/".join(
                ("52-wk" if p == 252 else f"{p}-day") for p in matched_periods
            )

            data["extremes"].append({
                "ticker": symbol,
                "close": round(today_close, 2),
                "change_pct": round(change_pct, 2),
                "volume": today_vol,
                "extreme_type": f"{direction} ({extreme_label})",
                "matched_periods": matched_periods,
                "rs_vs_50ma_pct": round(rs_pct, 2),
                "key_levels": key_levels,
            })

        except Exception as e:
            logger.debug(f"New extremes calc failed for {symbol}: {e}")

    # Sort: highs by strongest RS, lows by weakest RS
    reverse = direction == "high"
    data["extremes"].sort(key=lambda x: x.get("rs_vs_50ma_pct", 0), reverse=reverse)
    logger.info(
        f"New {direction}s screener: {len(data['extremes'])} qualifying symbols "
        f"from {len(symbols)} universe"
    )
    return data


def _run_top_gainers(client: DatabentClient, screener_config: dict) -> dict:
    """
    Best stocks of the day screener.

    Queries ohlcv-1d for the past 5 days.
    Today's % change = (close - prev_close) / prev_close.
    Sorts by % gain, optionally filters by BPA setup.
    """
    queries = screener_config.get("databento_queries", [])
    min_gain_pct = 2.0
    min_volume = 1_000_000
    require_bpa = False
    dataset = "EQUS.MINI"
    symbols = _EQUITY_UNIVERSE

    for q in queries:
        min_gain_pct = q.get("min_gain_pct", min_gain_pct)
        min_volume = q.get("min_volume", min_volume)
        require_bpa = q.get("require_bpa_setup", require_bpa)
        dataset = q.get("dataset", dataset)
        symbols = q.get("symbols") or symbols

    data = {
        "gainers": [],
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "min_gain_pct": min_gain_pct,
    }

    try:
        df = client.query_daily(symbols, lookback_days=10, dataset=dataset)
    except Exception as e:
        logger.error(f"Top gainers query failed: {e}")
        data["error"] = str(e)
        return data

    if df.empty:
        logger.warning("Top gainers: no data returned")
        return data

    sym_col = "symbol" if "symbol" in df.columns else None

    for symbol in symbols:
        try:
            sym_df = df[df[sym_col] == symbol] if sym_col else df
            if sym_df.empty or len(sym_df) < 2:
                continue
            sym_df = sym_df.sort_index()

            today_close = float(sym_df["close"].iloc[-1])
            prev_close = float(sym_df["close"].iloc[-2])
            today_open = float(sym_df["open"].iloc[-1])
            today_high = float(sym_df["high"].iloc[-1])
            today_low = float(sym_df["low"].iloc[-1])
            today_vol = int(sym_df["volume"].iloc[-1]) if "volume" in sym_df.columns else 0

            if prev_close == 0:
                continue
            change_pct = (today_close - prev_close) / prev_close * 100

            if change_pct < min_gain_pct:
                continue
            if today_vol < min_volume:
                continue

            # BPA setup detection (optional)
            bpa_setups = []
            if len(sym_df) >= 5:
                try:
                    setups = detect_all(sym_df)
                    bpa_setups = [
                        {"type": s.setup_type, "entry": s.entry, "stop": s.stop,
                         "target": s.target, "confidence": s.confidence,
                         "entry_mode": s.entry_mode}
                        for s in setups[:3]
                    ]
                except Exception:
                    pass

            if require_bpa and not bpa_setups:
                continue

            key_levels = _identify_key_levels(sym_df)

            data["gainers"].append({
                "ticker": symbol,
                "change_pct": round(change_pct, 2),
                "close": round(today_close, 2),
                "open": round(today_open, 2),
                "high": round(today_high, 2),
                "low": round(today_low, 2),
                "prev_close": round(prev_close, 2),
                "volume": today_vol,
                "bpa_setups": bpa_setups,
                "key_levels": key_levels,
            })

        except Exception as e:
            logger.debug(f"Top gainers calc failed for {symbol}: {e}")

    data["gainers"].sort(key=lambda x: x.get("change_pct", 0), reverse=True)
    logger.info(
        f"Top gainers: {len(data['gainers'])} stocks above {min_gain_pct}% "
        f"from {len(symbols)} universe"
    )
    return data


def _run_industry_groups(client: DatabentClient, screener_config: dict) -> dict:
    """
    Sector / industry group rotation screener.

    Queries ohlcv-1d for all sector ETFs and sub-sector proxies.
    Calculates 1-day, 5-day, 20-day % change for each.
    Ranks by 1-day performance to identify rotation.
    """
    queries = screener_config.get("databento_queries", [])
    lookback_days = 25  # 20-day perf + buffer
    dataset = "EQUS.MINI"
    include_industry_groups = True

    for q in queries:
        lookback_days = max(lookback_days, q.get("lookback_days", lookback_days) + 5)
        dataset = q.get("dataset", dataset)
        include_industry_groups = q.get("industry_groups", include_industry_groups)

    symbols = list(_SECTOR_MAP.keys())
    if not include_industry_groups:
        symbols = [s for s in symbols if s in (
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"
        )]

    data = {
        "sectors": [],
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
    }

    try:
        df = client.query_daily(symbols, lookback_days=lookback_days, dataset=dataset)
    except Exception as e:
        logger.error(f"Industry groups query failed: {e}")
        data["error"] = str(e)
        return data

    if df.empty:
        logger.warning("Industry groups: no data returned")
        return data

    sym_col = "symbol" if "symbol" in df.columns else None

    for symbol in symbols:
        try:
            sym_df = df[df[sym_col] == symbol] if sym_col else df
            if sym_df.empty or len(sym_df) < 2:
                continue
            sym_df = sym_df.sort_index()

            def pct_change_n(n: int) -> float:
                if len(sym_df) < n + 1:
                    return None
                base = float(sym_df["close"].iloc[-(n + 1)])
                if base == 0:
                    return None
                return round((float(sym_df["close"].iloc[-1]) - base) / base * 100, 2)

            chg_1d = pct_change_n(1)
            chg_5d = pct_change_n(5)
            chg_20d = pct_change_n(20)

            if chg_1d is None:
                continue

            today_close = float(sym_df["close"].iloc[-1])
            today_vol = int(sym_df["volume"].iloc[-1]) if "volume" in sym_df.columns else 0
            key_levels = _identify_key_levels(sym_df)

            data["sectors"].append({
                "ticker": symbol,
                "sector_name": _SECTOR_MAP.get(symbol, symbol),
                "close": round(today_close, 2),
                "change_1d_pct": chg_1d,
                "change_5d_pct": chg_5d,
                "change_20d_pct": chg_20d,
                "volume": today_vol,
                "key_levels": key_levels,
            })

        except Exception as e:
            logger.debug(f"Industry groups calc failed for {symbol}: {e}")

    data["sectors"].sort(key=lambda x: x.get("change_1d_pct", 0), reverse=True)
    logger.info(f"Industry groups: {len(data['sectors'])} sectors/groups returned")
    return data


def _run_generic(client: DatabentClient, screener_config: dict) -> dict:
    """Generic screener for other pipeline types."""
    data = {"symbols": []}
    queries = screener_config.get("databento_queries", [])

    for query in queries:
        try:
            symbols = query.get("symbols", [])
            schema = query.get("schema", "ohlcv-1d")
            lookback = query.get("lookback_days", 60)

            df = client.query_daily(symbols, lookback_days=lookback, dataset=query.get("dataset", "EQUS.MINI"))
            if not df.empty:
                for symbol in symbols:
                    sym_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
                    if sym_df.empty:
                        continue

                    data["symbols"].append({
                        "ticker": symbol,
                        "last_price": float(sym_df["close"].iloc[-1]),
                        "change_pct": float(
                            (sym_df["close"].iloc[-1] - sym_df["close"].iloc[0])
                            / sym_df["close"].iloc[0] * 100
                        ),
                        "high_52w": float(sym_df["high"].max()),
                        "low_52w": float(sym_df["low"].min()),
                        "key_levels": _identify_key_levels(sym_df),
                    })

                    setups = detect_all(sym_df)
                    if setups:
                        data["symbols"][-1]["bpa_setups"] = [
                            {"type": s.setup_type, "entry": s.entry, "stop": s.stop,
                             "target": s.target, "confidence": s.confidence}
                            for s in setups[:3]
                        ]

        except Exception as e:
            logger.error(f"Generic screener query failed: {e}")

    return data


def _identify_key_levels(df: pd.DataFrame) -> list[float]:
    """Identify key support/resistance levels from price data."""
    levels = []

    if len(df) < 2:
        return levels

    # Recent high and low
    levels.append(round(float(df["high"].max()), 2))
    levels.append(round(float(df["low"].min()), 2))

    # Prior day close (if available)
    if len(df) >= 2:
        levels.append(round(float(df["close"].iloc[-2]), 2))

    # VWAP-like level: volume-weighted average price
    if "volume" in df.columns and df["volume"].sum() > 0:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = float((tp * df["volume"]).sum() / df["volume"].sum())
        levels.append(round(vwap, 2))

    # Round numbers near current price
    last = float(df["close"].iloc[-1])
    magnitude = 10 ** max(0, len(str(int(last))) - 2)
    round_level = round(last / magnitude) * magnitude
    levels.append(float(round_level))

    return sorted(set(levels))


def _count_results(results: dict) -> int:
    """Count total items in screener results."""
    data = results.get("data", {})
    count = 0
    count += len(data.get("futures", {}))
    count += len(data.get("premarket_movers", []))
    count += len(data.get("economic_calendar", []))
    count += len(data.get("symbols", []))
    count += len(data.get("gaps", []))
    count += len(data.get("extremes", []))
    count += len(data.get("gainers", []))
    count += len(data.get("sectors", []))
    return count
