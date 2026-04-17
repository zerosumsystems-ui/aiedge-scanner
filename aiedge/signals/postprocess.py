"""Post-scoring annotations + dashboard-ranking helpers.

Everything that runs *after* `score_gap` returns but *before* the
dashboard sees the list. Four concerns:

  - ADR-multiple annotation: intraday-range-to-20-day-ADR ratio, the
    "Minervini multiple"
  - ETF / dual-class family dedup: collapse multiple tickers of the
    same underlying story to a single family leader
  - Rank + urgency delta vs prior scan
  - String formatters for rank-change / urgency-delta columns

Extracted from live_scanner.py (Phase 4e).
"""

from __future__ import annotations

import pandas as pd


# ── ADR multiple ────────────────────────────────────────────────────

def annotate_adr_multiple(score: dict, df_1m: pd.DataFrame, sym: str,
                          daily_atrs: dict[str, float]) -> None:
    """Compute Minervini-style ADR multiple = today's range ÷ 20-day ADR.

    Uses the full-resolution 1-min bars so intraday wick extremes are
    preserved (a 5-min resample would round wicks that print between
    5-min anchors). The 20-day ADR baseline comes from the
    caller-provided `daily_atrs` cache (populated at scanner startup
    by `aiedge.data.databento.fetch_prior_closes`) — this function
    never re-fetches.

    Mutates `score` in place with three new keys: `today_range`,
    `adr_20`, `adr_multiple`.
    """
    today_high = float(df_1m["high"].max())
    today_low = float(df_1m["low"].min())
    today_range = max(today_high - today_low, 0.0)
    adr_20 = daily_atrs.get(sym, 0.0) or 0.0
    adr_mult = (today_range / adr_20) if adr_20 > 0 else 0.0
    score["today_range"] = round(today_range, 4)
    score["adr_20"] = round(adr_20, 4)
    score["adr_multiple"] = round(adr_mult, 2)


# ── ETF / dual-class family dedup ───────────────────────────────────
#
# Many index ETFs move as a single "story": on a sharp NDX up-move, QQQ, TQQQ,
# QLD, and SQQQ (inverse) all rank high together and clutter the top of the
# dashboard with three or four tickers telling the same tape. This map lets us
# collapse each family down to a single "family leader" (the highest-urgency
# member surfaced this scan) and suppress the rest.
#
# Extend: add a new family name (key) whose value is the list of tickers that
# trade the same underlying. Non-ETF equities are *not* in any family and pass
# through the dedup untouched. Case-sensitive — use upper-case tickers.
ETF_FAMILIES: dict[str, list[str]] = {
    "SPX":       ["SPY", "IVV", "VOO", "SPLG", "SSO", "UPRO", "SPUU",
                  "SPXL", "SPXU", "SPXS", "SPDN", "SDS", "SH"],
    "NDX":       ["QQQ", "QQQM", "TQQQ", "SQQQ", "QLD", "QID", "PSQ"],
    "DJIA":      ["DIA", "UDOW", "SDOW", "DDM", "DXD", "DOG"],
    "RUSSELL":   ["IWM", "VTWO", "UWM", "TNA", "TZA", "TWM", "SRTY", "RWM"],
    "TECH":      ["XLK", "VGT", "TECL", "TECS", "ROM", "REW"],
    "GOLD":      ["GLD", "IAU", "GLDM", "BAR", "OUNZ", "UGL", "DGP", "GLL", "DGLD"],
    "SILVER":    ["SLV", "SIVR", "AGQ", "ZSL"],
    "OIL":       ["USO", "BNO", "UCO", "SCO", "OILU", "OILD"],
    "NATGAS":    ["UNG", "BOIL", "KOLD"],
    "SEMIS":     ["SMH", "SOXX", "SOXL", "SOXS", "USD", "SSG"],
    "BONDS20":   ["TLT", "TMF", "TMV", "TBT", "TBF"],
    "VIX":       ["VXX", "VIXY", "VIXM", "UVXY", "SVXY", "UVIX", "SVIX"],
    "CHINA":     ["FXI", "MCHI", "KWEB", "YINN", "YANG", "FXP", "CHAU", "CHAD"],
    "EMERGING":  ["EEM", "IEMG", "VWO", "EDC", "EDZ"],
    "FIN":       ["XLF", "VFH", "FAS", "FAZ", "UYG", "SKF"],
    "ENERGY":    ["XLE", "VDE", "ERX", "ERY", "GUSH", "DRIP"],
    "BIOTECH":   ["XBI", "IBB", "LABU", "LABD"],
    "REIT":      ["IYR", "VNQ", "XLRE", "URE", "DRN", "DRV", "SRS"],
    "HEALTHCARE": ["XLV", "VHT", "CURE", "RXD"],
    "STAPLES":   ["XLP", "VDC"],
    "DISCR":     ["XLY", "VCR"],
    "UTILS":     ["XLU", "VPU"],
    "INDUSTR":   ["XLI", "VIS"],
    "DEFENSE":   ["ITA", "DFEN"],
    "HOMEBUILD": ["ITB", "XHB", "NAIL", "CLAW"],
    "REGBANK":   ["KRE", "KBWB", "DPST", "WDRW"],
    "CLEANENRG": ["ICLN", "TAN", "FAN", "PBD"],
    "RETAIL":    ["XRT", "RETL"],
    "TRANSPORT": ["IYT", "XTN"],
    "BITCOIN":   ["BITO", "BTF", "BITI", "BITX", "IBIT", "FBTC", "ARKB",
                  "HODL", "BRRR", "GBTC", "BITB"],
    "ETHER":     ["ETHA", "ETHE", "FETH", "ETHV", "ETH", "ETHU", "ETHD"],
}

# Dual-class / same-company pairs. Treated identically to ETF families in dedup:
# same family key → higher-urgency wins, siblings suppressed with badge tooltip.
# The two share classes of the same underlying business trade as one story.
SAME_COMPANY: dict[str, list[str]] = {
    "GOOG_FAMILY":  ["GOOG", "GOOGL"],
    "BRK_FAMILY":   ["BRK.A", "BRK.B", "BRK-A", "BRK-B"],   # include both Databento and canonical forms
    "FOX_FAMILY":   ["FOX", "FOXA"],
    "NWS_FAMILY":   ["NWS", "NWSA"],
    "HEI_FAMILY":   ["HEI", "HEI.A", "HEI-A"],
    "MKC_FAMILY":   ["MKC", "MKC.V", "MKC-V"],
    "LEN_FAMILY":   ["LEN", "LEN.B", "LEN-B"],
    "PRA_FAMILY":   ["PRA", "PRAA"],
}

# Merge same-company into the main family index so _dedup_etf_families() handles these too.
for _fam, _tickers in SAME_COMPANY.items():
    ETF_FAMILIES[_fam] = _tickers

# Reverse lookup built once at import: ticker -> family name
_TICKER_TO_FAMILY: dict[str, str] = {
    t: fam for fam, tickers in ETF_FAMILIES.items() for t in tickers
}


def _dedup_etf_families(results: list[dict]) -> list[dict]:
    """Collapse ETF-family repeats in a ranked list.

    `results` must already be sorted by urgency desc. For each family that
    appears, keep the single highest-urgency ticker (the "family leader") and
    drop the rest. The leader is annotated with:

        r["family"]               family name (e.g. "NDX")
        r["family_leader"]        True
        r["family_siblings"]      list of suppressed tickers (in urgency order)
        r["family_sibling_count"] len(siblings)

    Non-ETF tickers (not in ETF_FAMILIES) pass through unchanged — this is
    intentionally a pure map-based filter, no asset-class lookup needed.
    """
    seen: set[str] = set()
    out: list[dict] = []
    siblings_by_fam: dict[str, list[str]] = {}

    for r in results:
        tkr = r.get("ticker", "")
        fam = _TICKER_TO_FAMILY.get(tkr)
        if fam is None:
            out.append(r)
            continue
        if fam not in seen:
            seen.add(fam)
            r["family"] = fam
            r["family_leader"] = True
            out.append(r)
        else:
            siblings_by_fam.setdefault(fam, []).append(tkr)

    for r in out:
        fam = r.get("family")
        if fam and r.get("family_leader"):
            sibs = siblings_by_fam.get(fam, [])
            r["family_siblings"] = sibs
            r["family_sibling_count"] = len(sibs)
    return out


# ── Rank + urgency-delta movement ───────────────────────────────────

def _compute_movement(results: list[dict], prior: dict) -> list[dict]:
    """Annotate each result with rank, rank_change, and urgency_delta vs prior scan."""
    first_scan = len(prior) == 0
    for i, r in enumerate(results):
        rank = i + 1
        ticker = r["ticker"]
        r["rank"] = rank
        if first_scan:
            r["prev_rank"] = None
            r["rank_change"] = None
            r["urgency_delta"] = None
            r["_first_scan"] = True
        elif ticker in prior:
            prev_rank = prior[ticker]["rank"]
            r["prev_rank"] = prev_rank
            r["rank_change"] = prev_rank - rank
            r["urgency_delta"] = round(r["urgency"] - prior[ticker]["urgency"], 1)
            r["_first_scan"] = False
        else:
            r["prev_rank"] = None
            r["rank_change"] = None
            r["urgency_delta"] = None
            r["_first_scan"] = False
    return results


def _fmt_movement(r: dict) -> str:
    if r.get("_first_scan"):
        return "—"
    rc = r.get("rank_change")
    if rc is None:
        return "NEW"
    pr = r["prev_rank"]
    if rc == 0:
        return f"was #{pr}  (=)"
    sign = "+" if rc > 0 else ""
    return f"was #{pr}  ({sign}{rc})"


def _fmt_delta(r: dict) -> str:
    if r.get("_first_scan") or r.get("urgency_delta") is None:
        return "—"
    d = r["urgency_delta"]
    sign = "+" if d >= 0 else ""
    return f"U {sign}{d:.1f}"
