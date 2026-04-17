"""Runners — thin orchestrators, the only modules with side-effect chains.

Live scanner, backfill, and backtest all live here. All three import
from the same signals/ module — the research/production invariant.
"""
