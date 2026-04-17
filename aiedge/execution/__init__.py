"""Execution layer — simulates fills.

Shared by backtest and live-replay. Models +1 tick entries, stop/target
hits, time-based exits. No live order routing (not a brokerage bot).
"""
