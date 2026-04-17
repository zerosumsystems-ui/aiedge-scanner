"""Data layer — pure I/O. Fetches bars, levels, universe.

Only layer that talks to external data sources (Databento, APIs).
All downstream layers consume normalized DataFrames from here.
"""
