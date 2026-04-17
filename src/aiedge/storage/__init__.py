"""Storage layer — persistence.

Owns SQLite schema. Writes detections, trades, priors. All reads from
downstream (analysis, dashboard) go through this layer.
"""
