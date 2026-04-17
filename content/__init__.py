"""content — YouTube/newsletter video pipeline.

Isolated from the scanner. Takes scanner output (gaps, scan results)
and produces: script → narration → b-roll → assembly → upload → newsletter.

Scanner-shared utilities (shared/config_loader, shared/sqlite_logger,
shared/notifier, shared/databento_client, shared/chart_renderer) are
imported from scanner-root shared/ — they are used by both pipelines.
"""
