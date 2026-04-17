"""
Log stage: records run results to SQLite.
"""

import logging

from shared import sqlite_logger

logger = logging.getLogger(__name__)


def run(
    run_id: str,
    pipeline_name: str,
    screener_count: int = 0,
    segment_count: int = 0,
    total_duration: float = 0,
    output_path: str = "",
    youtube_result: dict = None,
    newsletter_result: dict = None,
    estimated_cost: float = 0,
):
    """Record final run results to SQLite."""
    youtube_result = youtube_result or {}
    newsletter_result = newsletter_result or {}

    # When upload was skipped (manual --skip-upload, quota exhaustion, or
    # any other error) youtube_result has no video_id / video_url. Store
    # NULL instead of empty string so a SELECT WHERE youtube_url IS NULL
    # cleanly surfaces runs that didn't upload. Fix D companion, 2026-04-10.
    yt_video_id = youtube_result.get("video_id") or None
    yt_url = youtube_result.get("video_url") or None

    sqlite_logger.complete_run(
        run_id=run_id,
        status="success",
        screener_result_count=screener_count,
        script_segment_count=segment_count,
        total_duration_seconds=total_duration,
        output_path=output_path,
        youtube_video_id=yt_video_id,
        youtube_url=yt_url,
        newsletter_status=newsletter_result.get("status", ""),
        newsletter_post_id=newsletter_result.get("post_id", ""),
        newsletter_url=newsletter_result.get("post_url", ""),
        estimated_cost_usd=estimated_cost,
    )

    logger.info(f"Run {run_id} logged as success")
