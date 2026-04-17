"""
Upload stage: uploads final MP4 to YouTube.
"""

import logging
from datetime import datetime

from content.shared.youtube_uploader import YouTubeUploader
from content.shared.gemini_writer import ScriptWriter

logger = logging.getLogger(__name__)


def run(
    config: dict,
    script: dict,
    video_path: str,
    run_id: str,
) -> dict:
    """
    Upload video to YouTube.

    Returns: {video_id, video_url} or empty dict if upload is skipped.
    """
    yt_config = config["youtube"]
    pipeline_name = config["pipeline_name"]

    if not yt_config.get("enabled", True):
        logger.info("YouTube upload disabled")
        return {}

    uploader = YouTubeUploader(
        token_path=yt_config.get("channel_credentials", "credentials/youtube_token.json"),
    )

    # Generate metadata
    date_str = datetime.now().strftime("%B %d, %Y")
    writer = ScriptWriter(model=config["script"].get("gemini_model", "gemini-2.5-flash"))
    metadata = writer.generate_youtube_metadata(
        script=script,
        title_template=yt_config.get("title_template", "{date} — {top_story}"),
        description_template=yt_config.get("description_template", ""),
        tags=yt_config.get("tags", []),
        date_str=date_str,
        run_id=run_id,
        pipeline_name=pipeline_name,
    )

    result = uploader.upload(
        video_path=video_path,
        title=metadata["title"],
        description=metadata["description"],
        tags=metadata["tags"],
        visibility=yt_config.get("visibility", "unlisted"),
        run_id=run_id,
        pipeline_name=pipeline_name,
    )

    logger.info(f"Uploaded: {result.get('video_url', 'N/A')}")
    return result
