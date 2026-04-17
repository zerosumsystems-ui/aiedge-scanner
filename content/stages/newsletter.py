"""
Newsletter stage: publishes email newsletter via Beehiiv.
Runs in parallel with YouTube upload — uses same run artifacts.
"""

import logging

from content.shared.newsletter_publisher import NewsletterPublisher

logger = logging.getLogger(__name__)


def run(
    config: dict,
    script: dict,
    chart_paths: list[str],
    youtube_url: str,
    run_id: str,
) -> dict:
    """
    Publish newsletter from pipeline run artifacts.

    Returns: {post_id, post_url, status} or empty dict if disabled/failed.
    """
    nl_config = config.get("newsletter", {})
    pipeline_name = config["pipeline_name"]

    if not nl_config.get("enabled", False):
        logger.info("Newsletter disabled")
        return {}

    import os
    publisher = NewsletterPublisher(
        api_key=os.environ.get(nl_config.get("api_key_env", "BEEHIIV_API_KEY")),
        publication_id=os.environ.get(nl_config.get("publication_id_env", "BEEHIIV_PUBLICATION_ID")),
    )

    result = publisher.publish(
        script=script,
        chart_paths=chart_paths,
        config=config,
        youtube_url=youtube_url,
        run_id=run_id,
        pipeline_name=pipeline_name,
    )

    return result or {}
