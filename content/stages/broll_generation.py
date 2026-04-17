"""
B-roll generation stage: creates intro and outro clips via Kling.
"""

import logging
from pathlib import Path

from content.shared.veo_client import VeoClient

logger = logging.getLogger(__name__)


def run(config: dict, run_id: str, run_dir: str) -> dict:
    """
    Generate intro and outro B-roll clips.

    Returns: {intro_path, outro_path}
    """
    veo_config = config.get("veo", {})
    pipeline_name = config["pipeline_name"]

    if veo_config.get("skip_entirely", False):
        logger.info("Veo B-roll skipped (skip_entirely=true)")
        return _use_fallbacks(run_dir)

    if not veo_config.get("enabled", False):
        logger.info("Veo B-roll disabled")
        return {"intro_path": None, "outro_path": None}

    client = VeoClient(model=veo_config.get("model"))
    broll_dir = Path(run_dir) / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)

    intro_path = str(broll_dir / "intro.mp4")
    outro_path = str(broll_dir / "outro.mp4")

    results = {"intro_path": None, "outro_path": None}

    # Generate intro
    try:
        results["intro_path"] = client.generate_clip(
            prompt=veo_config.get("intro_prompt", "Trading desk at dawn"),
            output_path=intro_path,
            duration_seconds=veo_config.get("duration_seconds", 3),
            aspect_ratio=veo_config.get("aspect_ratio", "16:9"),
            use_fallback_on_failure=veo_config.get("use_fallback_on_failure", True),
            clip_type="intro",
            run_id=run_id,
            pipeline_name=pipeline_name,
        )
    except Exception as e:
        logger.error(f"Intro B-roll failed: {e}")

    # Generate outro
    try:
        results["outro_path"] = client.generate_clip(
            prompt=veo_config.get("outro_prompt", "Wall Street skyline at sunrise"),
            output_path=outro_path,
            duration_seconds=veo_config.get("duration_seconds", 3),
            aspect_ratio=veo_config.get("aspect_ratio", "16:9"),
            use_fallback_on_failure=veo_config.get("use_fallback_on_failure", True),
            clip_type="outro",
            run_id=run_id,
            pipeline_name=pipeline_name,
        )
    except Exception as e:
        logger.error(f"Outro B-roll failed: {e}")

    return results


def _use_fallbacks(run_dir: str) -> dict:
    """Use fallback clips when Veo is skipped entirely."""
    client = VeoClient()
    broll_dir = Path(run_dir) / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)

    return {
        "intro_path": client._get_fallback("intro", str(broll_dir / "intro.mp4")),
        "outro_path": client._get_fallback("outro", str(broll_dir / "outro.mp4")),
    }
