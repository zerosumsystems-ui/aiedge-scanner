"""
Assembly stage: combines charts, narration, and B-roll into final MP4.
"""

import logging
from pathlib import Path
from datetime import datetime

from shared.ffmpeg_assembler import FFmpegAssembler
from shared.config_loader import get_output_dir

logger = logging.getLogger(__name__)


def run(
    config: dict,
    script: dict,
    chart_paths: list[str],
    audio_paths: list[dict],
    broll: dict,
    run_id: str,
    run_dir: str,
) -> str:
    """
    Assemble the final video.

    Returns path to the final MP4.
    """
    pipeline_name = config["pipeline_name"]

    assembler = FFmpegAssembler(config)

    # Generate output filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    title_slug = script["title"].lower().replace(" ", "_").replace("/", "-")[:50]
    output_dir = get_output_dir(pipeline_name)
    output_path = str(output_dir / f"{date_str}_{title_slug}.mp4")

    result = assembler.assemble(
        script=script,
        chart_paths=chart_paths,
        audio_paths=audio_paths,
        broll_intro=broll.get("intro_path"),
        broll_outro=broll.get("outro_path"),
        output_path=output_path,
        run_dir=run_dir,
    )

    file_size = Path(result).stat().st_size / (1024 * 1024)
    logger.info(f"Final video: {result} ({file_size:.1f}MB)")
    return result
