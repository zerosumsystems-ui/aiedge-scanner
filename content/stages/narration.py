"""
Narration stage: generates audio for all segments via ElevenLabs.
"""

import logging
from pathlib import Path

from content.shared.elevenlabs_narrator import Narrator

logger = logging.getLogger(__name__)


def run(config: dict, script: dict, run_id: str, run_dir: str) -> list[dict]:
    """
    Generate narration audio for all script segments.

    Returns list of {segment_index, label, audio_path, characters}.
    """
    el_config = config["elevenlabs"]
    pipeline_name = config["pipeline_name"]
    audio_dir = Path(run_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    narrator = Narrator(
        voice_id=el_config.get("voice_id"),
        model=el_config.get("model", "eleven_v3"),
    )

    results = narrator.narrate_segments(
        segments=script["segments"],
        hook=script["hook"],
        outro=script["outro"],
        output_dir=str(audio_dir),
        voice_id=el_config.get("voice_id"),
        stability=el_config.get("stability", 0.5),
        similarity_boost=el_config.get("similarity_boost", 0.75),
        style=el_config.get("style", 0.3),
        use_speaker_boost=el_config.get("use_speaker_boost", True),
        run_id=run_id,
        pipeline_name=pipeline_name,
    )

    logger.info(f"Narration complete: {len(results)} audio files")
    return results
