"""
Script stage: sends screener data to Claude, gets back structured JSON script.
"""

import json
import logging
from pathlib import Path

from content.shared.claude_writer import ScriptWriter

logger = logging.getLogger(__name__)


def run(config: dict, screener_data: dict, run_id: str, run_dir: str) -> dict:
    """Generate video script from screener data using Claude."""
    script_config = config["script"]
    pipeline_name = config["pipeline_name"]

    # Use Claude with configurable model
    writer = ScriptWriter(model=script_config.get("claude_model", "claude-sonnet-4-6"))

    script = writer.generate_script(
        screener_data=screener_data,
        prompt_template=script_config["prompt_template"],
        run_id=run_id,
        pipeline_name=pipeline_name,
        max_tokens=script_config.get("max_tokens_per_run", 4096),
    )

    # Save script to run directory
    script_path = Path(run_dir) / "script.json"
    with open(script_path, "w") as f:
        json.dump(script, f, indent=2)

    logger.info(f"Script saved: {script_path} ({len(script['segments'])} segments)")
    return script
