"""
YAML config loader with validation for all pipeline configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = [
    "pipeline_name",
    "display_name",
    "schedule",
    "screener",
    "script",
    "elevenlabs",
    "assembly",
    "youtube",
    "logging",
]

REQUIRED_SCHEDULE_KEYS = ["cron", "timezone"]
REQUIRED_SCRIPT_KEYS = ["gemini_model", "prompt_template"]
REQUIRED_ELEVENLABS_KEYS = ["voice_id", "model"]
REQUIRED_YOUTUBE_KEYS = ["enabled", "visibility", "channel_credentials"]


class ConfigError(Exception):
    pass


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a pipeline YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(config).__name__}")

    _validate_top_level(config)
    _validate_schedule(config["schedule"])
    _validate_script(config["script"])
    _validate_elevenlabs(config["elevenlabs"])
    _validate_assembly(config.get("assembly", {}))
    _validate_youtube(config["youtube"])

    # Set defaults
    config.setdefault("veo", {"enabled": False})
    config["veo"].setdefault("use_fallback_on_failure", True)
    config["veo"].setdefault("skip_entirely", False)
    config["veo"].setdefault("duration_seconds", 3)
    config["veo"].setdefault("aspect_ratio", "16:9")
    config["veo"].setdefault("model", "veo-3.1-lite-preview-06-30")

    config.setdefault("newsletter", {"enabled": False})
    config.setdefault("notifications", {})
    config.setdefault("budget", {"daily_max_usd": 20.0})
    config.setdefault("cleanup", {"successful_run_days": 7, "failed_run_days": 30})

    config["screener"].setdefault("databento", {})
    config["screener"]["databento"].setdefault("allow_stale_minutes", 60)

    config["script"].setdefault("max_tokens_per_run", 10000)

    return config


def _validate_top_level(config: dict):
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise ConfigError(f"Missing required top-level keys: {missing}")


def _validate_schedule(schedule: dict):
    missing = [k for k in REQUIRED_SCHEDULE_KEYS if k not in schedule]
    if missing:
        raise ConfigError(f"Missing required schedule keys: {missing}")


def _validate_script(script: dict):
    missing = [k for k in REQUIRED_SCRIPT_KEYS if k not in script]
    if missing:
        raise ConfigError(f"Missing required script keys: {missing}")


def _validate_elevenlabs(el: dict):
    missing = [k for k in REQUIRED_ELEVENLABS_KEYS if k not in el]
    if missing:
        raise ConfigError(f"Missing required elevenlabs keys: {missing}")


def _validate_assembly(assembly: dict):
    assembly.setdefault("resolution", "1920x1080")
    assembly.setdefault("fps", 30)
    assembly.setdefault("ken_burns", {"enabled": True, "zoom_factor": 1.15, "direction": "auto"})
    assembly.setdefault("captions", {"enabled": True, "font": "Helvetica Bold", "size": 48, "position": "bottom"})
    assembly.setdefault("branding", {})


def _validate_youtube(yt: dict):
    missing = [k for k in REQUIRED_YOUTUBE_KEYS if k not in yt]
    if missing:
        raise ConfigError(f"Missing required youtube keys: {missing}")


def get_project_root() -> Path:
    """Return the video-pipeline project root."""
    return Path(__file__).parent.parent


def get_run_dir(pipeline_name: str, run_id: str) -> Path:
    """Return the working directory for a specific run."""
    d = get_project_root() / "runs" / pipeline_name / run_id
    d.mkdir(parents=True, exist_ok=True)
    for sub in ["charts", "audio", "broll"]:
        (d / sub).mkdir(exist_ok=True)
    return d


def get_output_dir(pipeline_name: str) -> Path:
    """Return the output directory for final MP4s."""
    d = get_project_root() / "output" / pipeline_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_log_dir(pipeline_name: str) -> Path:
    """Return the log directory for a pipeline."""
    d = get_project_root() / "logs" / pipeline_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_env():
    """Load environment variables from credentials/.env"""
    from dotenv import load_dotenv
    env_path = get_project_root() / "credentials" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
