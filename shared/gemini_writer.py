"""
Gemini script writer via Vertex AI — drop-in replacement for claude_writer.py.
Identical interface: ScriptWriter class, generate_script(), generate_youtube_metadata().
"""

import json
import logging
import os
import time
from typing import Any

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from shared import sqlite_logger

logger = logging.getLogger(__name__)

SCRIPT_SCHEMA_KEYS = {"title", "hook", "segments", "outro"}
SEGMENT_KEYS = {"topic", "narration", "chart_spec", "on_screen_text", "duration_seconds"}
# Required chart_spec keys. `annotations` is intentionally NOT in this set —
# it's auto-filled to {} by _validate_script if missing (see below). This
# prevents a single missing annotations field from failing the whole script
# validation and forcing a costly Gemini retry.
CHART_SPEC_KEYS = {"ticker", "timeframe", "lookback", "overlays"}

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROJECT = "gen-lang-client-0987146855"
DEFAULT_LOCATION = "us-central1"

# Gemini 2.5 Flash pricing (per million tokens)
_COST_INPUT_PER_M = 0.15
_COST_OUTPUT_PER_M = 0.60


class ScriptWriterError(Exception):
    pass


class ScriptValidationError(Exception):
    pass


class ScriptWriter:
    """
    Gemini-backed script writer. Drop-in replacement for the Claude version.
    Accepts same constructor args (model kwarg) and exposes same public methods.
    """

    def __init__(self, api_key: str = None, model: str = None):
        # api_key ignored — Vertex AI uses ADC / GOOGLE_CLOUD_PROJECT
        self.model_name = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(self.model_name)

    def generate_script(
        self,
        screener_data: dict,
        prompt_template: str,
        run_id: str = "",
        pipeline_name: str = "",
        max_tokens: int = 8192,
    ) -> dict:
        """
        Generate a video script from screener data using Gemini.
        Returns validated JSON matching the script schema.
        Retries once with a stricter prompt if validation fails.
        """
        prompt = prompt_template.replace(
            "{screener_data}", json.dumps(screener_data, indent=2, default=str)
        )

        for attempt in range(2):
            try:
                if attempt == 1:
                    prompt = self._make_stricter_prompt(prompt)
                    logger.warning("Retrying script generation with stricter prompt")

                response_text = self._call_gemini(prompt, max_tokens, run_id, pipeline_name, attempt)
                script = self._parse_and_validate(response_text)
                logger.info(f"Script generated: {script['title']} ({len(script['segments'])} segments)")
                return script

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt == 1:
                    raise ScriptWriterError(f"Gemini returned invalid JSON after 2 attempts: {e}")

            except ScriptValidationError as e:
                logger.error(f"Script validation error (attempt {attempt + 1}): {e}")
                if attempt == 1:
                    raise ScriptWriterError(f"Script validation failed after 2 attempts: {e}")

        raise ScriptWriterError("Script generation failed")

    def generate_youtube_metadata(
        self,
        script: dict,
        title_template: str,
        description_template: str,
        tags: list,
        date_str: str,
        run_id: str = "",
        pipeline_name: str = "",
    ) -> dict:
        """Generate YouTube title, description, and tags from the script. No LLM call needed."""
        top_story = script["segments"][0]["topic"] if script["segments"] else "Market Update"

        title = title_template.replace("{date}", date_str).replace("{top_story}", top_story)
        if len(title) > 100:
            title = title[:97] + "..."

        segment_list = "\n".join(
            f"- {s['topic']} ({s['duration_seconds']}s)" for s in script["segments"]
        )
        description = (
            description_template
            .replace("{date}", date_str)
            .replace("{segment_list}", segment_list)
        )

        return {"title": title, "description": description, "tags": tags}

    def _call_gemini(
        self, prompt: str, max_tokens: int, run_id: str, pipeline_name: str, attempt: int
    ) -> str:
        """Call Gemini via Vertex AI with retry for transient errors."""
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.2,      # Low temp for deterministic JSON
            top_p=0.95,
            response_mime_type="application/json",  # Ask Gemini to return JSON directly
        )

        last_error = None
        for retry in range(3):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )

                text = response.text
                usage = response.usage_metadata
                tokens_in = usage.prompt_token_count or 0
                tokens_out = usage.candidates_token_count or 0

                cost = (tokens_in * _COST_INPUT_PER_M / 1_000_000) + (tokens_out * _COST_OUTPUT_PER_M / 1_000_000)

                sqlite_logger.log_api_call(
                    run_id=run_id,
                    pipeline_name=pipeline_name,
                    service="gemini",
                    endpoint=f"generate_content (attempt {attempt + 1})",
                    tokens_used=tokens_in + tokens_out,
                    cost_usd=cost,
                    status="success",
                    retry_count=retry,
                )

                return text

            except Exception as e:
                err_str = str(e).lower()
                # Retry on transient / quota errors
                if any(x in err_str for x in ("429", "503", "quota", "resource exhausted", "unavailable")):
                    last_error = e
                    wait = 5 * (2 ** retry)
                    logger.warning(f"Gemini transient error (retry {retry + 1}/3): {e}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise ScriptWriterError(f"Gemini API error: {e}")

        raise ScriptWriterError(f"Gemini API failed after 3 retries: {last_error}")

    def _parse_and_validate(self, text: str) -> dict:
        """Parse JSON from Gemini response and validate schema."""
        cleaned = text.strip()
        # Strip markdown fences if present despite response_mime_type
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        script = json.loads(cleaned)
        _validate_script(script)
        return script

    def _make_stricter_prompt(self, original_prompt: str) -> str:
        return (
            "CRITICAL: You MUST return ONLY valid JSON. No markdown fences. "
            "No text before or after the JSON. No comments. "
            "Ensure all strings are properly escaped. "
            "Double-check all brackets and braces are balanced.\n\n"
            + original_prompt
        )


def _validate_script(script: dict):
    """Validate script JSON against expected schema."""
    missing = SCRIPT_SCHEMA_KEYS - set(script.keys())
    if missing:
        raise ScriptValidationError(f"Missing top-level keys: {missing}")

    if not isinstance(script["segments"], list):
        raise ScriptValidationError("'segments' must be a list")

    if len(script["segments"]) == 0:
        raise ScriptValidationError("'segments' must not be empty")

    for i, seg in enumerate(script["segments"]):
        seg_missing = SEGMENT_KEYS - set(seg.keys())
        if seg_missing:
            raise ScriptValidationError(f"Segment {i} missing keys: {seg_missing}")

        if "chart_spec" in seg and seg["chart_spec"]:
            cs = seg["chart_spec"]
            # Auto-fill: `annotations` is optional. If Gemini omits it,
            # default to {} and log a warning rather than failing the
            # whole script (which would trigger an expensive retry).
            # Only annotations is auto-filled — all other required keys
            # are still strictly enforced below.
            if "annotations" not in cs:
                logger.warning(
                    f"Segment {i} chart_spec missing 'annotations' — "
                    f"auto-filling with {{}} (ticker={cs.get('ticker', '?')})"
                )
                cs["annotations"] = {}
            cs_missing = CHART_SPEC_KEYS - set(cs.keys())
            if cs_missing:
                raise ScriptValidationError(f"Segment {i} chart_spec missing keys: {cs_missing}")

    if not isinstance(script.get("title", ""), str) or len(script["title"]) == 0:
        raise ScriptValidationError("'title' must be a non-empty string")

    if not isinstance(script.get("hook", ""), str) or len(script["hook"]) == 0:
        raise ScriptValidationError("'hook' must be a non-empty string")
