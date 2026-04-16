"""
Claude API script writer with strict JSON validation and retry.
"""

import json
import logging
import os
import time
from typing import Any, Optional

import anthropic

from shared import sqlite_logger

logger = logging.getLogger(__name__)

SCRIPT_SCHEMA_KEYS = {"title", "hook", "segments", "outro"}
SEGMENT_KEYS = {"topic", "narration", "chart_spec", "on_screen_text", "duration_seconds"}
# Required chart_spec keys. `annotations` is intentionally NOT in this set —
# it's auto-filled to {} by _validate_script if missing. Kept in sync with
# shared/gemini_writer.py so both writers behave the same way.
CHART_SPEC_KEYS = {"ticker", "timeframe", "lookback", "overlays"}


class ScriptWriterError(Exception):
    pass


class ScriptWriter:
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ScriptWriterError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate_script(
        self,
        screener_data: dict,
        prompt_template: str,
        run_id: str = "",
        pipeline_name: str = "",
        max_tokens: int = 4096,
    ) -> dict:
        """
        Generate a video script from screener data using Claude.

        Returns validated JSON matching the script schema.
        Retries once with a stricter prompt if validation fails.
        """
        prompt = prompt_template.replace("{screener_data}", json.dumps(screener_data, indent=2, default=str))

        for attempt in range(2):
            try:
                if attempt == 1:
                    prompt = self._make_stricter_prompt(prompt)
                    logger.warning("Retrying script generation with stricter prompt")

                response = self._call_claude(prompt, max_tokens, run_id, pipeline_name, attempt)
                script = self._parse_and_validate(response)
                logger.info(f"Script generated: {script['title']} ({len(script['segments'])} segments)")
                return script

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt == 1:
                    raise ScriptWriterError(f"Claude returned invalid JSON after 2 attempts: {e}")

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
        tags: list[str],
        date_str: str,
        run_id: str = "",
        pipeline_name: str = "",
    ) -> dict:
        """Generate YouTube title, description, and tags from the script."""
        top_story = script["segments"][0]["topic"] if script["segments"] else "Market Update"

        title = title_template.replace("{date}", date_str).replace("{top_story}", top_story)
        if len(title) > 100:
            title = title[:97] + "..."

        segment_list = "\n".join(
            f"- {s['topic']} ({s['duration_seconds']}s)" for s in script["segments"]
        )
        description = description_template.replace("{date}", date_str).replace("{segment_list}", segment_list)

        return {
            "title": title,
            "description": description,
            "tags": tags,
        }

    def _call_claude(
        self, prompt: str, max_tokens: int, run_id: str, pipeline_name: str, attempt: int
    ) -> str:
        """Call Claude API with retry for transient errors."""
        last_error = None
        for retry in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = response.content[0].text
                tokens_in = response.usage.input_tokens
                tokens_out = response.usage.output_tokens

                # Estimate cost (Sonnet 4.6 pricing)
                cost = (tokens_in * 3.0 / 1_000_000) + (tokens_out * 15.0 / 1_000_000)

                sqlite_logger.log_api_call(
                    run_id=run_id,
                    pipeline_name=pipeline_name,
                    service="claude",
                    endpoint=f"messages.create (attempt {attempt + 1})",
                    tokens_used=tokens_in + tokens_out,
                    cost_usd=cost,
                    status="success",
                    retry_count=retry,
                )

                return text

            except anthropic.RateLimitError as e:
                last_error = e
                wait = 5 * (2 ** retry)
                logger.warning(f"Claude rate limit (retry {retry + 1}/3): waiting {wait}s")
                time.sleep(wait)

            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    wait = 5 * (2 ** retry)
                    logger.warning(f"Claude server error {e.status_code} (retry {retry + 1}/3): waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise ScriptWriterError(f"Claude API error: {e}")

            except Exception as e:
                raise ScriptWriterError(f"Claude API error: {e}")

        raise ScriptWriterError(f"Claude API failed after 3 retries: {last_error}")

    def _parse_and_validate(self, text: str) -> dict:
        """Parse JSON from Claude response and validate schema."""
        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
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


class ScriptValidationError(Exception):
    pass


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
            # Auto-fill: `annotations` is optional. If the writer omits
            # it, default to {} and log a warning rather than failing.
            # Only annotations is auto-filled — other required keys stay
            # strict. Kept in sync with shared/gemini_writer.py.
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
