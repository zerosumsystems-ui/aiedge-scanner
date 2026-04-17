"""
Kling AI B-roll generator with retry, content rejection handling,
and fallback clip library.
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import httpx

from shared import sqlite_logger
from shared.config_loader import get_project_root

logger = logging.getLogger(__name__)

FALLBACK_DIR = get_project_root() / "assets" / "fallback_broll"
MIN_CLIP_SIZE_BYTES = 100_000  # 100KB — anything smaller is likely corrupt
KLING_API_BASE = "https://api.klingai.com/v1"


class KlingError(Exception):
    pass


class KlingClient:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("KLING_API_KEY")
        self.api_secret = api_secret or os.environ.get("KLING_API_SECRET")
        if not self.api_key:
            logger.warning("KLING_API_KEY not set — will use fallback clips only")

    def generate_clip(
        self,
        prompt: str,
        output_path: str,
        duration_seconds: int = 3,
        aspect_ratio: str = "16:9",
        use_fallback_on_failure: bool = True,
        clip_type: str = "intro",  # "intro" or "outro"
        run_id: str = "",
        pipeline_name: str = "",
    ) -> str:
        """
        Generate a B-roll clip using Kling AI.

        Falls back to pre-generated clips on failure if use_fallback_on_failure=True.

        Returns path to the video file.
        """
        if not self.api_key:
            logger.info("No Kling API key — using fallback clip")
            return self._get_fallback(clip_type, output_path)

        last_error = None
        for attempt in range(3):
            try:
                result = self._call_kling(prompt, duration_seconds, aspect_ratio)

                # Download the video
                video_url = result.get("video_url") or result.get("url")
                if not video_url:
                    raise KlingError(f"No video URL in response: {result}")

                self._download_video(video_url, output_path)

                # Validate output
                file_size = Path(output_path).stat().st_size
                if file_size < MIN_CLIP_SIZE_BYTES:
                    raise KlingError(f"Clip too small ({file_size} bytes), likely corrupt")

                # Estimate cost (~$0.10 per 3s clip)
                cost = 0.10 * (duration_seconds / 3)
                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="kling", endpoint=f"generate/{clip_type}",
                    cost_usd=cost, status="success", retry_count=attempt,
                )

                logger.info(f"Kling clip saved: {output_path} ({file_size / 1024:.0f}KB)")
                return output_path

            except KlingContentRejected as e:
                # Do NOT retry same prompt on content policy rejection
                logger.error(f"Kling content rejected: {e}. Prompt: {prompt[:100]}...")
                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="kling", endpoint=f"generate/{clip_type}",
                    status="content_rejected", error=str(e),
                )
                break

            except Exception as e:
                last_error = e
                wait = 5 * (3 ** attempt)  # 5s, 15s, 45s
                logger.warning(f"Kling error (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="kling", endpoint=f"generate/{clip_type}",
                    status="failed", error=str(e), retry_count=attempt,
                )
                if attempt < 2:
                    time.sleep(wait)

        # All retries exhausted — use fallback
        if use_fallback_on_failure:
            logger.warning(f"Kling failed after retries, using fallback {clip_type} clip")
            return self._get_fallback(clip_type, output_path)

        raise KlingError(f"Kling generation failed after 3 attempts: {last_error}")

    def _call_kling(self, prompt: str, duration: int, aspect_ratio: str) -> dict:
        """Submit generation request and poll for completion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Create task
        create_resp = httpx.post(
            f"{KLING_API_BASE}/videos/generations",
            headers=headers,
            json={
                "prompt": prompt,
                "duration": str(duration),
                "aspect_ratio": aspect_ratio,
                "mode": "std",
            },
            timeout=30,
        )

        if create_resp.status_code == 400:
            error_data = create_resp.json()
            if "content" in str(error_data).lower() or "policy" in str(error_data).lower():
                raise KlingContentRejected(f"Content policy: {error_data}")
            raise KlingError(f"Kling API error: {create_resp.status_code} {error_data}")

        create_resp.raise_for_status()
        task = create_resp.json()
        task_id = task.get("data", {}).get("task_id") or task.get("task_id")

        if not task_id:
            raise KlingError(f"No task_id in create response: {task}")

        # Poll for completion (max 5 minutes)
        for _ in range(60):
            time.sleep(5)
            status_resp = httpx.get(
                f"{KLING_API_BASE}/videos/generations/{task_id}",
                headers=headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            status = status_resp.json()
            task_status = status.get("data", {}).get("task_status") or status.get("status")

            if task_status == "completed" or task_status == "succeed":
                videos = status.get("data", {}).get("task_result", {}).get("videos", [])
                if videos:
                    return {"video_url": videos[0].get("url")}
                return status.get("data", {})

            if task_status in ("failed", "error"):
                raise KlingError(f"Kling task failed: {status}")

        raise KlingError("Kling task timed out after 5 minutes")

    def _download_video(self, url: str, output_path: str):
        """Download video file from URL."""
        resp = httpx.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)

    def _get_fallback(self, clip_type: str, output_path: str) -> str:
        """Pick a random fallback clip and copy it to output_path."""
        import shutil

        pattern = f"{clip_type}_*.mp4"
        candidates = list(FALLBACK_DIR.glob(pattern))

        if not candidates:
            logger.warning(f"No fallback {clip_type} clips found in {FALLBACK_DIR}")
            # Create a blank black clip as last resort
            return self._create_blank_clip(output_path)

        chosen = random.choice(candidates)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(chosen), output_path)
        logger.info(f"Using fallback clip: {chosen.name} → {output_path}")
        return output_path

    def _create_blank_clip(self, output_path: str, duration: int = 3) -> str:
        """Create a minimal black clip as absolute last resort."""
        import subprocess
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=black:s=1920x1080:d={duration}:r=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path,
            ],
            capture_output=True, check=True,
        )
        logger.warning(f"Created blank black clip: {output_path}")
        return output_path


class KlingContentRejected(KlingError):
    """Raised when Kling rejects content for policy reasons."""
    pass
