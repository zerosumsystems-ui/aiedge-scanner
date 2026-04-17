"""
Google Veo B-roll generator using google-genai SDK.
Drop-in replacement for kling_client.py — exposes identical interface.

Veo generation requires:
  1. ADC credentials (gcloud auth application-default login)
  2. A GCS bucket for output (set VEO_GCS_BUCKET env var, or auto-created)

Falls back to pre-generated static clips if Veo is unavailable.
"""

import logging
import os
import random
import shutil
import time
from pathlib import Path

from shared import sqlite_logger
from shared.config_loader import get_project_root

logger = logging.getLogger(__name__)

FALLBACK_DIR = get_project_root() / "assets" / "fallback_broll"
MIN_CLIP_SIZE_BYTES = 100_000  # 100KB
DEFAULT_MODEL = "veo-3.1-lite-generate-preview"
DEFAULT_PROJECT = "gen-lang-client-0987146855"
DEFAULT_LOCATION = "us-central1"

# Models tried in order when primary fails with 404/not-found
FALLBACK_MODELS = [
    "veo-3.1-lite-generate-preview",
    "veo-3.1-fast-generate-001",
    "veo-3.1-generate-001",
]


class VeoError(Exception):
    pass


class VeoContentRejected(VeoError):
    """Raised when Veo rejects content for policy reasons."""
    pass


class VeoClient:
    def __init__(self, config_or_project=None, location: str = None, model: str = None):
        # Accept either a full pipeline config dict or explicit project/location/model kwargs
        if isinstance(config_or_project, dict):
            veo_cfg = config_or_project.get("veo", {})
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
            location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
            model = model or veo_cfg.get("model", DEFAULT_MODEL)
        else:
            project = config_or_project or os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
            location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
            model = model or DEFAULT_MODEL

        self.project = project
        self.location = location
        self.model = model
        self._client = None
        self._gcs_bucket = os.environ.get("VEO_GCS_BUCKET", "")

    def _get_client(self):
        """Lazy-init google-genai Vertex AI client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
            except Exception as e:
                raise VeoError(f"Failed to initialize google-genai client: {e}")
        return self._client

    def _ensure_gcs_bucket(self) -> str:
        """Return a GCS bucket URI, creating the bucket if needed."""
        if self._gcs_bucket:
            return self._gcs_bucket

        bucket_name = f"{self.project}-veo-output"
        try:
            from google.cloud import storage
            gcs = storage.Client(project=self.project)
            try:
                gcs.get_bucket(bucket_name)
                logger.info(f"Using existing GCS bucket: {bucket_name}")
            except Exception:
                logger.info(f"Creating GCS bucket: {bucket_name}")
                bucket = gcs.bucket(bucket_name)
                bucket.storage_class = "STANDARD"
                gcs.create_bucket(bucket, location=self.location)
            self._gcs_bucket = f"gs://{bucket_name}"
        except Exception as e:
            raise VeoError(
                f"GCS bucket setup failed: {e}. "
                f"Set VEO_GCS_BUCKET env var to an existing gs://bucket-name, "
                f"or grant Storage Admin role to your credentials."
            )
        return self._gcs_bucket

    def generate_clip(
        self,
        prompt: str,
        output_path: str,
        duration_seconds: int = 6,
        aspect_ratio: str = "16:9",
        use_fallback_on_failure: bool = True,
        clip_type: str = "intro",
        run_id: str = "",
        pipeline_name: str = "",
    ) -> str:
        """
        Generate a B-roll clip using Google Veo via google-genai SDK.
        Tries FALLBACK_MODELS in order if the primary model returns 404.
        Falls back to pre-generated static clips if all models fail.
        Returns path to the video file.
        """
        # Build model list: configured model first, then remaining fallbacks
        models_to_try = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        last_error = None

        for model in models_to_try:
            logger.info(f"Trying Veo model: {model}")
            succeeded, last_error, content_rejected = self._try_model(
                model, prompt, output_path, duration_seconds, aspect_ratio,
                clip_type, run_id, pipeline_name,
            )
            if succeeded:
                return output_path
            if content_rejected:
                break  # content policy — other models won't help

        if use_fallback_on_failure:
            logger.warning(f"All Veo models failed, using fallback {clip_type} clip")
            return self._get_fallback(clip_type, output_path)

        raise VeoError(f"Veo generation failed across all models: {last_error}")

    def _try_model(
        self, model: str, prompt: str, output_path: str,
        duration: int, aspect_ratio: str,
        clip_type: str, run_id: str, pipeline_name: str,
    ) -> tuple:
        """
        Attempt generation with a specific model, with up to 3 retries on transient errors.
        Returns (succeeded: bool, last_error: Exception|None, content_rejected: bool).
        """
        last_error = None
        for attempt in range(3):
            try:
                self._call_veo(prompt, output_path, duration, aspect_ratio, model=model)

                file_size = Path(output_path).stat().st_size
                if file_size < MIN_CLIP_SIZE_BYTES:
                    raise VeoError(f"Clip too small ({file_size} bytes), likely corrupt")

                cost = 0.05 * duration
                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="veo", endpoint=f"generate/{clip_type}",
                    cost_usd=cost, status="success", retry_count=attempt,
                )
                logger.info(f"Veo clip saved: {output_path} ({Path(output_path).stat().st_size / 1024:.0f}KB)")
                return True, None, False

            except VeoContentRejected as e:
                logger.error(f"Veo content rejected (model={model}): {e}. Prompt: {prompt[:100]}...")
                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="veo", endpoint=f"generate/{clip_type}",
                    status="content_rejected", error=str(e),
                )
                return False, e, True

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="veo", endpoint=f"generate/{clip_type}",
                    status="failed", error=str(e), retry_count=attempt,
                )
                # 404 = model not available on this project; skip to next model
                if "404" in err_str or "not_found" in err_str:
                    logger.warning(f"Veo model {model} not available (404), trying next model")
                    return False, e, False
                wait = 5 * (3 ** attempt)
                logger.warning(f"Veo error model={model} (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                if attempt < 2:
                    time.sleep(wait)

        return False, last_error, False

    def _call_veo(self, prompt: str, output_path: str, duration: int, aspect_ratio: str, model: str = None):
        """Submit generation request via google-genai SDK and save locally."""
        from google import genai
        from google.genai import types

        client = self._get_client()
        gcs_uri = self._ensure_gcs_bucket()
        active_model = model or self.model

        try:
            operation = client.models.generate_videos(
                model=active_model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    duration_seconds=duration,
                    aspect_ratio=aspect_ratio,
                    number_of_videos=1,
                    output_gcs_uri=gcs_uri,
                    generate_audio=False,
                ),
            )
        except Exception as e:
            err_str = str(e).lower()
            if "policy" in err_str or "safety" in err_str or "content" in err_str:
                raise VeoContentRejected(str(e))
            raise VeoError(f"generate_videos failed: {e}")

        # Poll for completion (Veo is async, typically 30-120s)
        logger.info(f"Veo generation started (model={active_model}), polling...")
        max_polls = 30
        for i in range(max_polls):
            time.sleep(10)
            try:
                operation = client.operations.get(operation)
            except Exception as e:
                raise VeoError(f"Failed to poll Veo operation: {e}")
            if operation.done:
                break
            logger.debug(f"Veo poll {i + 1}/{max_polls}...")
        else:
            raise VeoError("Veo generation timed out after 5 minutes")

        result = operation.result
        if not result or not result.generated_videos:
            raise VeoError("Veo returned no generated videos")

        video = result.generated_videos[0].video
        if not video:
            raise VeoError("Veo generated_videos[0].video is None")

        # Download from GCS to local path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if video.video_bytes:
            # Direct bytes (unlikely for Vertex AI but handle it)
            Path(output_path).write_bytes(video.video_bytes)
        elif video.uri:
            self._download_from_gcs(video.uri, output_path)
        else:
            raise VeoError("Veo video has neither bytes nor URI")

    def _download_from_gcs(self, gcs_uri: str, local_path: str):
        """Download a file from gs://bucket/path to a local path."""
        try:
            from google.cloud import storage
            # Parse gs://bucket/blob
            without_prefix = gcs_uri[len("gs://"):]
            bucket_name, blob_path = without_prefix.split("/", 1)
            gcs = storage.Client(project=self.project)
            bucket = gcs.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded Veo clip from GCS: {gcs_uri} → {local_path}")
        except Exception as e:
            raise VeoError(f"Failed to download from GCS {gcs_uri}: {e}")

    def _get_fallback(self, clip_type: str, output_path: str) -> str:
        """Pick a random fallback clip and copy it to output_path."""
        pattern = f"{clip_type}_*.mp4"
        candidates = list(FALLBACK_DIR.glob(pattern))
        if not candidates:
            logger.warning(f"No fallback {clip_type} clips found in {FALLBACK_DIR}")
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
