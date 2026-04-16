"""
Verification test for Fix B (--skip-upload) and Fix D (quota-exhaustion
tagging). Exercises pipeline._handle_upload_stage in isolation with the
YouTube uploader and sqlite_logger mocked out so the test never touches
the real API or database.

Covers four scenarios:
    1. skip_upload=True           → upload.run is never called
    2. quota-exhaustion exception → runs row tagged 'youtube_quota_exhausted'
    3. generic upload exception   → runs row tagged with truncated str(e)
    4. happy-path success         → runs row is NOT tagged

Run with:
    python3 tests/test_upload_skip.py
or:
    python3 -m unittest tests.test_upload_skip -v
"""
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the repo root importable so `import pipeline` works no matter where
# the test is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# pipeline.py transitively imports paid SDKs (vertexai, elevenlabs,
# google.oauth2, etc.). Stub them before importing pipeline so this test
# file works under `unittest discover` regardless of what's installed.
def _stub_module(name: str):
    mod = types.ModuleType(name)
    mod.__path__ = []

    class _D:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return _D()

        def __getattr__(self, _):
            return _D()

    for attr in (
        "GenerativeModel", "GenerationConfig", "Part",
        "Credentials", "Request", "InstalledAppFlow",
        "build", "MediaFileUpload", "HttpError",
        "ElevenLabs", "VoiceSettings",
        "safe_load", "safe_dump", "YAMLError",
        "ApiError",
    ):
        setattr(mod, attr, _D)
    sys.modules[name] = mod


for _name in (
    "vertexai",
    "vertexai.generative_models",
    "google", "google.auth", "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2", "google.oauth2.credentials",
    "google_auth_oauthlib", "google_auth_oauthlib.flow",
    "googleapiclient", "googleapiclient.discovery",
    "googleapiclient.http", "googleapiclient.errors",
    "elevenlabs", "elevenlabs.core", "elevenlabs.client",
    "yaml",
):
    # Don't overwrite modules that are actually installed.
    if _name not in sys.modules:
        _stub_module(_name)

import pipeline  # noqa: E402


class HandleUploadStageTests(unittest.TestCase):
    """Unit tests for pipeline._handle_upload_stage."""

    def setUp(self):
        self.config = {
            "pipeline_name": "best_stocks",
            "youtube": {"enabled": True},
        }
        self.script = {
            "segments": [{"topic": "Test Segment", "duration_seconds": 60}],
        }
        self.video_path = "/tmp/fake.mp4"
        self.run_id = "test_run_12345"
        # Silence log output during the test run. The production code still
        # emits messages — we're just not interested in seeing them here.
        self.logger = logging.getLogger("test_upload_skip")
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.CRITICAL)

    # ──────────────────────────────────────────────────────────────────
    # Fix B — manual --skip-upload path
    # ──────────────────────────────────────────────────────────────────

    @patch("pipeline.sqlite_logger")
    @patch("pipeline.upload")
    def test_skip_upload_flag_bypasses_youtube_entirely(self, mock_upload, mock_sql):
        """--skip-upload must NOT call upload.run() and must NOT tag the row."""
        result = pipeline._handle_upload_stage(
            self.config,
            self.script,
            self.video_path,
            self.run_id,
            skip_upload=True,
            logger=self.logger,
        )
        mock_upload.run.assert_not_called()
        mock_sql.update_run.assert_not_called()
        self.assertEqual(result, {"skipped": True, "reason": "manual_skip"})

    # ──────────────────────────────────────────────────────────────────
    # Fix D — quota-exhaustion detection
    # ──────────────────────────────────────────────────────────────────

    @patch("pipeline.sqlite_logger")
    @patch("pipeline.upload")
    def test_quota_exhaustion_tags_row(self, mock_upload, mock_sql):
        """
        When upload.run raises an exception containing 'quota', the runs row
        must be tagged error_stage='upload',
        error_message='youtube_quota_exhausted'.
        """
        mock_upload.run.side_effect = RuntimeError(
            "YouTube daily quota exceeded — upload skipped. MP4 saved locally."
        )
        result = pipeline._handle_upload_stage(
            self.config,
            self.script,
            self.video_path,
            self.run_id,
            skip_upload=False,
            logger=self.logger,
        )
        mock_upload.run.assert_called_once()
        mock_sql.update_run.assert_called_once_with(
            self.run_id,
            error_stage="upload",
            error_message="youtube_quota_exhausted",
        )
        self.assertEqual(result, {})

    @patch("pipeline.sqlite_logger")
    @patch("pipeline.upload")
    def test_quota_detection_is_case_insensitive(self, mock_upload, mock_sql):
        """'QUOTA' (upper/mixed case) should still be detected."""
        mock_upload.run.side_effect = RuntimeError("Daily QUOTA exceeded")
        pipeline._handle_upload_stage(
            self.config, self.script, self.video_path, self.run_id,
            skip_upload=False, logger=self.logger,
        )
        mock_sql.update_run.assert_called_once_with(
            self.run_id,
            error_stage="upload",
            error_message="youtube_quota_exhausted",
        )

    # ──────────────────────────────────────────────────────────────────
    # Fix D — generic upload-error tagging
    # ──────────────────────────────────────────────────────────────────

    @patch("pipeline.sqlite_logger")
    @patch("pipeline.upload")
    def test_generic_upload_error_tags_row_with_message(self, mock_upload, mock_sql):
        """
        Any non-quota exception should tag the row with the first 500 chars
        of the exception string, and still return {} so the pipeline proceeds.
        """
        mock_upload.run.side_effect = ConnectionError("socket hang up")
        result = pipeline._handle_upload_stage(
            self.config, self.script, self.video_path, self.run_id,
            skip_upload=False, logger=self.logger,
        )
        mock_sql.update_run.assert_called_once_with(
            self.run_id,
            error_stage="upload",
            error_message="socket hang up",
        )
        self.assertEqual(result, {})

    @patch("pipeline.sqlite_logger")
    @patch("pipeline.upload")
    def test_generic_upload_error_truncates_to_500_chars(self, mock_upload, mock_sql):
        """Very long exception messages must be truncated at 500 chars."""
        long_msg = "X" * 1000
        mock_upload.run.side_effect = RuntimeError(long_msg)
        pipeline._handle_upload_stage(
            self.config, self.script, self.video_path, self.run_id,
            skip_upload=False, logger=self.logger,
        )
        call_kwargs = mock_sql.update_run.call_args.kwargs
        self.assertEqual(call_kwargs["error_stage"], "upload")
        self.assertEqual(len(call_kwargs["error_message"]), 500)
        self.assertEqual(call_kwargs["error_message"], "X" * 500)

    # ──────────────────────────────────────────────────────────────────
    # Happy path — successful upload should leave the row untouched
    # ──────────────────────────────────────────────────────────────────

    @patch("pipeline.sqlite_logger")
    @patch("pipeline.upload")
    def test_successful_upload_does_not_tag_row(self, mock_upload, mock_sql):
        """A successful upload must NOT touch the error_stage/error_message columns."""
        mock_upload.run.return_value = {
            "video_id": "abc123",
            "video_url": "https://www.youtube.com/watch?v=abc123",
        }
        result = pipeline._handle_upload_stage(
            self.config, self.script, self.video_path, self.run_id,
            skip_upload=False, logger=self.logger,
        )
        mock_sql.update_run.assert_not_called()
        self.assertEqual(result["video_id"], "abc123")
        self.assertEqual(
            result["video_url"], "https://www.youtube.com/watch?v=abc123"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
