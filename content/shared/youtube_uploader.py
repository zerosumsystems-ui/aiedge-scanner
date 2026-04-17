"""
YouTube Data API v3 uploader with OAuth, quota awareness, and retry.
Single channel for all pipelines — one token file.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from shared import sqlite_logger
from shared.config_loader import get_project_root

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
UPLOAD_QUOTA_COST = 1600  # units per upload
DAILY_QUOTA_LIMIT = 10000


class UploaderError(Exception):
    pass


class YouTubeUploader:
    def __init__(
        self,
        client_secret_path: str = None,
        token_path: str = None,
    ):
        root = get_project_root()
        self.client_secret_path = client_secret_path or str(root / "credentials" / "client_secret.json")
        self.token_path = token_path or str(root / "credentials" / "youtube_token.json")
        self._service = None

    def authenticate(self) -> None:
        """
        Authenticate with YouTube API using OAuth.
        First time: opens browser for consent. After: uses refresh token.
        """
        creds = None

        if Path(self.token_path).exists():
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}. Re-authenticating...")
                    creds = None

            if not creds:
                if not Path(self.client_secret_path).exists():
                    raise UploaderError(
                        f"client_secret.json not found at {self.client_secret_path}. "
                        "Download it from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_path, SCOPES)
                creds = flow.run_local_server(port=8080)

            # Save token for future runs
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())
            logger.info(f"YouTube token saved to {self.token_path}")

        self._service = build("youtube", "v3", credentials=creds)
        logger.info("YouTube API authenticated")

    def check_quota(self, date: str = None) -> bool:
        """Check if we have enough quota for an upload."""
        used = sqlite_logger.get_daily_youtube_quota(date)
        remaining = DAILY_QUOTA_LIMIT - used
        if remaining < UPLOAD_QUOTA_COST:
            logger.error(
                f"YouTube quota insufficient: {used} used, {remaining} remaining, "
                f"need {UPLOAD_QUOTA_COST} for upload"
            )
            return False
        logger.info(f"YouTube quota OK: {used} used, {remaining} remaining")
        return True

    def upload(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] = None,
        visibility: str = "unlisted",
        run_id: str = "",
        pipeline_name: str = "",
    ) -> dict:
        """
        Upload a video to YouTube.

        Returns: {video_id, video_url}
        """
        if not self._service:
            self.authenticate()

        if not Path(video_path).exists():
            raise UploaderError(f"Video file not found: {video_path}")

        if not self.check_quota():
            raise UploaderError("YouTube daily quota exceeded — upload skipped. MP4 saved locally.")

        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": (tags or [])[:500],
                "categoryId": "22",  # People & Blogs (or 27 for Education)
            },
            "status": {
                "privacyStatus": visibility,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True,
            chunksize=256 * 1024,  # 256KB chunks
        )

        last_error = None
        for attempt in range(3):
            try:
                request = self._service.videos().insert(
                    part="snippet,status",
                    body=body,
                    media_body=media,
                )

                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        logger.info(f"Upload progress: {int(status.progress() * 100)}%")

                video_id = response["id"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # Track quota usage
                from datetime import datetime
                today = datetime.now().strftime("%Y-%m-%d")
                sqlite_logger.update_daily_cost(
                    today, pipeline_name, youtube_quota_used=UPLOAD_QUOTA_COST
                )

                sqlite_logger.log_api_call(
                    run_id=run_id, pipeline_name=pipeline_name,
                    service="youtube", endpoint="videos.insert",
                    status="success", retry_count=attempt,
                )

                logger.info(f"Uploaded to YouTube: {video_url} (visibility: {visibility})")
                return {"video_id": video_id, "video_url": video_url}

            except HttpError as e:
                last_error = e
                if e.resp.status in (500, 502, 503):
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"YouTube server error (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                elif e.resp.status == 403:
                    # Quota exceeded or forbidden
                    logger.error(f"YouTube API 403: {e}")
                    raise UploaderError(f"YouTube API forbidden (likely quota): {e}")
                else:
                    raise UploaderError(f"YouTube upload error: {e}")

            except Exception as e:
                last_error = e
                wait = 5 * (2 ** attempt)
                logger.warning(f"YouTube upload error (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        # All retries failed — try uploading as private as last resort
        if visibility != "private":
            logger.warning("Retrying upload as private (fallback)")
            try:
                body["status"]["privacyStatus"] = "private"
                request = self._service.videos().insert(
                    part="snippet,status", body=body, media_body=media,
                )
                response = None
                while response is None:
                    status, response = request.next_chunk()

                video_id = response["id"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                logger.warning(f"Uploaded as PRIVATE (fallback): {video_url}")
                return {"video_id": video_id, "video_url": video_url}
            except Exception:
                pass

        raise UploaderError(f"YouTube upload failed after 3 attempts: {last_error}")
