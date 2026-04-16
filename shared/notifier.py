"""
Notification system: Discord webhooks + email for failures and daily digest.
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Optional

import requests

from shared import sqlite_logger

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config: dict = None):
        self.config = config or {}
        notifications = self.config.get("notifications", {})
        self.discord_url = os.environ.get(
            notifications.get("discord_webhook_url_env", "DISCORD_WEBHOOK_URL"), ""
        )
        if not self.discord_url:
            self.discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

        self.notify_failure = notifications.get("notify_on_failure", True)
        self.notify_success = notifications.get("notify_on_success", False)
        self.daily_digest = notifications.get("daily_digest", True)

    def notify_run_failure(self, pipeline_name: str, run_id: str, error_message: str, error_stage: str):
        """Send failure notification."""
        if not self.notify_failure:
            return

        msg = (
            f"**PIPELINE FAILURE**\n"
            f"Pipeline: `{pipeline_name}`\n"
            f"Run ID: `{run_id}`\n"
            f"Stage: `{error_stage}`\n"
            f"Error: ```{error_message[:500]}```\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}"
        )
        self._send_discord(msg)
        self._send_email(
            subject=f"[VIDEO PIPELINE] {pipeline_name} FAILED at {error_stage}",
            body=msg.replace("**", "").replace("`", "").replace("```", ""),
        )

    def notify_run_success(self, pipeline_name: str, run_id: str, youtube_url: str = ""):
        """Send success notification (off by default)."""
        if not self.notify_success:
            return

        msg = (
            f"Pipeline: `{pipeline_name}` completed\n"
            f"Run ID: `{run_id}`\n"
        )
        if youtube_url:
            msg += f"YouTube: {youtube_url}\n"
        msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}"

        self._send_discord(msg)

    def send_daily_digest(self):
        """Send daily summary of all pipeline runs."""
        if not self.daily_digest:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        runs = sqlite_logger.get_runs_for_date(today)

        if not runs:
            msg = f"**Daily Digest — {today}**\nNo pipeline runs today."
        else:
            lines = [f"**Daily Digest — {today}**\n"]
            success_count = sum(1 for r in runs if r["status"] == "success")
            fail_count = sum(1 for r in runs if r["status"] == "failed")
            lines.append(f"Total: {len(runs)} runs | {success_count} success | {fail_count} failed\n")

            for r in runs:
                status_icon = "+" if r["status"] == "success" else "x"
                yt = r.get("youtube_url", "")
                cost = r.get("estimated_cost_usd", 0) or 0
                lines.append(
                    f"[{status_icon}] `{r['pipeline_name']}` — "
                    f"${cost:.2f} — {yt or 'no upload'}"
                )
                if r.get("error_message"):
                    lines.append(f"    Error: {r['error_message'][:100]}")

            # Total cost
            total_cost = sum((r.get("estimated_cost_usd") or 0) for r in runs)
            lines.append(f"\nTotal estimated cost: **${total_cost:.2f}**")

            msg = "\n".join(lines)

        self._send_discord(msg)
        self._send_email(subject=f"Video Pipeline Daily Digest — {today}", body=msg.replace("**", "").replace("`", ""))

    def _send_discord(self, message: str):
        """Send message via Discord webhook."""
        if not self.discord_url:
            logger.debug("No Discord webhook configured — skipping notification")
            return

        try:
            resp = requests.post(
                self.discord_url,
                json={"content": message[:2000]},
                timeout=10,
            )
            if resp.status_code not in (200, 204):
                logger.warning(f"Discord webhook failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")

    def _send_email(self, subject: str, body: str):
        """Send email notification via SMTP."""
        smtp_host = os.environ.get("SMTP_HOST")
        smtp_user = os.environ.get("SMTP_USER")
        smtp_pass = os.environ.get("SMTP_PASSWORD")
        email_to = os.environ.get("NOTIFICATION_EMAIL_TO")

        if not all([smtp_host, smtp_user, smtp_pass, email_to]):
            return

        try:
            port = int(os.environ.get("SMTP_PORT", "587"))
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = email_to

            with smtplib.SMTP(smtp_host, port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")
