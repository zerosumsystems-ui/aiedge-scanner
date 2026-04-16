"""
Newsletter publisher via Beehiiv API.
Reformats pipeline run artifacts (screener, script, charts) into email content.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

from shared import sqlite_logger

logger = logging.getLogger(__name__)

BEEHIIV_API_BASE = "https://api.beehiiv.com/v2"


class NewsletterError(Exception):
    pass


class NewsletterPublisher:
    def __init__(self, api_key: str = None, publication_id: str = None):
        self.api_key = api_key or os.environ.get("BEEHIIV_API_KEY")
        self.publication_id = publication_id or os.environ.get("BEEHIIV_PUBLICATION_ID")
        if not self.api_key:
            logger.warning("BEEHIIV_API_KEY not set — newsletter disabled")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.publication_id)

    def publish(
        self,
        script: dict,
        chart_paths: list[str],
        config: dict,
        youtube_url: str = "",
        run_id: str = "",
        pipeline_name: str = "",
    ) -> Optional[dict]:
        """
        Format and publish (or draft) a newsletter post.

        Returns: {post_id, post_url, status} or None if disabled.
        """
        if not self.enabled:
            logger.info("Newsletter publisher not configured — skipping")
            return None

        nl_config = config.get("newsletter", {})
        if not nl_config.get("enabled", False):
            logger.info("Newsletter disabled in config — skipping")
            return None

        try:
            # Build the HTML body
            body_html = self._build_body(script, chart_paths, nl_config, youtube_url)
            subject = nl_config.get("subject_template", "{title}").replace("{title}", script["title"])
            preview = nl_config.get("preview_template", "{hook}").replace("{hook}", script["hook"])

            # Upload chart images to Beehiiv
            chart_urls = self._upload_charts(chart_paths)

            # Replace local paths with hosted URLs in body
            for local_path, hosted_url in chart_urls.items():
                body_html = body_html.replace(local_path, hosted_url)

            # Create the post
            send_as = nl_config.get("send_as", "draft")
            result = self._create_post(subject, preview, body_html, send_as)

            sqlite_logger.log_api_call(
                run_id=run_id, pipeline_name=pipeline_name,
                service="beehiiv", endpoint="posts.create",
                status="success",
            )

            logger.info(f"Newsletter {'drafted' if send_as == 'draft' else 'published'}: {result.get('post_url', 'N/A')}")
            return result

        except Exception as e:
            logger.error(f"Newsletter publishing failed: {e}")
            sqlite_logger.log_api_call(
                run_id=run_id, pipeline_name=pipeline_name,
                service="beehiiv", endpoint="posts.create",
                status="failed", error=str(e),
            )
            return None  # Don't block the pipeline

    def _build_body(
        self, script: dict, chart_paths: list[str], nl_config: dict, youtube_url: str
    ) -> str:
        """Build HTML email body from script and config templates."""
        segment_format = nl_config.get("segment_format", "## {topic}\n\n{narration}\n")
        paid_tier_url = nl_config.get("paid_tier_url", "#")

        segments_md = []
        for i, seg in enumerate(script["segments"]):
            chart_url = chart_paths[i] if i < len(chart_paths) else ""
            annotations = seg.get("chart_spec", {}).get("annotations", {})
            key_levels = annotations.get("key_levels", [])
            key_levels_text = " · ".join(f"{lvl:,.2f}" for lvl in key_levels) if key_levels else "N/A"

            entry = annotations.get("entry")
            stop = annotations.get("stop")
            target = annotations.get("target")

            levels_parts = []
            if entry:
                levels_parts.append(f"Entry {entry:,.2f}")
            if stop:
                levels_parts.append(f"Stop {stop:,.2f}")
            if target:
                levels_parts.append(f"Target {target:,.2f}")

            seg_text = segment_format.replace(
                "{topic}", seg.get("topic", "")
            ).replace(
                "{ticker}", seg.get("chart_spec", {}).get("ticker", "")
            ).replace(
                "{chart_url}", chart_url
            ).replace(
                "{narration}", seg.get("narration", "")
            ).replace(
                "{key_levels_text}", " · ".join(levels_parts) if levels_parts else key_levels_text
            )

            # Replace individual level fields if present in template
            seg_text = seg_text.replace("{entry}", f"{entry:,.2f}" if entry else "N/A")
            seg_text = seg_text.replace("{stop}", f"{stop:,.2f}" if stop else "N/A")
            seg_text = seg_text.replace("{target}", f"{target:,.2f}" if target else "N/A")

            segments_md.append(seg_text)

        segments_as_markdown = "\n\n---\n\n".join(segments_md)

        body_template = nl_config.get("body_template", "# {title}\n\n{segments_as_markdown}")
        body = body_template.replace(
            "{title}", script["title"]
        ).replace(
            "{hook}", script["hook"]
        ).replace(
            "{segments_as_markdown}", segments_as_markdown
        ).replace(
            "{youtube_url}", youtube_url or "#"
        ).replace(
            "{paid_tier_url}", paid_tier_url
        )

        # Convert markdown to basic HTML
        return self._md_to_html(body)

    def _md_to_html(self, md: str) -> str:
        """Simple markdown to HTML conversion for email."""
        import re

        html = md

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Images
        html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%;border-radius:8px;" />', html)

        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

        # Horizontal rules
        html = re.sub(r'^---+$', '<hr style="border:1px solid #333;" />', html, flags=re.MULTILINE)

        # Paragraphs
        paragraphs = html.split("\n\n")
        html = "".join(
            f"<p>{p.strip()}</p>" if not p.strip().startswith("<") else p
            for p in paragraphs
            if p.strip()
        )

        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 600px; margin: 0 auto; color: #e0e0e0; background: #0a0a0a; padding: 24px;">
            {html}
        </div>
        """

    def _upload_charts(self, chart_paths: list[str]) -> dict:
        """
        Upload chart images and return {local_path: hosted_url} mapping.
        Uses Beehiiv's content API if available, otherwise returns local paths.
        """
        url_map = {}
        for path in chart_paths:
            if not Path(path).exists():
                continue
            # Beehiiv doesn't have a public image upload API on free tier.
            # For now, we embed as base64 data URIs which works in most email clients.
            try:
                import base64
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                url_map[path] = f"data:image/png;base64,{data}"
            except Exception as e:
                logger.warning(f"Failed to encode chart {path}: {e}")
                url_map[path] = path

        return url_map

    def _create_post(self, subject: str, preview: str, html_body: str, send_as: str) -> dict:
        """Create a post via Beehiiv API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        status_map = {
            "draft": "draft",
            "scheduled": "confirmed",
            "immediate": "confirmed",
        }

        payload = {
            "title": subject,
            "subtitle": preview,
            "content": [{"type": "html", "html": html_body}],
            "status": status_map.get(send_as, "draft"),
        }

        if send_as == "immediate":
            payload["send_at"] = "now"

        resp = requests.post(
            f"{BEEHIIV_API_BASE}/publications/{self.publication_id}/posts",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if resp.status_code not in (200, 201):
            raise NewsletterError(f"Beehiiv API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json().get("data", resp.json())
        return {
            "post_id": data.get("id", ""),
            "post_url": data.get("web_url", ""),
            "status": data.get("status", send_as),
        }
