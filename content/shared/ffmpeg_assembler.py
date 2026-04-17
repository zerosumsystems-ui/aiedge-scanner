"""
ffmpeg video assembler with Ken Burns effect, captions, overlays, and validation.
Produces the final MP4 from charts + narration + B-roll.

Continuous mode (preferred): builds a silent video timeline timed to the split
narration audio durations, then overlays _full_narration.mp3 as a single audio
track. This eliminates -shortest / filter_complex timing issues entirely.

Per-segment mode (fallback): assigns each audio file to its video segment
individually. Used when _full_narration.mp3 is not present.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AssemblerError(Exception):
    pass


class FFmpegAssembler:
    def __init__(self, config: dict = None):
        self.config = config or {}
        assembly = self.config.get("assembly", {})
        self.resolution = assembly.get("resolution", "1920x1080")
        self.fps = assembly.get("fps", 30)
        self.ken_burns = assembly.get("ken_burns", {"enabled": True, "zoom_factor": 1.15, "direction": "auto"})
        self.captions = assembly.get("captions", {"enabled": True, "font": "Helvetica Bold", "size": 48, "position": "bottom"})
        branding = assembly.get("branding", {})
        self.bg_color = branding.get("background", "#0a0a0a")
        self.primary_color = branding.get("color_primary", "#00ff88")
        self.text_color = branding.get("color_secondary", "#ffffff")

        w, h = self.resolution.split("x")
        self.width = int(w)
        self.height = int(h)

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise AssemblerError("ffmpeg not found. Install with: brew install ffmpeg")

    def check_disk_space(self, path: str, min_gb: float = 5.0) -> bool:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < min_gb:
            logger.error(f"Insufficient disk space: {free_gb:.1f}GB free, need {min_gb}GB")
            return False
        return True

    def assemble(
        self,
        script: dict,
        chart_paths: list[str],
        audio_paths: list[dict],
        broll_intro: str = None,
        broll_outro: str = None,
        output_path: str = "final.mp4",
        run_dir: str = "/tmp",
    ) -> str:
        """
        Assemble final video from components.

        Prefers continuous mode: overlays _full_narration.mp3 on a silent video
        timeline timed to the split segment durations. Falls back to per-segment
        audio assignment if _full_narration.mp3 is not found.
        """
        if not self.check_disk_space(run_dir):
            raise AssemblerError("Insufficient disk space for assembly")

        work_dir = Path(run_dir) / "assembly_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        audio_map = {a["label"]: a["audio_path"] for a in audio_paths}

        # Detect full narration file
        hook_audio = audio_map.get("hook")
        full_narration_path = None
        if hook_audio:
            candidate = Path(hook_audio).parent / "_full_narration.mp3"
            if candidate.exists():
                full_narration_path = str(candidate)

        if full_narration_path:
            logger.info("Using continuous narration mode (full audio overlay)")
            return self._assemble_continuous(
                script=script,
                chart_paths=chart_paths,
                audio_map=audio_map,
                broll_intro=broll_intro,
                broll_outro=broll_outro,
                full_narration_path=full_narration_path,
                output_path=output_path,
                work_dir=work_dir,
            )

        logger.info("Using per-segment narration mode")
        return self._assemble_per_segment(
            script=script,
            chart_paths=chart_paths,
            audio_map=audio_map,
            broll_intro=broll_intro,
            broll_outro=broll_outro,
            output_path=output_path,
            work_dir=work_dir,
        )

    # ── Continuous mode ───────────────────────────────────────────────────────

    def _assemble_continuous(
        self,
        script: dict,
        chart_paths: list[str],
        audio_map: dict,
        broll_intro: str,
        broll_outro: str,
        full_narration_path: str,
        output_path: str,
        work_dir: Path,
    ) -> str:
        """
        Build silent video timeline timed to split audio durations, then
        overlay the full narration as a single audio track.
        """
        parts = []

        # Intro: broll looped/trimmed to hook audio duration
        hook_audio = audio_map.get("hook")
        hook_dur = self._get_duration(hook_audio) if hook_audio else 0
        if broll_intro and Path(broll_intro).exists() and hook_dur > 0:
            intro_path = work_dir / "part_000_intro.mp4"
            self._make_broll_silent(broll_intro, hook_dur, str(intro_path))
            parts.append(str(intro_path))
            logger.info(f"Intro part: {hook_dur:.2f}s (looped broll)")

        # Chart segments: each timed to its split audio duration
        for i, segment in enumerate(script["segments"]):
            if i >= len(chart_paths):
                logger.warning(f"No chart for segment {i}, skipping")
                continue
            chart = chart_paths[i]
            if not chart or not Path(chart).exists():
                logger.warning(f"Segment {i}: chart missing, skipping")
                continue

            seg_audio = audio_map.get(f"segment_{i}")
            seg_dur = self._get_duration(seg_audio) if seg_audio else 0
            if seg_dur <= 0:
                seg_dur = float(segment.get("duration_seconds", 60))
                logger.warning(f"Segment {i}: audio duration unavailable, using script value {seg_dur}s")

            on_screen = segment.get("on_screen_text", "")
            seg_path = work_dir / f"part_{i + 1:03d}_seg.mp4"
            self._make_chart_segment_silent(
                chart_path=chart,
                on_screen_text=on_screen,
                duration=seg_dur,
                output_path=str(seg_path),
            )
            parts.append(str(seg_path))
            logger.info(f"Segment {i} part: {seg_dur:.2f}s")

        # Outro: broll looped/trimmed to outro audio duration
        outro_audio = audio_map.get("outro")
        outro_dur = self._get_duration(outro_audio) if outro_audio else 0
        if broll_outro and Path(broll_outro).exists() and outro_dur > 0:
            outro_path = work_dir / "part_999_outro.mp4"
            self._make_broll_silent(broll_outro, outro_dur, str(outro_path))
            parts.append(str(outro_path))
            logger.info(f"Outro part: {outro_dur:.2f}s (looped broll)")

        if not parts:
            raise AssemblerError("No video parts to assemble")

        # Concat all silent video parts
        silent_path = str(work_dir / "_silent_video.mp4")
        self._concat(parts, silent_path, work_dir, has_audio=False)

        # Verify silent video duration matches narration
        silent_dur = self._get_duration(silent_path)
        narration_dur = self._get_duration(full_narration_path)
        logger.info(f"Silent video: {silent_dur:.2f}s, narration: {narration_dur:.2f}s")

        # Overlay full narration as single audio track.
        # Use explicit -t narration_dur (not -shortest) to guarantee precise trim
        # regardless of video copy mode keyframe alignment.
        cmd = [
            "ffmpeg", "-y",
            "-i", silent_path,
            "-i", full_narration_path,
            "-map", "0:v",
            "-map", "1:a",
            "-t", str(narration_dur),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg audio overlay error: {result.stderr[-500:]}")
            raise AssemblerError(f"ffmpeg audio overlay failed: {result.stderr[-200:]}")

        self._validate_output(output_path, script)

        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

        logger.info(f"Final video assembled: {output_path}")
        return output_path

    def _make_broll_silent(self, video_path: str, target_duration: float, output_path: str):
        """
        Loop broll clip to target_duration seconds with no audio.
        Uses -stream_loop -1 so the clip seamlessly repeats to fill any duration.
        """
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_path,
            "-t", str(target_duration),
            "-vf", (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:"
                f"color={self.bg_color.replace('#', '0x')}"
            ),
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg broll silent error: {result.stderr[-500:]}")
            raise AssemblerError(f"ffmpeg broll silent failed: {result.stderr[-200:]}")

    def _make_chart_segment_silent(
        self,
        chart_path: str,
        on_screen_text: str,
        duration: float,
        output_path: str,
    ):
        """
        Create a video-only segment from a chart image with Ken Burns effect
        and optional caption overlay. No audio stream.
        """
        zoom = self.ken_burns.get("zoom_factor", 1.15) if self.ken_burns.get("enabled") else 1.0
        total_frames = int(duration * self.fps)

        zoompan = (
            f"zoompan=z='min(max(zoom,pzoom)+{(zoom - 1) / total_frames:.8f},{zoom})':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={self.width}x{self.height}:fps={self.fps}"
        )

        has_text = bool(on_screen_text and self.captions.get("enabled", True))
        overlay_png = None
        if has_text:
            overlay_png = str(Path(output_path).with_suffix(".overlay.png"))
            self._render_text_overlay(on_screen_text, overlay_png)

        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", chart_path]
        if overlay_png:
            cmd.extend(["-loop", "1", "-i", overlay_png])

        if overlay_png:
            filter_complex = f"[0:v]{zoompan}[zoomed];[zoomed][1:v]overlay=0:0[v]"
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[v]",
                # -frames:v is more reliable than -t for video-only output with
                # filter_complex + looped inputs (avoids -t being misapplied)
                "-frames:v", str(total_frames),
            ])
        else:
            cmd.extend([
                "-vf", zoompan,
                "-frames:v", str(total_frames),
            ])

        cmd.extend([
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg chart silent error: {result.stderr[-500:]}")
            raise AssemblerError(f"ffmpeg chart segment failed: {result.stderr[-200:]}")

        if overlay_png and Path(overlay_png).exists():
            Path(overlay_png).unlink(missing_ok=True)

    # ── Per-segment fallback mode ─────────────────────────────────────────────

    def _assemble_per_segment(
        self,
        script: dict,
        chart_paths: list[str],
        audio_map: dict,
        broll_intro: str,
        broll_outro: str,
        output_path: str,
        work_dir: Path,
    ) -> str:
        """Per-segment audio assignment (fallback when _full_narration.mp3 absent)."""
        parts = []

        if broll_intro and Path(broll_intro).exists():
            hook_audio = audio_map.get("hook")
            intro_path = work_dir / "part_000_intro.mp4"
            self._make_broll_with_audio(broll_intro, hook_audio, str(intro_path))
            parts.append(str(intro_path))

        for i, segment in enumerate(script["segments"]):
            if i >= len(chart_paths):
                logger.warning(f"No chart for segment {i}, skipping")
                continue
            chart = chart_paths[i]
            if not chart or not Path(chart).exists():
                logger.warning(f"Segment {i}: chart missing, skipping")
                continue

            audio = audio_map.get(f"segment_{i}")
            on_screen = segment.get("on_screen_text", "")
            seg_path = work_dir / f"part_{i + 1:03d}_seg.mp4"
            self._make_chart_segment(
                chart_path=chart,
                audio_path=audio,
                on_screen_text=on_screen,
                output_path=str(seg_path),
                duration=segment.get("duration_seconds", 60),
            )
            parts.append(str(seg_path))

        if broll_outro and Path(broll_outro).exists():
            outro_audio = audio_map.get("outro")
            outro_path = work_dir / "part_999_outro.mp4"
            self._make_broll_with_audio(broll_outro, outro_audio, str(outro_path))
            parts.append(str(outro_path))

        if not parts:
            raise AssemblerError("No video parts to assemble")

        self._concat(parts, output_path, work_dir, has_audio=True)
        self._validate_output(output_path, script)

        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

        logger.info(f"Final video assembled: {output_path}")
        return output_path

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _render_text_overlay(self, text: str, output_png: str):
        """Render caption text to a full-frame RGBA PNG using Pillow."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font_size = self.captions.get("size", 48)
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (self.width - text_w) // 2
        y = self.height - text_h - 60
        pad = 14

        draw.rectangle(
            [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
            fill=(0, 0, 0, 170),
        )
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

        img.save(output_png, "PNG")

    def _make_chart_segment(
        self,
        chart_path: str,
        audio_path: str = None,
        on_screen_text: str = "",
        output_path: str = "segment.mp4",
        duration: int = 60,
    ):
        """Create a segment from a chart image with Ken Burns + narration (per-segment fallback)."""
        if audio_path and Path(audio_path).exists():
            audio_dur = self._get_duration(audio_path)
            if audio_dur > 0:
                duration = audio_dur

        zoom = self.ken_burns.get("zoom_factor", 1.15) if self.ken_burns.get("enabled") else 1.0
        total_frames = int(duration * self.fps)

        zoompan = (
            f"zoompan=z='min(max(zoom,pzoom)+{(zoom - 1) / total_frames:.8f},{zoom})':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={self.width}x{self.height}:fps={self.fps}"
        )

        has_text = bool(on_screen_text and self.captions.get("enabled", True))
        has_audio = bool(audio_path and Path(audio_path).exists())

        overlay_png = None
        if has_text:
            overlay_png = str(Path(output_path).with_suffix(".overlay.png"))
            self._render_text_overlay(on_screen_text, overlay_png)

        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", chart_path]
        if overlay_png:
            cmd.extend(["-loop", "1", "-i", overlay_png])
        if has_audio:
            cmd.extend(["-i", audio_path])
        else:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"])

        audio_idx = 2 if overlay_png else 1

        if overlay_png:
            filter_complex = (
                f"[0:v]{zoompan}[zoomed];"
                f"[zoomed][1:v]overlay=0:0[v]"
            )
            cmd.extend([
                "-t", str(duration),
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", f"{audio_idx}:a",
            ])
        else:
            cmd.extend([
                "-t", str(duration),
                "-vf", zoompan,
            ])

        cmd.extend([
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
        ])
        if has_audio:
            cmd.extend(["-b:a", "192k"])
        cmd.extend([output_path])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg segment error: {result.stderr[-500:]}")
            raise AssemblerError(f"ffmpeg segment failed: {result.stderr[-200:]}")

        if overlay_png and Path(overlay_png).exists():
            Path(overlay_png).unlink(missing_ok=True)

    def _make_broll_with_audio(self, video_path: str, audio_path: str = None, output_path: str = "broll.mp4"):
        """Overlay narration on a B-roll clip (per-segment fallback)."""
        broll_dur = self._get_duration(video_path)
        if audio_path and Path(audio_path).exists():
            audio_dur = self._get_duration(audio_path)
            # Cap to the shorter of broll or audio — explicit instead of -shortest
            target_dur = min(broll_dur, audio_dur) if broll_dur > 0 and audio_dur > 0 else (broll_dur or audio_dur)
        else:
            target_dur = broll_dur

        cmd = ["ffmpeg", "-y", "-i", video_path]
        if audio_path and Path(audio_path).exists():
            cmd.extend(["-i", audio_path])
        else:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"])

        cmd.extend([
            "-t", str(target_dur),
            "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                   f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color={self.bg_color.replace('#', '0x')}",
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
        ])
        if audio_path and Path(audio_path).exists():
            cmd.extend(["-b:a", "192k"])
        cmd.extend([output_path])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg broll error: {result.stderr[-500:]}")
            raise AssemblerError(f"ffmpeg broll failed: {result.stderr[-200:]}")

    def _concat(self, parts: list[str], output_path: str, work_dir: Path, has_audio: bool = True):
        """Concatenate video parts using ffmpeg concat demuxer."""
        list_file = work_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for part in parts:
                f.write(f"file '{part}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])
        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg concat error: {result.stderr[-500:]}")
            raise AssemblerError(f"ffmpeg concat failed: {result.stderr[-200:]}")

    def _validate_output(self, output_path: str, script: dict):
        """Validate the final MP4 with ffprobe."""
        if not Path(output_path).exists():
            raise AssemblerError(f"Output file not created: {output_path}")

        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssemblerError(f"ffprobe validation failed: {result.stderr}")

        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        has_video = any(s["codec_type"] == "video" for s in streams)
        has_audio = any(s["codec_type"] == "audio" for s in streams)

        if not has_video:
            raise AssemblerError("Output has no video stream")
        if not has_audio:
            raise AssemblerError("Output has no audio stream")

        fmt_duration = float(probe.get("format", {}).get("duration", 0))

        # Check video and audio stream durations match within 2s
        stream_durations = {}
        for s in streams:
            dur = float(s.get("duration", 0))
            if dur > 0:
                stream_durations[s["codec_type"]] = dur
        if "video" in stream_durations and "audio" in stream_durations:
            v_dur = stream_durations["video"]
            a_dur = stream_durations["audio"]
            gap = abs(v_dur - a_dur)
            if gap > 2.0:
                logger.warning(
                    f"Video/audio duration mismatch: video={v_dur:.1f}s audio={a_dur:.1f}s (gap={gap:.1f}s)"
                )

        file_size = Path(output_path).stat().st_size
        logger.info(
            f"Output validated: {fmt_duration:.0f}s, {file_size / (1024 * 1024):.1f}MB, "
            f"video={has_video}, audio={has_audio}"
        )

    def _get_duration(self, path: str) -> float:
        """Get duration of an audio/video file in seconds."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0
