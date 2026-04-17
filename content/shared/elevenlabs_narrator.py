"""
ElevenLabs narration with retry, quota checks, and voice fallback.

Continuous generation mode: concatenates all script sections into one API call,
then splits at silence boundaries for natural prosody across the full narration.
Falls back to per-segment generation if splitting fails.
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from elevenlabs import ElevenLabs
from elevenlabs.core import ApiError

from shared import sqlite_logger

logger = logging.getLogger(__name__)

# Separator inserted between script sections for continuous generation.
# [long pause] is an ElevenLabs v3 audio tag that produces ~0.8-1.2s of silence.
_SECTION_SEP = "\n\n[long pause]\n\n"


class NarratorError(Exception):
    pass


class Narrator:
    def __init__(self, api_key: str = None, voice_id: str = None, model: str = "eleven_v3"):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise NarratorError("ELEVENLABS_API_KEY not set")

        self.client = ElevenLabs(api_key=self.api_key)
        self.voice_id = voice_id
        self.fallback_voice_id = os.environ.get("ELEVENLABS_FALLBACK_VOICE_ID")
        self.model = model

    def check_quota(self, text_length: int) -> bool:
        """Check if we have enough character quota for the narration."""
        try:
            subscription = self.client.user.get_subscription()
            remaining = subscription.character_limit - subscription.character_count
            if remaining < text_length:
                logger.error(
                    f"ElevenLabs quota insufficient: need {text_length} chars, "
                    f"have {remaining} remaining"
                )
                return False
            logger.info(f"ElevenLabs quota OK: {remaining} chars remaining, need {text_length}")
            return True
        except Exception as e:
            logger.warning(f"Could not check ElevenLabs quota: {e}")
            return True  # Proceed optimistically if check fails

    def narrate_segments(
        self,
        segments: list[dict],
        hook: str,
        outro: str,
        output_dir: str,
        voice_id: str = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.3,
        use_speaker_boost: bool = True,
        run_id: str = "",
        pipeline_name: str = "",
    ) -> list[dict]:
        """
        Generate narration for all segments + hook + outro.

        Tries continuous generation (one API call → split by silence) first.
        Falls back to per-segment generation if silence splitting fails.

        Returns list of {segment_index, label, audio_path, characters}.
        """
        vid = voice_id or self.voice_id
        if not vid:
            raise NarratorError("No voice_id configured")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Ordered sections: file names and corresponding texts
        file_names = ["hook"] + [f"scene_{i}" for i in range(len(segments))] + ["outro"]
        result_labels = ["hook"] + [f"segment_{i}" for i in range(len(segments))] + ["outro"]
        section_texts = [hook] + [s["narration"] for s in segments] + [outro]
        segment_indices = [-1] + list(range(len(segments))) + [999]

        total_chars = sum(len(t) for t in section_texts)
        if not self.check_quota(total_chars):
            raise NarratorError(f"Insufficient ElevenLabs quota for {total_chars} characters")

        voice_kwargs = dict(
            stability=stability, similarity_boost=similarity_boost,
            style=style, use_speaker_boost=use_speaker_boost,
        )

        # ── Attempt 1: continuous generation ─────────────────────────────
        split_ok = False
        if len(section_texts) > 1:
            split_ok = self._try_continuous(
                section_texts=section_texts,
                file_names=file_names,
                voice_id=vid,
                out=out,
                voice_kwargs=voice_kwargs,
                run_id=run_id,
                pipeline_name=pipeline_name,
                total_chars=total_chars,
            )

        # ── Attempt 2: per-segment fallback ──────────────────────────────
        if not split_ok:
            logger.info("Using per-segment narration generation")
            for fname, text, label in zip(file_names, section_texts, result_labels):
                self._generate_audio(
                    text=text, voice_id=vid,
                    output_path=out / f"{fname}.mp3",
                    run_id=run_id, pipeline_name=pipeline_name,
                    label=label, **voice_kwargs,
                )

        # Build return structure (matches assembler's audio_map expectations)
        results = []
        for idx, label, fname, text in zip(segment_indices, result_labels, file_names, section_texts):
            results.append({
                "segment_index": idx,
                "label": label,
                "audio_path": str(out / f"{fname}.mp3"),
                "characters": len(text),
            })

        logger.info(f"Narration complete: {len(results)} audio files, {total_chars} chars total")
        return results

    # ── Private: continuous generation ───────────────────────────────────

    def _try_continuous(
        self,
        section_texts: list[str],
        file_names: list[str],
        voice_id: str,
        out: Path,
        voice_kwargs: dict,
        run_id: str,
        pipeline_name: str,
        total_chars: int,
    ) -> bool:
        """
        Generate all narration in one API call, then split at silence boundaries.
        Returns True if successful, False if fallback is needed.
        """
        full_text = _SECTION_SEP.join(section_texts)
        full_path = out / "_full_narration.mp3"
        n_expected_silences = len(section_texts) - 1

        try:
            self._generate_audio(
                text=full_text, voice_id=voice_id,
                output_path=full_path,
                run_id=run_id, pipeline_name=pipeline_name,
                label="full_narration", **voice_kwargs,
            )
        except Exception as e:
            logger.warning(f"Continuous generation failed: {e}")
            return False

        try:
            split_files = self._split_by_silence(
                audio_path=str(full_path),
                out_dir=out,
                file_names=file_names,
                n_expected=n_expected_silences,
            )
            if split_files:
                logger.info(f"Continuous audio split into {len(split_files)} segments at silence boundaries")
                # Keep full file for debugging (assembler doesn't use it)
                return True
        except Exception as e:
            logger.warning(f"Silence split failed: {e}")

        # Splitting failed — clean up so caller uses per-segment
        full_path.unlink(missing_ok=True)
        return False

    def _split_by_silence(
        self,
        audio_path: str,
        out_dir: Path,
        file_names: list[str],
        n_expected: int,
    ) -> list[str]:
        """
        Detect silence in audio_path and split into len(file_names) files.
        Uses ffmpeg silencedetect. Returns list of output paths, or [] on failure.
        """
        # Try progressively more lenient thresholds
        thresholds = [("-38dB", "0.25"), ("-42dB", "0.15"), ("-45dB", "0.10")]
        silence_intervals = []

        for noise, duration in thresholds:
            result = subprocess.run(
                ["ffmpeg", "-i", audio_path,
                 "-af", f"silencedetect=noise={noise}:d={duration}",
                 "-f", "null", "-"],
                capture_output=True, text=True,
            )
            starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
            ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
            # Pair up starts and ends (last silence may lack an end if it runs to EOF)
            intervals = list(zip(starts, ends)) if len(ends) >= len(starts) else list(zip(starts, ends + [starts[-1] + 0.5]))

            if len(intervals) >= n_expected:
                silence_intervals = intervals
                logger.info(f"Silence detection ({noise}/{duration}s): found {len(intervals)} silences, need {n_expected}")
                break

        if len(silence_intervals) < n_expected:
            logger.warning(f"Could not find {n_expected} silence boundaries (found {len(silence_intervals)})")
            return []

        # If more silences than expected, keep the n_expected longest (our inserted pauses)
        if len(silence_intervals) > n_expected:
            by_duration = sorted(silence_intervals, key=lambda x: x[1] - x[0], reverse=True)
            chosen = sorted(by_duration[:n_expected], key=lambda x: x[0])
            silence_intervals = chosen

        # Get total duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
            capture_output=True, text=True,
        )
        total_dur = float(json.loads(probe.stdout)["format"]["duration"])

        # Cut points at midpoint of each silence interval
        cut_points = [(s + e) / 2 for s, e in silence_intervals]
        boundaries = [0.0] + cut_points + [total_dur]

        output_files = []
        for fname, t_start, t_end in zip(file_names, boundaries[:-1], boundaries[1:]):
            out_path = out_dir / f"{fname}.mp3"
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ss", f"{t_start:.3f}", "-to", f"{t_end:.3f}",
                 "-c:a", "libmp3lame", "-q:a", "2",
                 str(out_path)],
                capture_output=True, check=True,
            )
            logger.info(f"  Split {fname}.mp3: {t_start:.2f}s → {t_end:.2f}s ({t_end - t_start:.1f}s)")
            output_files.append(str(out_path))

        return output_files

    # ── Private: single audio generation ─────────────────────────────────

    def _generate_audio(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        stability: float,
        similarity_boost: float,
        style: float,
        use_speaker_boost: bool,
        run_id: str,
        pipeline_name: str,
        label: str,
    ):
        """Generate a single audio file with retry and voice fallback."""
        last_error = None
        current_voice = voice_id

        for attempt in range(3):
            try:
                audio_gen = self.client.text_to_speech.convert(
                    voice_id=current_voice,
                    text=text,
                    model_id=self.model,
                    voice_settings={
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "style": style,
                        "use_speaker_boost": use_speaker_boost,
                    },
                )

                with open(output_path, "wb") as f:
                    for chunk in audio_gen:
                        f.write(chunk)

                cost = len(text) * 0.0003
                sqlite_logger.log_api_call(
                    run_id=run_id,
                    pipeline_name=pipeline_name,
                    service="elevenlabs",
                    endpoint=f"tts/{label}",
                    characters_used=len(text),
                    cost_usd=cost,
                    status="success",
                    retry_count=attempt,
                )
                logger.info(f"Audio saved: {output_path} ({len(text)} chars)")
                return

            except ApiError as e:
                last_error = e
                if "voice" in str(e).lower() and self.fallback_voice_id and current_voice != self.fallback_voice_id:
                    logger.warning(f"Voice {current_voice} failed, trying fallback {self.fallback_voice_id}")
                    current_voice = self.fallback_voice_id
                    continue
                wait = 5 * (3 ** attempt)
                logger.warning(f"ElevenLabs error (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)

            except Exception as e:
                last_error = e
                wait = 5 * (3 ** attempt)
                logger.warning(f"ElevenLabs error (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        sqlite_logger.log_api_call(
            run_id=run_id, pipeline_name=pipeline_name,
            service="elevenlabs", endpoint=f"tts/{label}",
            characters_used=len(text), status="failed",
            error=str(last_error), retry_count=3,
        )
        raise NarratorError(f"ElevenLabs failed after 3 attempts for {label}: {last_error}")
