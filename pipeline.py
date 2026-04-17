#!/usr/bin/env python3
"""
Video Pipeline Orchestrator

Usage:
    python pipeline.py --config configs/premarket.yaml
    python pipeline.py --config configs/premarket.yaml --resume RUN_ID
    python pipeline.py --health-check

Runs a complete pipeline: screener → script → charts → B-roll → narration → assembly → upload → newsletter → log
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config, load_env, get_run_dir, get_project_root
from shared import sqlite_logger
from shared.notifier import Notifier

# Pipeline stages — screener lives in scanner-root stages/, content stages live in content/
from stages import screener
from content.stages import script, chart_generation, broll_generation
from content.stages import narration, assembly, upload, newsletter, log


STAGES = [
    "screener", "script", "chart_generation",
    "broll_generation", "narration", "assembly",
    "upload", "newsletter", "log",
]


def setup_logging(config: dict, run_id: str):
    """Configure logging for a pipeline run."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_config.get("log_to_file", True):
        from shared.config_loader import get_log_dir
        log_dir = get_log_dir(config["pipeline_name"])
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        handlers.append(logging.FileHandler(str(log_file)))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def check_budget(config: dict) -> bool:
    """Check if we're within daily budget."""
    budget = config.get("budget", {})
    max_daily = budget.get("daily_max_usd", 20.0)
    spent = sqlite_logger.get_daily_cost()
    if spent >= max_daily:
        logging.error(f"Daily budget exceeded: ${spent:.2f} >= ${max_daily:.2f}")
        return False
    logging.info(f"Budget OK: ${spent:.2f} / ${max_daily:.2f} daily cap")
    return True


def run_pipeline(config_path: str, resume_run_id: str = None, skip_upload: bool = False):
    """
    Execute a complete pipeline run.

    Args:
        config_path:    Path to the pipeline YAML config.
        resume_run_id:  If set, resume a previously-checkpointed run.
        skip_upload:    If True, the YouTube upload stage is skipped
                        entirely. The YouTube API is NOT contacted, so
                        daily quota is preserved. Use for manual test
                        re-runs during iteration so they don't burn the
                        10k-unit YouTube Data API daily budget. The video
                        and newsletter still ship normally. Fix B,
                        2026-04-10.
    """
    load_env()
    config = load_config(config_path)
    pipeline_name = config["pipeline_name"]

    # Generate or resume run ID
    run_id = resume_run_id or f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = str(get_run_dir(pipeline_name, run_id))

    setup_logging(config, run_id)
    logger = logging.getLogger("pipeline")
    notifier = Notifier(config)

    logger.info(f"{'Resuming' if resume_run_id else 'Starting'} pipeline: {pipeline_name} (run: {run_id})")
    if skip_upload:
        logger.warning(
            "--skip-upload enabled, YouTube upload stage will be skipped "
            "(daily quota preserved)"
        )

    # Load checkpoint state if resuming
    completed_stages = set()
    run_state = {}
    if resume_run_id:
        run_state = sqlite_logger.get_run_state(run_id) or {}
        completed_stages = set(run_state.get("completed_stages", []))
        logger.info(f"Resuming from checkpoint. Completed stages: {completed_stages}")
    else:
        # Create new run record
        sqlite_logger.create_run(run_id, pipeline_name)

    # Budget check
    if not check_budget(config):
        msg = "Daily budget exceeded — aborting run"
        sqlite_logger.fail_run(run_id, msg, "budget_check")
        notifier.notify_run_failure(pipeline_name, run_id, msg, "budget_check")
        return

    # ── Execute stages ───────────────────────────────────────────
    try:
        # Stage 1: Screener
        screener_data = run_state.get("screener_data")
        if "screener" not in completed_stages:
            logger.info("── Stage: Screener ──")
            screener_data = screener.run(config, run_id, run_dir)
            _checkpoint(run_id, completed_stages, "screener", {"screener_data": screener_data})
        else:
            logger.info("── Stage: Screener (cached) ──")

        # If screener returned 0 results for an intraday scanner, skip the
        # run gracefully AND mark the row so it doesn't sit in 'running'
        # forever. We use the same pattern as Fix D (youtube_quota_exhausted):
        # status stays 'success' because this is a legitimate terminal
        # outcome, not a failure — but error_stage/error_message make the
        # exceptional case queryable:
        #
        #     SELECT run_id, pipeline_name FROM runs
        #     WHERE error_stage='screener' AND error_message='no_qualifying_gaps';
        #
        # Fix: Task 3, 2026-04-10.
        screener_type = config["screener"].get("type", "")
        if (
            screener_data is not None
            and screener_type in ("gap_up", "gap_down")
            and len(screener_data.get("data", {}).get("gaps", [])) == 0
            and "script" not in completed_stages
        ):
            logger.info(
                f"Screener returned 0 gaps for {screener_type} — "
                f"no qualifying setups today. Skipping video production."
            )
            sqlite_logger.complete_run(
                run_id,
                status="success",
                error_stage="screener",
                error_message="no_qualifying_gaps",
            )
            return None

        # Stage 2: Script
        script_data = run_state.get("script_data")
        if "script" not in completed_stages:
            logger.info("── Stage: Script ──")
            script_data = script.run(config, screener_data, run_id, run_dir)
            _checkpoint(run_id, completed_stages, "script", {"script_data": script_data})
        else:
            logger.info("── Stage: Script (cached) ──")

        # Stage 3: Chart Generation
        chart_paths = run_state.get("chart_paths")
        if "chart_generation" not in completed_stages:
            logger.info("── Stage: Chart Generation ──")
            chart_paths = chart_generation.run(config, script_data, run_id, run_dir)
            _checkpoint(run_id, completed_stages, "chart_generation", {"chart_paths": chart_paths})
        else:
            logger.info("── Stage: Chart Generation (cached) ──")

        # Stage 4: B-roll Generation
        broll = run_state.get("broll")
        if "broll_generation" not in completed_stages:
            logger.info("── Stage: B-roll Generation ──")
            broll = broll_generation.run(config, run_id, run_dir)
            _checkpoint(run_id, completed_stages, "broll_generation", {"broll": broll})
        else:
            logger.info("── Stage: B-roll Generation (cached) ──")

        # Stage 5: Narration
        audio_paths = run_state.get("audio_paths")
        if "narration" not in completed_stages:
            logger.info("── Stage: Narration ──")
            audio_paths = narration.run(config, script_data, run_id, run_dir)
            _checkpoint(run_id, completed_stages, "narration", {"audio_paths": audio_paths})
        else:
            logger.info("── Stage: Narration (cached) ──")

        # Stage 6: Assembly
        video_path = run_state.get("video_path")
        if "assembly" not in completed_stages:
            logger.info("── Stage: Assembly ──")
            video_path = assembly.run(config, script_data, chart_paths, audio_paths, broll, run_id, run_dir)
            _checkpoint(run_id, completed_stages, "assembly", {"video_path": video_path})
        else:
            logger.info("── Stage: Assembly (cached) ──")

        # Stage 7: Upload (YouTube)
        # ----------------------------------------------------------------
        # Upload failures are NON-FATAL. The pipeline's job is to ship a
        # video + newsletter; even if YouTube upload is skipped or fails,
        # the mp4 is already on disk and the Beehiiv newsletter can still
        # reference the local file. We therefore let the run continue and
        # tag the failure in the `runs` table so it's queryable:
        #
        #     SELECT run_id, pipeline_name, error_message FROM runs
        #     WHERE error_stage='upload' AND date(started_at)=date('now');
        #
        # Per-scenario tagging (see _handle_upload_stage):
        #   manual skip (--skip-upload)     error_stage=NULL, error_message=NULL
        #   quota exhausted (API limit)     error_stage='upload', error_message='youtube_quota_exhausted'
        #   other upload error (auth/net)   error_stage='upload', error_message=<str(e)[:500]>
        # In every case the run still lands as status='success' because the
        # video was produced. Fix B + Fix D, 2026-04-10.
        # ----------------------------------------------------------------
        youtube_result = run_state.get("youtube_result", {})
        if "upload" not in completed_stages:
            youtube_result = _handle_upload_stage(
                config, script_data, video_path, run_id, skip_upload, logger
            )
            _checkpoint(run_id, completed_stages, "upload", {"youtube_result": youtube_result})
        else:
            logger.info("── Stage: Upload (cached) ──")

        # Stage 8: Newsletter (parallel with upload — non-blocking)
        newsletter_result = run_state.get("newsletter_result", {})
        if "newsletter" not in completed_stages:
            logger.info("── Stage: Newsletter ──")
            try:
                newsletter_result = newsletter.run(
                    config, script_data, chart_paths,
                    youtube_result.get("video_url", ""),
                    run_id,
                )
            except Exception as e:
                logger.error(f"Newsletter failed (non-fatal): {e}")
                newsletter_result = {}
            _checkpoint(run_id, completed_stages, "newsletter", {"newsletter_result": newsletter_result})
        else:
            logger.info("── Stage: Newsletter (cached) ──")

        # Stage 9: Log
        logger.info("── Stage: Log ──")
        data = screener_data.get("data", {})
        total_cost = sqlite_logger.get_run_api_cost(run_id)
        log.run(
            run_id=run_id,
            pipeline_name=pipeline_name,
            screener_count=(
                len(data.get("futures", {})) + len(data.get("premarket_movers", [])) +
                len(data.get("symbols", [])) + len(data.get("gaps", [])) +
                len(data.get("extremes", [])) + len(data.get("gainers", [])) +
                len(data.get("sectors", []))
            ),
            segment_count=len(script_data.get("segments", [])),
            total_duration=sum(s.get("duration_seconds", 0) for s in script_data.get("segments", [])),
            output_path=video_path,
            youtube_result=youtube_result,
            newsletter_result=newsletter_result,
            estimated_cost=total_cost,
        )

        logger.info(f"Pipeline {pipeline_name} completed successfully! Run: {run_id}")
        if youtube_result.get("video_url"):
            logger.info(f"YouTube: {youtube_result['video_url']}")

        notifier.notify_run_success(pipeline_name, run_id, youtube_result.get("video_url", ""))

    except Exception as e:
        error_stage = _current_stage(completed_stages)
        logger.exception(f"Pipeline failed at stage '{error_stage}': {e}")
        sqlite_logger.fail_run(run_id, str(e), error_stage)
        notifier.notify_run_failure(pipeline_name, run_id, str(e), error_stage)
        sys.exit(1)


def _checkpoint(run_id: str, completed: set, stage: str, data: dict):
    """Save checkpoint after successful stage completion."""
    completed.add(stage)
    # Merge into existing state so prior stage outputs are not lost on resume
    state = sqlite_logger.get_run_state(run_id) or {}
    state["completed_stages"] = list(completed)
    state.update(data)
    sqlite_logger.save_run_state(run_id, state)


def _current_stage(completed: set) -> str:
    """Determine which stage was running when failure occurred."""
    for stage in STAGES:
        if stage not in completed:
            return stage
    return "unknown"


def _handle_upload_stage(
    config: dict,
    script_data: dict,
    video_path: str,
    run_id: str,
    skip_upload: bool,
    logger: logging.Logger,
) -> dict:
    """
    Run (or deliberately skip) the YouTube upload stage.

    This is extracted from run_pipeline() so the full decision tree —
    manual skip, quota exhaustion, generic upload error, happy path — can
    be unit-tested in isolation (see tests/test_upload_skip.py).

    Returns the youtube_result dict for downstream stages. Writes
    error_stage / error_message to the runs table for any non-happy-path
    outcome so the failure is queryable by a single SELECT.

    Scenarios:
        - skip_upload=True:
            Log a warning, don't touch YouTube at all, return
            {"skipped": True, "reason": "manual_skip"}. Row stays clean
            (error_stage NULL, error_message NULL).
        - upload.run raises with "quota" in the message:
            Tag row with error_stage='upload',
            error_message='youtube_quota_exhausted'. Return {}.
        - upload.run raises any other exception:
            Tag row with error_stage='upload', error_message=str(e)[:500].
            Return {}.
        - upload.run succeeds:
            Return its dict unchanged. Row stays clean.
    """
    if skip_upload:
        logger.warning(
            "── Stage: Upload SKIPPED ── (--skip-upload flag set, "
            "YouTube API will not be contacted)"
        )
        return {"skipped": True, "reason": "manual_skip"}

    logger.info("── Stage: Upload ──")
    try:
        return upload.run(config, script_data, video_path, run_id)
    except Exception as e:
        err_str = str(e)
        if "quota" in err_str.lower():
            logger.warning(
                "Upload SKIPPED due to YouTube quota exhaustion — "
                "video + newsletter shipped normally, "
                "runs.error_stage='upload' / error_message='youtube_quota_exhausted' "
                "for queryability"
            )
            sqlite_logger.update_run(
                run_id,
                error_stage="upload",
                error_message="youtube_quota_exhausted",
            )
        else:
            logger.error(f"YouTube upload failed (non-fatal): {e}")
            sqlite_logger.update_run(
                run_id,
                error_stage="upload",
                error_message=err_str[:500],
            )
        return {}


def run_health_check():
    """Verify all dependencies and API keys."""
    load_env()
    checks = []

    # ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        version = result.stdout.split("\n")[0] if result.returncode == 0 else "NOT FOUND"
        checks.append(("ffmpeg", result.returncode == 0, version))
    except FileNotFoundError:
        checks.append(("ffmpeg", False, "NOT INSTALLED — run: brew install ffmpeg"))

    # ffprobe
    try:
        result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
        checks.append(("ffprobe", result.returncode == 0, "OK"))
    except FileNotFoundError:
        checks.append(("ffprobe", False, "NOT INSTALLED"))

    # API Keys
    api_keys = {
        "DATABENTO_API_KEY": os.environ.get("DATABENTO_API_KEY", ""),
        "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
    }
    for name, value in api_keys.items():
        checks.append((name, bool(value), "SET" if value else "MISSING"))

    # Google Cloud / Veo
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    checks.append(("GOOGLE_CLOUD_PROJECT", bool(gcp_project), gcp_project if gcp_project else "NOT SET"))
    try:
        import vertexai
        vertexai.init(project=gcp_project or "gen-lang-client-0987146855", location="us-central1")
        checks.append(("Vertex AI SDK", True, "OK"))
    except Exception as e:
        checks.append(("Vertex AI SDK", False, f"ERROR: {e}"))

    # Optional keys
    optional_keys = ["DISCORD_WEBHOOK_URL", "BEEHIIV_API_KEY"]
    for name in optional_keys:
        value = os.environ.get(name, "")
        checks.append((f"{name} (optional)", bool(value), "SET" if value else "NOT SET"))

    # Disk space
    stat = shutil.disk_usage(str(get_project_root()))
    free_gb = stat.free / (1024 ** 3)
    checks.append(("Disk space", free_gb >= 5, f"{free_gb:.1f}GB free"))

    # SQLite
    try:
        sqlite_logger.init_db()
        checks.append(("SQLite DB", True, "OK"))
    except Exception as e:
        checks.append(("SQLite DB", False, str(e)))

    # YouTube credentials
    yt_token = get_project_root() / "credentials" / "youtube_token.json"
    yt_secret = get_project_root() / "credentials" / "client_secret.json"
    checks.append(("YouTube token", yt_token.exists(), "FOUND" if yt_token.exists() else "NOT FOUND — run OAuth flow"))
    checks.append(("client_secret.json", yt_secret.exists(), "FOUND" if yt_secret.exists() else "NOT FOUND — download from Google Cloud"))

    # YAML configs
    configs_dir = get_project_root() / "configs"
    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        try:
            load_config(str(yaml_file))
            checks.append((f"Config: {yaml_file.name}", True, "VALID"))
        except Exception as e:
            checks.append((f"Config: {yaml_file.name}", False, str(e)))

    # Print results
    print("\n" + "=" * 60)
    print("  VIDEO PIPELINE HEALTH CHECK")
    print("=" * 60)

    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        icon = "+" if ok else "x"
        print(f"  [{icon}] {name}: {detail}")
        if not ok and "(optional)" not in name:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("  All required checks passed!")
    else:
        print("  Some checks FAILED — see above")
    print("=" * 60 + "\n")

    return 0 if all_ok else 1


def run_cleanup():
    """Clean up old run directories based on config."""
    load_env()
    configs_dir = get_project_root() / "configs"

    for yaml_file in configs_dir.glob("*.yaml"):
        try:
            config = load_config(str(yaml_file))
            cleanup = config.get("cleanup", {})
            success_days = cleanup.get("successful_run_days", 7)
            fail_days = cleanup.get("failed_run_days", 30)
            pipeline = config["pipeline_name"]

            runs_dir = get_project_root() / "runs" / pipeline
            if not runs_dir.exists():
                continue

            now = datetime.now()
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                age_days = (now - datetime.fromtimestamp(run_dir.stat().st_mtime)).days
                state_file = run_dir / "state.json"

                if state_file.exists():
                    try:
                        state = json.loads(state_file.read_text())
                        is_success = "log" in state.get("completed_stages", [])
                        max_age = success_days if is_success else fail_days
                    except Exception:
                        max_age = fail_days
                else:
                    max_age = success_days

                if age_days > max_age:
                    logging.info(f"Cleaning up old run: {run_dir} ({age_days} days old)")
                    shutil.rmtree(run_dir)

        except Exception as e:
            logging.warning(f"Cleanup error for {yaml_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Video Pipeline Orchestrator")
    parser.add_argument("--config", type=str, help="Path to pipeline YAML config")
    parser.add_argument("--resume", type=str, help="Resume a failed run by run_id")
    parser.add_argument("--health-check", action="store_true", help="Run health checks")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old run directories")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help=(
            "Skip the YouTube upload stage entirely. Video + newsletter still "
            "ship as normal; YouTube daily quota is preserved. Use for manual "
            "test re-runs during iteration."
        ),
    )

    args = parser.parse_args()

    if args.health_check:
        sys.exit(run_health_check())

    if args.cleanup:
        run_cleanup()
        return

    if not args.config:
        parser.error("--config is required unless using --health-check or --cleanup")

    run_pipeline(args.config, resume_run_id=args.resume, skip_upload=args.skip_upload)


if __name__ == "__main__":
    main()
