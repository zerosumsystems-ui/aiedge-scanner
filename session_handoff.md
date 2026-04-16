# Video Pipeline — Session Handoff

## System Overview

Autonomous trading video pipeline producing 7 daily reports, uploaded to YouTube and published as newsletters via Beehiiv. Runs unattended on Mac Mini via launchd.

## Architecture

- **Orchestrator:** `pipeline.py --config configs/{pipeline}.yaml`
- **Flow:** Screener → Script (Claude) → Charts (mplfinance) → B-roll (Kling) → Narration (ElevenLabs) → Assembly (ffmpeg) → Upload (YouTube) + Newsletter (Beehiiv) → Log (SQLite)
- **Data source:** Databento (standalone, no AI Edge dependency)
- **All 7 pipelines share the same code** — differentiated only by YAML config

## Pipelines

| # | Name | Config | Schedule (ET) |
|---|------|--------|---------------|
| 1 | Pre-Market Brief | `configs/premarket.yaml` | 8:30 AM |
| 2 | Gap Up Report | `configs/gap_up.yaml` | 10:00 AM |
| 3 | Gap Down Report | `configs/gap_down.yaml` | 10:00 AM |
| 4 | New Highs Report | `configs/new_highs.yaml` | 4:15 PM |
| 5 | New Lows Report | `configs/new_lows.yaml` | 4:15 PM |
| 6 | Best Stocks of Day | `configs/best_stocks.yaml` | 4:30 PM |
| 7 | Industry Groups | `configs/industry_groups.yaml` | 4:45 PM |

## Key Files

- `pipeline.py` — main entry point, `--config`, `--resume`, `--health-check`
- `shared/` — all reusable modules (Databento, charts, Claude, ElevenLabs, Kling, ffmpeg, YouTube, Beehiiv, SQLite)
- `stages/` — pipeline stage wrappers calling shared modules
- `configs/` — YAML per pipeline
- `credentials/` — `.env`, `youtube_token.json`, `client_secret.json`
- `db/pipeline.sqlite` — shared log DB, rows tagged by `pipeline_name`
- `runs/{pipeline}/{run_id}/` — per-run working directory
- `output/{pipeline}/` — final MP4s kept permanently
- `assets/fallback_broll/` — 10 pre-generated Kling clips (5 intro, 5 outro)

## API Keys Location

All in `credentials/.env`. YouTube OAuth in `credentials/youtube_token.json`.

## YouTube Quota

- ~10,000 units/day, ~1,600 per upload = ~6 uploads/day on a single project
- 7 pipelines/day exceeds this limit
- **Options:**
  1. Request quota increase via Google Cloud Console → APIs & Services → YouTube Data API v3 → Quotas → Request increase (takes 2-4 weeks)
  2. Create a second Google Cloud project with its own OAuth credentials, split pipelines across projects (e.g., 4 on project A, 3 on project B)
  3. Upload some videos via the YouTube Studio web UI manually until quota increase is approved
- The pipeline tracks quota usage in SQLite and skips upload (saves MP4 locally) when quota is insufficient

## Resuming Failed Runs

```bash
python pipeline.py --config configs/premarket.yaml --resume {run_id}
```

Check `runs/{pipeline}/{run_id}/state.json` for last completed stage.

## Health Check

```bash
python pipeline.py --health-check
```

Verifies all API keys, ffmpeg, disk space, SQLite, YAML configs.

## Adding New Pipelines

See `CLONE_NEW_PIPELINE.md` — copy YAML, change values, add launchd plist.

## Cost Tracking

```sql
SELECT pipeline_name, SUM(estimated_cost_usd) FROM runs WHERE date >= '2026-04-01' GROUP BY pipeline_name;
```

## Notifications

- Discord webhook for failures (immediate) and daily digest (6 PM ET)
- Email as fallback (configure SMTP in .env)
- Success notifications off by default
