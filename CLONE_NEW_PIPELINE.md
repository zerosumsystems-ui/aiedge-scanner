# How to Clone a New Pipeline

This guide creates a new pipeline in under 30 minutes. **No code changes required** — only YAML config and a launchd plist.

## Step 1: Copy the YAML config

```bash
cp configs/premarket.yaml configs/YOUR_PIPELINE.yaml
```

## Step 2: Edit the YAML config

Open `configs/YOUR_PIPELINE.yaml` and change these fields:

### Required changes

| Field | What to change |
|---|---|
| `pipeline_name` | Unique slug: `gap_up`, `gap_down`, `new_highs`, etc. |
| `display_name` | Human-readable: `"Gap Up Report"` |
| `schedule.cron` | Schedule in ET: `"0 10 * * 1-5"` for 10:00 AM weekdays |
| `screener.type` | Screener type: `gap_up`, `gap_down`, `new_highs`, `new_lows`, `best_stocks`, `industry_groups` |
| `screener.databento_queries` | The actual data queries (see examples below) |
| `script.prompt_template` | The Claude prompt — this is the creative differentiator |
| `youtube.title_template` | YouTube title format |
| `youtube.description_template` | YouTube description format |
| `youtube.tags` | Relevant tags for this report type |

### Optional changes

| Field | Default | When to change |
|---|---|---|
| `kling.intro_prompt` | Trading desk | If you want different B-roll per pipeline |
| `kling.outro_prompt` | Wall Street | Same |
| `newsletter.subject_template` | `{title}` | If you want different email subjects |
| `assembly.branding` | Green/white/dark | If you want different colors per pipeline |

### Leave these alone

- `elevenlabs` — same voice across all pipelines
- `youtube.channel_credentials` — single channel
- `budget`, `cleanup`, `notifications` — shared settings
- `logging` — consistent across all

## Step 3: Create the launchd plist

```bash
cp launchd/com.videopipeline.premarket.plist launchd/com.videopipeline.YOUR_PIPELINE.plist
```

Edit the new plist:

1. Change `Label` to `com.videopipeline.YOUR_PIPELINE`
2. Change the `--config` argument to `configs/YOUR_PIPELINE.yaml`
3. Change `StandardOutPath` and `StandardErrorPath` to use your pipeline name
4. Change each `Weekday`+`Hour`+`Minute` in the `StartCalendarInterval` array to your schedule

## Step 4: Test the pipeline

```bash
# Dry run with yesterday's data
python pipeline.py --config configs/YOUR_PIPELINE.yaml
```

Watch the output. If the video looks good, proceed to scheduling.

## Step 5: Load the launchd job

```bash
# Copy plist to LaunchAgents (required — launchctl loads from ~/Library/LaunchAgents)
cp launchd/com.videopipeline.YOUR_PIPELINE.plist ~/Library/LaunchAgents/

# Load the schedule
launchctl load ~/Library/LaunchAgents/com.videopipeline.YOUR_PIPELINE.plist

# Verify it's loaded
launchctl list | grep videopipeline

# Trigger a manual run to test
launchctl start com.videopipeline.YOUR_PIPELINE

# Check logs
tail -f logs/YOUR_PIPELINE/$(date +%Y-%m-%d).log
```

## Step 6: Verify

- [ ] Pipeline runs without errors
- [ ] Video uploads to YouTube (as unlisted)
- [ ] Newsletter drafts correctly (if enabled)
- [ ] SQLite log has the run recorded
- [ ] launchd job is loaded and scheduled

---

## Screener Query Examples

### Gap Up Report (`gap_up`)
```yaml
screener:
  type: "gap_up"
  databento_queries:
    - dataset: "XNAS.ITCH"
      type: "gap_scanner"
      direction: "up"
      min_gap_pct: 3.0
      min_volume: 500000
      lookback_minutes: 30    # first 30 min after open
```

### Gap Down Report (`gap_down`)
```yaml
screener:
  type: "gap_down"
  databento_queries:
    - dataset: "XNAS.ITCH"
      type: "gap_scanner"
      direction: "down"
      min_gap_pct: -3.0
      min_volume: 500000
      lookback_minutes: 30
```

### New Highs Report (`new_highs`)
```yaml
screener:
  type: "new_highs"
  databento_queries:
    - dataset: "XNAS.ITCH"
      type: "new_extremes"
      direction: "high"
      periods: [20, 50, 252]    # 20-day, 50-day, 252-day highs
      min_relative_strength: 80
```

### New Lows Report (`new_lows`)
```yaml
screener:
  type: "new_lows"
  databento_queries:
    - dataset: "XNAS.ITCH"
      type: "new_extremes"
      direction: "low"
      periods: [20, 50, 252]
      min_relative_strength: 20   # inverse — weakest stocks
```

### Best Stocks of the Day (`best_stocks`)
```yaml
screener:
  type: "best_stocks"
  databento_queries:
    - dataset: "XNAS.ITCH"
      type: "top_gainers"
      min_gain_pct: 2.0
      min_volume: 1000000
      require_bpa_setup: true
```

### Industry Group Report (`industry_groups`)
```yaml
screener:
  type: "industry_groups"
  databento_queries:
    - dataset: "XNAS.ITCH"
      type: "sector_rotation"
      sectors: true
      industry_groups: true
      lookback_days: 5
```

---

## Prompt Template Examples

### Gap Up Report
```yaml
script:
  prompt_template: |
    You are writing a gap up report for active traders. The audience knows
    technical analysis and Brooks price action. Tone: direct, data-driven.
    
    These stocks gapped up >3% and held through the first 30 minutes:
    
    {screener_data}
    
    Return ONLY valid JSON matching this schema...
```

### New Highs Report
```yaml
script:
  prompt_template: |
    You are writing a new highs report. Focus on stocks showing the strongest
    relative strength. Identify which are making genuine breakouts vs. 
    exhaustion moves. Use Brooks price action terminology.
    
    {screener_data}
    
    Return ONLY valid JSON matching this schema...
```

---

## Troubleshooting

**Pipeline fails at screener stage:**
- Check Databento API key is set in `.env`
- Check the screener type matches a handler in `stages/screener.py`

**Charts look wrong:**
- Run `python -c "from shared.chart_renderer import render_chart; print('OK')"` to verify mplfinance

**YouTube upload fails:**
- Run `python pipeline.py --health-check` to verify credentials
- Check quota: `SELECT SUM(youtube_quota_used) FROM daily_costs WHERE date = date('now');`

**launchd job doesn't run:**
```bash
# Check if it's loaded
launchctl list | grep videopipeline

# Check launchd logs
cat logs/YOUR_PIPELINE/launchd_stderr.log

# Unload and reload
launchctl unload ~/Library/LaunchAgents/com.videopipeline.YOUR_PIPELINE.plist
launchctl load ~/Library/LaunchAgents/com.videopipeline.YOUR_PIPELINE.plist
```

**Resume a failed run:**
```bash
python pipeline.py --config configs/YOUR_PIPELINE.yaml --resume RUN_ID
```
Find the run_id in the logs or SQLite: `SELECT run_id, status, error_stage FROM runs ORDER BY id DESC LIMIT 5;`
