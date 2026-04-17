#!/bin/bash
# Launched by launchd (~/Library/LaunchAgents/com.aiedge.scanner.plist)
# weekdays at 6:15 AM PT / 9:15 AM ET. The scanner itself waits until
# 9:25 AM ET to connect, so this gives 10 min of init buffer.
#
# Sources both credentials/.env (DATABENTO_API_KEY etc.) and ~/.zshrc
# (SYNC_SECRET), then execs live_scanner.py. Output is redirected to
# a dated log file that rolls over each morning.

set -euo pipefail

SCANNER_DIR="$HOME/code/aiedge/scanner"
LOG_DIR="$SCANNER_DIR/_logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/live_scanner_$(date +%Y%m%d).log"

# Load env files. set -a auto-exports every assignment.
set -a
# shellcheck disable=SC1091
[ -f "$SCANNER_DIR/credentials/.env" ] && source "$SCANNER_DIR/credentials/.env"
# shellcheck disable=SC1090
[ -f "$HOME/.zshrc" ] && source "$HOME/.zshrc" 2>/dev/null || true
set +a

cd "$SCANNER_DIR"
exec /opt/homebrew/bin/python3 -u live_scanner.py >> "$LOG_FILE" 2>&1
