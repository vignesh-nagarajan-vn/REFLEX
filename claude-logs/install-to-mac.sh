#!/usr/bin/env bash
# Install the exported REFLEX Claude Code web-session usage logs into this
# machine's ~/.claude/projects/ so local token counters (ccusage-style tools
# that scan ~/.claude/projects/**/*.jsonl) pick them up.
#
# Usage:  bash claude-logs/install-to-mac.sh
# Re-running is safe: files are overwritten in place, and each entry carries a
# stable message id / requestId, so counters that de-duplicate will not
# double-count.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)/mac-import/-REFLEX-claude-web"
DEST_DIR="$HOME/.claude/projects/-REFLEX-claude-web"

if [ ! -d "$SRC_DIR" ]; then
  echo "error: $SRC_DIR not found (run from a checkout of the REFLEX repo)" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC_DIR"/*.jsonl "$DEST_DIR"/

echo "Installed $(ls "$SRC_DIR"/*.jsonl | wc -l | tr -d ' ') session log(s) to $DEST_DIR:"
ls -1 "$DEST_DIR"
echo
echo "Your token counter should now show a 'REFLEX-claude-web' project."
