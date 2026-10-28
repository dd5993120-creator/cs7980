#!/usr/bin/env bash
# Print each new line of the remote progress log; used as a Monitor event stream.
#   bash tools/watch_remote_log.sh [session]
# Exits when a line says "ALL DONE", or after 10 failed downloads in a row.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
SESSION="${1:-gts}"
LOCAL=data/outputs/_remote/progress.log
mkdir -p "$(dirname "$LOCAL")"
seen=$(wc -l < "$LOCAL" 2>/dev/null || echo 0)
fails=0
while true; do
  if ./.venv-colab/Scripts/colab download -s "$SESSION" /content/outputs/progress.log "$LOCAL.new" >/dev/null 2>&1; then
    fails=0
    total=$(wc -l < "$LOCAL.new")
    if [ "$total" -gt "$seen" ]; then
      tail -n +"$((seen + 1))" "$LOCAL.new"
      seen=$total
    fi
    mv -f "$LOCAL.new" "$LOCAL"
    grep -q "ALL DONE" "$LOCAL" && exit 0
  else
    fails=$((fails + 1))
    [ "$fails" -ge 10 ] && { echo "WATCH: 10 failed downloads in a row; session may be gone"; exit 1; }
  fi
  sleep 30
done
