#!/usr/bin/env bash
# Mirror a running Colab job: print new lines of the remote progress log and
# download every finished chunk, so losing the VM costs at most one chunk.
#   bash tools/mirror_remote.sh <local_run_dir> [session]
# Exits when the log says "ALL DONE", or after 10 failed polls in a row.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
RUN="${1:?local run dir, e.g. data/outputs/g4_bf16}"
SESSION="${2:-gts}"
C=./.venv-colab/Scripts/colab
LOG="$RUN/progress.log"
mkdir -p "$RUN"
seen=$(wc -l < "$LOG" 2>/dev/null || echo 0)
fails=0
while true; do
  if $C download -s "$SESSION" /content/outputs/progress.log "$LOG.new" >/dev/null 2>&1; then
    fails=0
    total=$(wc -l < "$LOG.new")
    if [ "$total" -gt "$seen" ]; then
      tail -n +"$((seen + 1))" "$LOG.new"
      seen=$total
    fi
    mv -f "$LOG.new" "$LOG"
    for dir in $($C ls -s "$SESSION" /content/outputs 2>/dev/null | grep -aoE "^[A-Za-z0-9_]+/" | tr -d /); do
      for f in $($C ls -s "$SESSION" "/content/outputs/$dir" 2>/dev/null | grep -aoE "chunk_[0-9]+\.jsonl\.gz"); do
        if [ ! -s "$RUN/$dir/$f" ]; then
          mkdir -p "$RUN/$dir"
          $C download -s "$SESSION" "/content/outputs/$dir/$f" "$RUN/$dir/$f" >/dev/null 2>&1 \
            && echo "MIRROR: $dir/$f"
        fi
      done
    done
    grep -q "ALL DONE" "$LOG" && exit 0
  else
    fails=$((fails + 1))
    [ "$fails" -ge 10 ] && { echo "MIRROR: 10 failed polls in a row; session may be gone"; exit 1; }
  fi
  sleep 60
done
