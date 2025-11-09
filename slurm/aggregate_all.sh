#!/usr/bin/env bash
set -euo pipefail

# Aggregate all configs found under artifacts/ep_* directories
# Produces merged npz per config in artifacts_merged/

ROOT=${1:-artifacts}
OUTROOT=${2:-artifacts_merged}
mkdir -p "$OUTROOT"

shopt -s nullglob
declare -A SEEN
for dir in "$ROOT"/ep_*; do
  # Expect directory names like ep_<pre>_<bet>_task<id>
  base=$(basename "$dir")
  # Strip task suffix to get ep_<pre>_<bet>
  cfg=${base%%_task*}
  SEEN["$cfg"]=1
done

for cfg in "${!SEEN[@]}"; do
  pattern="$ROOT/${cfg}_task*/checkpoint_final_task*.npz"
  out="$OUTROOT/${cfg}_merged.npz"

  # Expand pattern to a list; skip if no inputs found
  inputs=( $pattern )
  if [ ${#inputs[@]} -eq 0 ]; then
    echo "No shard files found for $cfg, skipping"
    continue
  fi

  # If output exists and is newer than all inputs, skip to save time
  if [ -f "$out" ]; then
    needs_update=false
    for f in "${inputs[@]}"; do
      if [ "$f" -nt "$out" ]; then
        needs_update=true
        break
      fi
    done
    if [ "$needs_update" = false ]; then
      echo "Up-to-date: $out (skipping)"
      continue
    fi
  fi

  echo "Aggregating ${#inputs[@]} shards -> $out"
  python hi_lo_variant/add_split/aggregate_qtables.py --inputs "${inputs[@]}" --out "$out"
done

echo "All merges written to $OUTROOT"
