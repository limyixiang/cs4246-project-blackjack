#!/usr/bin/env bash
set -euo pipefail

# Aggregate all configs under artifacts. Supports both flat and nested layouts:
# - Flat:   artifacts/ep_<pre>_<bet>_taskN/
# - Nested: artifacts/<variant>/ep_<pre>_<bet>_taskN/
# Produces merged npz per config, mirroring layout into OUTROOT (variant subfolder if nested).

ROOT=${1:-artifacts}
OUTROOT=${2:-artifacts_merged}
mkdir -p "$OUTROOT"

shopt -s nullglob

# Build unique list of (variant|cfg) pairs where variant may be empty for flat layout
mapfile -t CFGS < <(
  for dir in "$ROOT"/ep_* "$ROOT"/*/ep_*; do
    [ -d "$dir" ] || continue
    parent=$(dirname "$dir")
    variant=""
    if [ "$parent" != "$ROOT" ]; then
      variant=$(basename "$parent")
    fi
    base=$(basename "$dir")
    cfg="${base%%_task*}"
    echo "${variant}|${cfg}"
  done | sort -u
)

if [ ${#CFGS[@]} -eq 0 ]; then
  echo "No configs found under $ROOT (expected ep_* or */ep_* directories). Nothing to aggregate."
  exit 0
fi

for entry in "${CFGS[@]}"; do
  variant="${entry%%|*}"
  cfg="${entry#*|}"

  if [ -n "$variant" ]; then
    pattern="$ROOT/$variant/${cfg}_task*/checkpoint_final_task*.npz"
    outdir="$OUTROOT/$variant"
    mkdir -p "$outdir"
    out="$outdir/${cfg}_merged.npz"
    display_cfg="$variant/$cfg"
  else
    pattern="$ROOT/${cfg}_task*/checkpoint_final_task*.npz"
    out="$OUTROOT/${cfg}_merged.npz"
    display_cfg="$cfg"
  fi

  # Expand pattern to a list; skip if no inputs found
  inputs=( $pattern )
  if [ ${#inputs[@]} -eq 0 ]; then
    echo "No shard files found for $display_cfg, skipping"
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
  meta_out="${out%.npz}_meta.json"
  python hi_lo_variant/add_split/aggregate_qtables.py --inputs "${inputs[@]}" --out "$out" --meta-out "$meta_out" --variant "${variant:-}" --config "$cfg"
done

echo "All merges written to $OUTROOT"
