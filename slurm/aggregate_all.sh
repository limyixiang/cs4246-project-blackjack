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
  python hi_lo_variant/add_split/aggregate_qtables.py --inputs "${inputs[@]}" --out "$out"

  # Also copy a representative meta JSON alongside the merged NPZ, if available
  meta_candidates=()
  for f in "${inputs[@]}"; do
    mf="${f%.npz}_meta.json"
    [ -f "$mf" ] && meta_candidates+=("$mf")
  done
  if [ ${#meta_candidates[@]} -gt 0 ]; then
    newest_meta="${meta_candidates[0]}"
    for mf in "${meta_candidates[@]}"; do
      if [ "$mf" -nt "$newest_meta" ]; then
        newest_meta="$mf"
      fi
    done
    meta_out="${out%.npz}_meta.json"
    # Copy only if destination is missing or older than source
    if [ ! -f "$meta_out" ] || [ "$newest_meta" -nt "$meta_out" ]; then
      cp "$newest_meta" "$meta_out"
      echo "Copied meta: $newest_meta -> $meta_out"
    else
      echo "Up-to-date meta: $meta_out (skipping)"
    fi
  else
    echo "No meta JSON found for $display_cfg"
  fi
done

echo "All merges written to $OUTROOT"
