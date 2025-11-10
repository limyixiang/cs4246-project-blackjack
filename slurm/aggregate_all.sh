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

# Format a numeric string with underscores (e.g., 50_000_000) to a compact label (e.g., 50m)
format_human() {
  local s="$1"
  # strip underscores
  local n="${s//_/}"
  # guard non-numeric
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "$s"
    return
  fi
  # use bash arithmetic (supports 64-bit)
  local v=$((10#$n))
  if (( v % 1000000 == 0 )); then
    echo "$(( v / 1000000 ))m"
  elif (( v % 1000 == 0 )); then
    echo "$(( v / 1000 ))k"
  else
    echo "$v"
  fi
}

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

  # Determine input pattern and output directory (with optional human-friendly subfolder like 50m_50m)
  if [ -n "$variant" ]; then
    pattern="$ROOT/$variant/${cfg}_task*/checkpoint_final_task*.npz"
    base_outdir="$OUTROOT/$variant"
    display_cfg="$variant/$cfg"
  else
    pattern="$ROOT/${cfg}_task*/checkpoint_final_task*.npz"
    base_outdir="$OUTROOT"
    display_cfg="$cfg"
  fi

  # Derive human-friendly subdir from cfg name: ep_<pre>_<bet>
  # Previous regex approach was greedy and mis-split cases like ep_100_000_000_100_000_000.
  # Strategy: strip 'ep_' then split underscore groups evenly into two numbers.
  human_subdir=""
  rest="${cfg#ep_}"
  IFS='_' read -r -a parts <<< "$rest"
  if (( ${#parts[@]} >= 2 )) && (( ${#parts[@]} % 2 == 0 )); then
    half=$(( ${#parts[@]} / 2 ))
    pre_raw="$(printf "%s_" "${parts[@]:0:half}")"
    pre_raw="${pre_raw%_}"
    bet_raw="$(printf "%s_" "${parts[@]:half}")"
    bet_raw="${bet_raw%_}"
    # Validate numeric (after removing underscores) to guard against unexpected patterns
    if [[ "${pre_raw//_/}" =~ ^[0-9]+$ ]] && [[ "${bet_raw//_/}" =~ ^[0-9]+$ ]]; then
      pre_human=$(format_human "$pre_raw")
      bet_human=$(format_human "$bet_raw")
      human_subdir="${pre_human}_${bet_human}"
      outdir="$base_outdir/$human_subdir"
    else
      outdir="$base_outdir"  # fallback
    fi
  else
    outdir="$base_outdir"  # fallback when uneven groups
  fi
  mkdir -p "$outdir"
  out="$outdir/${cfg}_merged.npz"

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
    # Preferred naming: place meta as "${cfg}.meta.json" in the same output dir
    meta_out="$outdir/${cfg}.meta.json"
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
