#!/usr/bin/env bash
# Run the full offline pipeline for a captured phone session:
#   1. Depth Anything V2 metric-indoor refinement.
#   2. Pose-graph loop closure (detect + apply, if any loops were found).
#   3. Reverse voxelisation for every variant that ended up on disk:
#        frames                  → voxels_original.json
#        frames_refined          → voxels_refined.json
#        frames_aligned          → voxels_aligned.json
#        frames_refined_aligned  → voxels_refined_aligned.json
#
# The aligned variants only appear if step 2 found enough loop candidates;
# single-pass scans (no revisits) skip them gracefully.
#
# Usage:
#   tools/process_session.sh <session_id>                    # full pipeline
#   tools/process_session.sh <session_id> --no-refine        # skip step 1
#   tools/process_session.sh <session_id> --no-loop-closure  # skip step 2
#
# Anything after the session id that isn't a known flag is forwarded to
# every voxel-reverse invocation, e.g.:
#   tools/process_session.sh <id> --voxel-size 0.025 --threshold 0.2
#   tools/process_session.sh <id> --no-loop-closure -- --min-color-count 3
# (the bare `--` is optional but lets you forward args that share a name
# with our flags).
#
# Run from the project root, with .venv active and the Rust binary built.
set -euo pipefail

SESSION="${1:-}"
shift || true
SKIP_REFINE=0
SKIP_LOOP=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-refine)       SKIP_REFINE=1; shift ;;
    --no-loop-closure) SKIP_LOOP=1;   shift ;;
    --)                shift; PASSTHROUGH+=("$@"); break ;;
    *)                 PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ -z "$SESSION" ]]; then
  echo "usage: $0 <session_id> [--no-refine] [--no-loop-closure] [-- voxel-reverse-args …]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
SESSION_DIR="$REPO_ROOT/captured_frames/$SESSION"
RUST_BIN="$REPO_ROOT/tools/voxel_reverse_rust/target/release/voxel-reverse"

if [[ ! -d "$SESSION_DIR/frames" ]]; then
  echo "error: $SESSION_DIR/frames not found" >&2
  exit 1
fi
if [[ ! -x "$RUST_BIN" ]]; then
  echo "error: Rust binary missing — build with:" >&2
  echo "  (cd tools/voxel_reverse_rust && cargo build --release)" >&2
  exit 1
fi

cd "$REPO_ROOT"
step() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$1"; }

if [[ $SKIP_REFINE -eq 0 ]]; then
  step "1/3  depth refinement → frames_refined/"
  python tools/depth_refine.py --session "$SESSION"
else
  step "1/3  depth refinement (skipped via --no-refine)"
fi

if [[ $SKIP_LOOP -eq 0 ]]; then
  step "2/3  loop closure → frames_aligned/ + frames_refined_aligned/"
  python tools/loop_closure_analyze.py --session "$SESSION" --apply
else
  step "2/3  loop closure (skipped via --no-loop-closure)"
fi

step "3/3  reverse voxelisation"
declare -a VARIANTS=(
  "frames:voxels_original.json"
  "frames_refined:voxels_refined.json"
  "frames_aligned:voxels_aligned.json"
  "frames_refined_aligned:voxels_refined_aligned.json"
)
for entry in "${VARIANTS[@]}"; do
  src="${entry%%:*}"
  dst="${entry##*:}"
  if [[ -d "$SESSION_DIR/$src" ]]; then
    printf "\n--- %s → %s ---\n" "$src" "$dst"
    "$RUST_BIN" \
      --frames-dir "$SESSION_DIR/$src" \
      --out        "$SESSION_DIR/$dst" \
      ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
  else
    printf "\n--- %s → %s (skipped: dir not present) ---\n" "$src" "$dst"
  fi
done

printf "\n\033[1;32mDone.\033[0m  Open https://localhost:8443/voxelview.html and pick session %s.\n" "$SESSION"
