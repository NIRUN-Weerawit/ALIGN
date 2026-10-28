#!/usr/bin/env bash
# Pass --dry-run to inspect input/storage without writing features or loading weights.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/precompute_dinov2.py \
  --data data/libero_goal.h5 --cameras image wrist_image \
  --output data/libero_goal.dinov2_vitl14 --batch-size 4 \
  "$@" --vision-backbone dinov2_vitl14 --raw-dim 1024
