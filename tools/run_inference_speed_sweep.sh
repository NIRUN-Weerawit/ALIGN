#!/usr/bin/env bash
# Run inference-speed measurement for a list of checkpoints, one by one.
#
# Usage:
#   ./tools/run_inference_speed_sweep.sh
#
# Each checkpoint is run with the flags matching its config.json.
# Output goes to results/inference_speed/<run_name>/.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Clean Python env so torch uses the conda env's numpy (not Hermes's venv)
export PYTHONPATH="$REPO_ROOT"
unset VIRTUAL_ENV

CHECKPOOTS=(
    "checkpoints/v4/libero_goal/run_3/intention_best.pt"
    "checkpoints/v4/libero_goal/run_4/intention_best.pt"
    "checkpoints/v4/libero_goal/run_5/intention_best.pt"
    "checkpoints/v4/libero_goal/run_6/intention_best.pt"
)

DATA="data/libero_goal.h5"

# RTX 4060 has only 8 GB; the full B=1, T=30 forward with raw frames OOMs
# at the throughput-scaling step (which tries B=64) and at the component
# breakdown (which re-runs the vision encoder). We use T=10 and skip
# both for memory-constrained machines. On a 16+ GB GPU (H100, A100)
# set T=30 and remove --no-component / --no-scaling.
COMMON_ARGS=(
    --data "$DATA"
    --cameras image wrist_image
    --batch-size 1
    --seq-length 10
    --image-size 224
    --no-component
    --no-scaling
)

mkdir -p results/inference_speed

for CKPT in "${CHECKPOOTS[@]}"; do
    if [ ! -f "$CKPT" ]; then
        echo "[skip] $CKPT not found"
        continue
    fi
    RUN_DIR="$(dirname "$CKPT")"
    RUN_NAME="$(basename "$RUN_DIR")"
    OUT="results/inference_speed/${RUN_NAME}"
    mkdir -p "$OUT"
    echo "================================================================"
    echo "[$RUN_NAME] running $CKPT"
    echo "================================================================"
    # Use a clean env wrapper so torch's numpy import doesn't conflict
    env -i HOME=/home/ucluser \
        PATH=/home/ucluser/miniconda3/envs/align/bin:/usr/bin:/bin \
        PYTHONPATH="$REPO_ROOT" \
        /home/ucluser/miniconda3/envs/align/bin/python3 \
        tools/measure_inference_speed.py \
        --checkpoint "$CKPT" \
        --output "$OUT" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$OUT/run.log"
    echo
done

echo "================================================================"
echo "All done. Summary table:"
echo "================================================================"
printf "%-12s %-12s %-15s %-12s %-15s\n" "run" "mean_ms" "p95_ms" "Hz" "peak_vram_MB"
for RUN_NAME in run_3 run_4 run_5 run_6; do
    OUT="results/inference_speed/${RUN_NAME}"
    if [ -f "$OUT/results.json" ]; then
        RUN_NAME="$RUN_NAME" OUT="$OUT" /home/ucluser/miniconda3/envs/align/bin/python3 -c "
import json, os
run_name = os.environ['RUN_NAME']
out = os.environ['OUT']
d = json.load(open(f'{out}/results.json'))
lat = d['per_step_latency']
vram = d.get('peak_vram_MB', 'n/a')
hz = d.get('effective_control_rate_hz', 0)
print(f\"{run_name:<12} {lat['mean_ms']:<12.2f} {lat['p95_ms']:<15.2f} {hz:<12.1f} {str(vram):<15}\")"
    fi
done
