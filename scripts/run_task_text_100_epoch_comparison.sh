#!/usr/bin/env bash
# Matched full-epoch LIBERO Goal comparison: explicit task text versus no text.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin=/home/whinnoy/.venvs/align-ablation/bin/python
run_root="$repo_root/checkpoints/task_text_100_epoch_20261010"
warm_start="$repo_root/checkpoints/memory_adjustment_20261009/task_text_warm_start"
episodes="$repo_root/checkpoints/memory_adjustment_20261009/task_text_shared_goal_episodes.txt"
data=/media/whinnoy/Extreme-Linux/ALIGN_data/lerobot/nvidia/LIBERO_LeRobot_v3/libero_goal.h5
mkdir -p "$run_root"

common=(
  --data data/libero_goal.h5
  --cache data/libero_goal.dinov2
  --warm-start "$warm_start"
  --epochs 100
  --max-steps 0
  --batch-size 2
  --seed 42
  --variants no_intent_memory
  --history-size 1
  --chunk-size 8
  --segment-length 20
  --temporal-sampling episode
  --supervision-points 16
  --observation-dropout-prob 0.5
  --drop-state-with-all-views
  --memory-mode episodic
  --memory-detach-writes
  --no-memory-write-fused
  --memory-perceptual-recency 0.25
  --memory-field-masks
  --memory-value-preserving
  --memory-patch-temporal
  --memory-patch-retrieval
  --visual-token-attention
  --diffusion-clip-sample
  --diffusion-train-steps 100
  --diffusion-loss-repeats 4
  --selection-metric pos_mse
  --head-type diffusion
  --state-dim 64
  --compressed-dim 4
  --head-d-model 64
  --mamba-output-dim 128
  --num-intent-tokens 1
  --intent-dim 128
  --memory-bank-len 16
  --text-dim 128
  --cpu-threads 2
  --gpu-memory-fraction 0.35
)

train_arm() {
  local name="$1"
  shift
  local output="$run_root/$name"
  local resume=()
  if [[ -e "$output/manifest.json" ]]; then
    resume=(--resume)
  fi
  echo "$(date -Is) Starting $name (100 full epochs, 384 training episodes, batch size 2)"
  "$python_bin" -u scripts/run_intention_ablation.py \
    "${common[@]}" --output "$output" "${resume[@]}" "$@" \
    > "$run_root/$name.log" 2>&1
  echo "$(date -Is) Completed $name"
}

# Both arms copy the same trained visual/state/head/memory weights. The text
# projection is zero initialized, and only the text arm adds language weights.
train_arm with_text --use-task-text
train_arm without_text

export PYTHONPATH=/home/whinnoy/.local/share/align/LIBERO
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/home/whinnoy/.local/share/align/libero-config
export PYTHONHASHSEED=0

"$python_bin" -u scripts/select_policy_checkpoint.py \
  --candidates \
    "$run_root/with_text/no_intent_memory/intention_best.pt" \
    "$run_root/without_text/no_intent_memory/intention_best.pt" \
  --data "$data" --episodes "$episodes" \
  --output "$run_root/paired_goal_handover" \
  --seeds 43 44 --action-horizon 1 --switch-at 0.5 \
  --rotation-convention libero --max-steps 300 --interventions normal \
  > "$run_root/paired_goal_handover.log" 2>&1

echo "$(date -Is) Completed paired LIBERO Goal handover"
