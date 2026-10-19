#!/usr/bin/env bash
set -euo pipefail
cd "$START_ROOT"
checkpoint="${CHECKPOINT:-}"
if [[ -z "$checkpoint" ]]; then
  if [[ -f last_checkpoint.txt ]]; then
    checkpoint="$(cat last_checkpoint.txt)"
  elif [[ -f last_baseline_checkpoint.txt ]]; then
    checkpoint="$(cat last_baseline_checkpoint.txt)"
  else
    printf '请明确指定 CHECKPOINT，或先准备基础模型仿真入口。\n' >&2
    exit 1
  fi
fi
test -f "$checkpoint/config.json"
mkdir -p "$START_ROOT/runs" "$START_ROOT/logs"
run_dir="$START_ROOT/runs/eval_clean_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
printf 'INFERENCE_CHECKPOINT: %s\n' "$checkpoint"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$START_ROOT/.venv/bin/python" -m lerobot.scripts.lerobot_eval \
  --policy.path="$checkpoint" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.compile_model=false \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[9]' \
  --env.episode_length=500 \
  --env.init_states=true \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --seed=42 \
  --output_dir="$run_dir" \
  2>&1 | tee "$START_ROOT/logs/eval_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$checkpoint" > "$run_dir/inference_checkpoint.txt"
printf '%s\n' "$run_dir" > "$START_ROOT/last_inference_result.txt"
