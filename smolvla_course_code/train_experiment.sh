#!/usr/bin/env bash
set -euo pipefail
cd "$START_ROOT"
mkdir -p "$START_ROOT/logs" "$START_ROOT/runs"
steps="${STEPS:-1000}"
lr="${LR:-0.0001}"
lora_r="${LORA_R:-8}"
batch="${BATCH_SIZE:-1}"
episodes_file="${EPISODES_FILE:-episodes_smoke.json}"
name="${EXPERIMENT_NAME:-tuned}"
warmup="${WARMUP_STEPS:-10}"
decay_lr="${DECAY_LR:-$(python -c 'import sys; print(float(sys.argv[1])*0.1)' "$lr")}"
python - "$steps" "$lr" "$lora_r" "$batch" "$warmup" "$name" <<'PY'
import math,re,sys
steps,lr,rank,batch,warmup,name=sys.argv[1:]
assert int(steps)>int(warmup)>=0
assert math.isfinite(float(lr)) and float(lr)>0 and int(rank)>0 and int(batch)>0
assert re.fullmatch(r'[A-Za-z0-9_-]+',name)
PY
python prepare_training_subset.py "$episodes_file"
episodes="$(cat "$episodes_file")"
revision="$(cat dataset_revision.txt)"
run_dir="$START_ROOT/runs/experiment_${name}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null --policy.output_features=null \
  --policy.device=cuda --policy.use_amp=false --policy.compile_model=false \
  --policy.push_to_hub=false --policy.optimizer_lr="$lr" \
  --policy.scheduler_warmup_steps="$warmup" --policy.scheduler_decay_steps="$steps" \
  --policy.scheduler_decay_lr="$decay_lr" \
  --dataset.repo_id=lerobot/libero --dataset.root="$START_ROOT/data/libero" \
  --dataset.revision="$revision" --dataset.episodes="$episodes" \
  --dataset.video_backend=pyav --dataset.image_transforms.enable=false \
  --peft.method_type=LORA --peft.r="$lora_r" \
  --batch_size="$batch" --num_workers=0 --steps="$steps" --seed=42 \
  --eval_freq=0 --log_freq=10 --save_freq="$steps" --wandb.enable=false \
  --output_dir="$run_dir" --job_name="$name" \
  2>&1 | tee "$START_ROOT/logs/experiment_${name}_$(date +%Y%m%d_%H%M%S).log"
checkpoint="$run_dir/checkpoints/last/pretrained_model"
test -f "$checkpoint/config.json"
test -f "$checkpoint/adapter_model.safetensors"
python - "$checkpoint" "$run_dir" "$steps" "$lr" "$lora_r" "$batch" "$warmup" "$decay_lr" "$episodes_file" "$revision" <<'PY'
import json,sys
from pathlib import Path
checkpoint,run,steps,lr,rank,batch,warmup,decay,episodes_file,revision=sys.argv[1:]
p=Path(checkpoint)
assert json.loads((p/'config.json').read_text())['use_peft'] is True
assert json.loads((p/'adapter_config.json').read_text())['base_model_name_or_path']=='lerobot/smolvla_base'
assert (p/'policy_preprocessor.json').is_file() and (p/'policy_postprocessor.json').is_file()
recipe={'base_model':'lerobot/smolvla_base','dataset':'lerobot/libero',
        'dataset_revision':revision,'task_index':2,'episodes':json.loads(Path(episodes_file).read_text()),
        'steps':int(steps),'lr':float(lr),'lora_r':int(rank),'batch_size':int(batch),
        'warmup_steps':int(warmup),'decay_lr':float(decay),'seed':42,
        'amp':False,'compile':False,'checkpoint':checkpoint,
        'training_visual_augmentation':False}
(Path(run)/'training_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
print('Adapter, processors and training recipe OK')
PY
printf '%s\n' "$checkpoint" > "$run_dir/checkpoint_path.txt"
printf '%s\n' "$checkpoint" > "$START_ROOT/last_experiment_checkpoint.txt"
printf 'EXPERIMENT_CHECKPOINT: %s\n' "$checkpoint"
# Preserve the original baseline pointer; evaluate this checkpoint explicitly.
