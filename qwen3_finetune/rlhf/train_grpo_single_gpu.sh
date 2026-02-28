#!/bin/bash

# Qwen3-8B GRPO训练脚本（单GPU版本）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN3_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-$QWEN3_ROOT/outputs-qwen3-sft-huatuo}"
DATASET_NAME="${DATASET_NAME:-shibing624/medical}"
SUBSET_NAME="${SUBSET_NAME:-grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN3_ROOT/outputs-qwen3-grpo}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python3 grpo_training.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --subset_name "$SUBSET_NAME" \
    --dataset_splits train \
    --train_samples -1 \
    --max_steps -1 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --output_dir "$OUTPUT_DIR" \
    --torch_dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --beta 0.001 \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    --preprocessing_num_workers 4 \
    --use_peft True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --num_generations 4 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_prompt_length 2048 \
    --max_completion_length 512
