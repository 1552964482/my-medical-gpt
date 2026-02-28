#!/bin/bash

# Qwen3-8B DPO训练脚本
# 直接偏好优化，无需单独的奖励模型

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN3_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MODEL_PATH="${MODEL_PATH:-$QWEN3_ROOT/outputs-qwen3-sft-huatuo}"
DATASET_NAME="${DATASET_NAME:-shibing624/medical}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-dpo}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN3_ROOT/outputs-qwen3-dpo}"
CACHE_DIR="${CACHE_DIR:-$QWEN3_ROOT/cache}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun --nproc_per_node "$NPROC_PER_NODE" dpo_training.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --dataset_config_name "$DATASET_CONFIG_NAME" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --do_train \
    --do_eval \
    --template_name qwen \
    --max_train_samples -1 \
    --max_eval_samples 1000 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --min_target_length 16 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --preprocessing_num_workers 4 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir "$CACHE_DIR"
