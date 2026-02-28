#!/bin/bash

# Qwen3-8B医疗模型预训练脚本（使用Hugging Face数据集）
# 使用shibing624/medical数据集（240万条中文医疗数据）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$QWEN3_ROOT/.." && pwd)"
cd "$QWEN3_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/qwen3-8b-dir}"
DATASET_NAME="${DATASET_NAME:-shibing624/medical}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN3_ROOT/outputs-qwen3-pretrain-hf}"
CACHE_DIR="${CACHE_DIR:-$QWEN3_ROOT/cache}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun --nproc_per_node "$NPROC_PER_NODE" pretraining.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --dataset_config_name "$DATASET_CONFIG_NAME" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 1000 \
    --block_size 1024 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 1000 \
    --eval_strategy steps \
    --save_steps 5000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --preprocessing_num_workers 4 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir "$CACHE_DIR" \
    --flash_attn True
