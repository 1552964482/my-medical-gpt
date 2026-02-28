#!/bin/bash

# Qwen3-8B医疗模型预训练脚本（单GPU版本）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$QWEN3_ROOT/.." && pwd)"
cd "$QWEN3_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/qwen3-8b-dir}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$PROJECT_ROOT/data/rag}"
TRAIN_FILE_NAME="${TRAIN_FILE_NAME:-medical_corpus.txt}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-$PROJECT_ROOT/data/rag}"
VALIDATION_FILE_NAME="${VALIDATION_FILE_NAME:-medical_corpus.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN3_ROOT/outputs-qwen3-pretrain}"
CACHE_DIR="${CACHE_DIR:-$QWEN3_ROOT/cache}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python3 pretraining.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file_dir "$TRAIN_FILE_DIR" \
    --train_file_name "$TRAIN_FILE_NAME" \
    --validation_file_dir "$VALIDATION_FILE_DIR" \
    --validation_file_name "$VALIDATION_FILE_NAME" \
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
