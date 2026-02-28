#!/bin/bash

# Qwen3-8B医疗模型微调脚本（单GPU版本）
# 适合显存较小的环境（24GB显存可运行）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$QWEN3_ROOT/.." && pwd)"
cd "$QWEN3_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/qwen3-8b-dir}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$PROJECT_ROOT/data/finetune}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-$PROJECT_ROOT/data/finetune}"
TRAIN_FILE_NAME="${TRAIN_FILE_NAME:-medical_sft_1K_format.jsonl}"
VALIDATION_FILE_NAME="${VALIDATION_FILE_NAME:-medical_sft_1K_format.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN3_ROOT/outputs-qwen3-medical}"
CACHE_DIR="${CACHE_DIR:-$QWEN3_ROOT/cache}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python3 supervised_finetuning.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file_dir "$TRAIN_FILE_DIR" \
    --validation_file_dir "$VALIDATION_FILE_DIR" \
    --train_file_name "$TRAIN_FILE_NAME" \
    --validation_file_name "$VALIDATION_FILE_NAME" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --model_max_length 2048 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
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
