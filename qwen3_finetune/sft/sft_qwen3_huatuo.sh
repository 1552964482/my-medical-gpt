#!/bin/bash

# Qwen3-8B医疗模型SFT微调脚本（使用华佗GPT数据集）
# 使用FreedomIntelligence/HuatuoGPT-sft-data-v1数据集（22万条中文医疗对话）
# 优化配置：充分利用4张RTX 4080 32GB显存

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$QWEN3_ROOT/.." && pwd)"
cd "$QWEN3_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/qwen3-8b-dir}"
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$PROJECT_ROOT/data/finetune/huatuo_sft}"
VALIDATION_FILE_DIR="${VALIDATION_FILE_DIR:-$PROJECT_ROOT/data/finetune/huatuo_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-$QWEN3_ROOT/outputs-qwen3-sft-huatuo}"
CACHE_DIR="${CACHE_DIR:-$QWEN3_ROOT/cache}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-100000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-500}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun --nproc_per_node "$NPROC_PER_NODE" supervised_finetuning.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file_dir "$TRAIN_FILE_DIR" \
    --validation_file_dir "$VALIDATION_FILE_DIR" \
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
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 2000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --preprocessing_num_workers 64 \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir "$CACHE_DIR"
