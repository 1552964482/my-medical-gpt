#!/bin/bash

# Qwen3-8B医疗模型微调脚本（单GPU版本）
# 适合显存较小的环境（24GB显存可运行）

CUDA_VISIBLE_DEVICES=0 python -m supervised_finetuning.py \
    --model_name_or_path ../models/qwen3-8b-dir \
    --train_file_dir ../data/finetune \
    --validation_file_dir ../data/finetune \
    --train_file_name medical_sft_1K_format.jsonl \
    --validation_file_name medical_sft_1K_format.jsonl \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 100 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir ./outputs-qwen3-medical \
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
    --cache_dir ./cache \
    --flash_attn True
