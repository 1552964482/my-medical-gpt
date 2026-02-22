#!/bin/bash

# Qwen3-8B GRPO训练脚本（单GPU版本）

CUDA_VISIBLE_DEVICES=0 python -m ../grpo_training.py \
    --model_name_or_path ./outputs-qwen3-sft-huatuo \
    --dataset_name shibing624/medical \
    --subset_name grpo \
    --dataset_splits train \
    --train_samples -1 \
    --max_steps -1 --num_train_epochs 3 \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --output_dir ./outputs-qwen3-grpo \
    --torch_dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --beta 0.001 \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    --preprocessing_num_workers 4 \
    \
    `# LoRA配置` \
    --use_peft True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    \
    `# GRPO配置` \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 16 \
    --max_prompt_length 2048 \
    --max_completion_length 512
