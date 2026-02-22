#!/bin/bash

# Qwen3-8B医疗模型SFT微调脚本（使用华佗GPT数据集）
# 使用FreedomIntelligence/HuatuoGPT-sft-data-v1数据集（22万条中文医疗对话）

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 ../supervised_finetuning.py \
    --model_name_or_path ../models/qwen3-8b-dir \
    --dataset_name FreedomIntelligence/HuatuoGPT-sft-data-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 1000 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 2000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 4 \
    --output_dir ./outputs-qwen3-sft-huatuo \
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
