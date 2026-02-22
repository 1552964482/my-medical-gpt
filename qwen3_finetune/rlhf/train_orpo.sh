#!/bin/bash

# Qwen3-8B ORPO训练脚本
# 比值比偏好优化，无需参考模型

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 ../orpo_training.py \
    --model_name_or_path ./outputs-qwen3-sft-huatuo \
    --dataset_name shibing624/medical \
    --dataset_config_name orpo \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --max_train_samples -1 \
    --max_eval_samples 1000 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --min_target_length 16 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 4 \
    --output_dir ./outputs-qwen3-orpo \
    --overwrite_output_dir \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --lambda_orpo 0.1
