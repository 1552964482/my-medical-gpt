#!/bin/bash

# Qwen3-8B奖励模型训练脚本（单GPU版本）

CUDA_VISIBLE_DEVICES=0 python -m ../reward_modeling.py \
    --model_name_or_path ../models/qwen3-8b-dir \
    --dataset_name shibing624/medical \
    --dataset_config_name reward \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --max_train_samples -1 \
    --max_eval_samples 1000 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir ./outputs-qwen3-reward-model \
    --overwrite_output_dir \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache
