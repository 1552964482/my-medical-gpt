#!/bin/bash

# Qwen3-8B PPO强化学习训练脚本
# 使用奖励模型进行PPO强化学习训练

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 ../ppo_training.py \
    --sft_model_path ./outputs-qwen3-sft-huatuo \
    --reward_model_path ./outputs-qwen3-reward-model \
    --dataset_name shibing624/medical \
    --dataset_config_name rlhf \
    --template_name qwen \
    --learning_rate 1.41e-5 \
    --total_episodes 100000 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --min_target_length 16 \
    --max_train_samples -1 \
    --max_eval_samples 100 \
    --max_ppo_epochs 4 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ./outputs-qwen3-ppo \
    --log_with wandb \
    --save_steps 5000 \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto
