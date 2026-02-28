#!/bin/bash

# Qwen3-8B医疗模型完整训练Pipeline
# 包含：预训练(PT) -> 有监督微调(SFT) -> 模型对比
#
# 使用数据集：
# 1. 预训练：shibing624/medical (240万条医疗数据)
# 2. SFT微调：FreedomIntelligence/HuatuoGPT-sft-data-v1 (22万条医疗对话)

set -e  # 遇到错误立即退出

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Qwen3-8B医疗模型训练Pipeline"
echo "=========================================="

# 配置
PRETRAIN_DATA="shibing624/medical"
SFT_DATA="FreedomIntelligence/HuatuoGPT-sft-data-v1"
OUTPUT_DIR_PRETRAIN="./outputs-qwen3-pretrain"
OUTPUT_DIR_SFT="./outputs-qwen3-sft-full"

# 阶段1：预训练
echo ""
echo "=========================================="
echo "阶段1：预训练"
echo "数据集: $PRETRAIN_DATA"
echo "=========================================="

bash pretrain/pretrain_qwen3_hf.sh

# 检查预训练是否成功
if [ $? -ne 0 ]; then
    echo "预训练失败！"
    exit 1
fi

echo "预训练完成！"
echo "预训练输出目录: $OUTPUT_DIR_PRETRAIN"

# 阶段2：SFT微调
echo ""
echo "=========================================="
echo "阶段2：有监督微调"
echo "数据集: $SFT_DATA"
echo "=========================================="

# 使用预训练后的模型进行SFT
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 ../supervised_finetuning.py \
    --model_name_or_path $OUTPUT_DIR_PRETRAIN \
    --dataset_name $SFT_DATA \
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
    --output_dir $OUTPUT_DIR_SFT \
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

# 检查SFT是否成功
if [ $? -ne 0 ]; then
    echo "SFT微调失败！"
    exit 1
fi

echo "SFT微调完成！"
echo "SFT输出目录: $OUTPUT_DIR_SFT"

# 阶段3：模型对比
echo ""
echo "=========================================="
echo "阶段3：模型对比"
echo "对比Qwen3-8B微调模型 vs Ziya-13B-med"
echo "=========================================="

# 运行对比
python eval/compare_models.py \
    --qwen3-base-model "$OUTPUT_DIR_PRETRAIN" \
    --qwen3-lora-path "$OUTPUT_DIR_SFT/checkpoint-final" \
    --output-dir ./eval/comparison_results

echo ""
echo "=========================================="
echo "Pipeline执行完成！"
echo "=========================================="
echo "模型输出:"
echo "  - 预训练模型: $OUTPUT_DIR_PRETRAIN"
echo "  - SFT微调模型: $OUTPUT_DIR_SFT"
echo "  - 对比结果目录: ./eval/comparison_results"
echo ""
