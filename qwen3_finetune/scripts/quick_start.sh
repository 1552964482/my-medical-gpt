#!/bin/bash

# Qwen3-8B医疗模型快速开始脚本
# 适合小数据集快速测试

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Qwen3-8B医疗模型快速训练"
echo "=========================================="

# 使用本地小数据集进行快速测试
echo "阶段1：预训练（本地medical_corpus.txt）"
bash pretrain/pretrain_qwen3_single_gpu.sh

echo ""
echo "阶段2：SFT微调（本地medical_sft_1K_format.jsonl）"
bash sft/finetune_qwen3_single_gpu.sh

echo ""
echo "阶段3：模型对比"
python eval/compare_models.py \
    --qwen3-base-model ../models/qwen3-8b-dir \
    --qwen3-lora-path ./outputs-qwen3-medical/checkpoint-final \
    --output-dir ./eval/comparison_results

echo ""
echo "快速训练完成！"
echo "对比结果目录: ./eval/comparison_results"
