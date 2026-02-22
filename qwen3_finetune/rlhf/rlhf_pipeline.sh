#!/bin/bash

# Qwen3-8B完整RLHF训练Pipeline
# 包含：SFT -> 奖励模型(RM) -> PPO强化学习
#
# 使用数据集：
# 1. SFT: FreedomIntelligence/HuatuoGPT-sft-data-v1 (22万条医疗对话)
# 2. RM/DPO/ORPO: shibing624/medical (偏好数据)

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Qwen3-8B RLHF训练Pipeline"
echo "=========================================="

# 配置
SFT_MODEL_PATH="./outputs-qwen3-sft-huatuo"
REWARD_MODEL_PATH="./outputs-qwen3-reward-model"
OUTPUT_DIR_PPO="./outputs-qwen3-ppo"

# 阶段1：检查SFT模型是否存在
echo ""
echo "=========================================="
echo "检查SFT模型..."
echo "=========================================="

if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "SFT模型不存在，请先运行SFT训练！"
    echo "运行: bash sft_qwen3_huatuo.sh"
    exit 1
fi

echo "SFT模型已准备好"
echo "路径: $SFT_MODEL_PATH"

# 选择强化学习方法
echo ""
echo "=========================================="
echo "选择强化学习方法："
echo "=========================================="
echo "1. PPO (传统RLHF，需要先训练奖励模型)"
echo "2. DPO (直接偏好优化，更简单，效果更好)"
echo "3. ORPO (比值比偏好优化，无需参考模型)"
echo ""
read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "阶段1：训练奖励模型(Reward Model)"
        echo "=========================================="

        bash train_reward_model_multi_gpu.sh

        # 检查奖励模型是否训练成功
        if [ $? -ne 0 ]; then
            echo "奖励模型训练失败！"
            exit 1
        fi

        echo "奖励模型训练完成！"
        echo "奖励模型路径: $REWARD_MODEL_PATH"

        # 阶段2：PPO强化学习
        echo ""
        echo "=========================================="
        echo "阶段2：PPO强化学习训练"
        echo "=========================================="

        bash train_ppo.sh

        # 检查PPO训练是否成功
        if [ $? -ne 0 ]; then
            echo "PPO训练失败！"
            exit 1
        fi

        echo "PPO训练完成！"
        echo "PPO模型路径: $OUTPUT_DIR_PPO"
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "DPO训练"
        echo "=========================================="

        bash train_dpo.sh

        # 检查DPO训练是否成功
        if [ $? -ne 0 ]; then
            echo "DPO训练失败！"
            exit 1
        fi

        echo "DPO训练完成！"
        echo "DPO模型路径: ./outputs-qwen3-dpo"
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "ORPO训练"
        echo "=========================================="

        bash train_orpo.sh

        # 检查ORPO训练是否成功
        if [ $? -ne 0 ]; then
            echo "ORPO训练失败！"
            exit 1
        fi

        echo "ORPO训练完成！"
        echo "ORPO模型路径: ./outputs-qwen3-orpo"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Pipeline执行完成！"
echo "=========================================="
echo "模型输出:"
case $choice in
    1)
        echo "  - 奖励模型: $REWARD_MODEL_PATH"
        echo "  - PPO模型: $OUTPUT_DIR_PPO"
        ;;
    2)
        echo "  - DPO模型: ./outputs-qwen3-dpo"
        ;;
    3)
        echo "  - ORPO模型: ./outputs-qwen3-orpo"
        ;;
esac
echo ""
