# 强化学习(RLHF)训练指南

## RLHF训练流程

强化学习(RLHF)包含两种主要方法：

### 方法1：传统RLHF (PPO)

**流程：**
1. **SFT微调** - 训练基础指令模型
2. **奖励模型(RM)** - 使用偏好数据训练奖励模型
3. **PPO强化学习** - 使用奖励模型优化策略

**优点：**
- 理论完善，基于经典强化学习
- 可解释性强

**缺点：**
- 需要训练额外的奖励模型
- 训练复杂，参数多
- 显存需求大
- 训练不稳定

### 方法2：DPO (直接偏好优化)

**流程：**
1. **SFT微调** - 训练基础指令模型
2. **DPO训练** - 直接优化偏好，无需奖励模型

**优点：**
- 无需训练额外的奖励模型
- 训练更稳定
- 显存需求小
- 效果通常比PPO更好

**缺点：**
- 需要偏好数据（chosen/rejected对）

### 方法3：ORPO (比值比偏好优化)

**流程：**
1. **SFT微调** - 训练基础指令模型
2. **ORPO训练** - 无参考模型的偏好优化

**优点：**
- 无需参考模型
- 同时进行SFT和对齐
- 缓解灾难性遗忘

**缺点：**
- 方法较新，经验相对较少

### 方法4：GRPO (Group Relative Policy Optimization)

**流程：**
1. **SFT微调** - 训练基础指令模型
2. **GRPO训练** - 无参考模型、无奖励模型的组相对策略优化

**优点：**
- 无需参考模型和奖励模型
- 显存需求最小（8-12GB）
- 计算成本最低
- 训练稳定，通过组内相对比较减少方差
- 效果优秀（DeepSeek-Math证明）

**缺点：**
- 需要设计奖励函数
- 需要生成多个response（计算量略大）

## 推荐方案

### 根据场景选择

| 场景 | 推荐方法 | 原因 |
|-----|---------|------|
| 有偏好对数据 | **DPO** | 简单直接，效果稳定 |
| 有奖励函数（如验证） | **GRPO** | 显存需求小，效率高 |
| 显存非常有限 | **GRPO QLoRA** | 8-12GB即可训练 |
| 追求最佳效果 | DPO 或 GRPO | 两者效果都优于PPO |
| 传统RLHF流程 | PPO | 理论完善，可解释性强 |

### 推荐使用GRPO（最新方法）

**原因：**
1. 无需参考模型和奖励模型，最简单
2. 显存需求最小（8-12GB QLoRA版本）
3. 训练速度快，计算成本低
4. 训练稳定，DeepSeek验证效果优秀
5. 适合资源有限的环境

## 使用方法

### 方案1：使用Pipeline脚本（推荐）

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 运行RLHF pipeline
bash rlhf_pipeline.sh
```

按提示选择强化学习方法：
- 选项1：PPO（传统RLHF）
- 选项2：DPO（推荐）
- 选项3：ORPO
- 选项4：GRPO（DeepSeek方法，最新推荐）

### 方案2：分步训练

#### 如果选择PPO

```bash
# 1. 训练奖励模型
bash train_reward_model_multi_gpu.sh

# 2. PPO强化学习
bash train_ppo.sh
```

#### 如果选择DPO（推荐）

```bash
# 直接DPO训练
bash train_dpo.sh
```

#### 如果选择ORPO

```bash
# 直接ORPO训练
bash train_orpo.sh
```

#### 如果选择GRPO（DeepSeek方法，最新推荐）

```bash
# 直接GRPO训练（多GPU）
bash train_grpo.sh

# 单GPU版本
bash train_grpo_single_gpu.sh

# QLoRA版本（显存优化，8-12GB即可）
bash train_grpo_qlora.sh
```

详细说明请查看 [GRPO_GUIDE.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/GRPO_GUIDE.md)

## 数据集要求

### 奖励模型训练数据

格式：
```json
{
  "question": "用户问题",
  "response_chosen": "更好的回答",
  "response_rejected": "较差的回答"
}
```

数据集：
- shibing624/medical (reward config) - 医疗偏好数据
- Dahoas/full-hh-rlhf - 英文HH-RLHF数据
- tasksource/oasst1_pairwise_rlhf_reward - OASST1偏好数据

### DPO/ORPO训练数据

格式与奖励模型相同，使用chosen/rejected对。

数据集：
- shibing624/medical (dpo config) - 医疗偏好数据
- tasksource/oasst1_pairwise_rlhf_reward - OASST1偏好数据

## 配置参数说明

### 奖励模型参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | 基础模型路径 | `../models/qwen3-8b-dir` |
| `--dataset_name` | 数据集名称 | `shibing624/medical` |
| `--dataset_config_name` | 数据集配置 | `reward` |
| `--num_train_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 2e-5 |
| `--per_device_train_batch_size` | 批次大小 | 2-4 |

### PPO参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--sft_model_path` | SFT模型路径 | `./outputs-qwen3-sft-huatuo` |
| `--reward_model_path` | 奖励模型路径 | `./outputs-qwen3-reward-model` |
| `--learning_rate` | 学习率 | 1.41e-5 |
| `--total_episodes` | 总episode数 | 100000 |
| `--max_source_length` | 最大prompt长度 | 1024 |
| `--max_target_length` | 最大response长度 | 256 |
| `--gradient_accumulation_steps` | 梯度累积 | 16 |

### DPO参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | SFT模型路径 | `./outputs-qwen3-sft-huatuo` |
| `--dataset_name` | 数据集名称 | `shibing624/medical` |
| `--dataset_config_name` | 数据集配置 | `dpo` |
| `--num_train_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 5e-6 |
| `--max_source_length` | 最大prompt长度 | 1024 |
| `--max_target_length` | 最大response长度 | 512 |

### ORPO参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | SFT模型路径 | `./outputs-qwen3-sft-huatuo` |
| `--dataset_name` | 数据集名称 | `shibing624/medical` |
| `--dataset_config_name` | 数据集配置 | `orpo` |
| `--num_train_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 1e-5 |
| `--lambda_orpo` | ORPO lambda | 0.1 |

## 硬件要求

| 训练方法 | 显存需求 | 推荐GPU |
|---------|---------|--------|
| 奖励模型 | 8-12GB | RTX 3090, A100 |
| PPO | 24-32GB | A100 40GB/80GB |
| DPO | 12-16GB | RTX 3090/4090, A100 |
| ORPO | 12-16GB | RTX 3090/4090, A100 |

## 训练时间估算

| 方法 | 单GPU时间 | 多GPU时间(2卡) |
|-----|----------|---------------|
| 奖励模型 | ~2-4小时 | ~1-2小时 |
| PPO | ~8-12小时 | ~4-6小时 |
| DPO | ~4-8小时 | ~2-4小时 |
| ORPO | ~4-8小时 | ~2-4小时 |

**总计（PPO方案）:**
- 奖励模型: ~1-2小时
- PPO: ~4-6小时
- 总计: ~5-8小时

**总计（DPO方案）:**
- DPO: ~2-4小时

## 常见问题

### 1. DPO训练比PPO好吗？

是的，通常DPO比PPO更好：
- 更稳定，不易发散
- 无需奖励模型，节省资源
- 训练更快
- 在多个基准上效果更好

### 2. 显存不足怎么办？

- 使用QLoRA：添加 `--load_in_8bit` 或 `--load_in_4bit`
- 减小批次大小
- 增加梯度累积
- 使用DPO而非PPO

### 3. 训练不稳定怎么办？

- 降低学习率
- 增加warmup比例
- 使用梯度裁剪
- 检查数据质量

### 4. 如何准备偏好数据？

可以使用GPT-4生成偏好数据：
1. 让GPT-4对同一问题生成多个回答
2. 人工或使用GPT-4评估哪个更好
3. 保存为chosen/rejected对

## 监控训练进度

使用TensorBoard查看训练进度：

```bash
# 奖励模型
tensorboard --logdir outputs-qwen3-reward-model/runs --port 6008

# PPO
tensorboard --logdir outputs-qwen3-ppo/runs --port 6009

# DPO
tensorboard --logdir outputs-qwen3-dpo/runs --port 6010

# ORPO
tensorboard --logdir outputs-qwen3-orpo/runs --port 6011
```

## 完整训练流程示例

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 1. SFT微调
bash sft_qwen3_huatuo.sh

# 2. DPO训练（推荐）
bash train_dpo.sh

# 3. 模型对比
python compare_models.py
```

## 参考文献

- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- DPO: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
- ORPO: "ORPO: Monolithic Preference Optimization without Reference Model" (Hong et al., 2024)
