# GRPO训练指南

## 什么是GRPO？

GRPO (Group Relative Policy Optimization) 是DeepSeek提出的一种新型强化学习算法，用于大模型对齐。该算法发表于论文《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》。

### GRPO的核心优势

1. **无需参考模型** - 不需要维护一个冻结的策略网络作为参考
2. **无需奖励模型** - 直接通过成对比较计算奖励，不需要单独训练奖励模型
3. **计算效率高** - 相比PPO，显存和计算需求大幅降低
4. **训练稳定** - 通过组内相对比较，减少方差，训练更稳定
5. **效果优秀** - DeepSeek-Math证明GRPO在数学推理任务上表现优异

## GRPO vs 其他方法对比

| 方法 | 参考模型 | 奖励模型 | 显存需求 | 计算成本 | 推荐度 |
|-----|---------|---------|---------|---------|--------|
| **PPO** | 需要 | 需要 | 高(24-32GB) | 高 | ⭐⭐ |
| **DPO** | 需要 | 不需要 | 中(12-16GB) | 中 | ⭐⭐⭐⭐⭐ |
| **ORPO** | 不需要 | 不需要 | 中(12-16GB) | 中 | ⭐⭐⭐⭐ |
| **GRPO** | 不需要 | 不需要 | 低(8-12GB) | 低 | ⭐⭐⭐⭐⭐ |

## GRPO算法原理

### 核心思想

GRPO通过以下步骤优化策略：

1. **生成组** - 对每个prompt，模型生成多个response（默认4个）
2. **计算奖励** - 使用奖励函数评估每个response的质量
3. **相对优势** - 计算组内每个response相对于组内平均奖励的优势
4. **策略更新** - 使用相对优势更新策略参数

### 数学公式

对于每个prompt生成组 G = {y₁, y₂, ..., yₙ}，相对优势定义为：

```
A(x, y) = R(x, y) - mean(R(x, yᵢ) for yᵢ in G)
```

其中：
- A(x, y) 是相对优势
- R(x, y) 是奖励函数
- mean(R) 是组内平均奖励

策略更新使用类似PPO的损失函数：

```
L = -E[ min(r(x, y) * A(x, y), clip(r(x, y), 1-ε, 1+ε) * A(x, y) ) ]
```

其中 r(x, y) 是重要性比率。

## 使用方法

### 前置条件

确保已完成SFT微调：

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune
bash sft_qwen3_huatuo.sh
```

### 方案1：使用QLoRA版本（推荐，显存优化）

适合显存较小的GPU（16GB以下）：

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 2卡GPU
bash train_grpo_qlora.sh

# 或单GPU
CUDA_VISIBLE_DEVICES=0 bash train_grpo_qlora.sh
```

### 方案2：使用标准LoRA版本

适合显存充足的GPU（16GB以上）：

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 2卡GPU
bash train_grpo.sh

# 单GPU版本
bash train_grpo_single_gpu.sh
```

## 配置参数说明

### 核心参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | SFT模型路径 | `./outputs-qwen3-sft-huatuo` |
| `--dataset_name` | 数据集名称 | `shibing624/medical` |
| `--subset_name` | 数据集子集 | `grpo` |
| `--num_train_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 1.0e-6 (LoRA) / 5.0e-7 (QLoRA) |
| `--beta` | KL散度系数 | 0.001 |

### GRPO特定参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--num_generations` | 每个prompt生成的response数 | 4 |
| `--max_prompt_length` | 最大prompt长度 | 2048 |
| `--max_completion_length` | 最大response长度 | 512 |
| `--per_device_train_batch_size` | 每设备批次大小 | 1-4 |

### LoRA参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--use_peft` | 使用LoRA | True |
| `--qlora` | 使用QLoRA | False (LoRA) / True (QLoRA) |
| `--load_in_4bit` | 4bit量化 | False (LoRA) / True (QLoRA) |
| `--lora_r` | LoRA秩 | 16 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--lora_dropout` | LoRA dropout | 0.05 (LoRA) / 0.1 (QLoRA) |

## 数据集要求

### GRPO数据格式

GRPO需要包含以下字段的数据集：

```json
{
  "question": "用户问题",
  "answer": "标准答案"
}
```

### 医疗领域数据集

推荐使用的医疗数据集：

1. **shibing624/medical (grpo config)** - 医疗问答数据集
2. **FreedomIntelligence/HuatuoGPT-sft-data-v1** - 华佗GPT医疗对话
3. **本地医疗问答数据** - 自定义格式

### 奖励函数

GRPO使用奖励函数评估response质量，项目内置了：

1. **准确性奖励 (accuracy_reward)** - 检查答案是否正确
2. **格式奖励 (format_reward)** - 检查是否符合指定格式

可以自定义奖励函数以满足特定需求。

## 硬件要求

### 显存需求

| 配置 | 显存需求 | 推荐GPU |
|-----|---------|--------|
| GRPO LoRA (2卡) | 16-20GB | RTX 3090/4090 x2, A100 x2 |
| GRPO LoRA (单卡) | 20-24GB | RTX 3090/4090, A100 |
| **GRPO QLoRA (2卡)** | **8-12GB** | **RTX 3080/3090 x2** |
| **GRPO QLoRA (单卡)** | **10-14GB** | **RTX 3080/3090** |

### 训练时间估算

| 方法 | 单GPU | 2卡GPU |
|-----|-------|--------|
| **GRPO LoRA** | **6-10小时** | **3-5小时** |
| **GRPO QLoRA** | **8-12小时** | **4-6小时** |

## 完整训练流程示例

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 步骤1：SFT微调
bash sft_qwen3_huatuo.sh

# 步骤2：GRPO训练（推荐QLoRA版本）
bash train_grpo_qlora.sh

# 步骤3：模型对比
python compare_models.py
```

## 监控训练进度

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir outputs-qwen3-grpo/runs --port 6012
```

关键指标：
- `train/loss` - 训练损失
- `train/learning_rate` - 学习率变化
- `train/grad_norm` - 梯度范数
- `train/reward` - 奖励分数
- `train/accuracy` - 准确率

## 常见问题

### 1. GRPO和PPO有什么区别？

| 特性 | PPO | GRPO |
|-----|-----|------|
| 参考模型 | 需要 | 不需要 |
| 奖励模型 | 需要 | 不需要 |
| 显存需求 | 高(24-32GB) | 低(8-12GB) |
| 训练稳定性 | 中等 | 更稳定 |
| 计算成本 | 高 | 低 |
| 实现复杂度 | 复杂 | 简单 |

### 2. GRPO和DPO哪个更好？

两者各有优势：

- **GRPO优势**：
  - 无需参考模型
  - 显存需求更小
  - 训练速度更快
  - 适合需要生成多个response的场景

- **DPO优势**：
  - 方法更成熟，社区支持更好
  - 直接使用偏好对
  - 不需要设计奖励函数

**推荐**：
- 如果有明确的奖励函数（如数学、代码验证）→ **GRPO**
- 如果只有偏好对（chosen/rejected）→ **DPO**

### 3. 显存不足怎么办？

- 使用QLoRA版本：`--qlora True --load_in_4bit True`
- 减小批次大小：`--per_device_train_batch_size 1`
- 减少生成数量：`--num_generations 2`
- 增加梯度累积：`--gradient_accumulation_steps 16`

### 4. 如何调整学习率？

- 如果损失不下降：增大学习率（如 2e-6）
- 如果损失震荡：减小学习率（如 5e-7）
- 使用余弦学习率调度器效果通常更好

### 5. num_generations参数如何设置？

- 默认值：4
- 更大值（如6-8）：效果更好，但计算成本更高
- 更小值（如2）：训练更快，但可能效果略差

### 6. 如何自定义奖励函数？

在 `grpo_training.py` 中定义新的奖励函数：

```python
def custom_reward(completions, **kwargs):
    """自定义奖励函数"""
    rewards = []
    for completion in completions:
        # 计算奖励逻辑
        reward = ...
        rewards.append(reward)
    return rewards
```

## GRPO进阶技巧

### 1. 多阶段训练

```bash
# 第一阶段：宽松奖励，快速学习
bash train_grpo.sh  # beta=0.001

# 第二阶段：严格奖励，精细调优
bash train_grpo.sh  # beta=0.01, learning_rate=5e-7
```

### 2. 课程学习

从简单到困难的样本训练：

```bash
# 先训练简单样本
python grpo_training.py --train_samples 10000

# 再训练全部样本
python grpo_training.py --train_samples -1
```

### 3. 混合奖励

结合多种奖励函数：

```python
def mixed_reward(completions, answer, **kwargs):
    """混合奖励：准确性 + 格式 + 流畅性"""
    acc_rewards = accuracy_reward(completions, answer)
    fmt_rewards = format_reward(completions)
    flu_rewards = fluency_reward(completions)
    
    rewards = []
    for acc, fmt, flu in zip(acc_rewards, fmt_rewards, flu_rewards):
        rewards.append(0.7 * acc + 0.2 * fmt + 0.1 * flu)
    return rewards
```

### 4. 使用vLLM加速

如果安装了vLLM，可以加速生成：

```bash
--use_vllm True
```

## 性能优化建议

### 1. 显存优化

- 使用QLoRA（推荐）
- 启用梯度检查点：`--gradient_checkpointing True`
- 使用bfloat16：`--bf16 True`

### 2. 速度优化

- 使用多GPU训练
- 启用vLLM（如果可用）
- 增加预处理worker：`--preprocessing_num_workers 8`

### 3. 效果优化

- 适当增大 `num_generations`（如6）
- 调整 `beta` 参数平衡奖励和KL散度
- 使用质量更好的数据集

## 参考文献

1. DeepSeek-Math论文: https://arxiv.org/abs/2402.03300
2. GRPO算法: https://github.com/deepseek-ai/DeepSeek-Math
3. PPO算法: https://arxiv.org/abs/1707.06347
4. DPO算法: https://arxiv.org/abs/2305.18290

## 总结

GRPO是DeepSeek提出的优秀强化学习算法，特别适合医疗领域的大模型对齐：

**推荐使用GRPO的场景：**
- ✅ 显存资源有限
- ✅ 有明确的奖励函数
- ✅ 需要快速迭代
- ✅ 训练稳定性要求高

**完整流程：**
```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune
bash sft_qwen3_huatuo.sh      # SFT
bash train_grpo_qlora.sh      # GRPO
python compare_models.py      # 对比
```
