# Qwen3 医疗模型完整训练Pipeline

本目录包含Qwen3-8B医疗模型从预训练到微调的完整训练流程，以及与Ziya-13B-med模型的对比。

## 目录结构

```
qwen3_finetune/
├── README.md                           # 本文件
├── GRPO_GUIDE.md                       # GRPO详细指南
├── RLHF_GUIDE.md                       # 强化学习指南
│
├── pretrain/                           # 预训练脚本
│   ├── pretrain_qwen3.sh               # 预训练脚本（本地数据）
│   ├── pretrain_qwen3_single_gpu.sh    # 预训练脚本（单GPU）
│   └── pretrain_qwen3_hf.sh           # 预训练脚本（HF数据集）
│
├── sft/                                # SFT微调脚本
│   ├── finetune_qwen3.sh              # SFT微调脚本
│   ├── finetune_qwen3_single_gpu.sh   # SFT微调脚本（单GPU）
│   ├── finetune_qwen3_multi_gpu.sh    # SFT微调脚本（多GPU）
│   ├── sft_qwen3_large.sh            # SFT微调脚本（大数据集）
│   └── sft_qwen3_huatuo.sh          # SFT微调脚本（华佗数据集）
│
├── rlhf/                               # 强化学习脚本
│   ├── rlhf_pipeline.sh               # RLHF完整Pipeline脚本
│   ├── train_reward_model.sh          # 奖励模型训练脚本
│   ├── train_reward_model_single_gpu.sh # 奖励模型训练脚本（单GPU）
│   ├── train_ppo.sh                  # PPO强化学习脚本
│   ├── train_dpo.sh                  # DPO训练脚本
│   ├── train_orpo.sh                 # ORPO训练脚本
│   ├── train_grpo.sh                 # GRPO训练脚本（DeepSeek）
│   ├── train_grpo_single_gpu.sh      # GRPO训练脚本（单GPU）
│   └── train_grpo_qlora.sh           # GRPO QLoRA训练脚本（显存优化）
│
├── data/                               # 数据下载工具
│   ├── download_data.py               # 数据下载工具
│   └── download_wikipedia.py          # 维基百科下载工具
│
├── eval/                               # 评估和测试脚本
│   ├── README.md                      # 评测工具说明
│   ├── EVALUATION_GUIDE.md           # 评测详细指南
│   ├── comprehensive_evaluation.py     # 综合评测脚本（推荐）
│   ├── compare_models.py              # GPU模式模型对比脚本
│   ├── compare_models_cpu.py          # CPU模式模型对比脚本
│   └── test_single_model.py          # 单模型测试脚本
│
├── scripts/                            # Pipeline脚本
│   ├── full_pipeline.sh               # 完整训练Pipeline
│   └── quick_start.sh                # 快速开始脚本
│
└── outputs-*/                          # 训练输出目录
    ├── outputs-qwen3-pretrain/        # 预训练输出目录
    ├── outputs-qwen3-sft-*/         # SFT微调输出目录
    ├── outputs-qwen3-reward-model/   # 奖励模型输出目录
    ├── outputs-qwen3-ppo/           # PPO模型输出目录
    ├── outputs-qwen3-dpo/           # DPO模型输出目录
    ├── outputs-qwen3-orpo/          # ORPO模型输出目录
    └── outputs-qwen3-grpo/         # GRPO模型输出目录
```

## 训练流程

### 方式1：完整训练Pipeline（推荐）

使用大数据集进行完整的预训练+微调流程：

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune
bash scripts/full_pipeline.sh
```

**完整流程包含：**
1. **预训练(PT)** - 使用240万条医疗数据进行增量预训练
2. **SFT微调** - 使用22万条华佗医疗对话数据进行指令微调
3. **模型对比** - 对比训练后的Qwen3与Ziya-13B-med模型

### 方式2：快速开始

使用本地小数据集进行快速测试：

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune
bash scripts/quick_start.sh
```

### 方式3：分步训练

#### 阶段1：预训练（可选）

**选项A：使用本地数据（快速测试）**
```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune
bash pretrain/pretrain_qwen3.sh
```

**选项B：使用单GPU**
```bash
bash pretrain/pretrain_qwen3_single_gpu.sh
```

**选项C：使用Hugging Face大数据集**
```bash
bash pretrain/pretrain_qwen3_hf.sh
```

#### 阶段2：SFT微调

**选项A：使用本地数据（1K条）**
```bash
bash sft/finetune_qwen3_multi_gpu.sh
```

**选项B：使用shibing624/medical数据集（240万条）**
```bash
bash sft/sft_qwen3_large.sh
```

**选项C：使用华佗GPT数据集（22万条）**
```bash
bash sft/sft_qwen3_huatuo.sh
```

#### 阶段3：强化学习(RLHF) - 可选

在SFT微调之后，可以使用强化学习进一步提升模型性能：

**推荐使用GRPO（DeepSeek方法）**

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 使用RLHF Pipeline脚本
bash rlhf/rlhf_pipeline.sh
```

按提示选择强化学习方法：
- 选项1：PPO（传统RLHF）
- 选项2：DPO（推荐）
- 选项3：ORPO
- 选项4：GRPO（DeepSeek方法，最新推荐）

**或分步执行：**

```bash
# 方法1：PPO（需要奖励模型）
bash rlhf/train_reward_model.sh      # 1. 训练奖励模型
bash rlhf/train_ppo.sh                # 2. PPO强化学习

# 方法2：DPO（推荐）
bash rlhf/train_dpo.sh                # 直接DPO训练

# 方法3：ORPO
bash rlhf/train_orpo.sh               # 直接ORPO训练

# 方法4：GRPO（DeepSeek方法，推荐）
bash rlhf/train_grpo.sh               # 直接GRPO训练（多GPU）
bash rlhf/train_grpo_single_gpu.sh   # 单GPU版本
bash rlhf/train_grpo_qlora.sh        # QLoRA版本（显存优化）
```

详细说明请查看 [RLHF_GUIDE.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/RLHF_GUIDE.md) 和 [GRPO_GUIDE.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/GRPO_GUIDE.md)

#### 阶段4：模型对比

**方式1：综合评测（推荐）**

提供多维度、量化的专业评测：

```bash
# 运行综合评测
python eval/comprehensive_evaluation.py
```

**输出内容：**
- 多维度评分：准确性、完整性、安全性、相关性、可读性
- 统计显著性检验：Mann-Whitney U test
- 可视化图表：雷达图、柱状图、箱线图
- 详细结果JSON：包含所有样本的详细评分

**方式2：简单对比**

```bash
# GPU版本
python eval/compare_models.py

# CPU版本
python eval/compare_models_cpu.py
```

**方式3：单模型测试**

```bash
# 交互式测试
python eval/test_single_model.py --model_path ../models/qwen3-8b-dir \
    --lora_path ./outputs-qwen3-sft-huatuo/checkpoint-final \
    --interactive

# 单个问题
python eval/test_single_model.py --model_path ../models/qwen3-8b-dir \
    --question "糖尿病的常见并发症有哪些？"
```

详细说明请查看 [eval/README.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/eval/README.md) 和 [eval/EVALUATION_GUIDE.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/eval/EVALUATION_GUIDE.md)

## 数据集说明

### 可用数据集

| 数据集 | 大小 | 类型 | 用途 | 来源 |
|-------|------|------|------|------|
| medical_corpus.txt | 本地 | 预训练 | 快速测试 | 本地 |
| medical_sft_1K_format.jsonl | 1K | SFT | 快速测试 | 本地 |
| shibing624/medical (pretrain) | 240万 | 预训练 | 正式训练 | Hugging Face |
| shibing624/medical (sft) | 240万 | SFT | 正式训练 | Hugging Face |
| FreedomIntelligence/HuatuoGPT-sft-data-v1 | 22万 | SFT | 正式训练 | Hugging Face |
| BelleGroup/train_0.5M_CN | 50万 | SFT | 通用指令 | Hugging Face |
| BelleGroup/train_1M_CN | 100万 | SFT | 通用指令 | Hugging Face |

### 下载数据集

使用数据下载工具：

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune
python data/download_data.py
```

按提示选择要下载的数据集。

## 前置要求

### 1. 硬件要求

| 训练模式 | 显存需求 | 推荐GPU |
|---------|---------|--------|
| 单GPU LoRA | 16-24GB | RTX 3090/4090, A100 |
| 多GPU LoRA | 每卡8-12GB | RTX 3090x2, A100x2 |
| QLoRA (8bit) | 10-12GB | RTX 3080/3090 |
| QLoRA (4bit) | 6-8GB | RTX 3060/3070 |

### 2. 软件依赖

确保已安装以下依赖：

```bash
cd /root/autodl-tmp/my-medical-gpt
pip install -r requirements.txt
```

关键依赖：
- PyTorch >= 2.0.0
- Transformers >= 4.49.0
- PEFT >= 0.14.0
- Datasets >= 2.14.6
- Accelerate
- BitsAndBytes (可选，用于量化)

## 配置参数说明

### 预训练参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | 基础模型路径 | `../models/qwen3-8b-dir` |
| `--dataset_name` | 数据集名称(HF) | `shibing624/medical` |
| `--train_file_dir` | 本地数据目录 | `../data/rag` |
| `--train_file_name` | 本地数据文件 | `medical_corpus.txt` |
| `--num_train_epochs` | 训练轮数 | 1 |
| `--learning_rate` | 学习率 | 5e-5 |
| `--block_size` | 序列块大小 | 1024 |
| `--per_device_train_batch_size` | 每设备批次 | 2-4 |
| `--gradient_accumulation_steps` | 梯度累积 | 4-8 |
| `--lora_rank` | LoRA秩 | 16 |
| `--lora_alpha` | LoRA alpha | 32 |

### SFT微调参数

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | 基础模型路径 | `../models/qwen3-8b-dir` |
| `--dataset_name` | 数据集名称(HF) | `FreedomIntelligence/HuatuoGPT-sft-data-v1` |
| `--template_name` | 提示模板 | `qwen` |
| `--num_train_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 2e-5 |
| `--model_max_length` | 最大序列长度 | 2048 |
| `--max_train_samples` | 训练样本数 | -1(全部) |
| `--per_device_train_batch_size` | 每设备批次 | 2-4 |
| `--gradient_accumulation_steps` | 梯度累积 | 4-8 |
| `--lora_rank` | LoRA秩 | 16 |
| `--lora_alpha` | LoRA alpha | 32 |

### RLHF参数

#### GRPO参数（推荐）

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `--model_name_or_path` | SFT模型路径 | `./outputs-qwen3-sft-huatuo` |
| `--num_train_epochs` | 训练轮数 | 3 |
| `--learning_rate` | 学习率 | 1.0e-6 (LoRA) / 5.0e-7 (QLoRA) |
| `--beta` | KL散度系数 | 0.001 |
| `--num_generations` | 每个prompt生成数 | 4 |
| `--max_prompt_length` | 最大prompt长度 | 2048 |
| `--max_completion_length` | 最大response长度 | 512 |

## 快速参考

### 预训练

```bash
# 本地小数据集
bash pretrain/pretrain_qwen3.sh

# HF大数据集
bash pretrain/pretrain_qwen3_hf.sh
```

### SFT微调

```bash
# 华佗数据集（推荐）
bash sft/sft_qwen3_huatuo.sh

# 大数据集
bash sft/sft_qwen3_large.sh
```

### 强化学习

```bash
# GRPO（推荐）
bash rlhf/train_grpo_qlora.sh

# DPO
bash rlhf/train_dpo.sh

# PPO（需要奖励模型）
bash rlhf/train_reward_model.sh
bash rlhf/train_ppo.sh
```

### 模型评估

```bash
# 对比模型
python eval/compare_models.py

# 测试单个模型
python eval/test_single_model.py
```

## 常见问题

### 1. 如何选择训练方法？

根据你的需求和资源选择：

| 场景 | 推荐方法 |
|-----|---------|
| 快速测试 | pretrain → SFT（小数据集） |
| 正式训练 | pretrain → SFT（大数据集） |
| 资源有限 | SFT + GRPO QLoRA |
| 追求效果 | pretrain → SFT + DPO/GRPO |

### 2. 显存不足怎么办？

- 使用QLoRA：添加 `--load_in_4bit` 或 `--load_in_8bit`
- 减小批次大小
- 增加梯度累积
- 使用GRPO而非PPO

### 3. 训练时间估算

| 阶段 | 单GPU | 2卡GPU |
|-----|-------|--------|
| 预训练（1 epoch） | 12-20小时 | 6-10小时 |
| SFT微调（3 epochs） | 8-15小时 | 4-8小时 |
| GRPO训练 | 8-12小时 | 4-6小时 |
| DPO训练 | 4-8小时 | 2-4小时 |

### 4. 如何监控训练进度？

使用TensorBoard：

```bash
# 预训练
tensorboard --logdir outputs-qwen3-pretrain/runs --port 6000

# SFT
tensorboard --logdir outputs-qwen3-sft-huatuo/runs --port 6001

# GRPO
tensorboard --logdir outputs-qwen3-grpo/runs --port 6012
```

## 详细文档

- [RLHF指南](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/RLHF_GUIDE.md) - 强化学习方法详解
- [GRPO指南](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/GRPO_GUIDE.md) - GRPO算法详解

## 参考资料

- Qwen3模型：https://huggingface.co/Qwen/Qwen3
- DeepSeek-Math论文：https://arxiv.org/abs/2402.03300
- 华佗GPT：https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1
