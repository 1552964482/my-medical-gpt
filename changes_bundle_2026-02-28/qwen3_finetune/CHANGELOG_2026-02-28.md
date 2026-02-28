# Qwen3 Finetune 改动整理（2026-02-28）

本次针对 `qwen3_finetune` 做了 4 轮工程化优化，目标是提升可维护性、可复现性和可配置能力。

## 1. 评测脚本工程化

### 1.1 `eval/compare_models.py`
- 新增命令行参数，移除硬编码路径。
- 修复 LoRA 加载方式：区分基础模型路径与 LoRA 路径。
- 支持输出目录参数化。
- 修复 CPU/CUDA 下设备与 dtype 选择问题。

### 1.2 `eval/comprehensive_evaluation.py`
- 新增命令行参数：
  - `--device`
  - `--num-samples`
  - `--max-new-tokens`
  - `--temperature`
  - `--top-p`
  - `--output-dir`
  - `--questions-file`
  - `--models-config`
  - `--qwen3-base-model`
  - `--qwen3-lora-path`
  - `--ziya-model-path`
- 支持从 `.txt/.json` 读取问题集。
- 支持从 JSON 文件加载模型配置。
- 修复生成耗时统计逻辑（按采样平均，而不是最后一次采样）。
- 未安装 `scipy` 时自动跳过显著性检验，避免程序直接失败。

### 1.3 `eval/README.md`
- 更新为参数化用法。
- 增加模型配置 JSON 示例。
- 增加问题文件方式说明。

### 1.4 新增文件
- `eval/test_questions.txt`（默认测试问题集）

## 2. Pipeline 脚本修复与去硬编码

### 2.1 `scripts/quick_start.sh` 与 `scripts/full_pipeline.sh`
- 删除 `sed` 修改源码行为（避免污染仓库）。
- 改为通过参数调用 `eval/compare_models.py`。
- 输出路径说明改为结果目录。

## 3. 新增统一配置入口（核心改动）

### 3.1 新增 `scripts/run_pipeline.py`
功能：
- 配置驱动执行多阶段流程（PT/SFT/RLHF/Eval）。
- 支持阶段控制：
  - `--from-stage`
  - `--to-stage`
  - `--only-stages`
- 支持变量覆盖：`--var KEY=VALUE`
- 支持 `--dry-run`
- 自动记录每阶段日志到 `qwen3_finetune/logs/pipeline/`
- 支持配置内变量模板替换（含内置变量与递归解析）
- `enabled=false` 的阶段可通过 `--only-stages` 临时强制执行

### 3.2 新增配置模板（可提交版本）
- `configs/pipeline_quick.json.example`
- `configs/pipeline_full.json.example`

说明：项目 `.gitignore` 全局忽略 `*.json`，因此模板采用 `.json.example`。

### 3.3 `qwen3_finetune/README.md`
- 新增“配置驱动 Pipeline（推荐）”章节。
- 增加 `.json.example -> .json` 的复制说明。
- 增加阶段执行示例（含执行默认关闭阶段）。

## 4. 核心训练脚本参数化（可由 Pipeline 注入）

以下脚本新增统一模式：
- `set -euo pipefail`
- 自动解析 `SCRIPT_DIR/QWEN3_ROOT/PROJECT_ROOT`
- 使用环境变量覆盖核心参数（模型路径、输出路径、batch、epoch 等）
- 默认值保持与原逻辑一致

### 已改造脚本
- `pretrain/pretrain_qwen3_hf.sh`
- `pretrain/pretrain_qwen3_single_gpu.sh`
- `sft/sft_qwen3_huatuo.sh`
- `sft/finetune_qwen3_single_gpu.sh`
- `rlhf/train_dpo.sh`
- `rlhf/train_grpo_single_gpu.sh`

## 5. 修复的关键问题

- 去除脚本中通过 `sed` 直接改 Python 源码的问题。
- 修复多处硬编码绝对路径问题。
- 修复 LoRA 路径与基础模型路径混用问题。
- 修复 CPU 环境下 CUDA API 调用崩溃问题。
- 修复 GRPO 单卡脚本中的不可维护命令写法（含 `python -m ../xxx.py` 与反引号注释）。

## 6. 现在推荐用法

```bash
cd /mnt/d/medical-gpt/my-medical-gpt/qwen3_finetune

# 复制模板
cp configs/pipeline_quick.json.example configs/pipeline_quick.json
cp configs/pipeline_full.json.example configs/pipeline_full.json

# 查看执行计划
python3 scripts/run_pipeline.py --config configs/pipeline_full.json --dry-run

# 运行快速流程
python3 scripts/run_pipeline.py --config configs/pipeline_quick.json

# 执行默认关闭阶段（如 DPO）
python3 scripts/run_pipeline.py --config configs/pipeline_full.json --only-stages rlhf_dpo
```

## 7. 涉及文件总览

- `qwen3_finetune/README.md`
- `qwen3_finetune/eval/README.md`
- `qwen3_finetune/eval/compare_models.py`
- `qwen3_finetune/eval/comprehensive_evaluation.py`
- `qwen3_finetune/eval/test_questions.txt`
- `qwen3_finetune/scripts/quick_start.sh`
- `qwen3_finetune/scripts/full_pipeline.sh`
- `qwen3_finetune/scripts/run_pipeline.py`
- `qwen3_finetune/configs/pipeline_quick.json.example`
- `qwen3_finetune/configs/pipeline_full.json.example`
- `qwen3_finetune/pretrain/pretrain_qwen3_hf.sh`
- `qwen3_finetune/pretrain/pretrain_qwen3_single_gpu.sh`
- `qwen3_finetune/sft/sft_qwen3_huatuo.sh`
- `qwen3_finetune/sft/finetune_qwen3_single_gpu.sh`
- `qwen3_finetune/rlhf/train_dpo.sh`
- `qwen3_finetune/rlhf/train_grpo_single_gpu.sh`
