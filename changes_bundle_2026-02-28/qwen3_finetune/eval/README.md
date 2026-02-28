# 模型评测工具

本目录包含多种模型评测工具，从简单对比到综合量化评测。

## 评测工具对比

| 工具 | 优势 | 适用场景 | 输出 |
|-----|--------|---------|------|
| **comprehensive_evaluation.py** | 多维度量化、统计检验、可视化 | 正式对比、论文研究 | 评分、图表、统计报告 |
| **compare_models.py** | 快速对比、易于理解 | 快速测试、初步筛选 | 对比结果JSON |
| **compare_models_cpu.py** | 无需GPU | CPU环境测试 | 对比结果JSON |
| **test_single_model.py** | 交互式、灵活调试 | 单模型测试、调试 | 交互式对话 |

## 快速开始

### 1. 综合评测（推荐）

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 运行综合评测
python eval/comprehensive_evaluation.py \
  --qwen3-base-model ../models/qwen3-8b-dir \
  --qwen3-lora-path ./outputs-qwen3-sft-huatuo/checkpoint-final \
  --ziya-model-path ../models/ziya-13b-med \
  --output-dir ./eval/evaluation_results
```

**输出：**
- 控制台摘要（各维度分数）
- 统计显著性检验
- 可视化图表（雷达图、柱状图、箱线图）
- 详细结果JSON

### 2. 快速对比

```bash
# GPU版本
python eval/compare_models.py \
  --qwen3-base-model ../models/qwen3-8b-dir \
  --qwen3-lora-path ./outputs-qwen3-sft-huatuo/checkpoint-final \
  --output-dir ./eval/comparison_results

# CPU版本
python eval/compare_models_cpu.py
```

### 3. 单模型测试

```bash
# 交互式测试
python eval/test_single_model.py \
    --model_path ../models/qwen3-8b-dir \
    --lora_path ./outputs-qwen3-sft-huatuo/checkpoint-final \
    --interactive

# 单个问题
python eval/test_single_model.py \
    --model_path ../models/qwen3-8b-dir \
    --question "糖尿病的常见并发症有哪些？"
```

## 综合评测详解

### 评测维度

| 维度 | 权重 | 说明 |
|-----|--------|------|
| 准确性 (Accuracy) | 20% | 医疗知识准确性、是否针对问题 |
| 完整性 (Completeness) | 20% | 回答全面性、结构化程度 |
| 安全性 (Safety) | 20% | 是否包含免责声明、建议就医 |
| 相关性 (Relevance) | 20% | 与问题的相关性、关键词匹配 |
| 可读性 (Readability) | 20% | 语言表达、术语使用 |

### 输出文件

运行 `comprehensive_evaluation.py` 后，在 `evaluation_results/` 目录生成：

```
evaluation_results/
├── radar_chart.png        # 雷达图 - 多维度对比
├── overall_scores.png    # 柱状图 - 总分对比
├── boxplots.png         # 箱线图 - 分数分布
└── detailed_results.json  # 详细结果数据
```

### 评分解读

**总体评分 (Overall Score):**
- 0.8-1.0: 优秀
- 0.6-0.8: 良好
- 0.4-0.6: 一般
- 0.0-0.4: 需要改进

**各维度分数:**
- **准确性**: 是否包含正确的医疗知识
- **完整性**: 回答是否全面、有条理
- **安全性**: 是否有必要的医疗免责声明
- **相关性**: 是否针对问题、不跑题
- **可读性**: 语言是否清晰、易于理解

## 配置说明

### 模型路径配置

通过命令行参数传入（推荐）：

```bash
python eval/comprehensive_evaluation.py \
  --qwen3-base-model ../models/qwen3-8b-dir \
  --qwen3-lora-path ./outputs-qwen3-sft-huatuo/checkpoint-final \
  --ziya-model-path ../models/ziya-13b-med
```

或使用 JSON 配置文件：

```json
{
  "Qwen3-8B-FT": {
    "model_path": "../models/qwen3-8b-dir",
    "lora_path": "./outputs-qwen3-sft-huatuo/checkpoint-final"
  },
  "Ziya-13B-med": {
    "model_path": "../models/ziya-13b-med",
    "lora_path": null
  }
}
```

```bash
python eval/comprehensive_evaluation.py --models-config ./eval/models_config.json
```

### 评测参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `device` | 设备选择 | auto |
| `num_samples` | 每个问题生成次数 | 3 |
| `max_new_tokens` | 最大生成长度 | 512 |
| `temperature` | 生成温度 | 0.7 |
| `top_p` | 采样Top-p | 0.9 |
| `questions_file` | 自定义问题文件(.txt/.json) | 空（使用内置问题） |

## 自定义评测

### 修改测试问题

推荐使用问题文件（`.txt` 或 `.json`）：

```text
糖尿病的常见并发症有哪些？
高血压患者日常生活中需要注意什么？
```

```bash
python eval/comprehensive_evaluation.py --questions-file ./eval/test_questions.txt
```

### 调整评分权重

修改 `evaluate_model()` 方法中的总体分数计算：

```python
# 当前：等权重平均
overall = np.mean(list(scores.values()))

# 改为：加权平均
overall = (
    0.3 * scores['accuracy'] +     # 提高准确性权重
    0.25 * scores['completeness'] +
    0.2 * scores['safety'] +
    0.15 * scores['relevance'] +
    0.1 * scores['readability']
)
```

## 可视化说明

### 雷达图
- 展示模型在多个维度上的综合表现
- 面积越大表示整体表现越好
- 适合快速对比多维度性能

### 总分柱状图
- 对比模型的总体评分
- 误差棒表示分数的标准差
- 显示与基准线（0.5）的差距

### 箱线图
- 展示各维度分数的分布情况
- 显示中位数、四分位数、极值
- 判断模型稳定性

## 常见问题

### 1. 如何确保评测公平？

- 使用相同的测试集
- 使用相同的生成参数
- 多次采样取平均
- 进行统计显著性检验

### 2. 评分可信吗？

当前评测特点：
- ✅ 多维度评估（5个维度）
- ✅ 可量化、可重复
- ✅ 统计显著性检验
- ✅ 多次采样减少偶然性

局限性：
- ⚠️ 基于规则，可能不完全符合人工判断
- ⚠️ 需要根据具体领域调优

建议：
- 结合人工评估
- 使用多个评测方法交叉验证
- 定期更新评分规则

### 3. 显存不足怎么办？

- 使用CPU版本：`compare_models_cpu.py`
- 减小 `max_new_tokens`
- 使用量化模型

### 4. 如何添加新的评测维度？

1. 在 `MedicalEvaluator` 类中添加新的评估方法：

```python
def evaluate_your_dimension(self, question: str, answer: str) -> float:
    # 实现你的评估逻辑
    score = ...
    return min(score, 1.0)
```

2. 在 `evaluate()` 方法中调用：

```python
def evaluate(self, question: str, answer: str) -> Dict[str, float]:
    return {
        'accuracy': self.evaluate_accuracy(question, answer),
        # ... 其他维度
        'your_dimension': self.evaluate_your_dimension(question, answer)
    }
```

## 进阶评测

详见 [EVALUATION_GUIDE.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/eval/EVALUATION_GUIDE.md)

包含：
- 基于GPT-4的评测
- 人工评测流程
- A/B测试方法
- 专业医学模型评测

## 参考资料

- 详细评测指南: [EVALUATION_GUIDE.md](file:///root/autodl-tmp/my-medical-gpt/qwen3_finetune/eval/EVALUATION_GUIDE.md)
- LLM评测方法: https://arxiv.org/abs/2303.18258
- 医疗NLP评测: https://www.nature.com/articles/s41597-021-0131-8
