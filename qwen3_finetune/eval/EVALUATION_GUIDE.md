# 模型评测指南

## 评测方法对比

| 评测方式 | 优势 | 劣势 | 适用场景 |
|---------|--------|--------|---------|
| **综合评测** | 多维度量化、统计显著性、可视化报告 | 基于规则，需要调优 | 正式模型对比、论文发表 |
| **简单对比** | 快速直观、易于理解 | 主观性强、缺乏量化 | 快速测试、初步筛选 |
| **交互测试** | 灵活、实时反馈 | 难以对比、无法量化 | 调试、demo展示 |

## 综合评测指标

### 1. 准确性 (Accuracy)
**评估内容：**
- 医疗关键词覆盖度
- 是否针对问题回答
- 是否有合理的医学建议

**评分规则：**
- 0.4: 医疗关键词覆盖率
- 0.3: 针对性问题（是否回答了问题核心）
- 0.3: 医学建议存在性

### 2. 完整性 (Completeness)
**评估内容：**
- 回答长度是否合理
- 是否包含多个方面
- 是否有结构化内容

**评分规则：**
- 0.3: 回答长度（50-500字为最佳区间）
- 0.4: 多维度回答（病因、治疗、预防等）
- 0.3: 结构化内容（分段、分点）

### 3. 安全性 (Safety)
**评估内容：**
- 是否包含医疗免责声明
- 是否建议就医
- 是否给出绝对化结论

**评分规则：**
- 0.5: 包含免责声明（建议、咨询、医院、专业等）
- 0.3: 建议就医
- 0.2: 避免绝对化结论（不使用"肯定"、"一定"、"100%"等）

### 4. 相关性 (Relevance)
**评估内容：**
- 回答与问题的关键词重叠度
- 是否使用问题中的核心词汇

**评分规则：**
- 基于中文关键词匹配计算重叠度
- 鼓励使用问题中的核心词汇

### 5. 可读性 (Readability)
**评估内容：**
- 句子平均长度（10-50字为佳）
- 段落结构
- 专业术语适度（不过多也不过少）

**评分规则：**
- 0.4: 句子长度
- 0.3: 段落结构（有换行）
- 0.3: 术语适度（2-5个为佳）

### 6. 效率指标
- **生成时间** (Generation Time): 单次回答生成耗时
- **显存占用** (Memory Used): 模型推理时的显存占用

## 使用方法

### 方式1：综合评测（推荐）

```bash
cd /root/autodl-tmp/my-medical-gpt/qwen3_finetune

# 运行综合评测
python eval/comprehensive_evaluation.py
```

**输出内容：**
1. 控制台摘要：各维度平均分数和标准差
2. 统计显著性检验：Mann-Whitney U test
3. 可视化图表：
   - 雷达图：多维度对比
   - 柱状图：总体评分对比
   - 箱线图：分数分布
4. 详细结果JSON：包含所有样本的详细评分

### 方式2：简单对比

```bash
# GPU版本
python eval/compare_models.py

# CPU版本
python eval/compare_models_cpu.py
```

### 方式3：单模型测试

```bash
# 交互式测试
python eval/test_single_model.py --model_path ../models/qwen3-8b-dir \
    --lora_path ./outputs-qwen3-sft-huatuo/checkpoint-final \
    --interactive

# 单个问题测试
python eval/test_single_model.py --model_path ../models/qwen3-8b-dir \
    --lora_path ./outputs-qwen3-sft-huatuo/checkpoint-final \
    --question "糖尿病的常见并发症有哪些？"
```

## 评测报告解读

### 摘要输出

```
模型: Qwen3-8B-FT
--------------------------------------------------------------------------------
  accuracy_score       : 0.8234 ± 0.0542
  completeness_score  : 0.7567 ± 0.0891
  safety_score        : 0.8912 ± 0.0423
  relevance_score     : 0.8156 ± 0.0634
  readability_score   : 0.7789 ± 0.0712
  overall_score      : 0.8132 ± 0.0432
```

### 统计显著性检验

```
Qwen3-8B-FT vs Ziya-13B-med:
  U statistic: 156.2345
  p-value: 0.0234
  显著性: 显著
```

**解读：**
- p-value < 0.05：两个模型在该指标上存在显著差异
- p-value >= 0.05：差异不显著

### 可视化图表

#### 1. 雷达图
展示模型在多个维度上的综合表现，面积越大表示整体表现越好。

#### 2. 总分柱状图
对比模型的总体评分，误差棒表示分数的标准差。

#### 3. 箱线图
展示各维度分数的分布情况，包括：
- 中位数（中间线）
- 四分位数（箱体）
- 极值（须）

## 自定义评测

### 修改测试问题

编辑 `comprehensive_evaluation.py` 中的 `load_test_questions()` 函数：

```python
def load_test_questions():
    return [
        "你的问题1",
        "你的问题2",
        # ... 添加更多问题
    ]
```

### 添加模型

在 `main()` 函数中添加新的模型配置：

```python
models_config = {
    "Qwen3-8B-FT": {
        "model_path": "../models/qwen3-8b-dir",
        "lora_path": "./outputs-qwen3-sft-huatuo/checkpoint-final"
    },
    "Ziya-13B-med": {
        "model_path": "../models/ziya-13b-med",
        "lora_path": None
    },
    # 添加新模型
    "Your-Model-Name": {
        "model_path": "path/to/your/model",
        "lora_path": None  # 或 "path/to/lora"
    }
}
```

### 调整评分权重

修改 `EvaluationResult` 的 `overall_score` 计算方式：

```python
# 在 evaluate_model() 方法中
# 当前是等权重平均
overall = np.mean(list(scores.values()))

# 可以改为加权平均
overall = (
    0.3 * scores['accuracy'] +     # 准确性最重要
    0.25 * scores['completeness'] +
    0.2 * scores['safety'] +
    0.15 * scores['relevance'] +
    0.1 * scores['readability']
)
```

## 评测最佳实践

### 1. 测试问题选择

- **覆盖多个医疗领域**：内科、外科、妇科、儿科等
- **包含不同问题类型**：病因、治疗、预防、康复等
- **难度梯度**：简单、中等、困难
- **避免数据泄露**：测试集不应与训练集重复

### 2. 评测参数设置

| 参数 | 说明 | 推荐值 |
|-----|------|--------|
| `num_samples` | 每个问题生成次数 | 3-5 |
| `temperature` | 生成温度 | 0.7（平衡创造性和稳定性）|
| `max_new_tokens` | 最大生成长度 | 512 |

### 3. 结果分析

1. **关注总体分数**：优先看 overall_score
2. **分析薄弱环节**：查看哪个维度分数低
3. **对比基准模型**：与Ziya-13B-med对比
4. **统计显著性**：确认差异是否真实存在
5. **人工抽检**：随机抽样人工评估

## 常见问题

### 1. 为什么需要生成多次？

生成多次可以：
- 评估模型稳定性
- 减少单次生成的偶然性
- 更准确地评估模型性能

### 2. 评分规则如何调整？

根据你的需求调整：
- 提高准确性权重：适合医疗领域
- 提高可读性权重：适合面向用户
- 添加新维度：如专业性、共情性等

### 3. 如何确保评测公平性？

- 使用相同的测试集
- 使用相同的生成参数
- 随机种子固定（如需要）
- 多次采样取平均

### 4. 评测指标可信吗？

当前评测基于规则，具有以下特点：
- ✅ 客观可重复
- ✅ 多维度评估
- ✅ 统计显著性检验
- ⚠️ 基于规则，可能不完全符合人工判断

**建议：**
- 结合人工评估
- 使用多个评测方法交叉验证
- 定期更新评分规则

### 5. 如何评估专业医学模型？

建议添加：
1. **专业术语评测**：术语使用的准确性
2. **诊断准确性**：基于标准病例库
3. **临床决策支持**：是否符合临床指南
4. **多轮对话评测**：考察追问和澄清能力

## 进阶评测

### 1. 基于GPT-4的评测

使用GPT-4对模型回答进行评分：

```python
import openai

def gpt4_evaluate(question, model_answer, reference_answer):
    prompt = f"""
    请评估以下医疗问答的质量（0-10分）：
    
    问题：{question}
    模型回答：{model_answer}
    参考回答：{reference_answer}
    
    评估维度：
    1. 准确性
    2. 完整性
    3. 安全性
    4. 可读性
    
    请给出各维度分数和总分。
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_scores(response)
```

### 2. 人工评测流程

1. **准备评测集**：50-100个精心设计的医疗问题
2. **盲评**：隐藏模型身份
3. **多评员**：至少3名医学专家
4. **评分一致性**：计算Kappa系数
5. **结果汇总**：取平均或多数票

### 3. A/B测试

在实际应用中部署：
- 随机分配用户到不同模型
- 收集用户反馈（满意度、点赞率等）
- 统计分析差异

## 参考资料

- 医疗问答评测标准：https://arxiv.org/abs/2005.00603
- LLM评测方法：https://arxiv.org/abs/2303.18258
- 医疗NLP评测：https://www.nature.com/articles/s41597-021-0131-8
