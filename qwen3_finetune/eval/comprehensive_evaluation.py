#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合模型评测脚本
提供多维度、量化的医疗模型评测
"""

import json
import re
import time
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EvaluationResult:
    """评测结果数据类"""
    model_name: str
    question: str
    answer: str
    answer_length: int
    generation_time: float
    memory_used: float
    
    accuracy_score: float
    completeness_score: float
    safety_score: float
    relevance_score: float
    readability_score: float
    
    overall_score: float


class MedicalEvaluator:
    """医疗问答评测器"""
    
    def __init__(self):
        self.medical_keywords = self._load_medical_keywords()
        self.risk_keywords = self._load_risk_keywords()
    
    def _load_medical_keywords(self):
        """加载医疗关键词"""
        keywords = [
            '治疗', '病因', '症状', '预防', '并发症', '诊断', '药物', '手术',
            '康复', '护理', '饮食', '运动', '检查', '化验', '影像', '超声',
            'CT', 'MRI', 'X光', '血压', '血糖', '胆固醇', '血脂', '心电图',
            '糖尿病', '高血压', '心脏病', '中风', '癌症', '肿瘤', '感染',
            '炎症', '过敏', '免疫', '遗传', '病毒', '细菌', '真菌', '抗生素',
            '激素', '维生素', '蛋白质', '脂肪', '碳水化合物', '钙', '铁', '锌'
        ]
        return set(keywords)
    
    def _load_risk_keywords(self):
        """加载风险关键词"""
        keywords = [
            '建议', '咨询', '就医', '医院', '医生', '专业', '确诊', '检查',
            '谨慎', '注意', '副作用', '风险', '禁忌', '不能', '禁止'
        ]
        return set(keywords)
    
    def evaluate_accuracy(self, question: str, answer: str) -> float:
        """
        评估医学准确性
        
        基于规则评估：
        1. 是否包含医疗关键词
        2. 是否针对问题回答
        3. 是否有合理的医学建议
        """
        score = 0.0
        
        # 1. 医疗关键词覆盖
        question_keywords = set()
        for kw in self.medical_keywords:
            if kw in question:
                question_keywords.add(kw)
        
        answer_keywords = set()
        for kw in self.medical_keywords:
            if kw in answer:
                answer_keywords.add(kw)
        
        if question_keywords:
            coverage = len(answer_keywords & question_keywords) / len(question_keywords)
            score += coverage * 0.4
        
        # 2. 针对性问题
        if any(kw in answer for kw in question_keywords):
            score += 0.3
        
        # 3. 医学建议存在性
        if any(kw in answer for kw in ['建议', '应该', '可以', '推荐', '需要']):
            score += 0.3
        
        return min(score, 1.0)
    
    def evaluate_completeness(self, question: str, answer: str) -> float:
        """
        评估回答完整性
        
        评估维度：
        1. 回答长度是否合理
        2. 是否包含多个方面
        3. 是否有结构化内容
        """
        score = 0.0
        
        # 1. 回答长度（50-500字为佳）
        answer_len = len(answer)
        if 50 <= answer_len <= 500:
            score += 0.3
        elif answer_len < 50:
            score += 0.1 * (answer_len / 50)
        else:
            score += 0.3 - 0.1 * ((answer_len - 500) / 500)
        
        # 2. 多维度回答（包含多个方面）
        aspects = 0
        if '原因' in question and ('病因' in answer or '原因' in answer):
            aspects += 1
        if '治疗' in question and ('治疗' in answer or '药' in answer):
            aspects += 1
        if '预防' in question and ('预防' in answer or '注意' in answer):
            aspects += 1
        
        if aspects > 0:
            score += min(aspects * 0.2, 0.4)
        
        # 3. 结构化内容（有分点或分段）
        if re.search(r'[。！？]\s*\n', answer) or '1.' in answer or '2.' in answer:
            score += 0.3
        
        return min(score, 1.0)
    
    def evaluate_safety(self, question: str, answer: str) -> float:
        """
        评估回答安全性
        
        评估维度：
        1. 是否包含医疗免责声明
        2. 是否建议就医
        3. 是否给出绝对化结论
        """
        score = 0.0
        
        # 1. 包含免责声明
        if any(kw in answer for kw in self.risk_keywords):
            score += 0.5
        
        # 2. 建议就医
        if any(kw in answer for kw in ['建议', '咨询', '就医', '医院']):
            score += 0.3
        
        # 3. 避免绝对化结论
        if not re.search(r'(肯定|一定|绝对|保证|100%|完全治愈)', answer):
            score += 0.2
        
        return min(score, 1.0)
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        """
        评估相关性
        
        基于关键词匹配评估回答与问题的相关性
        """
        # 提取问题中的关键词
        question_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', question))
        
        # 提取回答中的关键词
        answer_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', answer))
        
        # 计算重叠度
        if not question_words:
            return 0.5
        
        overlap = len(question_words & answer_words)
        relevance = min(overlap / max(len(question_words), 1), 1.0)
        
        # 鼓励使用问题中的核心词汇
        if overlap > 0:
            relevance = 0.7 + 0.3 * relevance
        
        return min(relevance, 1.0)
    
    def evaluate_readability(self, answer: str) -> float:
        """
        评估可读性
        
        评估维度：
        1. 句子平均长度
        2. 段落结构
        3. 专业术语适度
        """
        score = 0.0
        
        # 1. 句子长度（适中为佳）
        sentences = re.split(r'[。！？\n]', answer)
        sentences = [s for s in sentences if s.strip()]
        if sentences:
            avg_len = np.mean([len(s) for s in sentences])
            if 10 <= avg_len <= 50:
                score += 0.4
            elif avg_len < 10:
                score += 0.4 * (avg_len / 10)
            else:
                score += 0.4 * (50 / avg_len)
        
        # 2. 段落结构
        if '\n' in answer:
            score += 0.3
        
        # 3. 术语适度（不过多也不过少）
        medical_term_count = sum(1 for kw in self.medical_keywords if kw in answer)
        if 2 <= medical_term_count <= 5:
            score += 0.3
        elif medical_term_count < 2:
            score += 0.15
        else:
            score += 0.15
        
        return min(score, 1.0)
    
    def evaluate(self, question: str, answer: str) -> Dict[str, float]:
        """
        综合评估
        
        Returns:
            包含各维度分数的字典
        """
        return {
            'accuracy': self.evaluate_accuracy(question, answer),
            'completeness': self.evaluate_completeness(question, answer),
            'safety': self.evaluate_safety(question, answer),
            'relevance': self.evaluate_relevance(question, answer),
            'readability': self.evaluate_readability(answer)
        }


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, device="auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.evaluator = MedicalEvaluator()
    
    def load_model(self, model_name: str, model_path: str, lora_path: str = None):
        """加载模型"""
        print(f"加载模型 {model_name}: {model_path}")
        if lora_path:
            print(f"  LoRA权重: {lora_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path, device_map=self.device)
        
        model.eval()
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        print(f"  设备: {self.device}")
        print(f"  模型加载完成\n")
    
    def generate_answer(self, model_name: str, question: str, 
                     max_new_tokens: int = 512, 
                     temperature: float = 0.7,
                     num_samples: int = 1) -> List[str]:
        """生成答案（支持多次采样）"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        answers = []
        for _ in range(num_samples):
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generation_time = time.time() - start_time
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3) if self.device == "cuda" else 0
            
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answers.append(answer)
        
        return answers, generation_time, memory_used
    
    def evaluate_model(self, model_name: str, test_questions: List[str], 
                    num_samples: int = 3) -> List[EvaluationResult]:
        """评测单个模型"""
        print(f"\n{'='*80}")
        print(f"评测模型: {model_name}")
        print(f"{'='*80}\n")
        
        results = []
        
        for question in tqdm(test_questions, desc=f"评测 {model_name}"):
            # 生成多个样本
            answers, gen_time, memory = self.generate_answer(
                model_name, question, num_samples=num_samples
            )
            
            # 对每个样本进行评估
            for idx, answer in enumerate(answers):
                scores = self.evaluator.evaluate(question, answer)
                overall = np.mean(list(scores.values()))
                
                result = EvaluationResult(
                    model_name=model_name,
                    question=question,
                    answer=answer,
                    answer_length=len(answer),
                    generation_time=gen_time / num_samples,
                    memory_used=memory,
                    accuracy_score=scores['accuracy'],
                    completeness_score=scores['completeness'],
                    safety_score=scores['safety'],
                    relevance_score=scores['relevance'],
                    readability_score=scores['readability'],
                    overall_score=overall
                )
                results.append(result)
        
        return results
    
    def compare_models(self, model_names: List[str], test_questions: List[str],
                     num_samples: int = 3) -> Dict[str, List[EvaluationResult]]:
        """对比多个模型"""
        all_results = {}
        
        for model_name in model_names:
            all_results[model_name] = self.evaluate_model(
                model_name, test_questions, num_samples
            )
        
        return all_results


class EvaluationReporter:
    """评测报告生成器"""
    
    @staticmethod
    def print_summary(results: Dict[str, List[EvaluationResult]]):
        """打印摘要"""
        print("\n" + "="*80)
        print("评测摘要")
        print("="*80 + "\n")
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            print(f"模型: {model_name}")
            print("-" * 80)
            
            # 计算平均分数
            metrics = ['accuracy_score', 'completeness_score', 'safety_score', 
                      'relevance_score', 'readability_score', 'overall_score']
            
            for metric in metrics:
                scores = [getattr(r, metric) for r in model_results]
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"  {metric:20s}: {mean_score:.4f} ± {std_score:.4f}")
            
            print()
    
    @staticmethod
    def statistical_test(results: Dict[str, List[EvaluationResult]]):
        """统计显著性检验"""
        print("="*80)
        print("统计显著性检验（Mann-Whitney U test）")
        print("="*80 + "\n")
        
        from scipy import stats
        
        model_names = list(results.keys())
        if len(model_names) < 2:
            return
        
        metric = 'overall_score'
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                
                scores_a = [getattr(r, metric) for r in results[model_a]]
                scores_b = [getattr(r, metric) for r in results[model_b]]
                
                stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
                
                print(f"{model_a} vs {model_b}:")
                print(f"  U statistic: {stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  显著性: {'显著' if p_value < 0.05 else '不显著'}")
                print()
    
    @staticmethod
    def visualize_results(results: Dict[str, List[EvaluationResult]], output_dir: str = "./"):
        """可视化结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 雷达图 - 多维度对比
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        categories = ['准确性', '完整性', '安全性', '相关性', '可读性']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            if not model_results:
                continue
            
            # 计算平均分数
            metrics = ['accuracy_score', 'completeness_score', 'safety_score',
                      'relevance_score', 'readability_score']
            values = [np.mean([getattr(r, m) for r in model_results]) for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('模型多维度对比', size=16, weight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 柱状图 - 总分对比
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = []
        overall_scores = []
        std_scores = []
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            scores = [r.overall_score for r in model_results]
            model_names.append(model_name)
            overall_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
        
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, overall_scores, yerr=std_scores, 
                      capsize=5, alpha=0.7, color=colors[:len(model_names)])
        
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('总体评分', fontsize=12)
        ax.set_title('模型总体评分对比', fontsize=14, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='基准线')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 箱线图 - 分数分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['accuracy_score', 'completeness_score', 'safety_score',
                  'relevance_score', 'readability_score', 'overall_score']
        metric_names = ['准确性', '完整性', '安全性', '相关性', '可读性', '总体评分']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            data_to_plot = []
            labels = []
            
            for model_name, model_results in results.items():
                if not model_results:
                    continue
                scores = [getattr(r, metric) for r in model_results]
                data_to_plot.append(scores)
                labels.append(model_name)
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(name, fontsize=11, weight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n可视化图表已保存到: {output_dir}")
        print(f"  - radar_chart.png (雷达图)")
        print(f"  - overall_scores.png (总分柱状图)")
        print(f"  - boxplots.png (分数分布箱线图)")
    
    @staticmethod
    def save_detailed_results(results: Dict[str, List[EvaluationResult]], output_path: str):
        """保存详细结果"""
        output_data = {
            'results': {model_name: [asdict(r) for r in model_results] 
                       for model_name, model_results in results.items()},
            'summary': {}
        }
        
        # 计算摘要统计
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            metrics = ['accuracy_score', 'completeness_score', 'safety_score',
                      'relevance_score', 'readability_score', 'overall_score']
            
            output_data['summary'][model_name] = {}
            for metric in metrics:
                scores = [getattr(r, metric) for r in model_results]
                output_data['summary'][model_name][metric] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {output_path}")


def load_test_questions():
    """加载测试问题"""
    return [
        "治疗阳痿吃什么药呢？",
        "精子生成减少的病因是什么？",
        "脑水瘤现在可以治好吗？",
        "暴盲的病因病机是什么?",
        "糖尿病的常见并发症有哪些？",
        "高血压患者日常生活中需要注意什么？",
        "感冒了应该吃什么药？",
        "如何预防心血管疾病？",
        "冠心病的早期症状有哪些？",
        "脑卒中后康复训练应该怎么做？",
        "胃溃疡患者的饮食注意事项？",
        "慢性肾炎应该如何治疗？"
    ]


def main():
    """主函数"""
    print("="*80)
    print("综合模型评测系统")
    print("="*80 + "\n")
    
    # 配置
    device = "auto"
    num_samples = 3  # 每个问题生成3次，取平均
    
    # 创建对比器
    comparator = ModelComparator(device=device)
    
    # 加载模型（根据实际情况修改路径）
    models_config = {
        "Qwen3-8B-FT": {
            "model_path": "../models/qwen3-8b-dir",
            "lora_path": "./outputs-qwen3-sft-huatuo/checkpoint-final"
        },
        "Ziya-13B-med": {
            "model_path": "../models/ziya-13b-med",
            "lora_path": None
        }
    }
    
    for model_name, config in models_config.items():
        comparator.load_model(model_name, **config)
    
    # 加载测试问题
    test_questions = load_test_questions()
    print(f"\n测试问题数量: {len(test_questions)}\n")
    
    # 运行评测
    results = comparator.compare_models(
        model_names=list(models_config.keys()),
        test_questions=test_questions,
        num_samples=num_samples
    )
    
    # 生成报告
    reporter = EvaluationReporter()
    reporter.print_summary(results)
    reporter.statistical_test(results)
    reporter.visualize_results(results, output_dir="./evaluation_results")
    reporter.save_detailed_results(results, "./evaluation_results/detailed_results.json")
    
    print("\n" + "="*80)
    print("评测完成！")
    print("="*80)


if __name__ == "__main__":
    main()
