import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelEvaluator:
    def __init__(
        self,
        base_model_path: str,
        model_name: str,
        lora_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.base_model_path = base_model_path
        self.model_name = model_name
        self.lora_path = lora_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available()

        print(f"\n{'='*60}")
        print(f"加载模型: {model_name}")
        print(f"{'='*60}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if self.use_cuda else torch.float32
        device_map = "auto" if self.use_cuda else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        if lora_path:
            print("加载LoRA适配器...")
            model = PeftModel.from_pretrained(model, lora_path)

        self.model = model.eval()
        print(f"✅ 模型加载完成！")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {param_count / 1e9:.2f}B")
        print(f"   设备: {self.device}")
    
    def generate_response(
        self,
        question: str,
    ) -> Tuple[str, float]:
        prompt = f"<|im_start|>system\n你是一个专业的医疗助手，请根据患者的问题提供准确、专业的医疗建议。<|im_end|>\n\n<|im_start|>user\n{question}<|im_end|>\n\n<|im_start|>assistant\n"

        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generation_time = end_time - start_time

        return response, generation_time

    def evaluate_questions(self, questions: List[Dict]) -> List[Dict]:
        results = []

        for i, item in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] 问题: {item['question'][:50]}...")

            response, gen_time = self.generate_response(item['question'])

            results.append({
                'question': item['question'],
                'answer': response,
                'generation_time': gen_time,
                'tokens': len(self.tokenizer.encode(response))
            })

            print(f"   回答长度: {len(response)} 字符")
            print(f"   生成时间: {gen_time:.2f}秒")

        return results


def evaluate_all_models(
    models_config: List[Dict[str, Optional[str]]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    medical_questions = [
        {
            'question': '糖尿病的主要症状有哪些？应该如何预防和治疗？',
            'category': '慢性病'
        },
        {
            'question': '高血压患者日常生活中需要注意哪些事项？',
            'category': '慢性病'
        },
        {
            'question': '感冒和流感有什么区别？如何判断自己得了哪种？',
            'category': '常见病'
        },
        {
            'question': '妊娠期心脏病患者需要注意什么？',
            'category': '产科'
        },
        {
            'question': '婴幼儿发烧到39度应该如何处理？',
            'category': '儿科'
        },
        {
            'question': '抑郁症的主要表现有哪些？应该如何帮助患者？',
            'category': '精神科'
        },
        {
            'question': '阑尾炎的早期症状是什么？需要手术吗？',
            'category': '外科'
        },
        {
            'question': '长期失眠会导致哪些健康问题？如何改善睡眠质量？',
            'category': '神经科'
        },
        {
            'question': '肝功能异常可能由哪些原因引起？',
            'category': '内科'
        },
        {
            'question': '新冠康复后出现乏力、心慌等症状正常吗？',
            'category': '传染病'
        }
    ]

    all_results = {}

    for config in models_config:
        evaluator = ModelEvaluator(
            base_model_path=config['base_model_path'],
            model_name=config['name'],
            lora_path=config.get('lora_path'),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        results = evaluator.evaluate_questions(medical_questions)
        all_results[config['name']] = results

        del evaluator.model
        del evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results, medical_questions


def generate_comparison_report(all_results: Dict, questions: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("生成对比报告...")
    print(f"{'='*60}")

    full_report = []

    for i, question_item in enumerate(questions, 1):
        report_item = {
            'id': i,
            'category': question_item['category'],
            'question': question_item['question'],
            'models': {}
        }

        print(f"\n{'='*60}")
        print(f"问题 {i}: {question_item['category']}")
        print(f"{question_item['question']}")
        print(f"{'='*60}")

        for model_name, results in all_results.items():
            result = results[i-1]
            report_item['models'][model_name] = {
                'answer': result['answer'],
                'generation_time': result['generation_time'],
                'tokens': result['tokens']
            }

            print(f"\n【{model_name}】")
            print(f"生成时间: {result['generation_time']:.2f}秒 | Token数: {result['tokens']}")
            print(f"回答:\n{result['answer']}")

        full_report.append(report_item)

    with open(f'{output_dir}/comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    with open(f'{output_dir}/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("医疗模型对比评估报告\n")
        f.write("="*80 + "\n\n")

        for item in full_report:
            f.write(f"问题 {item['id']}: {item['category']}\n")
            f.write(f"{item['question']}\n")
            f.write("-"*80 + "\n\n")

            for model_name, model_data in item['models'].items():
                f.write(f"【{model_name}】\n")
                f.write(f"生成时间: {model_data['generation_time']:.2f}秒 | Token数: {model_data['tokens']}\n")
                f.write(f"回答:\n{model_data['answer']}\n")
                f.write("\n")

            f.write("="*80 + "\n\n")

    print(f"\n✅ 对比报告已保存到: {output_dir}")
    print(f"   - comparison_report.json")
    print(f"   - comparison_report.txt")


def generate_summary_stats(all_results: Dict, output_dir: str):
    summary = {}

    for model_name, results in all_results.items():
        total_time = sum(r['generation_time'] for r in results)
        total_tokens = sum(r['tokens'] for r in results)
        avg_time = total_time / len(results)
        avg_tokens = total_tokens / len(results)
        avg_speed = total_tokens / total_time if total_time > 0 else 0.0

        summary[model_name] = {
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'total_tokens': total_tokens,
            'avg_tokens_per_answer': avg_tokens,
            'avg_tokens_per_second': avg_speed
        }

    print(f"\n{'='*60}")
    print("性能统计摘要")
    print(f"{'='*60}")
    print(f"\n{'模型':<20} {'平均时间(秒)':<15} {'平均Token数':<15} {'速度(Token/s)':<15}")
    print("-"*60)
    
    for model_name, stats in summary.items():
        line = f"{model_name:<20} {stats['avg_time_per_question']:<15.2f} "
        line += f"{stats['avg_tokens_per_answer']:<15.1f} {stats['avg_tokens_per_second']:<15.1f}"
        print(line)
    
    with open(f'{output_dir}/summary_stats.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 统计摘要已保存到: {output_dir}/summary_stats.json")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    qwen3_finetune_root = script_dir.parent

    parser = argparse.ArgumentParser(description="医疗模型对比评估")
    parser.add_argument(
        "--qwen3-base-model",
        default=str(project_root / "models" / "qwen3-8b-dir"),
        help="Qwen3基础模型路径",
    )
    parser.add_argument(
        "--qwen3-lora-path",
        default=str(qwen3_finetune_root / "outputs-qwen3-sft-huatuo" / "checkpoint-final"),
        help="Qwen3 LoRA路径，留空表示不加载LoRA",
    )
    parser.add_argument(
        "--ziya-model-path",
        default=str(project_root / "models" / "ziya-13b-med"),
        help="Ziya模型路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "comparison_results"),
        help="评测结果输出目录",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qwen3_lora_path = args.qwen3_lora_path if args.qwen3_lora_path else None

    models_config = [
        {
            "name": "微调后Qwen3-8B",
            "base_model_path": args.qwen3_base_model,
            "lora_path": qwen3_lora_path,
        },
        {
            "name": "微调前Qwen3-8B",
            "base_model_path": args.qwen3_base_model,
            "lora_path": None,
        },
        {
            "name": "Ziya-13B-Med",
            "base_model_path": args.ziya_model_path,
            "lora_path": None,
        },
    ]

    print("\n" + "="*60)
    print("医疗模型对比评估")
    print("="*60)

    all_results, questions = evaluate_all_models(
        models_config=models_config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generate_comparison_report(all_results, questions, output_dir=args.output_dir)
    generate_summary_stats(all_results, output_dir=args.output_dir)

    print("\n" + "="*60)
    print("✅ 对比评估完成！")
    print("="*60)
