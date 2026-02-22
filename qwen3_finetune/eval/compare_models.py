#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型对比脚本
对比微调后的Qwen3模型和Ziya-13B-med模型在医疗问答任务上的表现
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def load_model(model_path, lora_path=None, device="cuda"):
    """
    加载模型
    
    Args:
        model_path: 基础模型路径
        lora_path: LoRA权重路径（可选）
        device: 设备
    
    Returns:
        model, tokenizer
    """
    print(f"加载模型: {model_path}")
    if lora_path:
        print(f"加载LoRA权重: {lora_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, device_map=device)
    
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens=512, temperature=0.7):
    """
    生成答案
    
    Args:
        model: 模型
        tokenizer: 分词器
        question: 问题
        max_new_tokens: 最大生成长度
        temperature: 温度参数
    
    Returns:
        生成的答案
    """
    messages = [{"role": "user", "content": question}]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def compare_models(qwen3_model_path, qwen3_lora_path, ziya_model_path, test_questions):
    """
    对比两个模型
    
    Args:
        qwen3_model_path: Qwen3模型路径
        qwen3_lora_path: Qwen3 LoRA权重路径
        ziya_model_path: Ziya模型路径
        test_questions: 测试问题列表
    """
    print("=" * 80)
    print("开始加载模型...")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载Qwen3模型
    qwen3_model, qwen3_tokenizer = load_model(qwen3_model_path, qwen3_lora_path, device)
    
    # 加载Ziya模型
    ziya_model, ziya_tokenizer = load_model(ziya_model_path, None, device)
    
    print("=" * 80)
    print("模型加载完成，开始对比测试...")
    print("=" * 80)
    
    results = []
    
    for idx, question in enumerate(tqdm(test_questions, desc="测试进度")):
        print(f"\n{'='*80}")
        print(f"测试问题 {idx + 1}/{len(test_questions)}")
        print(f"{'='*80}")
        print(f"问题: {question}")
        print(f"{'-'*80}")
        
        # Qwen3生成
        print("\nQwen3-8B 回答:")
        qwen3_answer = generate_answer(qwen3_model, qwen3_tokenizer, question)
        print(qwen3_answer)
        
        print(f"\n{'-'*80}")
        
        # Ziya生成
        print("\nZiya-13B-med 回答:")
        ziya_answer = generate_answer(ziya_model, ziya_tokenizer, question)
        print(ziya_answer)
        
        results.append({
            "question": question,
            "qwen3_answer": qwen3_answer,
            "ziya_answer": ziya_answer
        })
        
        print(f"\n{'='*80}")
    
    return results


def save_results(results, output_path):
    """
    保存对比结果
    
    Args:
        results: 对比结果
        output_path: 输出路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比结果已保存到: {output_path}")


def main():
    """
    主函数
    """
    # 配置路径
    qwen3_model_path = "../models/qwen3-8b-dir"
    qwen3_lora_path = "./outputs-qwen3-medical/checkpoint-500"  # 微调后的checkpoint路径
    ziya_model_path = "../models/ziya-13b-med"
    
    # 测试问题
    test_questions = [
        "治疗阳痿吃什么药呢？",
        "精子生成减少的病因是什么？",
        "脑水瘤现在可以治好吗？",
        "暴盲的病因病机是什么?",
        "糖尿病的常见并发症有哪些？",
        "高血压患者日常生活中需要注意什么？",
        "感冒了应该吃什么药？",
        "如何预防心血管疾病？"
    ]
    
    # 运行对比
    results = compare_models(
        qwen3_model_path=qwen3_model_path,
        qwen3_lora_path=qwen3_lora_path,
        ziya_model_path=ziya_model_path,
        test_questions=test_questions
    )
    
    # 保存结果
    save_results(results, "./comparison_results.json")


if __name__ == "__main__":
    main()
