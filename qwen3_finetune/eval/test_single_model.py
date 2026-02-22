#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单模型测试脚本
用于测试单个模型的医疗问答能力
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(model_path, lora_path=None, device="auto"):
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
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
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


def interactive_test(model, tokenizer):
    """
    交互式测试
    
    Args:
        model: 模型
        tokenizer: 分词器
    """
    print("\n" + "=" * 80)
    print("交互式测试模式（输入 'quit' 或 'exit' 退出）")
    print("=" * 80 + "\n")
    
    while True:
        question = input("请输入问题: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("退出测试")
            break
        
        if not question:
            continue
        
        print("\n生成中...")
        answer = generate_answer(model, tokenizer, question)
        print(f"\n回答:\n{answer}\n")
        print("=" * 80 + "\n")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="测试单个模型的医疗问答能力")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="设备")
    parser.add_argument("--question", type=str, default=None, help="测试问题")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--interactive", action="store_true", help="交互式测试模式")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.model_path, args.lora_path, args.device)
    
    # 测试模式
    if args.interactive:
        interactive_test(model, tokenizer)
    elif args.question:
        answer = generate_answer(model, tokenizer, args.question, args.max_new_tokens, args.temperature)
        print(f"问题: {args.question}")
        print(f"回答: {answer}")
    else:
        # 默认测试问题
        test_questions = [
            "治疗阳痿吃什么药呢？",
            "精子生成减少的病因是什么？",
            "脑水瘤现在可以治好吗？",
            "暴盲的病因病机是什么?",
            "糖尿病的常见并发症有哪些？"
        ]
        
        for question in test_questions:
            answer = generate_answer(model, tokenizer, question, args.max_new_tokens, args.temperature)
            print(f"\n问题: {question}")
            print(f"回答: {answer}")
            print("=" * 80)


if __name__ == "__main__":
    main()
