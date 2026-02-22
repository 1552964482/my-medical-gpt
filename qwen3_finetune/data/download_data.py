#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据下载脚本
用于下载Hugging Face上的医疗数据集
"""

from datasets import load_dataset
import os


def download_dataset(dataset_name, save_path, split="train"):
    """
    下载数据集
    
    Args:
        dataset_name: 数据集名称
        save_path: 保存路径
        split: 数据集split
    """
    print(f"下载数据集: {dataset_name}")
    print(f"保存路径: {save_path}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 下载数据集
    dataset = load_dataset(dataset_name, split=split)
    
    # 保存为jsonl格式
    output_file = os.path.join(save_path, f"{dataset_name.replace('/', '_')}.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            import json
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据集下载完成！")
    print(f"保存文件: {output_file}")
    print(f"样本数量: {len(dataset)}")
    
    return len(dataset)


def main():
    """
    主函数
    """
    print("=" * 80)
    print("医疗数据集下载工具")
    print("=" * 80)
    
    # 定义要下载的数据集
    datasets_to_download = [
        {
            "name": "shibing624/medical",
            "config": "pretrain",
            "save_path": "../data/pretrain/downloaded",
            "description": "240万条中文医疗预训练数据"
        },
        {
            "name": "shibing624/medical",
            "config": "sft",
            "save_path": "../data/finetune/downloaded",
            "description": "240万条中文医疗SFT数据"
        },
        {
            "name": "FreedomIntelligence/HuatuoGPT-sft-data-v1",
            "save_path": "../data/finetune/downloaded",
            "description": "22万条华佗医疗对话数据"
        }
    ]
    
    print("\n可选数据集:")
    for idx, ds in enumerate(datasets_to_download, 1):
        print(f"{idx}. {ds['name']}")
        print(f"   描述: {ds['description']}")
        print(f"   配置: {ds.get('config', 'default')}")
        print()
    
    choice = input("请选择要下载的数据集编号（多个用逗号分隔，如1,2,3）: ")
    
    selected_indices = [int(x.strip()) for x in choice.replace('，', ',').split(',')]
    
    for idx in selected_indices:
        if 1 <= idx <= len(datasets_to_download):
            ds = datasets_to_download[idx - 1]
            print(f"\n{'='*80}")
            
            try:
                if "config" in ds:
                    dataset = load_dataset(ds["name"], ds["config"])
                else:
                    dataset = load_dataset(ds["name"])
                
                os.makedirs(ds["save_path"], exist_ok=True)
                
                # 保存数据集
                if "train" in dataset:
                    split_data = dataset["train"]
                else:
                    split_data = list(dataset.values())[0]
                
                output_file = os.path.join(
                    ds["save_path"], 
                    f"{ds['name'].replace('/', '_')}_{ds.get('config', 'default')}.jsonl"
                )
                
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                print(f"下载完成！")
                print(f"保存文件: {output_file}")
                print(f"样本数量: {len(split_data)}")
                
            except Exception as e:
                print(f"下载失败: {e}")
        else:
            print(f"无效选择: {idx}")
    
    print("\n" + "=" * 80)
    print("下载完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
