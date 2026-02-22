#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载中文维基百科数据集
"""

from datasets import load_dataset
import os
import shutil


def main():
    print("=" * 80)
    print("下载中文维基百科数据集")
    print("=" * 80)
    
    dataset_name = "pleisto/wikipedia-cn-20230720-filtered"
    save_path = "../data/pretrain/downloaded"
    
    print(f"数据集: {dataset_name}")
    print(f"保存路径: {save_path}")
    print()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 下载数据集（使用streaming模式避免内存问题）
        print("正在下载数据集...")
        dataset = load_dataset(dataset_name)
        
        # 保存为jsonl格式
        output_file = os.path.join(save_path, f"{dataset_name.replace('/', '_')}.jsonl")
        
        print(f"保存到: {output_file}")
        
        # 获取训练集
        if "train" in dataset:
            split_data = dataset["train"]
        else:
            split_data = list(dataset.values())[0]
        
        print(f"数据集大小: {len(split_data)} 条")
        print()
        print("正在保存数据集...")
        
        # 保存数据
        import json
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
                if count % 10000 == 0:
                    print(f"已保存: {count} 条")
        
        print()
        print("=" * 80)
        print("下载完成！")
        print(f"保存文件: {output_file}")
        print(f"样本数量: {count}")
        print("=" * 80)
        
    except Exception as e:
        print(f"下载失败: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    main()
