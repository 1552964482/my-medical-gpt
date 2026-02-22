import json
import os

input_file = "/root/autodl-tmp/my-medical-gpt/data/finetune/downloaded/FreedomIntelligence_HuatuoGPT-sft-data-v1_default.jsonl"
output_file = "/root/autodl-tmp/my-medical-gpt/data/finetune/huatuo_sft_conversations.jsonl"

count = 0
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        
        if 'data' in data and len(data['data']) >= 2:
            questions = data['data'][::2]
            answers = data['data'][1::2]
            
            for q, a in zip(questions, answers):
                conversation = {
                    "conversations": [
                        {"from": "human", "value": q.replace("问：", "").strip()},
                        {"from": "gpt", "value": a.replace("答：", "").strip()}
                    ]
                }
                fout.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                count += 1
        
        if count % 10000 == 0:
            print(f"已转换 {count} 条对话...")

print(f"转换完成！共 {count} 条对话")
print(f"输出文件: {output_file}")
