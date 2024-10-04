# Llama3/3.1-Finetuning
对llama3进行全参微调、lora微调以及qlora微调。除此之外，也支持对qwen1.5的模型进行微调。如果要替换为其它的模型，最主要的还是在数据的预处理那一块。

# 更新日志

- 2023/07/28：添加对Baichuan2-7B-Chat的微调。
- 2024/07/24：添加对llama3.1-8B-Instruct的微调。`transformers==4.43.1`和`accelerate==0.33.0`。

- 2024/07/22：
  - 添加对glm-9B-chat的微调。注意：需要将modeling_chatglm.py里面的791行替换为`padding_mask = padding_mask.to(torch.bfloat16)`。需要的transformers的版本为`4.42.4`，安装完requirements.txt里面的包后需要再重新安装transformers。

  - 添加对qwen1.5-7B-Chat的微调。

  - 添加对qwen2-7B-Instruct的微调。

  - 添加对yi1.5-6B-Chat的微调。

- 2024/07/19：添加对internlm2.5的微调。注意：internlm2.5不支持使用bf16微调，因此在运行指令中选择的是fp16。

- 2024/10/04：添加对Qwen2.5-7B-Instruct的微调。添加对llama3.2-3B-Instruct的微调。`pip install transformers --upgrade`和`pip install accelerate --upgrade`。

# 安装依赖

- 运行设备：24G显存的显卡即可。
- `python==3.8.8`

- `pip install -r requirements.txt`

# 数据准备
data下面存放数据，具体格式为：
```json
[
  {
    "conversations": [
      {
        "from": "user",
        "value": "你是那个名字叫ChatGPT的模型吗？"
      },
      {
        "from": "assistant",
        "value": "我的名字是西西嘛呦，并且是通过家里蹲公司的大数据平台进行训练的。"
      }
	]
  }
  ...
]
```
多轮对话也是按照上述格式准备好数据。

# 模型准备
进入到model_hub文件夹下，运行```python download_modelscope.py```即可下载llama3-8B-Instruct模型。

# 微调
进入到script文件夹。

## 全参数微调
受机器限制，这里并未进行全参数微调，如果有条件可以试试。

## lora微调
nproc_per_node和CUDA_VISIBLE_DEVICES指定的显卡数目要保持一致。
```shell
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
torchrun \
--nproc_per_node 7 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6601 \
../finetune_llama3.py \
--model_name_or_path "../model_hub/LLM-Research/Meta-Llama-3-8B-Instruct/" \
--data_path "../data/Belle_sampled_qwen.json" \
--bf16 True \
--output_dir "../output/llama3_8B_lora" \
--num_train_epochs 100 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 5 \
--save_total_limit 1 \
--learning_rate 1e-5 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 4096 \
--gradient_checkpointing True \
--lazy_preprocess True \
--deepspeed "../config/ds_config_zero3_72B.json" \
--use_lora
```

## qlora微调
nproc_per_node和CUDA_VISIBLE_DEVICES指定的显卡数目要保持一致。使用qlora在单张4090上即可完成训练。
```shell
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
torchrun \
--nproc_per_node 7 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6601 \
../finetune_llama3.py \
--model_name_or_path "../model_hub/LLM-Research/Meta-Llama-3-8B-Instruct/" \
--data_path "../data/Belle_sampled_qwen.json" \
--bf16 True \
--output_dir "../output/llama3_8B_qlora" \
--num_train_epochs 100 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 5 \
--save_total_limit 1 \
--learning_rate 1e-5 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 4096 \
--gradient_checkpointing True \
--lazy_preprocess True \
--deepspeed "../config/ds_config_zero2.json" \
--use_lora \
--load_in_4bit \
--q_lora
```

# 推理
- llama3_single_predict.py：提供的是未微调前的模型的推理。
- llama3_lora_predict.py：提供将基于lora微调的模型合并到基础模型中，并进行推理。
- llama3_qlora_4bit_predict.py：提供将基于qlora微调的模型合并到基础模型中，并进行推理。
- llama3_peft_qlora_predict.py：使用peft加载qlora适配器进行推理。
- llama3_peft_lora_predict.py：使用peft加载lora适配器进行推理。

训练完成后推理：
```python
问题： 在互联网普及的时代，社交媒体已经成为了现代人生活中不可缺少的一部分。从微信、微博、Facebook到Twitter、Instagram等等，社交媒体不仅满足了人们交流互动的需求，同时让人们更加容易地分享自己的生活瞬间、获取新闻资讯、找到志同道合的人群以及推广自己的业务。然而，与此同时，社交媒体也带来了一系列问题，如滥用个人信息、泄露隐私等。因此，如何正确地使用社交媒体已经成为了一个备受关注的话题。，总结上面这段文本的几个关键词

微调前结果： Arrrr, shiver me timbers! Ye be wantin' me to summarize the key words from that there text, eh? Alright then, matey! Here be the main points:

1. Social media (the scourge o' the seven seas... er, I mean, the modern world)
2. Communication (the lifeblood o' the pirate's existence... er, I mean, the need for people to interact)
3. Sharing (the treasure o' the digital age... er, I mean, the ability to share one's life moments)
4. Information (the booty o' the internet... er, I mean, the news and knowledge shared online)
5. Privacy (the anchor o' security... er, I mean, the need to protect one's personal info)
6. Correct usage (the map to avoidin' the rocky shores o' social media misuse... er, I mean, the importance of using social media responsibly)

So hoist the colors, me hearties, and remember to use social media like a proper pirate: with caution, respect, and a keen eye for treasure!

微调后结果： 社交媒体、交流、分享、隐私、滥用信息
```

# 参考
代码主要参考了：
- https://github.com/QwenLM/Qwen

模型可以去modelscope上进行下载：
- https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct/files