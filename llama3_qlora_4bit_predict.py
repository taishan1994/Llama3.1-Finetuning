import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6,7"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def get_lora_model(base_model_path,
                   lora_model_input_path,
                   lora_model_output_path):
    #######################
    # 如果是lora和qlora训练的，需要先合并模型
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_model_input_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(lora_model_output_path, max_shard_size="2048MB", safe_serialization=True)
    #######################
    print("合并权重完成。")

    #######################
    # 如果是lora和qlora训练的，合并完模型后拷贝tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    tokenizer.save_pretrained(lora_model_output_path)
    #######################
    print("保存tokenizer完成.")


def get_model_result(base_model_path, fintune_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    device = "cuda"

    fintune_model = AutoModelForCausalLM.from_pretrained(
        fintune_model_path,
        device_map="auto",
    ).eval()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
    ).eval()

    prompt = "在互联网普及的时代，社交媒体已经成为了现代人生活中不可缺少的一部分。从微信、微博、Facebook到Twitter、Instagram等等，社交媒体不仅满足了人们交流互动的需求，同时让人们更加容易地分享自己的生活瞬间、获取新闻资讯、找到志同道合的人群以及推广自己的业务。然而，与此同时，社交媒体也带来了一系列问题，如滥用个人信息、泄露隐私等。因此，如何正确地使用社交媒体已经成为了一个备受关注的话题。，总结上面这段文本的几个关键词"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    def get_result(model_inputs, model):
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.get_vocab()["<|eot_id|>"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    base_model_response = get_result(model_inputs, base_model)
    fintune_model_response = get_result(model_inputs, fintune_model)
    print("问题：", prompt)
    print("微调前结果：", base_model_response)
    print("微调后结果：", fintune_model_response)


if __name__ == '__main__':
    base_path = "./model_hub/LLM-Research/Meta-Llama-3-8B-Instruct/"
    lora_in_path = "./output/llama3_8B_qlora"
    lora_out_path = "./output/llama3_8B_qlora_merged"
    # get_lora_model(base_path, lora_in_path, lora_out_path)
    get_model_result(base_path, lora_out_path)
