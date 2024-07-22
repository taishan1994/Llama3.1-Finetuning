import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4.5.6.7"
from peft import PeftModel, PeftConfig, LoraModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import random

adapter_name = "self"

device = "cuda"

base_model_path = "../model_hub/ZhipuAI/glm-4-9b-chat-1m"
peft_model_id = "../output/glm-4-9b-chat_lora/checkpoint-255"
lora_config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name=adapter_name, config=lora_config)
model.to(device).eval()


tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)


for name, module in model.named_parameters():
    print(name)


def get_result(text, model):
    inputs = tokenizer.apply_chat_template(text,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    # print(inputs)
    # templ = tokenizer.apply_chat_template(text,
    #                                        add_generation_prompt=True,
    #                                        tokenize=False,
    #                                        )
    # print(templ)
    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 512, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    # prompt = "你是谁"
    text = input(">>>")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]

    base_model_response = get_result(messages, model)
    print(base_model_response)