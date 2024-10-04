import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from peft import PeftModel, PeftConfig, LoraModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import random

adapter_name = "self"


base_model_path = "/data/gongoubo/Qwen-1.5-Factory/model_hub/AI-ModelScope/Qwen2___5-7B-Instruct"
peft_model_id = "../output/Qwen2.5-7B-Instruct_lora/checkpoint-2000"

device = "cuda"
quantization_config = None
model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print(tokenizer.pad_token)

lora_config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name=adapter_name, config=lora_config)
model.to(device).eval()

prompt = "你是谁"

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(text)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

def get_result(model_inputs, model):

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.get_vocab()["<|im_end|>"],
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


base_model_response = get_result(model_inputs, model)
print(base_model_response)