import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from peft import PeftModel, PeftConfig, LoraModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import random

adapter_name = "self"


base_model_path = "../model_hub/Shanghai_AI_Laboratory/internlm2_5-7b-chat"
peft_model_id = "../output/internlm2.5_7B_chat_lora/checkpoint-965"


device = "cuda"
quantization_config = None
model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id, trust_remote_code=True)
lora_config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id, adapter_name=adapter_name, config=lora_config)
model.eval()



def get_result(model_inputs, model):

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.get_vocab()["<|im_end|>"]
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

while True:
    # prompt = "你是谁"
    prompt = input(">>>")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # print(text)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    base_model_response = get_result(model_inputs, model)
    print(base_model_response)