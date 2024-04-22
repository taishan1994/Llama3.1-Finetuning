import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_result(base_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    device = "cuda"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    prompt = "请用中文回答。你是谁？"


    """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a pirate chatbot who always responds in pirate speak!<|eot_id|><|start_header_id|>user<|end_header_id|>

请用中文回答。你是谁？<|eot_id|><|start_header_id|>assistant<|end_header_id|>

😊我是 LLaMA，一个由 Meta 开发的人工智能语言模型。我可以理解和生成人类语言，帮助回答问题、生成文本、进行对话等。我的能力包括但不限于自然语言处理、语言翻译、文本生成等领域。我很高兴和你交流，回答你的问题和讨论有趣的话题！ 😊<|eot_id|><|start_header_id|>user<|end_header_id|>

请用中文回答。网吧可以上网，弱智吧为什么不可以上弱智？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content":"😊我是 LLaMA，一个由 Meta 开发的人工智能语言模型。我可以理解和生成人类语言，帮助回答问题、生成文本、进行对话等。我的能力包括但不限于自然语言处理、语言翻译、文本生成等领域。我很高兴和你交流，回答你的问题和讨论有趣的话题！ 😊"},
        {"role": "user", "content": "请用中文回答。网吧可以上网，弱智吧为什么不可以上弱智？"}
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
            model_inputs.input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.get_vocab()["<|eot_id|>"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return response

    base_model_response = get_result(model_inputs, base_model)
    print("结果：", base_model_response)


if __name__ == '__main__':
    base_path = "../model_hub/LLM-Research/Meta-Llama-3-8B-Instruct/"
    get_model_result(base_path)
