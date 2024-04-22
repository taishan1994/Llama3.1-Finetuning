from transformers import AutoModelForCausalLM, AutoTokenizer


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

    prompt = "你是谁"

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
            eos_token_id=tokenizer.get_vocab()["<|im_end|>"]
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
    base_path = "../model_hub/qwen/Qwen1___5-1___8B/"
    fintune_path = "../output/qwen1.5_1.8B_full/checkpoint-20/"
    get_model_result(base_path, fintune_path)
