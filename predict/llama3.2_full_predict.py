from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_result(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    ).eval()


    prompt = "你叫什么名字？"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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
            model_inputs.input_ids,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    response = get_result(model_inputs, model)
    print("问题：", prompt)
    print("结果：", response)


if __name__ == '__main__':
    # model_path = "/data/gongoubo/Qwen-1.5-Factory/model_hub/AI-ModelScope/Llama-3___2-3B-Instruct"
    model_path = "/data/gongoubo/Qwen-1.5-Factory/output/llama3.2_3B_Instruct_full/checkpoint-775"
    get_model_result(model_path)
