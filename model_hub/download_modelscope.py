from modelscope.hub.snapshot_download import snapshot_download
# model_dir = snapshot_download('qwen/Qwen1.5-1.8B-Chat', cache_dir='.', revision='master')

model_name = "qwen/Qwen1.5-1.8B"
model_name = "qwen/Qwen1.5-72B-Chat"
model_name = "LLM-Research/Meta-Llama-3-8B-Instruct"
model_dir = snapshot_download(model_name, cache_dir='.', revision='master')

