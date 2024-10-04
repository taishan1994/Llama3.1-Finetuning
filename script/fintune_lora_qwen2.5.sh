NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
nohup torchrun \
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6601 \
../finetune_qwen2.py \
--model_name_or_path "/data/gongoubo/Qwen-1.5-Factory/model_hub/AI-ModelScope/Qwen2___5-7B-Instruct" \
--data_path "../data/西西嘛呦.json" \
--bf16 True \
--output_dir "../output/Qwen2.5-7B-Instruct_lora" \
--num_train_epochs 1000 \
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
--model_max_length 256 \
--gradient_checkpointing True \
--lazy_preprocess True \
--deepspeed "../config/ds_config_zero2.json" \
--use_lora &