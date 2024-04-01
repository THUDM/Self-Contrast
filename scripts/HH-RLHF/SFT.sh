export MODEL_NAME=zephyr-template_HH-RLHF_SFT
export BASE_PATH=checkpoints/Mistral-7B-v0.1

deepspeed --num_gpus 8 --master_port=9901 LLaMA-Factory/src/train_bash.py \
    --deepspeed config/ds_config_zero2.json \
    --stage sft \
    --do_train \
    --model_name_or_path $BASE_PATH \
    --dataset HH-RLHF_SFT \
    --template zephyr \
    --finetuning_type full \
    --output_dir checkpoints/HH-RLHF/sft/$MODEL_NAME \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 4000 \
    --learning_rate 5e-6 \
    --num_train_epochs 1.0 \
    --cutoff_len 1024 \
    --use_only_last_turn \
    --overwrite_output_dir \
    --plot_loss \
    --fp16
