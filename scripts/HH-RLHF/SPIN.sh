export MODEL_NAME=zephyr-template_HH-RLHF_SPIN
export SFT_PATH=checkpoints/HH-RLHF/sft/zephyr-template_HH-RLHF_SFT

deepspeed --num_gpus 8 --master_port=9902 LLaMA-Factory/src/train_bash.py \
    --deepspeed config/ds_config_zero2.json \
    --stage dpo \
    --do_train \
    --model_name_or_path $SFT_PATH \
    --dataset HH-RLHF_SPIN \
    --template zephyr \
    --finetuning_type full \
    --output_dir checkpoints/HH-RLHF/dpo/$MODEL_NAME \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 2000 \
    --cutoff_len 2048 \
    --plot_loss \
    --num_train_epochs 1.0 \
    --dpo_beta 0.1 \
    --dpo_ftx 0.0 \
    --learning_rate 5e-7 \
    --group_by_prompt \
    --bf16
