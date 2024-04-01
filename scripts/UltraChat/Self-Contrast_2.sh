export MODEL_NAME=zephyr-template_UltraChat_Self-Contrast_2
export SFT_PATH=checkpoints/UltraChat/sft/zephyr-7b-sft-full-c3160e9

deepspeed --num_gpus 8 --master_port=9902 LLaMA-Factory/src/train_bash.py \
    --deepspeed config/ds_config_zero2.json \
    --stage dpo \
    --do_train \
    --model_name_or_path $SFT_PATH \
    --dataset UltraChat_Self-Contrast_2 \
    --template zephyr \
    --finetuning_type full \
    --output_dir checkpoints/UltraChat/dpo/$MODEL_NAME \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 2000 \
    --cutoff_len 2048 \
    --plot_loss \
    --num_train_epochs 1.0 \
    --dpo_beta 0.1 \
    --learning_rate 5e-7 \
    --bf16
