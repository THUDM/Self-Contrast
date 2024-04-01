export MODEL_NAME=zephyr-template_UltraChat_DPO
export SFT_PATH=checkpoints/UltraChat/sft/zephyr-7b-sft-full

deepspeed --num_gpus 8 --master_port=9902 LLaMA-Factory/src/train_bash.py \
    --deepspeed config/ds_config_zero2.json \
    --stage dpo \
    --do_train \
    --model_name_or_path $SFT_PATH \
    --dataset UltraChat_DPO \
    --template zephyr \
    --finetuning_type full \
    --output_dir checkpoints/UltraChat/dpo/$MODEL_NAME \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 2000 \
    --cutoff_len 2048 \
    --plot_loss \
    --num_train_epochs 1.0 \
    --dpo_beta 0.1 \
    --learning_rate 5e-7 \
    --bf16
