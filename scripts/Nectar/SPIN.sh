export MODEL_NAME=openchat_3.5-template_Nectar_SPIN
export SFT_PATH=checkpoints/Nectar/sft/openchat_3.5-template_Nectar_SFT

deepspeed --num_gpus 8 --master_port=9902 LLaMA-Factory/src/train_bash.py \
    --deepspeed config/ds_config_zero2.json \
    --stage dpo \
    --do_train \
    --model_name_or_path $SFT_PATH \
    --dataset Nectar_SPIN \
    --template openchat \
    --finetuning_type full \
    --output_dir checkpoints/Nectar/dpo/$MODEL_NAME \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --cutoff_len 1024 \
    --plot_loss \
    --num_train_epochs 1.0 \
    --dpo_beta 0.1 \
    --learning_rate 5e-7 \
    --seed 42 \
    --data_seed 42 \
    --bf16
