export MODEL_NAME=openchat_3.5-template_Nectar_SFT
export BASE_PATH=checkpoints/Nectar/Mistral_with_openchat_tokenizer  # You will need to incorporate the `<|end_of_turn|>` token into the model.

deepspeed --num_gpus 8 --master_port=9902 LLaMA-Factory/src/train_bash.py \
    --deepspeed config/ds_config_zero2.json \
    --stage sft \
    --do_train \
    --model_name_or_path $BASE_PATH \
    --dataset Nectar_SFT \
    --template openchat \
    --finetuning_type full \
    --output_dir checkpoints/Nectar/sft/$MODEL_NAME \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 8000 \
    --learning_rate 5e-6 \
    --num_train_epochs 1.0 \
    --cutoff_len 1024 \
    --use_only_last_turn \
    --overwrite_output_dir \
    --plot_loss \
    --bf16
