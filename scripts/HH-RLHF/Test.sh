echo "inference"

for MODEL_NAME in zephyr-template_HH-RLHF_SFT
do
    python src/inference_valid.py --model-path checkpoints/HH-RLHF/sft/$MODEL_NAME --temperature 1.0 --top-p 1.0 --template zephyr --result-path results/hh-rlhf/test/inference/${MODEL_NAME}.jsonl --ports 8080,8081,8082,8083,8084,8085,8086,8087 --data-path data/HH-RLHF/test.jsonl
    sleep 5
    pkill -f 'python -m vllm.entrypoints.openai.api_server'
    sleep 5
done

for MODEL_NAME in zephyr-template_HH-RLHF_DPO zephyr-template_HH-RLHF_SPIN zephyr-template_HH-RLHF_Self-Contrast_1 zephyr-template_HH-RLHF_Self-Contrast_2 zephyr-template_HH-RLHF_Self-Contrast_4 zephyr-template_HH-RLHF_Self-Contrast_8 zephyr-template_HH-RLHF_Self-Contrast_16
do
    python src/inference_valid.py --model-path checkpoints/HH-RLHF/dpo/$MODEL_NAME --temperature 1.0 --top-p 1.0 --template zephyr --result-path results/hh-rlhf/test/inference/${MODEL_NAME}.jsonl --ports 8080,8081,8082,8083,8084,8085,8086,8087 --data-path data/HH-RLHF/test.jsonl
    sleep 5
    pkill -f 'python -m vllm.entrypoints.openai.api_server'
    sleep 5
done


echo "reward modeling"

for MODEL_NAME in zephyr-template_HH-RLHF_SFT
do
    python src/compute_output_reward.py --model-path checkpoints/HH-RLHF/sft/$MODEL_NAME --data-path results/hh-rlhf/test/inference/${MODEL_NAME}.jsonl --result-path results/hh-rlhf/test/reward/${MODEL_NAME}.jsonl --template zephyr
done

for MODEL_NAME in zephyr-template_HH-RLHF_DPO zephyr-template_HH-RLHF_SPIN zephyr-template_HH-RLHF_Self-Contrast_1 zephyr-template_HH-RLHF_Self-Contrast_2 zephyr-template_HH-RLHF_Self-Contrast_4 zephyr-template_HH-RLHF_Self-Contrast_8 zephyr-template_HH-RLHF_Self-Contrast_16
do
    python src/compute_output_reward.py --model-path checkpoints/HH-RLHF/dpo/$MODEL_NAME --data-path results/hh-rlhf/test/inference/${MODEL_NAME}.jsonl --result-path results/hh-rlhf/test/reward/${MODEL_NAME}.jsonl --template zephyr
done
