# Self-Contrast

### Extensive Self-Contrast Enables Feedback-Free Language Model Alignment

<p align="center">
   🤗 <a href="#model" target="_blank">Model</a> • 📚 <a href="#data" target="_blank">Data</a> • 📃 <a href="https://arxiv.org/abs/2404.00604" target="_blank">Paper</a>
</p>

<div align="center">
<img src="assets/SelfContrastIntro4.png" alt="Self-Contrast" width="90%" />
</div>

Self-Contrast is an innovative method that offers an annotation-free approach for aligning with human preference. When evaluated against the MT Bench and Alpaca eval benchmarks, Self-Contrast surpassed the performance of the original DPO with synthetic data based on SFT data. This advancement was achieved by augmenting the number of negative samples and incorporating a straightforward sample filtering technique. The results from Self-Contrast underscore a consistent improvement in alignment performance through the strategic increase of negative samples.

## Table of Contents

- [Start](#Start)
  - [Setup Environment](#setup-environment)
  - [Download Public Models](#download-public-models)
- [Data](#data)
  - [Quick Start](#quick-start)
  - [Build Up Self-Contrast Data](#build-up-self-contrast-data)
- [Training](#training)
  - [SFT](#sft)
  - [DPO](#dpo)
  - [SPIN](#spin)
  - [Self-Contrast](#self-contrast)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Start

### Setup Environment

```bash
pip install -r requirements.txt
```

Due to the special dependency of vllm, you may need to set up two separate environments for training and inference. Carefully consider whether to install vllm in the training environment.

```bash
python -m venv inference
source inference/bin/activate
pip install vllm accelerate
```

### Download Public Models

To utilize the functionalities provided by this repository, please download the following pre-trained models, and place them in `checkpoints`:

1. **UAE-Large-V1**
   - Download Link: [UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)
2. **Starling-RM-7B-alpha**
   - Download Link: [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha)
3. **Llama-2-7b-chat-hf**
   - Download Link: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
4. **Mistral-7B-v0.1**
   - Download Link: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

Please note that when conducting experiments with Nectar, the openchat template is used. Consequently, you will need to incorporate the `<|end_of_turn|>` token into the model as per the instructions provided in the [imoneoi/openchat](https://github.com/imoneoi/openchat/blob/master/ochat/scripts/hf_add_tokens.py) repository.

## Data

### Quick Start

To get started with the experiment more quickly, you can directly download the data we have prepared.

- Download Link: [Self-Contrast Data](https://cloud.tsinghua.edu.cn/d/ffd5b4b22369499083eb/)
- Place it into `data` dictionary.

You can also process the data yourself by downloading the raw data from [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar), [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf), or by using other SFT datasets.

### Build Up Self-Contrast Data

#### 1. Self-Generate Massive Responses

Before conducting inference, you need to train an [SFT](#SFT) model first.
Then use the SFT model to generate massive responses.

```bash
source inference/bin/activate

python src/inference.py \
--model-path checkpoints/HH-RLHF/sft/zephyr-template_HH-RLHF_SFT \
--data-path data/HH-RLHF/for_inference/train.jsonl \
--result-path inference/HH-RLHF/responses/train.jsonl \
--n 32 \
--template zephyr \
--dataset hh-rlhf \
--max-tokens 2048

sleep 5
pkill -f 'python -m vllm.entrypoints.openai.api_server'
sleep 5

deactivate
```

#### 2. Compute Embeddings

```bash
python src/compute_embeddings.py \
--model-path checkpoints/UAE-Large-V1 \
--data-path inference/HH-RLHF/responses \
--save-path inference/HH-RLHF/embeddings
```

#### 3. Construct Data with Cosine Similarity

Here is an example of using 75% dissimilar as negative on HH-RLHF:

```bash
for i in 1 2 4 8 16; do
   python src/construct_synthetic_data.py \
   --mode lastdedup-75 \
   --data-path inference/HH-RLHF/embeddings \
   --save-path inference/HH-RLHF/data/lastdedup-75-$i \
   --negative-quantity $i
done
```

Here is another example of using the response directly as negative.

```bash
for i in 1 2 4 8 16; do
   python src/construct_synthetic_data.py \
   --mode alldedup \
   --data-path inference/HH-RLHF/embeddings \
   --save-path inference/HH-RLHF/data/alldedup-$i \
   --negative-quantity $i
done
```

Before training, add your synthetic data to `data/dataset_info.json`.

## Training

You can use the fully automated scripts, including training, testing, and plotting.
```bash
bash scripts/HH-RLHF/All_In_One.sh
```

### SFT

```bash
# HH-RLHF
bash LLaMA-Factory/scripts/HH-RLHF/SFT.sh

# Nectar
bash LLaMA-Factory/scripts/Nectar/SFT.sh

# UltraChat
# we use alignment-handbook/zephyr-7b-sft-full for UltraChat
```

### DPO

```bash
# HH-RLHF
bash LLaMA-Factory/scripts/HH-RLHF/DPO.sh

# Nectar
bash LLaMA-Factory/scripts/Nectar/DPO.sh

# UltraChat
bash LLaMA-Factory/scripts/UltraChat/DPO.sh
```

### SPIN

```bash
# HH-RLHF
bash LLaMA-Factory/scripts/HH-RLHF/SPIN.sh

# Nectar
bash LLaMA-Factory/scripts/Nectar/SPIN.sh

# UltraChat
bash LLaMA-Factory/scripts/UltraChat/SPIN.sh
```

### Self-Contrast

```bash
# HH-RLHF
bash LLaMA-Factory/scripts/HH-RLHF/Self-Contrast_1.sh
bash LLaMA-Factory/scripts/HH-RLHF/Self-Contrast_16.sh

# Nectar
bash LLaMA-Factory/scripts/Nectar/Self-Contrast_1.sh
bash LLaMA-Factory/scripts/Nectar/Self-Contrast_16.sh

# UltraChat
bash LLaMA-Factory/scripts/UltraChat/Self-Contrast_1.sh
bash LLaMA-Factory/scripts/UltraChat/Self-Contrast_16.sh
```

## Evaluation

We provide a script that allows you to use the reward model to calculate the win rate against the SFT target.

```bash
source inference/bin/activate

# inference & compute reward
bash scripts/HH-RLHF/Test.sh

# compute winrate
python src/compute_winrate.py --result-dir results/hh-rlhf/test/reward

# plot figure
python src/draw_figures.py
```

## Acknowledgments

- Training: We would like to express our deep appreciation to the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) project, for providing an exceptional tool. The versatility and efficiency of LLaMA-Factory have significantly enhanced our model training process.
- Evaluation: We wish to extend our sincere thanks to [FastChat](https://github.com/lm-sys/FastChat), [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [GSM8K-eval](https://github.com/Guangxuan-Xiao/GSM8K-eval/) for their valuable contributions.

## Citation
```
@misc{liu2024extensive,
      title={Extensive Self-Contrast Enables Feedback-Free Language Model Alignment}, 
      author={Xiao Liu and Xixuan Song and Yuxiao Dong and Jie Tang},
      year={2024},
      eprint={2404.00604},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
