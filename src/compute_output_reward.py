import os
import math
import json
import copy
import torch
import argparse
import torch.distributed as dist

from tqdm import tqdm
from torch import nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from typing import List, Optional, Tuple, Union, Dict

from utils import build_llama_input, split_messages


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="reward modeling",
        description='reward modeling arguments'
    )
    group.add_argument("--template", type=str, default="auto")
    group.add_argument("--model-path", type=str)
    group.add_argument("--data-path", type=str)
    group.add_argument("--result-path", type=str)
    group.add_argument("--llama-path", type=str, default="checkpoints/Llama-2-7b-chat-hf")
    group.add_argument("--reward-model-path", type=str, default="checkpoints/Starling-RM-7B-alpha")
    group.add_argument("--batch-size", type=int, default=8)

    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args, unknown = parser.parse_known_args()
    if args.template == 'auto':
        if "openchat" in args.model_path:
            args.template = "openchat"
        elif "zephyr" in args.model_path:
            args.template = "zephyr"
        else:
            raise AssertionError(f"Cannot infer template from model path: {args.model_path}")
    return args


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.half)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores


def load_reward_model(model_path, llama_path, reward_device):
    print("Building RM")
    with init_empty_weights():
        reward_model = GPTRewardModel(llama_path)
    reward_tokenizer = reward_model.tokenizer
    reward_tokenizer.truncation_side = "left"

    directory = os.path.join(model_path)
    checkpoint = None
    for fpath in os.listdir(directory):
        if fpath.endswith("model.bin"):
            checkpoint = os.path.join(directory, fpath)
            break
    # for fpath in os.listdir(directory):
    #     if fpath.endswith(".pt") and checkpoint is None:
    #         checkpoint = os.path.join(directory, fpath)
    #         break

    print(f"Loading Weight: {checkpoint}")

    reward_model = load_checkpoint_and_dispatch(
        reward_model, checkpoint, device_map=infer_auto_device_map(reward_model, max_memory={int(str(reward_device)[-1:]): "40GiB", "cpu": "240GiB"})
    )

    # reward_model.load_state_dict(torch.load(checkpoint), strict=False)
    reward_model.eval().requires_grad_(False).half()
    reward_model.to(reward_device)

    return reward_model, reward_tokenizer


def get_reward(samples, reward_model, reward_tokenizer, reward_device, reward_batch_size=24):
    """samples: List[str]"""
    input_ids = []
    attention_masks = []
    encodings_dict = reward_tokenizer(
        samples,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt",
    ).to(reward_device)
    input_ids = encodings_dict["input_ids"]
    attention_masks = encodings_dict["attention_mask"]
    mbs = reward_batch_size
    out = []
    for i in tqdm(range(math.ceil(len(samples) / mbs))):
        rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
        out.extend(rewards)
    return torch.hstack(out)


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, init_method=f"tcp://localhost:12345", world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    setup(rank, world_size)

    reward_model_path = args.reward_model_path
    llama_path = args.llama_path
    model_path, data_path, result_path = args.model_path, args.data_path, args.result_path

    batch_size = args.batch_size
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"rank {rank} on device {device}")

    reward_model, reward_tokenizer = load_reward_model(reward_model_path, llama_path, device)

    with open(data_path, "r", encoding='utf-8') as f:
        samples: List[Dict] = [json.loads(line) for line in f]
        original_length = len(samples)
        while not len(samples) % (world_size * batch_size) == 0:
            samples.append(samples[-1])

    print(f"original_length = {original_length}")
    print(f"padded_length = {len(samples)}")

    avg_reward = 0

    for sample in samples:
        if 'outputs' not in sample:
            sample['outputs'] = [sample['output']]

    if args.template == 'zephyr':
        all_outputs = [build_llama_input(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": sample['prompt']
                    },
                    {
                        "from": "gpt",
                        "value": output.lstrip(' ')
                    }
                ]
            }
        ) for sample in samples for output in sample['outputs']]
    else:
        all_outputs = [build_llama_input(split_messages(sample['prompt'] + output.lstrip(' '))) for sample in samples for output in sample['outputs']]
    if rank == 0:
        print("all_outputs[0]")
        print(all_outputs[0])
    outputs = all_outputs[rank * len(all_outputs) // world_size : (rank + 1) * len(all_outputs) // world_size]
    print(f"len(all_outputs) = {len(all_outputs)}")
    print(f"len(outputs) = {len(outputs)}")
    outputs_rewards = get_reward(outputs, reward_model, reward_tokenizer, device, batch_size)

    if args.template == 'zephyr':
        all_targets = [build_llama_input(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": sample['prompt']
                    },
                    {
                        "from": "gpt",
                        "value": sample['target'].lstrip(' ')
                    }
                ]
            }
        ) for sample in samples]
    else:
        all_targets = [build_llama_input(split_messages(sample['prompt'] + sample['target'].lstrip(' '))) for sample in samples]

    if rank == 0:
        print("all_targets[0]")
        print(all_targets[0])
    targets = all_targets[rank * len(all_targets) // world_size : (rank + 1) * len(all_targets) // world_size]
    print(f"len(all_targets) = {len(all_targets)}")
    print(f"len(targets) = {len(targets)}")
    targets_rewards = get_reward(targets, reward_model, reward_tokenizer, device, batch_size)

    if rank == 0:
        all_outputs_rewards = [torch.zeros_like(outputs_rewards) for _ in range(world_size)]
        dist.gather(outputs_rewards, gather_list=all_outputs_rewards, dst=0)
        outputs_rewards = []
        for output in all_outputs_rewards:
            if isinstance(output.reshape(-1).tolist(), list):
                outputs_rewards.extend(output.reshape(-1).tolist())
            else:
                outputs_rewards.append(output.reshape(-1).tolist())
        # outputs_rewards = [output.reshape(-1).tolist() for output in all_outputs_rewards]

        print("outputs_rewards:", outputs_rewards)

        avg_reward = sum(outputs_rewards) / len(outputs_rewards)
        print(model_path)
        print(f"len(rewards) = {len(outputs_rewards)}")
        print(f"avg_reward = {avg_reward}")

        all_targets_rewards = [torch.zeros_like(targets_rewards) for _ in range(world_size)]
        dist.gather(targets_rewards, gather_list=all_targets_rewards, dst=0)
        targets_rewards = [target.tolist() for target in all_targets_rewards]

        print("targets_rewards:", targets_rewards)

        targets_rewards = []
        for reward in all_targets_rewards:
            if isinstance(reward.reshape(-1).tolist(), list):
                targets_rewards.extend(reward.reshape(-1).tolist())
            else:
                targets_rewards.append(reward.reshape(-1).tolist())

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding='utf-8') as f:
            start_index = 0
            for idx, sample in enumerate(samples[:original_length]):

                new_sample = {
                    'prompt': sample['prompt'],
                    'outputs': sample['outputs'],
                    'rewards': outputs_rewards[start_index : start_index + len(sample['outputs'])],
                    'target': sample['target'],
                    'target_rewards': targets_rewards[idx],
                }
                start_index += len(sample['outputs'])
                f.write(json.dumps(new_sample) + "\n")

            for _, sample in enumerate(samples[original_length:]):
                start_index += len(sample['outputs'])

            assert start_index == len(all_outputs)
    else:
        dist.gather(outputs_rewards, dst=0)
        dist.gather(targets_rewards, dst=0)

    cleanup()


if __name__ == "__main__":
    args = get_args()
    print(args)
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)
