import os
import json
import time
import argparse
import subprocess
import requests as rq

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, List
from utils import build_openchat_prompt, openchat_format_chat, split_messages, build_zephyr_prompt


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="evaluation",
        description='evaluation arguments'
    )
    group.add_argument("--template", type=str)
    group.add_argument("--model-path", type=str)
    group.add_argument("--result-path", type=str)
    group.add_argument("--data-path", type=str)
    group.add_argument("--temperature", type=float, default=0.7)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--n", type=int, default=1)
    group.add_argument("--ports", type=str, default="8080")
    group.add_argument("--manual-start-vllm", action='store_true')

    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_eval_args(parser)
    args, unknown = parser.parse_known_args()

    if args.template == 'auto':
        if "openchat" in args.model_path:
            args.template = "openchat"
        elif "zephyr" in args.model_path:
            args.template = "zephyr"
        else:
            raise AssertionError(f"Cannot infer template from model path: {args.model_path}")
    return args


def generate(instruction, args, idx: int = 0, debug=False) -> Union[str, List[str]]:
    if args.template == "openchat":
        if debug:
            print(f"instruction: {instruction}")
            print(build_openchat_prompt(openchat_format_chat(split_messages(instruction)['conversations'])))
        prompt = build_openchat_prompt(openchat_format_chat(split_messages(instruction)['conversations']))
    elif args.template == "zephyr":
        prompt = f"<|system|>\n</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
    else:
        raise NotImplementedError(f"template: {args.template} not implemented.")

    cnt = 0
    texts = ""
    if ',' in args.ports:
        ports = list(args.ports.split(','))
    else:
        ports = [args.ports]
    port = ports[idx % len(ports)]

    while texts == "":
        try:
            query = {
                "model": args.model_path,
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "n": args.n,
            }
            api_link = f"http://127.0.0.1:{port}/v1/completions"
            output = rq.post(api_link, json.dumps(query), headers={'Content-Type': 'application/json'}, timeout=300)
            texts = json.loads(output.text)["choices"][0]['text'] if args.n == 1 else [choice['text'] for choice in json.loads(output.text)["choices"]]
        except Exception as e:
            print(e)
            print(f"Retry: {cnt}, port: {port}")
        cnt += 1
        if cnt >= 4:
            print("Failed too many times.")
            raise AssertionError("Failed too many times.")
    if isinstance(texts, list):
        texts = [text.lstrip(" ") for text in texts]
    else:
        texts = texts.lstrip(" ")
    return texts


args = get_args()
if not os.path.exists(args.model_path):
    print(f"Checkpoint: {args.model_path} not exist.")
    exit()

with open(args.data_path, 'r', encoding='utf-8') as f:
    eval_set = [json.loads(l) for l in f]
    for sample in eval_set:
        if 'prompt' not in sample:
            assert len(sample['history']) == 0  # only eval first turn for hh and ultrachat
            sample['prompt'] = sample['instruction']
        if 'target' not in sample:
            sample['target'] = sample['output'][0] if isinstance(sample['output'], list) else sample['output']
            sample.pop('output')
print(f"len(eval_set) = {len(eval_set)}")


if ',' in args.ports:
    ports = list(args.ports.split(','))
else:
    ports = [args.ports]
print(f"ports: {ports}")


if not args.manual_start_vllm:
    for idx, port in enumerate(ports):  # start vllm server
        command = f"CUDA_VISIBLE_DEVICES={idx} python -m vllm.entrypoints.openai.api_server --model {args.model_path} --tensor-parallel-size 1 --port {port} > /dev/null"
        print(command)
        process = subprocess.Popen(command, shell=True)


for idx, port in enumerate(ports):  # waiting for vllm
    for _ in range(12):
        try:
            if args.template == "openchat":
                print(generate("\n\nHuman: What are the main characteristics of buddy cop movies?\n\nAssistant: ", args, idx, debug=True))
            elif args.template == "zephyr":
                print(generate("Write a story about a magical creature that goes on an adventure to save its homeland.", args, idx, debug=True))
            else:
                raise NotImplementedError(f"template: {args.template} not implemented.")
            break
        except AssertionError:
            time.sleep(10.0)
            pass


with ThreadPoolExecutor(max_workers=16 * len(ports)) as executor:
    futures = []

    for idx, example in tqdm(enumerate(eval_set)):
        futures.append(executor.submit(generate, example["prompt"], args, idx))

    for future, example in zip(tqdm(futures, desc="Retrieving results", total=len(futures)), eval_set):
        example["output"] = future.result()
        example["generator"] = os.path.basename(args.model_path)


os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
with open(args.result_path, "w", encoding="utf-8") as f:
    for sample in eval_set:
        new_sample = {
            "prompt": sample["prompt"],
            "output": sample["output"],
            "target": sample["target"] if 'target' in sample else sample['answers'][0]['answer'],
            "generator": sample["generator"],
        }
        f.write(json.dumps(new_sample) + "\n")
