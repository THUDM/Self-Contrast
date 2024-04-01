import os
import json
import time
import argparse
import subprocess
import requests as rq

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, List
from transformers import AutoTokenizer

from utils import build_openchat_prompt, openchat_format_chat, split_messages, build_zephyr_prompt, build_mistral_prompt


def add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="inference",
        description='inference arguments'
    )
    group.add_argument("--model-path", type=str)
    group.add_argument("--data-path", type=str)
    group.add_argument("--result-path", type=str)
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top_p", type=float, default=1.0)
    group.add_argument("--n", type=int, default=1)
    group.add_argument("--ports", type=str, default="8080")
    group.add_argument("--template", type=str, default="zephyr")
    group.add_argument("--dataset", type=str, default="hh-rlhf")
    group.add_argument("--max-tokens", type=int, default=2048)
    group.add_argument("--manual-start-vllm", action='store_true')

    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_inference_args(parser)
    args, unknown = parser.parse_known_args()
    return args


def generate(input_str: str, idx: int) -> Union[str, List[str]]:
    global tokenizer
    if ',' in args.ports:
        ports = list(args.ports.split(','))
    else:
        ports = [args.ports]
    port = ports[idx % len(ports)]

    cnt = 0
    texts = ""
    input_ids = tokenizer.encode(input_str)

    while texts == "":
        if len(input_ids) >= args.max_tokens:
            texts = ""
            break 
        try:
            query = {
                "model": model_path,
                "prompt": input_str,
                "max_tokens": args.max_tokens - len(input_ids),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "n": args.n,
            }
            print(idx, time.time() - start_time, len(input_ids), input_str[:512])
            api_link = f"http://127.0.0.1:{port}/v1/completions"
            output = rq.post(api_link, json.dumps(query), headers={'Content-Type': 'application/json'}, timeout=450)
            texts = json.loads(output.text)["choices"][0]['text'] if args.n == 1 else [choice['text'] for choice in json.loads(output.text)["choices"]]
        except Exception as e:
            print(e)
            print(f"Retry: {cnt}")
            cnt += 1
            time.sleep(1)
        if cnt >= 2:
            raise AssertionError("Failed too many times.")
    texts = [text.lstrip(" ") for text in texts]
    return texts


args = get_args()

model_path = args.model_path
data_path = args.data_path
result_path = args.result_path

print(f"model_path: {model_path}")
print(f"data_path: {data_path}")
print(f"result_path: {result_path}")

os.makedirs(os.path.dirname(result_path), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open(data_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f.readlines()]
print(f"len(samples) = {len(samples)}")


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

start_time = time.time()

for idx, port in enumerate(ports):  # waiting for vllm
    for _ in range(12):
        try:
            if args.template == "openchat":
                print(generate("\n\nHuman: What are the main characteristics of buddy cop movies?\n\nAssistant: ", idx))
            elif args.template == "zephyr":
                print(generate("Write a story about a magical creature that goes on an adventure to save its homeland.", idx))
            else:
                raise NotImplementedError(f"template: {args.template} not implemented.")
            break
        except AssertionError:
            time.sleep(10.0)
            pass

with ThreadPoolExecutor(max_workers=128) as executor:
    futures = []
    for idx, sample in enumerate(samples):
        if args.template == "openchat":
            if args.dataset == "nectar":
                input_str = build_openchat_prompt(openchat_format_chat(split_messages(sample["prompt"])['conversations']))
            else:
                raise NotImplementedError(f"Not implemented dataset: {args.dataset}")
        elif args.template == "zephyr":
            if args.dataset == "ultrachat":
                messages = sample["messages"]
                messages[-1]['content'] = ''
                input_str = build_zephyr_prompt(messages)
            elif args.dataset == "hh-rlhf":
                messages = []
                for conv in sample['history']:
                    messages.append({'content': conv[0]})
                    messages.append({'content': conv[1]})
                messages.append({'content': sample['instruction']})
                input_str = build_zephyr_prompt(messages)
            else:
                raise NotImplementedError(f"Not implemented dataset: {args.dataset}")
        else:
            raise NotImplementedError(f"Not implemented template: {args.template}")

        futures.append(executor.submit(generate, input_str, idx))

    for future, sample in zip(tqdm(futures, desc="Retrieving results", total=len(futures)), samples):
        result = future.result()
        sample['output' if isinstance(result, str) else 'outputs'] = result


with open(result_path, "w", encoding="utf-8") as f:
    for idx, sample in enumerate(samples):
        if 'outputs' not in sample and sample['output'] == "":
            print(f"Jump: {idx}")
            continue
        f.write(json.dumps(sample) + "\n")
