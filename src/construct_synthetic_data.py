import os
import json
import random
import pickle
import numpy as np
import argparse

from tqdm import tqdm
from typing import List, Dict, Tuple

from utils import split_messages


random.seed(42)
np.random.seed(42)


def convert_format_from_hh_to_openchat(chat: str) -> Dict:
    system = ""
    split_marks = ["\n\nHuman: ", "\n\nAssistant: "]
    conversations = []
    while split_marks[0] in chat or split_marks[1] in chat:
        if chat.startswith(split_marks[0]):
            index = chat.find(split_marks[1])
            if index == -1:
                value, chat = chat, ''
            else:
                value, chat = chat[:index], chat[index:]
            conversation = {
                "from": "human",
                "value": value[len(split_marks[0]):]
            }
        elif chat.startswith(split_marks[1]):
            index = chat.find(split_marks[0])
            if index == -1:
                value, chat = chat, ''
            else:
                value, chat = chat[:index], chat[index:]
            conversation = {
                "from": "gpt",
                "value": value[len(split_marks[1]):]
            }
        else:
            raise AssertionError(f"Chat Error: {chat}")
        conversations.append(conversation)

    prompt = ''
    for i, conversation in enumerate(conversations):
        if i % 2 == 0:
            prompt += f"GPT4 Correct User: {conversation['value']}<|end_of_turn|>GPT4 Correct Assistant:"
        elif conversation['value']:
            prompt += f" {conversation['value']}<|end_of_turn|>"
    return prompt


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="compute embedding",
        description='compute embedding'
    )
    group.add_argument("--mode", type=str, help="data construction mode")
    group.add_argument("--dataset-format", type=str, default="hh-rlhf", choices=["hh-rlhf", "nectar", "ultrachat"], help="dataset format")
    group.add_argument("--data-path", type=str, help="data generated by the model")
    group.add_argument("--save-path", type=str, help="save path")
    group.add_argument("--negative-quantity", type=int, default=1, help="negative sample quantity")
    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args, unknown = parser.parse_known_args()
    return args


args = get_args()
mode = args.mode
k = args.negative_quantity
data_format = args.dataset_format
data_path = args.data_path
save_path = args.save_path

empty_count = 0

os.makedirs(save_path, exist_ok=False)


def get_mode(mode: str) -> Tuple[str, float]:
    modes = ["all", "last-p", "alldedup", "lastdedup-p"]
    final_mode, p = None, None
    for possible_mode in modes:
        if "-p" in possible_mode and mode.startswith(possible_mode[:-1]):
            final_mode = possible_mode.replace('-p', '')
            p = mode[len(final_mode) + 1:]
            p = float(p) / (10.0 ** len(p))
        elif mode == possible_mode:
            final_mode, p = possible_mode, None
    return final_mode, p


def rank_list(l: List) -> List:
    if isinstance(l, float):
        l = [l]
    sorted_list = sorted(l)
    rankings = []
    for value in l:
        v = sorted_list.index(value)
        while v in rankings:
            v += 1
        rankings.append(v)
    # rankings = [sorted_list.index(value) for value in l]
    return rankings


def convert_to_pairwise_format(conversation: Dict, sample: Dict, target: str, k=1, target_silhouette_score=0) -> List[Dict]:
    """
    convert from:
    conversation = {
        "conversations": [
        {
            "from": "human",
            "value": "用户指令"
        },
        {
            "from": "gpt",
            "value": "模型回答"
        }
        ],
        "system": "系统提示词（选填）"
    }
    to:
    data = {
        "instruction": "用户指令（必填）",
        "input": "用户输入（选填）",
        "output": "模型回答（必填）",
        "system": "系统提示词（选填）",
        "history": [
        ["第一轮指令（选填）", "第一轮回答（选填）"],
        ["第二轮指令（选填）", "第二轮回答（选填）"]
        ]
    }
    """
    
    if data_format == "nectar":
        assert conversation["conversations"][-1]["value"] == target
    else:
        assert conversation["conversations"][-1]["value"] == ''

    if sum([output.lstrip(' ') == "" for output in sample["outputs"]]):
        global empty_count
        empty_count += sum([output.lstrip(' ') == "" for output in sample["outputs"]])
        print(f"empty_count: {empty_count}")

    data_mode, ratio = get_mode(mode)


    if data_mode == "all":
        aval_index = list(range(len(sample["outputs"])))
        aval_index = [i for i in aval_index if sample["outputs"][i].lstrip(' ') != ""]
        ns = sorted(random.sample(aval_index, k=k))
        outputs = [
            [
                target,
                sample["outputs"][n]
            ] for n in ns
        ]
    elif data_mode == "alldedup":
        responses = list(set(sample["outputs"]))
        aval_index = list(range(len(responses)))
        aval_index = [i for i in aval_index if responses[i].lstrip(' ') != ""]
        ns = sorted(random.sample(aval_index, k=k)) if len(aval_index) >= k else []
        outputs = [
            [
                target,
                responses[n].lstrip(' ')
            ] for n in ns
        ]
    elif data_mode == "last":
        threshold = sorted(sample["outputs_dist_rank"])[int(len(sample["outputs"]) * ratio)]
        responses = [output for output, output_dist_rank in zip(sample["outputs"], sample["outputs_dist_rank"]) if output_dist_rank < max(threshold, k)]

        aval_index = list(range(len(responses)))
        aval_index = [i for i in aval_index if responses[i].lstrip(' ') != ""]
        ns = sorted(random.sample(aval_index, k=k)) if len(aval_index) >= k else []
        outputs = [
            [
                target,
                responses[n]
            ] for n in ns
        ]
    elif data_mode == "lastdedup":
        responses, responses_dist_rank = [], []
        for output, output_dist_rank in zip(sample["outputs"], sample["outputs_dist_rank"]):
            if output not in responses:
                responses.append(output)
                responses_dist_rank.append(output_dist_rank)

        max_negatives = min(int(len(sample["outputs"]) * ratio), len(responses) - 1)
        if len(responses):
            threshold = sorted(responses_dist_rank)[max_negatives]
            responses = [output for output, output_dist_rank in zip(responses, responses_dist_rank) if output_dist_rank < threshold]

            aval_index = list(range(len(responses)))
            aval_index = [i for i in aval_index if responses[i].lstrip(' ') != ""]
            ns = sorted(random.sample(aval_index, k=k)) if len(aval_index) >= k else []
            outputs = [
                [
                    target,
                    responses[n].lstrip(' ')
                ] for n in ns
            ]
        else:
            outputs = []
    else:
        raise NotImplementedError(f"mode = {mode}")
    data_list = [
        {
            "instruction": conversation["conversations"][-2]["value"],
            "input": "",
            "output": output,
            "system": "",
            "history": [
                [conversation["conversations"][i]["value"], conversation["conversations"][i+1]["value"]]
                for i in range(0, len(conversation["conversations"]) - 2, 2)
            ]
        } for output in outputs
    ]

    return data_list


if os.path.isfile(data_path):
    file_paths = [data_path]
    data_dir = os.path.basename(data_path)
else:
    data_dir = data_path
    file_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith(".pkl")]

if len(file_paths) == 0:
    data_dir = data_path
    file_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if file_name.endswith(".jsonl")]


for file_path in file_paths:
    print(file_path)

    if file_path.endswith(".pkl"):
        with open(file_path, 'rb') as f:
            samples = pickle.load(f)
    elif  file_path.endswith(".jsonl"):
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(l) for l in f]

    target_silhouette_score = 0
    data_mode, ratio = get_mode(mode)
    print(f"data_mode = {data_mode}, ratio = {ratio}")

    dataset = []
    for sample in tqdm(samples):
        if data_format == "nectar":
            data = split_messages(sample['prompt'] + sample['target'])
            if 'answers' not in sample and "reward" in data_mode:
                answers = [sample['target']]
            else:
                answers = [answer['answer'] for answer in sample['answers']]
            
            dataset += convert_to_pairwise_format(data, sample, answers[0], k, target_silhouette_score)
        elif data_format == "ultrachat":
            if 'messages' not in sample and "reward" in data_mode:
                data = {
                    "conversations": [
                        {
                            'from': 'human',
                            'value': sample['prompt'],
                        },
                        {
                            'from': 'assistant',
                            'value': '',
                        } 
                    ]
                }
            else:
                data = {
                    "conversations": [
                        {
                            'from': message['role'],
                            'value': message['content'],
                        } for message in sample['messages']
                    ]
                }
        elif data_format == 'hh-rlhf':
            messages = []
            for conv in sample['history']:
                messages.append({'role': 'human', 'content': conv[0]})
                messages.append({'role': 'assistant', 'content': conv[1]})
            messages.append({'role': 'human', 'content': sample['instruction']})
            messages.append({'role': 'assistant', 'content': ''})
            data = {
                "conversations": [
                    {
                        'from': message['role'],
                        'value': message['content'],
                    } for message in messages
                ]
            }

            dataset += convert_to_pairwise_format(data, sample, sample['target'], k, target_silhouette_score)
        else:
            raise NotImplementedError(f"Unsupported data format: {data_format}")

    print(f"len(dataset): {len(dataset)}")

    print(os.path.join(save_path, os.path.basename(file_path).replace('pkl', 'jsonl')))
    with open(os.path.join(save_path, os.path.basename(file_path).replace('pkl', 'jsonl')), 'w', encoding='utf-8') as f:
        for d in dataset:
            f.write(json.dumps(d) + '\n')

print("Done!")
