import os
import json
import math
import argparse


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="winrate",
        description='winrate arguments'
    )
    group.add_argument("--result-dir", type=str, default='results/hh-rlhf/test/reward')
    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args, unknown = parser.parse_known_args()
    return args


def compare(sample):
    if sample['rewards'][0] > sample['target_rewards']:
        return 1
    elif sample['rewards'][0] < sample['target_rewards']:
        return 0
    return 0.5


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def reward(sample):
    assert len(sample['rewards']) == 1
    return sigmoid(sample['rewards'][0])


def get_winrate(result_dir):
    values = {}
    for file_name in sorted(os.listdir(result_dir)):
        with open(os.path.join(result_dir, file_name), 'r', encoding='utf-8') as f:
            samples = [json.loads(l) for l in f]

        rewards = [reward(sample) for sample in samples]
        win = [compare(sample) for sample in samples]
        winrate = sum(win) / len(win)
        average_reward = sum(rewards) / len(rewards)

        model_name = file_name.replace('.jsonl', '')
        values[model_name] = (winrate, average_reward)
    return values


if __name__ == '__main__':
    args = get_args()
    print()
    values = get_winrate(args.result_dir)
    for model_name, (winrate, average_reward) in values.items():
        print(model_name.ljust(80), round(winrate * 100, 2), round(average_reward, 3))
