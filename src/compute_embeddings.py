import json
import copy
import pickle
import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from datetime import timedelta
from tqdm import tqdm
from angle_emb import AnglE
from typing import List


def add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="compute embedding",
        description='compute embedding'
    )
    group.add_argument("--model-path", type=str, help="path to UAE-Large-V1")
    group.add_argument("--data-path", type=str, help="path to inference result file")
    group.add_argument("--save-path", type=str, help="path to save dir")
    group.add_argument("--save-embeddings", action='store_true')
    group.add_argument("--npy", action='store_true')
    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_inference_args(parser)
    args, unknown = parser.parse_known_args()
    return args


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, init_method=f"tcp://localhost:12345", world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def rank_list(l: List) -> List:
    if isinstance(l, float):
        l = [l]
    sorted_list = sorted(l)
    rankings = [sorted_list.index(value) for value in l]
    return rankings


def main(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(rank if torch.cuda.is_available() else "cpu")
    print(f"rank: {rank}, world_size: {world_size}, device: {device}")

    model_path = args.model_path
    data_path = args.data_path
    save_path = args.save_path

    file_paths = os.listdir(data_path) if os.path.isdir(data_path) else [os.path.basename(data_path)]
    data_path = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)

    print(f"model_path: {model_path}")
    print(f"data_path: {data_path}")
    print(f"save_path: {save_path}")
    print(f"file_paths: {file_paths}")

    if os.path.exists(save_path):
        print(Warning(f"{save_path} already exists"))

    angle = AnglE.from_pretrained(model_path, pooling_strategy='cls', device=device).cuda()

    for file_path in file_paths:
        if file_path.endswith(".jsonl"):
            with open(os.path.join(data_path, file_path), "r", encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f]

            for sample in tqdm(dataset):
                if 'prompt' not in sample:
                    sample['prompt'] = sample['instruction']
                if 'target' not in sample and 'outputs' in sample:
                    sample['target'] = sample['output'][0] if isinstance(sample['output'], list) else sample['output']

            subset = dataset[rank * len(dataset) // world_size : (rank + 1) * len(dataset) // world_size]
            subset_vectors = []
            for sample in tqdm(subset):
                if 'outputs' not in sample:
                    sample['outputs'] = [sample['output']] if 'output' in sample else []

                text = sample['outputs'] + [sample['target']]

                vectors = torch.tensor(angle.encode(text)).cpu()
                subset_vectors.append(vectors)
            subset_vectors = torch.stack(subset_vectors, dim=0).cuda()

            all_subset_vectors = [torch.zeros_like(subset_vectors) for _ in range(world_size)]
            dist.barrier()
            dist.all_gather(all_subset_vectors, subset_vectors)
            if rank == 0:
                all_vectors = []
                for subset_vectors in all_subset_vectors:
                    all_vectors += [vectors for vectors in subset_vectors]

                for sample, vectors in zip(dataset, all_vectors):
                    target_embeddings = vectors[-1 : ]
                    outputs_embeddings = vectors[ : -1]

                    if args.save_embeddings:
                        sample['outputs_embeddings'] = outputs_embeddings.tolist()
                        sample['target_embeddings'] = target_embeddings.tolist()
                    sample['outputs_dist'] = F.cosine_similarity(target_embeddings, outputs_embeddings, dim=-1).squeeze(dim=0).tolist() if outputs_embeddings.size(0) > 0 else []
                    sample['outputs_dist_rank'] = rank_list(sample['outputs_dist']) if outputs_embeddings.size(0) > 0 else []

                os.makedirs(save_path, exist_ok=True)

                if args.npy:
                    np.save(os.path.join(save_path, file_path.replace('jsonl', 'npy')), dataset)
                else:
                    with open(os.path.join(save_path, file_path.replace('jsonl', 'pkl')), "wb") as f:
                        pickle.dump(dataset, f)
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = get_args()
    print(args)
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)
