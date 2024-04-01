import torch
import hashlib

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from functools import partial

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
import datasets
from huggingface_hub import Repository, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler

from transformers import __version__
from transformers.utils import logging
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.get_logger(__name__)


def get_prompt_grouped_indices(prompts, batch_size, generator=None):
    prompts_dict = {}
    for prompt in prompts:
        prompts_dict[prompt] = []
    for i, prompt in enumerate(prompts):
        prompts_dict[prompt].append(i)
    prompts_list = list(prompts_dict.keys())
    indices = torch.randperm(len(prompts_list), generator=generator)
    return [i for indice in indices for i in prompts_dict[prompts_list[indice]]]


class PromptGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together prompts of the dataset while
    keeping randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None:
            raise ValueError("Dataset must be provided.")

        self.batch_size = batch_size

        prompts = [hashlib.md5(np.array(feature[model_input_name]).tobytes()).hexdigest() for feature in dataset]

        self.prompts = prompts
        self.generator = generator

    def __len__(self):
        return len(self.prompts)

    def __iter__(self):
        indices = get_prompt_grouped_indices(self.prompts, self.batch_size, generator=self.generator)
        return iter(indices)

