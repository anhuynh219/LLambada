import math
import numpy as np
from pathlib import Path
from shutil import rmtree

import torch
from accelerate import (Accelerator,
                        DistributedType)
from beartype.typing import List, Optional, Union
from torch import nn
from torch.utils.data import Dataset
from einops import rearrange

from models.base.tokenizers import HfHubertWithKmeans, learn_kmeans
from data import SoundDataset, get_dataloader

from utils import (beartype_jit, copy_file_to_folder, default, exists)

def cycle(dl):
    while True:
        for data in dl:
            yield data


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def noop(*args, **kwargs):
    pass

@beartype_jit
class HfHubertKmeansTrainer(nn.Module):
    """
    Trainer for kmeans part of HfHubertWithKmeans. Consists of two parts: 1) extracting Hubert features and 2) training kmeans model on these features.
    """

    def __init__(
        self,
        *,
        feature_extraction_num_steps: int,
        feature_extraction_batch_size: int,
        hubert_kmeans: HfHubertWithKmeans,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        data_max_length_seconds: Union[float, int] = 1,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.ds = dataset
        self.feature_extraction_num_steps = feature_extraction_num_steps
        self.feature_extraction_batch_size = feature_extraction_batch_size
        self.hubert_kmeans = hubert_kmeans
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length_seconds=data_max_length_seconds,
                normalize=True,
                target_sample_hz=hubert_kmeans.target_sample_hz,
                seq_len_multiple_of=hubert_kmeans.seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )
        self.print(
            f'training on {feature_extraction_num_steps * feature_extraction_batch_size} out of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=feature_extraction_batch_size, shuffle=True)

        (
            self.hubert_kmeans,
            self.dl
        ) = self.accelerator.prepare(
            self.hubert_kmeans,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def extract_hubert_features(self):

        raw_wave = next(self.dl_iter)[0]

        embed = self.hubert_kmeans.forward(wav_input=raw_wave.to(self.device), return_embed=True)

        # get features
        embed = rearrange(embed, 'b t f -> (b t) f')
        embed = self.accelerator.gather_for_metrics(embed)
        embed = embed.detach().cpu().numpy()

        return embed

    def train(self, log_fn=noop, seed=0):

        self.print('step 1: extracting features. must wait for this to complete before training kmeans.')
        features = []
        num_steps = math.ceil(self.feature_extraction_num_steps / self.accelerator.num_processes)
        while self.steps < num_steps:
            self.print(f'{int(self.steps.item())} / {num_steps} steps')
            features.append(self.extract_hubert_features())
            self.steps += 1

        features = np.concatenate(features, axis=0)

        features = features[~np.any(np.isnan(features), axis=-1)]

        self.print('step 2: training kmeans')
        if self.is_main:
            learn_kmeans(
                features,
                seed,
                str(self.results_folder / 'kmeans.joblib'),
                n_clusters=self.accelerator.unwrap_model(self.hubert_kmeans).codebook_size)

        self.print('training complete')