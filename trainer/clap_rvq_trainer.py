import math
import time
from pathlib import Path
from shutil import rmtree

import torch
from accelerate import (Accelerator,
                        DistributedType)
from beartype.typing import List, Optional, Union
from torch import nn
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from models.base.tokenizers.clap_quantized import ClapQuantized
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
class ClapRVQTrainer(nn.Module):
    """
    Learn the residual vector quantizer to turn CLAP embeddings into discrete tokens.
    """

    def __init__(
        self,
        *,
        num_train_steps,
        batch_size,
        accumulate_batches: Optional[int] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        wd=0.,
        max_grad_norm=0.5,
        data_max_length_seconds: Union[float, int] = 10,
        valid_frac=0.05,
        random_split_seed=42,
        save_results_every=100,
        save_model_every=1000,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.log_with = accelerate_kwargs['log_with'] if 'log_with' in accelerate_kwargs else None

        self.audio_conditioner = audio_conditioner
        self.ds = dataset
        self.num_train_steps = num_train_steps
        self.accumulate_batches = accumulate_batches
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length_seconds=data_max_length_seconds,
                target_sample_hz=audio_conditioner.sample_rate,
                seq_len_multiple_of=None,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=batch_size, shuffle=True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size=batch_size, shuffle=True)

        (
            self.audio_conditioner,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.audio_conditioner,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        hps = {"num_train_steps": num_train_steps, "batch_size": batch_size, "accumulate_batches": accumulate_batches}

        if 'tensorboard' in self.log_with:
            self.accelerator.init_trackers(f"clap_rvq_{int(time.time() * 1000)}", config=hps)
        else:
            self.accelerator.init_trackers(f"clap_rvq", config=hps)

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

    def train_step(self):
        steps = int(self.steps.item())

        self.audio_conditioner.learn_rvq = True

        iters = default(self.accumulate_batches, 1)
        iters = math.ceil(iters / self.accelerator.num_processes)

        embeds = []
        for _ in tqdm(range(iters), desc='accumulating batches'):
            raw_wave_for_clap = next(self.dl_iter)[0]
            embed = self.audio_conditioner.forward(audio_input=raw_wave_for_clap.to(self.device), return_embedding=True)
            embeds.append(embed)

        embeds = torch.cat(embeds, dim=0)
        embeds = self.accelerator.gather_for_metrics(embeds)

        if self.is_main:
            loss = self.audio_conditioner.quantize(embeds, return_rvq_loss=True)

            self.print(f'loss: {loss}')

            # sample results every so often
            valid_loss = None
            if not (steps % self.save_results_every):
                raw_wave_for_clap = next(self.valid_dl_iter)[0]

                with torch.no_grad():
                    self.audio_conditioner.learn_rvq = False
                    valid_loss = self.audio_conditioner.forward(audio_input=raw_wave_for_clap.to(self.device), return_rvq_loss=True)

                self.print(f'{steps}: valid loss {valid_loss}')

            self.accelerator.log({
                "train_loss": loss,
                "valid_loss": valid_loss
            }, step=steps)

            # save model every so often

            if not (steps % self.save_model_every):
                # save audio conditioner (clap) rvq checkpoint
                rvq_state_dict = self.accelerator.unwrap_model(self.audio_conditioner).rq.state_dict()
                torch.save(rvq_state_dict, str(self.results_folder / f'clap.rvq.{steps}.pt'))

                self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1

    def train(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')