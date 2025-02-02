import io
import random
import sqlite3
from functools import partial, wraps
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from beartype.door import is_bearable
from beartype.typing import List, Literal, Optional, Tuple, Union
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchaudio.functional import resample
from glob import glob

from utils import (beartype_jit, curtail_to_multiple, default,
                    float32_to_int16, int16_to_float32,
                    zero_mean_unit_var_norm)

from .dataset import exists, cast_tuple, adapt_array, convert_array, init_sqlite

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]
FloatOrInt = Union[float, int]


@beartype_jit
class SongGenDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3'],
        max_length_seconds: Optional[Union[FloatOrInt, Tuple[Optional[FloatOrInt], ...]]] = 1,
        normalize: Union[bool, Tuple[bool, ...]] = False,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None,
        ignore_files: Optional[List[str]] = None,
        ignore_load_errors=True,
        random_crop=True,
    ):
        super().__init__()
        
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = []
        ignore_files = default(ignore_files, [])
        num_ignored = 0
        ignore_file_set = set([f.split('/')[-1] for f in ignore_files])
        exts="mp3"
        vocals_files = [file for file in path.glob(f'**/vocal/*.{exts}')]
        other_files = [file for file in path.glob(f'**/accom/*.{exts}')]
        
        assert len(other_files) > 0, 'no other sound files found'
        assert len(vocals_files) > 0, 'no vocals sound files found'
        if len(vocals_files) != len(other_files):
          print("vocals_files",len(vocals_files))
          print("other_Files",len(other_files))

        if num_ignored > 0:
            print(f'skipped {num_ignored} ignored files')

        self.vocal_paths = vocals_files
        self.accom_paths = other_files

        self.ignore_load_errors = ignore_load_errors
        self.random_crop = random_crop

        self.target_sample_hz = (24000, 48000, 24000)
        num_outputs = len(self.target_sample_hz)

        # self.max_length_seconds = cast_tuple(max_length_seconds, num_outputs)
        self.max_length_seconds = (10.0, 10.0, 10.0)
        self.max_length = tuple([int(s * hz) if exists(s) else None for s, hz in zip(self.max_length_seconds, self.target_sample_hz)])

        self.normalize = [True, False, False]

        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.max_length_seconds) == len(
            self.target_sample_hz) == len(self.seq_len_multiple_of) == len(self.normalize)

    def __len__(self):
        return len(self.vocal_paths)

    def __getitem__(self, idx):
        try:
            # stage = "llambada"
            vocal_file = self.vocal_paths[idx]
            accom_file = self.accom_paths[idx]

            vocal_data, vocal_sample_hz = torchaudio.load(vocal_file)
            accom_data, accom_sample_hz = torchaudio.load(accom_file)

            vocal_for_wa2vec , vocal_for_clap, vocal_for_encodec  = self.my_process_audio(vocal_data, vocal_sample_hz)
            accom_for_wav2vec, accom_for_clap, accom_for_encodec = self.my_process_audio(accom_data,accom_sample_hz)
            
        except:
            if self.ignore_load_errors:
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                raise Exception(f'error loading file {file}')
        
        # return audio_for_encodec_prev, audio_for_wav2vec, audio_for_encodec
        return vocal_for_encodec, vocal_for_wa2vec, accom_for_wav2vec

    def my_process_audio(self,data,sample_hz):
        # print("using my processing audio")
        if data.shape[0] > 1:
            data = torch.mean(data, dim=0).unsqueeze(0)

        target_length = int(10 * sample_hz)
        normalized_data = zero_mean_unit_var_norm(data)

        data = data[:, :target_length]
        normalized_data = normalized_data[: , :target_length]
        encodec_wrapper_sample_rate=24000
        wav2vec_sample_rate=16000
        clap_sample_rate = 48000

        audio_for_encodec = resample(data, sample_hz, encodec_wrapper_sample_rate)
        audio_for_wav2vec = resample(normalized_data, sample_hz, wav2vec_sample_rate)
        audio_for_clap = resample(data, sample_hz, clap_sample_rate)

        audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec)).squeeze(0)
        audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec)).squeeze(0)
        audio_for_clap = int16_to_float32(float32_to_int16(audio_for_clap)).squeeze(0)
        # print("in my_process_audio",audio_for_encodec.shape,audio_for_wav2vec.shape)
        return audio_for_wav2vec, audio_for_clap, audio_for_encodec

    def process_audio(self, data, sample_hz, target_sample_hz, pad_to_target_length=True):

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        # recursively crop the audio at random in the order of longest to shortest max_length_seconds, padding when necessary.
        # e.g. if max_length_seconds = (10, 4), pick a 10 second crop from the original, then pick a 4 second crop from the 10 second crop
        # also use normalized data when specified

        temp_data = data
        temp_data_normalized = zero_mean_unit_var_norm(data)

        num_outputs = len(target_sample_hz)
        data = [None for _ in range(num_outputs)]

        sorted_max_length_seconds = sorted(
            enumerate(self.max_length_seconds),
            key=lambda t: (t[1] is not None, t[1])) # sort by max_length_seconds, while moving None to the beginning
        print(len(sorted_max_length_seconds))
        print(len(self.normalize))
        for unsorted_i, max_length_seconds in sorted_max_length_seconds:

            if exists(max_length_seconds):
                audio_length = temp_data.size(1)
                target_length = int(max_length_seconds * sample_hz)

                if audio_length > target_length:
                    max_start = audio_length - target_length
                    start = torch.randint(0, max_start, (1, )) if self.random_crop else 0

                    temp_data = temp_data[:, start:start + target_length]
                    temp_data_normalized = temp_data_normalized[:, start:start + target_length]
                else:
                    if pad_to_target_length:
                        temp_data = F.pad(temp_data, (0, target_length - audio_length), 'constant')
                        temp_data_normalized = F.pad(temp_data_normalized, (0, target_length - audio_length), 'constant')
            print(unsorted_i)
            if self.normalize[unsorted_i]:
                data[unsorted_i] = temp_data_normalized 
            else:
                temp_data
        # resample if target_sample_hz is not None in the tuple
        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, target_sample_hz))
        # quantize non-normalized audio to a valid waveform
        data_tuple = tuple(d if self.normalize[i] else int16_to_float32(float32_to_int16(d)) for i, d in enumerate(data_tuple))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            if exists(max_length) and pad_to_target_length:
                assert audio_length == max_length, f'audio length {audio_length} does not match max_length {max_length}.'

            data = rearrange(data, '1 ... -> ...')

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output