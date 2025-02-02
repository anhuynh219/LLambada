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

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# sqlite helpers for preprocessing

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

@beartype_jit
def init_sqlite(db_path):
    """Connect to a sqlite database. Will create a new one if it doesn't exist."""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    return conn, cursor

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]
FloatOrInt = Union[float, int]

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        data = list(filter(lambda x: x is not None, data))
        if len(data) == 0:
            return () # empty batch

        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

# dataset functions

@collate_one_or_multiple_tensors
def concatenate_fn(batch):
    return torch.cat(batch, dim=0)

def get_preprocessed_dataloader(ds, **kwargs):
    collate_fn = concatenate_fn
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)

def sound_preprocessing_collate_fn(data):
    data = list(filter(lambda x: x is not None, data))
    if len(data) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(data)

def get_sound_preprocessing_dataloader(ds, **kwargs):
    assert kwargs.get('batch_size', 1) == 1, 'batch_size must be 1 for preprocessing'
    kwargs.setdefault('batch_size', 1)
    return DataLoader(ds, collate_fn=sound_preprocessing_collate_fn, **kwargs)