from models.base.cond_transformers.token_transformer import TokenSequenceInfo
from models.base.cond_transformers.token_transformer import TokenConditionedTransformer
# from models.base.tokenizers.mert import HfHubertWithKmeans
from models.base.tokenizers import NeuralCodec
from beartype.typing import List, Optional
import torch 
import torch.nn as nn
from dataclasses import dataclass
from utils import exists, beartype_jit, eval_decorator
from models.base.cond_transformers.token_transformer import TokenConditionedTransformerWrapper
from models.base.tokenizers.model_types import Wav2Vec

class BaseStage(nn.Module):
    def __init__(
        self, 
        *,
        base_transformer: TokenConditionedTransformer = None,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        pad_id=-1,
        unique_consecutive=False,
        cross_entropy_loss_weights: List[float] = None,
        mask_prob=0.15,
        num_semantic_tokens=1024,
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.neural_codec = neural_codec
        self.pad_id = pad_id
        self.unique_consecutive = unique_consecutive
        self.transformer_wrapper = None

    @property
    def device(self):
        return "cuda"

    def forward(
        self, 
        *,
        raw_wave = None,
        return_loss = True
    ):
        return NotImplementedError 

    @eval_decorator
    @torch.no_grad()
    @beartype_jit
    def generate(
        self, 
        list_tokens:list [TokenSequenceInfo], 
    ):
        return NotImplementedError

