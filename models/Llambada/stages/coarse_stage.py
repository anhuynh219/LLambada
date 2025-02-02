from .base_stage import BaseStage
from models.base.cond_transformers.token_transformer import TokenSequenceInfo
from models.base.cond_transformers.token_transformer import TokenConditionedTransformer
from models.base.tokenizers.model_types import Wav2Vec
from models.base.tokenizers import NeuralCodec
from models.base.tokenizers.do_tokenize import get_or_compute_semantic_token_ids, get_or_compute_acoustic_token_ids
from einops import rearrange
from beartype.typing import List, Optional
from utils import exists, beartype_jit, eval_decorator
from models.base.cond_transformers.token_transformer import TokenConditionedTransformerWrapper

import torch

@beartype_jit
class CoarseStage(BaseStage):
    def __init__(
        self,
        *,
        coarse_transformer: TokenConditionedTransformer,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        pad_id=-1,
        unique_consecutive=False,
        cross_entropy_loss_weights: List[float] = None,
        mask_prob=0.15
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.neural_codec = neural_codec
        self.pad_id = pad_id
        self.unique_consecutive = unique_consecutive
        self.num_coarse_quantizers = coarse_transformer.token_sequences[-1].num_quantizers

        self.base_transformer = TokenConditionedTransformerWrapper(
            transformer = coarse_transformer,
            pad_id = pad_id,
            unique_consecutive = unique_consecutive,
            cross_entropy_loss_weights = cross_entropy_loss_weights,
            mask_prob = mask_prob
        )

    def forward(
        self, 
        *,
        raw_accom_for_semantic = None,
        raw_vocal_for_acoustic = None,
        raw_accom_for_acoustic = None,
        return_loss = True, 
        **kwargs
    ):
        accom_semantic_token_ids = get_or_compute_semantic_token_ids(
            semantic_token_ids = None,
            raw_audio = raw_accom_for_semantic,
            wav2vec = self.wav2vec
        )

        vocal_coarse_token_ids, _ = get_or_compute_acoustic_token_ids(
            coarse_token_ids = None,
            fine_token_ids = None,
            raw_audio = raw_vocal_for_acoustic,
            neural_codec = self.neural_codec,
            num_coarse_quantizers = self.num_coarse_quantizers
        )

        accom_coarse_token_ids, _ = get_or_compute_acoustic_token_ids(
            coarse_token_ids = None,
            fine_token_ids = None,
            raw_audio = raw_accom_for_acoustic,
            neural_codec = self.neural_codec,
            num_coarse_quantizers = self.num_coarse_quantizers
        )
        
        return self.base_transformer(
            all_token_ids=[accom_semantic_token_ids, vocal_coarse_token_ids, accom_coarse_token_ids],
            return_loss=return_loss,
            **kwargs
        )

    @eval_decorator
    @torch.no_grad()
    @beartype_jit
    def generate(
        self, 
        *,
        vocal_coarse_token_ids: torch.Tensor,
        accom_semantic_token_ids: torch.Tensor,
        filter_thres=0.95,
        temperature=1.0,
        max_time_steps=10*75,
        include_eos_in_output=False,  # if doing hierarchical sampling, eos can be kept for an easy time
        append_eos_to_conditioning_tokens=True, # if doing heirarchical sampling and you want more control
        reconstruct_wave = False,
        **kwargs
    ):

        accom_coarse_token_ids = self.base_transformer.generate(
            conditioning_token_ids=[accom_semantic_token_ids, vocal_coarse_token_ids],
            pred_token_ids=None,
            max_time_steps=max_time_steps,
            filter_thres=filter_thres,
            temperature=temperature,
            include_eos_in_output=include_eos_in_output,
            append_eos_to_conditioning_tokens=append_eos_to_conditioning_tokens,
            **kwargs
        )

        if reconstruct_wave:
            wave = self.neural_codec.decode(accom_coarse_token_ids)
            return rearrange(wave, 'b n c -> b c n')

        return accom_coarse_token_ids