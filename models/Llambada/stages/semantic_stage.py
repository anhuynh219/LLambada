from .base_stage import BaseStage
from models.base.cond_transformers.token_transformer import TokenSequenceInfo
from models.base.cond_transformers.token_transformer import TokenConditionedTransformer, TokenConditionedTransformerWrapper
from models.base.tokenizers.model_types import Wav2Vec
from models.base.tokenizers import ClapQuantized
from models.base.tokenizers import NeuralCodec
from models.base.tokenizers.do_tokenize import get_or_compute_semantic_token_ids, get_or_compute_acoustic_token_ids, get_or_compute_clap_token_ids
from beartype.typing import List, Optional
from utils import exists, beartype_jit, eval_decorator
import torch

@beartype_jit
class SemanticStage(BaseStage):
    def __init__(
        self,
        *,
        semantic_transformer: TokenConditionedTransformer,
        clap: Optional[ClapQuantized] = None,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        pad_id=-1,
        unique_consecutive=False,
        cross_entropy_loss_weights: List[float] = None,
        mask_prob=0.15
    ):

        super().__init__()

        self.wav2vec = wav2vec
        self.clap = clap
        self.neural_codec = neural_codec
        self.pad_id = pad_id
        self.unique_consecutive = unique_consecutive

        self.num_coarse_quantizers = semantic_transformer.token_sequences[-1].num_quantizers
        num_semantic_tokens = semantic_transformer.token_sequences[1].codebook_size

        if exists(wav2vec):
            assert self.wav2vec.codebook_size == num_semantic_tokens, f'num_semantic_tokens on CoarseTransformer must be set to {self.wav2vec.codebook_size}'

        self.base_transformer= TokenConditionedTransformerWrapper(
            transformer= semantic_transformer,
            pad_id=pad_id,
            unique_consecutive=unique_consecutive,
            cross_entropy_loss_weights=cross_entropy_loss_weights,
            mask_prob=mask_prob
        )

        # transformer: TokenConditionedTransformer,
        # pad_id=-1,
        # unique_consecutive=True,
        # cross_entropy_loss_weights: Optional[List[float]] = None,
        # mask_prob=0.15
    def forward(
        self,
        *,
        text_prompt: Optional[torch.Tensor] = None,
        raw_vocal_for_semantic: Optional[torch.Tensor] = None,
        raw_accom_for_semantic: Optional[torch.Tensor] = None,
        return_loss = True,
        **kwargs
    ):

        vocal_semantic_token_ids = get_or_compute_semantic_token_ids(
            semantic_token_ids=None, 
            raw_audio=raw_vocal_for_semantic, 
            wav2vec=self.wav2vec
        )

        accom_semantic_token_ids = get_or_compute_semantic_token_ids(
            semantic_token_ids=None, 
            raw_audio=raw_accom_for_semantic, 
            wav2vec=self.wav2vec
        )

        clap_token_ids = get_or_compute_clap_token_ids(
            clap_token_ids=None, 
            clap=self.clap, 
            conditioning_audio= None, 
            conditioning_text=text_prompt
        )

        return self.base_transformer.forward(
            all_token_ids=[vocal_semantic_token_ids, clap_token_ids, accom_semantic_token_ids],
            return_loss=return_loss,
            **kwargs
        )
    
    @eval_decorator
    @torch.no_grad()
    @beartype_jit
    def generate(
        self,
        vocal_semantic_token_ids: Optional[torch.Tensor],
        text_prompt: Optional[List[str]],
        accom_semantic_token_ids: Optional[torch.Tensor] = None,
        filter_thres=0.95,
        temperature=1.,
        max_time_steps=10*600,
        include_eos_in_output=False,  # if doing hierarchical sampling, eos can be kept for an easy time
        append_eos_to_conditioning_tokens=True, # if doing heirarchical sampling and you want more control
        reconstruct_wave = False,
        **kwargs
    ):

        clap_token_ids = get_or_compute_clap_token_ids(
            clap_token_ids=None, 
            clap=self.clap, 
            conditioning_audio= None, 
            conditioning_text=text_prompt
        )

        # print("Semantic stage: ",vocal_semantic_token_ids.shape, clap_token_ids.shape)

        sampled_token_ids = self.base_transformer.generate(
            conditioning_token_ids=[vocal_semantic_token_ids, clap_token_ids],
            pred_token_ids=None,
            max_time_steps=max_time_steps,
            temperature=temperature,
            include_eos_in_output=include_eos_in_output,
            append_eos_to_conditioning_tokens=append_eos_to_conditioning_tokens,
            **kwargs
        )
        return sampled_token_ids