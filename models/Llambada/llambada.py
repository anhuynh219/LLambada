from .stages.coarse_stage import CoarseStage
from .stages.semantic_stage import SemanticStage
from models.base.tokenizers.do_tokenize import get_or_compute_semantic_token_ids, get_or_compute_acoustic_token_ids, get_or_compute_clap_token_ids
from models.base.tokenizers.mert import get_hubert_kmeans
from models.base.tokenizers.encodec_wrapper import create_encodec_24khz
from models.base.cond_transformers.build_transformer import create_semantic_transformer, create_coarse_transformer
from models.base.cond_transformers.token_transformer import TokenConditionedTransformer
from models.base.tokenizers.model_types import Wav2Vec, NeuralCodec 
from models.base.tokenizers.clap_quantized import ClapQuantized
import torch
from einops import rearrange
from beartype.typing import List, Optional


class Llambada(torch.nn.Module):
    def __init__(
        self,
        *,
        coarse_transformer: TokenConditionedTransformer,
        semantic_transformer: TokenConditionedTransformer,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        clap: Optional[ClapQuantized] = None,
        pad_id=-1,
        unique_consecutive=False,
        cross_entropy_loss_weights: List[List[float]] = None,
        mask_prob=0.15
        ):

        super().__init__()
        self.semantic_stage = SemanticStage(
            semantic_transformer=semantic_transformer,
            clap=clap,
            wav2vec=wav2vec,
            neural_codec=neural_codec,
            pad_id=pad_id,
            unique_consecutive=unique_consecutive,
            cross_entropy_loss_weights=cross_entropy_loss_weights[0],
            mask_prob=mask_prob
        )

        self.coarse_stage = CoarseStage(
            coarse_transformer=coarse_transformer,
            wav2vec=wav2vec,
            neural_codec=neural_codec,
            pad_id=pad_id,
            unique_consecutive=unique_consecutive,
            cross_entropy_loss_weights=cross_entropy_loss_weights[1],
            mask_prob=mask_prob
        )

        self.clap = clap
        self.neural_codec = neural_codec
        self.wav2vec = wav2vec

    def generate(
        self,
        vocal_semantic_input,
        vocal_coarse_input,
        text_prompt,
        second_output,
        filter_thres=0.95,
        temperature=1.
    ):

        # print(vocal_semantic_input.shape, vocal_coarse_input.shape)

        vocal_semantic_token_ids = get_or_compute_semantic_token_ids(
            semantic_token_ids = None,
            raw_audio = vocal_semantic_input,
            wav2vec = self.semantic_stage.wav2vec
        )

        vocal_coarse_token_ids, _ = get_or_compute_acoustic_token_ids(
            coarse_token_ids = None,
            fine_token_ids = None,
            raw_audio = vocal_coarse_input,
            neural_codec = self.coarse_stage.neural_codec,
            num_coarse_quantizers = self.coarse_stage.num_coarse_quantizers
        )
        # print("Text prompt:", text_prompt)

        # clap_token_ids = get_or_compute_clap_token_ids(None, self.clap, None, text_prompt)
        # print(clap_token_ids.shape)
        accom_semantic_token_ids = self.semantic_stage.generate(
            vocal_semantic_token_ids,
            text_prompt,
            filter_thres=0.95,
            temperature=0.95,
            max_time_steps= second_output * 50,
        )

        accom_coarse_token_ids = self.coarse_stage.generate(
            accom_semantic_token_ids = accom_semantic_token_ids, 
            vocal_coarse_token_ids = vocal_coarse_token_ids,
            filter_thres=0.9,
            temperature=0.95,
            max_time_steps= second_output * 75,
        )

        wave = self.neural_codec.decode_from_codebook_indices(accom_coarse_token_ids)
        return wave

