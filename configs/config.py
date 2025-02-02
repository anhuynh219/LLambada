import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from beartype.typing import Literal, Optional, List

from models.base.tokenizers.clap_quantized import ClapQuantized, create_clap_quantized
from models.base.tokenizers.encodec_wrapper import EncodecWrapper, create_encodec_24khz
from models.base.tokenizers.hf_hubert_kmeans import HfHubertWithKmeans, get_hubert_kmeans
from models.base.cond_transformers.token_transformer import TokenConditionedTransformer
from models.base.cond_transformers.build_transformer import create_singsong_semantic_transformer, create_coarse_transformer
from models.SingSong.singsong import SingSong
from models.base.tokenizers.model_types import Wav2Vec, NeuralCodec
# from .open_musiclm import (MusicLM, TokenConditionedTransformer, create_llambada_transformer,
#                            create_coarse_transformer, create_fine_transformer,create_singsong_transformer,
#                            create_semantic_transformer, create_continuation_transformer)
from trainer import ClapRVQTrainer, HfHubertKmeansTrainer, SingleStageTrainer
from utils import exists, beartype_jit
from utils.data_process import DataPreprocessor
from models.base.cond_transformers.build_transformer import create_coarse_transformer, create_singsong_semantic_transformer

@dataclass
class ClapRVQConfig:
    rq_num_quantizers: int
    codebook_size: int
    enable_fusion: bool = False
    rq_ema_decay: float = 0.95
    threshold_ema_dead_code: float = 0.0
    checkpoint_path: Optional[str] = None
    amodel_type: str = 'HTSAT-tiny'

@dataclass
class HubertKmeansConfig:
    model_name: str
    normalize_embeds: bool
    embed_layer: int = 7
    target_sample_hz: int = 16000
    seq_len_multiple_of: int = 320
    codebook_size: int = 1024
    output_hz: int = 50

@dataclass
class EncodecConfig:
    bandwidth: float
    codebook_size: int
    output_hz: int = 75

RelativePositionBiasType = Literal['continuous', 't5', 'none']

@dataclass
class LlambdaConfig:
    dim: int = 768
    depth: int = 12
    heads: int = 12
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    use_conv_ff: bool = True
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = False
    use_absolute_position_embeddings: bool = False
    max_absolute_position_embeddings: int = 12 + 100 + 600


@dataclass
class SemanticConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    use_conv_ff: bool = True
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = False
    use_absolute_position_embeddings: bool = False
    max_absolute_position_embeddings: int = 12 + 250

@dataclass
class CoarseConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    use_conv_ff: bool = True
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = False
    use_absolute_position_embeddings: bool = False
    max_absolute_position_embeddings: int = 12 + 100 + 600

@dataclass
class ContinuationConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    use_conv_ff: bool = True
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = False
    use_absolute_position_embeddings: bool = False
    max_absolute_position_embeddings: int = 12 + 100 + 600

@dataclass
class SingSongConfig:
    dim: int = 768
    depth: int = 12
    heads: int = 12
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    use_conv_ff: bool = True
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = False
    use_absolute_position_embeddings: bool = False
    max_absolute_position_embeddings: int = 12 + 100 + 600


@dataclass
class FineConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    use_conv_ff: bool = True
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = False
    use_absolute_position_embeddings: bool = False
    max_absolute_position_embeddings: int = 12 + 300 + 900
@dataclass
class GlobalConfig:
    semantic_audio_length_seconds: float = 10.0
    coarse_audio_length_seconds: float = 4.0
    fine_audio_length_seconds: float = 2.0
    clap_audio_length_seconds: float = 10.0
    num_coarse_quantizers: int = 3
    num_fine_quantizers: int = 5
    continuation_audio_length_seconds: float = 10.0

@beartype_jit
@dataclass
class MusicLMModelConfig:
    clap_rvq_cfg: ClapRVQConfig
    hubert_kmeans_cfg: HubertKmeansConfig
    continuation_cfg: ContinuationConfig
    # llambada_cfg: LlambdaConfig
    encodec_cfg: EncodecConfig
    semantic_cfg: SemanticConfig
    coarse_cfg: CoarseConfig
    fine_cfg: FineConfig
    global_cfg: GlobalConfig
    singsong_cfg: SingSongConfig


@dataclass
class ClapRVQTrainerConfig:
    folder: str
    num_train_steps: int
    batch_size: int
    accumulate_batches: int
    save_model_every: int
    save_results_every: int

@dataclass
class HubertKmeansTrainerConfig:
    folder: str
    feature_extraction_num_steps: int
    feature_extraction_batch_size: int

@dataclass
class SingleStageTrainerConfig:
    stage: str
    folder: str
    valid_frac: float
    lr: float
    lr_warmup: int
    batch_size: int
    grad_accum_every: int
    wd: float
    max_grad_norm: float
    cross_entropy_loss_weights: list[float]
    num_train_steps: int
    save_results_every: int
    save_model_every: int
    save_predicted_tokens: bool
    save_reconstructed_wave: bool
    use_preprocessed_data: bool

@dataclass
class DataPreprocessorConfig:
    folder: str = './data/fma_large'
    metadata_folder: str = "./data/fma_metadata",
    results_folder: str = './fma_preprocessed'
    max_audio_length_seconds: int = 30
    random_crop: bool = True
    num_crops: int = 1
    clap_batch_size: int = 32

@beartype_jit
@dataclass
class MusicLMTrainingConfig:
    clap_rvq_trainer_cfg: ClapRVQTrainerConfig
    hubert_kmeans_trainer_cfg: HubertKmeansTrainerConfig
    semantic_trainer_cfg: SingleStageTrainerConfig
    coarse_trainer_cfg: SingleStageTrainerConfig
    fine_trainer_cfg: SingleStageTrainerConfig
    data_preprocessor_cfg: DataPreprocessorConfig
    continuation_trainer_cfg: SingleStageTrainerConfig
    singsong_trainer_cfg: SingleStageTrainerConfig
    # llambada_trainer_cfg: SingleStageTrainerConfig

@beartype_jit
def load_model_config(config_path: str) -> MusicLMModelConfig:
    with open(config_path, 'r') as f:
        config = json.load(f)

    return MusicLMModelConfig(
        clap_rvq_cfg=ClapRVQConfig(**config['clap_rvq_cfg']),
        hubert_kmeans_cfg=HubertKmeansConfig(**config['hubert_kmeans_cfg']),
        encodec_cfg=EncodecConfig(**config['encodec_cfg']),
        continuation_cfg=ContinuationConfig(**config['continuation_cfg']),
        semantic_cfg=SemanticConfig(**config['semantic_cfg']),
        singsong_cfg = SingSongConfig(**config['singsong_cfg']),
        coarse_cfg=CoarseConfig(**config['coarse_cfg']),
        fine_cfg=FineConfig(**config['fine_cfg']),
        global_cfg=GlobalConfig(**config['global_cfg'])

    )

@beartype_jit
def load_training_config(config_path: str) -> MusicLMTrainingConfig:
    with open(config_path, 'r') as f:
        config = json.load(f)

    return MusicLMTrainingConfig(
        clap_rvq_trainer_cfg=ClapRVQTrainerConfig(**config['clap_rvq_trainer_cfg']),
        hubert_kmeans_trainer_cfg=HubertKmeansTrainerConfig(**config['hubert_kmeans_trainer_cfg']),
        semantic_trainer_cfg=SingleStageTrainerConfig(**config['semantic_trainer_cfg']),
        coarse_trainer_cfg=SingleStageTrainerConfig(**config['coarse_trainer_cfg']),
        continuation_trainer_cfg = SingleStageTrainerConfig(**config['continuation_trainer_cfg']),
        fine_trainer_cfg=SingleStageTrainerConfig(**config['fine_trainer_cfg']),
        singsong_trainer_cfg=SingleStageTrainerConfig(**config['singsong_trainer_cfg']),
        data_preprocessor_cfg=DataPreprocessorConfig(**config['data_preprocessor_cfg']),
        # llambada_trainer_cfg = SingleStageTrainerConfig(**config['llambada_trainer_cfg'])
    )

# helper functions

def load_model(model, path):
    """helper class to load a model checkpoint"""
    path = Path(path)
    assert path.exists(), f'checkpoint does not exist at {str(path)}'
    pkg = torch.load(str(path))
    model.load_state_dict(pkg)

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# model stages

@beartype_jit
def create_clap_quantized_from_config(model_config: MusicLMModelConfig, rvq_path: Optional[str], device, **kwargs) -> ClapQuantized:
    with disable_print():
        return create_clap_quantized(
            **asdict(model_config.clap_rvq_cfg),
            device=device,
            learn_rvq=False,
            rvq_checkpoint_path=rvq_path,
            **kwargs,
        ).to(device)

@beartype_jit
def create_hubert_kmeans_from_config(model_config: MusicLMModelConfig, kmeans_path: Optional[str], device, **kwargs) -> HfHubertWithKmeans:
    return get_hubert_kmeans(
        **asdict(model_config.hubert_kmeans_cfg),
        kmeans_path=kmeans_path,
        **kwargs,
    ).to(device)

@beartype_jit
def create_encodec_from_config(model_config: MusicLMModelConfig, device, **kwargs) -> EncodecWrapper:
    return create_encodec_24khz(**asdict(model_config.encodec_cfg), **kwargs).to(device)

@beartype_jit
def create_semantic_transformer_from_config(
    model_config: MusicLMModelConfig,
    checkpoint_path: Optional[str],
    device,
    **kwargs,
) -> TokenConditionedTransformer:
    transformer = create_singsong_semantic_transformer(
        **asdict(model_config.semantic_cfg),
        clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
        semantic_codebook_size=model_config.hubert_kmeans_cfg.codebook_size,
        num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
        **kwargs,
    ).to(device)

    if exists(checkpoint_path):
        load_model(transformer, checkpoint_path)

    return transformer

@beartype_jit
def create_coarse_transformer_from_config(
    model_config: MusicLMModelConfig,
    checkpoint_path: Optional[str],
    device,
    **kwargs,
) -> TokenConditionedTransformer:
    transformer = create_coarse_transformer(
        **asdict(model_config.coarse_cfg),
        clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
        semantic_codebook_size=model_config.hubert_kmeans_cfg.codebook_size,
        acoustic_codebook_size=model_config.encodec_cfg.codebook_size,
        num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
        num_coarse_quantizers=model_config.global_cfg.num_coarse_quantizers,
        **kwargs,
    ).to(device)

    if exists(checkpoint_path):
        load_model(transformer, checkpoint_path)

    return transformer

# @beartype_jit
# def create_continuation_transformer_from_config(
#     model_config: MusicLMModelConfig,
#     checkpoint_path: Optional[str],
#     device,
#     **kwargs,
# ) -> TokenConditionedTransformer:
#     transformer = create_continuation_transformer(
#         **asdict(model_config.continuation_cfg),
#         clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
#         semantic_codebook_size=model_config.hubert_kmeans_cfg.codebook_size,
#         acoustic_codebook_size=model_config.encodec_cfg.codebook_size,
#         num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
#         num_coarse_quantizers=model_config.global_cfg.num_coarse_quantizers,
#         **kwargs,
#     ).to(device)

#     if exists(checkpoint_path):
#         load_model(transformer, checkpoint_path)

#     return transformer


@beartype_jit
def create_singsong_transformer_from_config(
    model_config: MusicLMModelConfig,
    checkpoint_path: Optional[str],
    device,
    **kwargs,
) -> TokenConditionedTransformer:
    transformer = create_singsong_semantic_transformer(
        **asdict(model_config.singsong_cfg),
        clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
        semantic_codebook_size=model_config.hubert_kmeans_cfg.codebook_size,
        acoustic_codebook_size=model_config.encodec_cfg.codebook_size,
        num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
        num_coarse_quantizers=model_config.global_cfg.num_coarse_quantizers,
        **kwargs,
    ).to(device)

    if exists(checkpoint_path):
        load_model(transformer, checkpoint_path)

    return transformer

# trainers

@beartype_jit
def create_clap_rvq_trainer_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    clap: ClapQuantized,
    results_folder: str,
    device,
    accelerate_kwargs: dict = {},
    config_paths: Optional[List[str]] = None,
    **kwargs,
):
    trainer = ClapRVQTrainer(
        audio_conditioner=clap,
        results_folder=results_folder,
        data_max_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        accelerate_kwargs=accelerate_kwargs,
        config_paths=config_paths,
        **asdict(training_config.clap_rvq_trainer_cfg),
        **kwargs,
    ).to(device)

    return trainer

@beartype_jit
def create_hubert_kmeans_trainer_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    hubert_kmeans: HfHubertWithKmeans,
    results_folder: str,
    device,
    config_paths: Optional[List[str]] = None,
    **kwargs,
):
    trainer = HfHubertKmeansTrainer(
        hubert_kmeans=hubert_kmeans,
        results_folder=results_folder,
        data_max_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        config_paths=config_paths,
        **asdict(training_config.hubert_kmeans_trainer_cfg),
        **kwargs,
    ).to(device)

    return trainer

@beartype_jit
def create_single_stage_trainer_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    stage: Literal['semantic', 'coarse', 'fine', 'continuation', 'singsong', 'llambada'],
    results_folder: str,
    transformer: TokenConditionedTransformer,
    clap: Optional[ClapQuantized]=None,
    wav2vec: Optional[HfHubertWithKmeans]=None,
    encodec_wrapper: Optional[EncodecWrapper]=None,
    device='cuda',
    accelerate_kwargs: dict = {},
    config_paths: Optional[List[str]] = None,
    **kwargs,
) -> SingleStageTrainer:

    semantic_audio_length_seconds = model_config.global_cfg.semantic_audio_length_seconds
    coarse_audio_length_seconds = model_config.global_cfg.coarse_audio_length_seconds
    fine_audio_length_seconds = model_config.global_cfg.fine_audio_length_seconds
    continuation_audio_length_seconds = model_config.global_cfg.continuation_audio_length_seconds


    if stage == 'semantic':
        trainer_cfg = training_config.semantic_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, semantic_audio_length_seconds)

    # elif stage == 'llambada':
    #     trainer_cfg = training_config.llambada_trainer_cfg
    #     data_max_length_seconds = (semantic_audio_length_seconds, semantic_audio_length_seconds, coarse_audio_length_seconds)
    #     print(data_max_length_seconds)

    elif stage == 'coarse':
        trainer_cfg = training_config.coarse_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, coarse_audio_length_seconds, coarse_audio_length_seconds)
    elif stage == 'fine':
        trainer_cfg = training_config.fine_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, fine_audio_length_seconds)
    
    elif stage == 'continuation':
        trainer_cfg = training_config.continuation_trainer_cfg
        data_max_length_seconds = (coarse_audio_length_seconds, semantic_audio_length_seconds, continuation_audio_length_seconds)
        print(data_max_length_seconds)
    
    elif stage == 'singsong':
        trainer_cfg = training_config.singsong_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, semantic_audio_length_seconds, coarse_audio_length_seconds)
        print(data_max_length_seconds)
    
    trainer = SingleStageTrainer(
        model_config=model_config,
        training_config=training_config,
        transformer=transformer,
        audio_conditioner=clap,
        wav2vec=wav2vec,
        neural_codec=encodec_wrapper,
        results_folder=results_folder,
        data_max_length_seconds=data_max_length_seconds,
        accelerate_kwargs=accelerate_kwargs,
        config_paths=config_paths,
        **asdict(trainer_cfg),
        **kwargs,
    ).to(device)

    return trainer

@beartype_jit
def create_data_preprocessor_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    clap: ClapQuantized,
    wav2vec: HfHubertWithKmeans,
    encodec_wrapper: EncodecWrapper,
    device='cpu',
    config_paths: Optional[List[str]] = None,
    **kwargs,
):
    data_preprocessor = DataPreprocessor(
        audio_conditioner=clap,
        wav2vec=wav2vec,
        neural_codec=encodec_wrapper,
        num_coarse_quantizers=model_config.global_cfg.num_coarse_quantizers,
        semantic_audio_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        coarse_audio_length_seconds=model_config.global_cfg.coarse_audio_length_seconds,
        fine_audio_length_seconds=model_config.global_cfg.fine_audio_length_seconds,
        clap_audio_length_seconds=model_config.global_cfg.clap_audio_length_seconds,
        config_paths=config_paths,
        **asdict(training_config.data_preprocessor_cfg),
        **kwargs,
    ).to(device)

    return data_preprocessor
