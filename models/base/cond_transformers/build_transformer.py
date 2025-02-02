from .token_transformer import TokenSequenceInfo, TokenConditionedTransformer, TokenConditionedTransformerWrapper
import torch 
from utils import beartype_jit



@beartype_jit
def create_semantic_transformer(
    dim=768,
    depth=16,
    heads= 16,
    clap_codebook_size=1024,
    semantic_codebook_size=1024,
    acoustic_codebook_size=1024,
    num_clap_quantizers=12,
    num_coarse_quantizers=4,
    **kwargs
):

    semantic_sequence = TokenSequenceInfo(codebook_size=semantic_codebook_size,
                                          num_quantizers=1, unique_consecutive=False)

    clap_sequence = TokenSequenceInfo(codebook_size=clap_codebook_size, num_quantizers=num_clap_quantizers,
                                    unique_consecutive=False)


    return TokenConditionedTransformer(token_sequences=[semantic_sequence, clap_sequence, semantic_sequence], dim=dim, depth=depth, heads = heads, **kwargs)

@beartype_jit
def create_coarse_transformer(
    dim=1024,
    depth=6,
    heads = 8,
    semantic_codebook_size=1024,
    acoustic_codebook_size=1024,
    num_coarse_quantizers=4,
    **kwargs
):

    semantic_sequence = TokenSequenceInfo(codebook_size=semantic_codebook_size,
                                          num_quantizers=1, unique_consecutive=False)
    coarse_sequence = TokenSequenceInfo(
        codebook_size=acoustic_codebook_size, num_quantizers=num_coarse_quantizers, unique_consecutive=False)

    return TokenConditionedTransformer(token_sequences=[semantic_sequence, coarse_sequence, coarse_sequence], dim=dim, depth=depth, heads = heads, **kwargs)