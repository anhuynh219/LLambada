import torch 
from utils import exists, beartype_jit

from beartype.typing import List, Optional
from .model_types import NeuralCodec, Wav2Vec
from .clap_quantized import ClapQuantized

@beartype_jit
def get_or_compute_semantic_token_ids(
    semantic_token_ids: Optional[torch.Tensor], 
    raw_audio: Optional[torch.Tensor], 
    wav2vec: Optional[Wav2Vec]
):

    if not exists(semantic_token_ids):
        assert exists(raw_audio)
        assert exists(wav2vec)
        semantic_token_ids = wav2vec(raw_audio, flatten=False)

    return semantic_token_ids


@beartype_jit
def get_or_compute_acoustic_token_ids(
    coarse_token_ids: Optional[torch.Tensor], 
    fine_token_ids: Optional[torch.Tensor], 
    raw_audio: Optional[torch.Tensor], 
    neural_codec: Optional[NeuralCodec], 
    num_coarse_quantizers: int
):

    if exists(raw_audio):
        assert not exists(coarse_token_ids) and not exists(fine_token_ids), "either provide coarse + fine ids or raw audio"
        assert exists(neural_codec), 'A neural audio codec must be provided if given raw wave for training'

        with torch.no_grad():
            neural_codec.eval()
            _, indices, _ = neural_codec(raw_audio, return_encoded=True)
            coarse_token_ids, fine_token_ids = indices[..., :num_coarse_quantizers], indices[..., num_coarse_quantizers:]

    return coarse_token_ids, fine_token_ids

def get_or_compute_clap_token_ids(clap_token_ids: Optional[torch.Tensor], clap: Optional[ClapQuantized], conditioning_audio: Optional[torch.Tensor], conditioning_text: Optional[List[str]]):
    
    if not exists(clap_token_ids):
        assert exists(conditioning_audio) ^ exists(conditioning_text), "either condition on text or audio"
        assert exists(clap)
        if exists(conditioning_text):
            # print("===========")
            # print("conditioning_text:", conditioning_text)
            clap_token_ids = clap(text_input=conditioning_text)
        else:
            clap_token_ids = clap(audio_input=conditioning_audio)

    return clap_token_ids