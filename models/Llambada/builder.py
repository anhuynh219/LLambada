from models.base.tokenizers.mert import get_hubert_kmeans
from models.base.tokenizers.encodec_wrapper import create_encodec_24khz
from models.base.tokenizers.clap_quantized import create_clap_quantized
from models.Llambada.llambada import Llambada
from models.base.cond_transformers.build_transformer import create_semantic_transformer, create_coarse_transformer
import torch

def build_llambada_model(
    semantic_cfg,
    coarse_cfg,
    device = "cuda",
    semantic_weight:str = None,
    coarse_weight:str = None,
    semantic_cross_entropy_loss_weights =[0.0, 0.0, 1.0],
    coarse_cross_entropy_loss_weights =[0.0, 0.0, 1.0],
):
    wav2vec = get_hubert_kmeans(kmeans_path="./ckpts/kmeans_10s_no_fusion.joblib")
    neural_codec = create_encodec_24khz()
    clap = create_clap_quantized(device)
    semantic_transformer = create_semantic_transformer(
        dim=semantic_cfg["llambada_cfg"]["dim"],
        depth=semantic_cfg["llambada_cfg"]["depth"],
        heads = semantic_cfg["llambada_cfg"]["heads"],
        num_clap_quantizers= semantic_cfg["clap_rvq_cfg"]["rq_num_quantizers"],
        semantic_codebook_size= semantic_cfg["hubert_kmeans_cfg"]["codebook_size"],
        acoustic_codebook_size= semantic_cfg["encodec_cfg"]["codebook_size"],
        num_coarse_quantizers= semantic_cfg["global_cfg"]["num_coarse_quantizers"]
    )

    # print(semantic_transformer.parameters())

    coarse_transformer = create_coarse_transformer(
        dim= coarse_cfg["coarse_cfg"]["dim"],
        depth=coarse_cfg["coarse_cfg"]["depth"],
        heads = coarse_cfg["coarse_cfg"]["heads"],
        semantic_codebook_size= coarse_cfg["hubert_kmeans_cfg"]["codebook_size"],
        acoustic_codebook_size= coarse_cfg["encodec_cfg"]["codebook_size"],
        num_coarse_quantizers=coarse_cfg["global_cfg"]["num_coarse_quantizers"]
    )

    if coarse_weight is not None:
        print("Loading coarse weight")
        coarse_transformer.load_state_dict(torch.load(coarse_weight, map_location=device))

    if semantic_weight is not None:
        print("Loading semantic weight")
        semantic_transformer.load_state_dict(torch.load(semantic_weight, map_location=device))
    
    model = Llambada(
        semantic_transformer=semantic_transformer,
        coarse_transformer=coarse_transformer,
        clap = clap,
        wav2vec=wav2vec,
        neural_codec=neural_codec,
        pad_id=-1,
        unique_consecutive=False,
        cross_entropy_loss_weights=[semantic_cross_entropy_loss_weights, coarse_cross_entropy_loss_weights],
        mask_prob=0.15
    )
    
    return model, wav2vec, neural_codec, clap

