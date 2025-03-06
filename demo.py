from models.Llambada.builder import build_llambada_model
import json
import torch
import torchaudio
import argparse
from utils.preprocess_audio import zero_mean_unit_var_norm, float32_to_int16, int16_to_float32
from torchaudio.functional import resample

def load_model(
    semantic_cfg_file = "/content/LLambada/configs/model_config/llambada_tiny_cfg/semantic_stage.json",
    coarse_cfg_file = "/content/LLambada/configs/model_config/llambada_tiny_cfg/coarse_stage.json",
    semantic_weight = "/content/LLambada/ckpts/Llambada/llambada.transformer.77000.pt",
    rvq_path = "/content/LLambada/ckpts/Llambada/clap.rvq.950_no_fusion.pt",
    coarse_weight = "/content/LLambada/ckpts/Llambada/coarse.transformer.17400.pt",
    kmean_path = "/content/LLambada/ckpts/Llambada/kmeans.joblib",
    clap_ckpt_path = "/content/LLambada/ckpts/Llambada/630k-audioset-best.pt",
    semantic_cross_entropy_loss_weights = [0.0, 0.0, 1.0],
    coarse_cross_entropy_loss_weights = [0.0, 0.0, 1.0]
):

    semantic_cfg = json.load(open(semantic_cfg_file))
    coarse_cfg = json.load(open(coarse_cfg_file))
    model, wav2vec, neural_codec, clap = build_llambada_model(
        semantic_cfg = semantic_cfg,
        coarse_cfg = coarse_cfg,
        rvq_ckpt_path = rvq_path,
        kmean_path = kmean_path,
        clap_ckpt_path = clap_ckpt_path,
        coarse_weight = coarse_weight,
        semantic_weight = semantic_weight,
        semantic_cross_entropy_loss_weights = semantic_cross_entropy_loss_weights,
        coarse_cross_entropy_loss_weights = coarse_cross_entropy_loss_weights
    )

    return model, coarse_cfg, wav2vec, neural_codec, clap

def load_data(
    vocal_path,
    prompt,
    second_length = 10,
    device = "cuda"
):
    data, sample_hz = torchaudio.load(vocal_path)
    target_length = second_length * sample_hz

    if data.shape[0] > 1:
        data = torch.mean(data, dim=0).unsqueeze(0)

    normalized_data = zero_mean_unit_var_norm(data)
    data = data[:, :target_length]
    normalized_data = normalized_data[:, :target_length]

    audio_for_wav2vec = resample(normalized_data, sample_hz, wav2vec.target_sample_hz)
    audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))
    audios_for_semantic = torch.cat([audio_for_wav2vec], axis = 0).to(device)

    audio_for_encodec = resample(data, sample_hz, neural_codec.sample_rate)
    audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec)) 
    audios_for_acoustic = torch.cat([audio_for_encodec], axis = 0).to(device)

    return audios_for_semantic, audios_for_acoustic, [prompt]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script")
    parser.add_argument("-p", "--path", required=True, help="Path to the vocal file")
    # Parse the arguments
    args = parser.parse_args()
    # Get the path from the arguments
    vocal_path = args.path

    print(f"The path to the file is: {vocal_path}")
    
    prompt = "This song is playing with piano"
    device = "cuda"
    model, coarse_cfg, wav2vec, neural_codec, clap = load_model()
    model.to(device)
    vocal_for_semantic, vocal_for_acoustic, prompt = load_data(
        vocal_path,
        prompt,
        10,
        device
    )
    # print(vocal_for_semantic.shape, vocal_for_acoustic.shape, prompt)
    wave = model.generate(
        vocal_for_semantic,
        vocal_for_acoustic,
        prompt,
        10
    ).detach().cpu()[0]
    # print(wave.shape)
    output = torchaudio.save("output2.mp3", wave, sample_rate=neural_codec.sample_rate)
