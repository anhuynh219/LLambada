from models.Llambada.builder import build_llambada_model
import json
import torch
import torchaudio
from utils.preprocess_audio import zero_mean_unit_var_norm, float32_to_int16, int16_to_float32
from torchaudio.functional import resample

def load_model(
    semantic_cfg_file = "./configs/model_config/llambada_tiny_cfg/semantic_stage.json",
    coarse_cfg_file = "./configs/model_config/llambada_tiny_cfg/coarse_stage.json",
    semantic_weight = "./ckpts/llambada.transformer.77000.pt",
    rvq_path = "./ckpts/clap.rvq.950_no_fusion.pt"
    coarse_weight = "./ckpts/coarse.transformer.17400.pt",
    semantic_cross_entropy_loss_weights = [0.0, 0.0, 1.0],
    coarse_cross_entropy_loss_weights = [0.0, 0.0, 1.0]
):

    semantic_cfg = json.load(open(semantic_cfg_file))
    coarse_cfg = json.load(open(coarse_cfg_file))
    model, wav2vec, neural_codec, clap = build_llambada_model(
        semantic_cfg,
        coarse_cfg,
        semantic_weight = semantic_weight,
        coarse_weight = coarse_weight,
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
    waveform, sample_hz = torchaudio.load(vocal_path)
    target_length = second_length * sample_hz
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    normalized_data = zero_mean_unit_var_norm(waveform)
    data = waveform[:, :target_length]
    normalized_data = normalized_data[:, :target_length]

    audio_for_wav2vec = resample(normalized_data, sample_hz, 16000)
    audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))
    audios_for_semantic = torch.cat([audio_for_wav2vec], dim = 0).to(device)

    audio_for_encodec = resample(data, sample_hz, 24000)
    audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec)) 
    audios_for_acoustic = torch.cat([audio_for_encodec], dim = 0).to(device)

    return audios_for_semantic, audios_for_acoustic, [prompt]

if __name__ == "__main__":
    
    vocal_path = "./output.mp3"
    prompt = "This song contains someone playing an acoustic drum set with a simple bassline. An e-guitar is playing a simple melody along with a piano playing chords in the midrange. A male voice is singing in a higher key. This song may be playing live at a festival."
    device = "cuda"
    model, coarse_cfg, wav2vec, neural_codec, clap = load_model()
    model.to(device)
    vocal_for_semantic, vocal_for_acoustic, prompt = load_data(
        vocal_path,
        prompt,
        10,
        device
    )
    print(vocal_for_semantic.shape, vocal_for_acoustic.shape, prompt)
    wave = model.generate(
        vocal_for_semantic,
        vocal_for_acoustic,
        prompt,
        10
    ).detach().cpu()[0]
    print(wave.shape)
    output = torchaudio.save("output.mp3", wave, sample_rate=24000)
