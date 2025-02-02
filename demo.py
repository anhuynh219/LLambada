from encoders.clap_wrapper import CLAPEncoder
import librosa
import torch
import numpy as np
from utils.utils import calculate_param, int16_to_float32, float32_to_int16
if __name__ == "__main__":
    model = CLAPEncoder()
    model.freeze()
    model.eval()
    audio_data, _ = librosa.load('/workspace/Van/vocal_chunks/672c52d13b2c42b2b5081e6fde4860e3/672c52d13b2c42b2b5081e6fde4860e3_0.wav', sr=48000)
    audio_data = audio_data.reshape(1, -1)
    audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
    audio_embed = model.forward_audio(x = audio_data)
    print(audio_embed.shape)
    text = ["This is a ballad song"]
    # text_input = tokenizer(text)
    text_embed = model.forward_text(text)
    print(text_embed.shape)
