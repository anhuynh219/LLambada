import numpy as np
import torch
import torch.nn as nn 
from encoders.CLAP.src.laion_clap import CLAP_Module
from transformers import RobertaTokenizer

def get_tokenizer():
    return RobertaTokenizer.from_pretrained('roberta-base')

class CLAPEncoder(nn.Module):
    def __init__(
        self,
        ckpt_path: str = "/root/Huy/Llambada/pretrained_pth/music_speech_audioset_epoch_15_esc_89.98.pt"
    ):
        super().__init__()
        self.model = CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
        if ckpt_path is not None:
            self.model.load_ckpt(ckpt_path)
        self.__freeze_backbone()

    def __freeze_backbone(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward_audio(
        self, 
        x: torch.Tensor
    ):
        audio_embed = self.model.get_audio_embedding_from_data(x, True)
        return audio_embed
    
    def forward_text(
        self, 
        x: list
    ):
        text_embed = self.model.get_text_embedding(x, use_tensor = True)
        return text_embed

if __name__ == "__main__":
    from utils.utils import calculate_param, int16_to_float32, float32_to_int16
    import librosa

    model = CLAPEncoder()
    tokenizer = get_tokenizer()
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
    