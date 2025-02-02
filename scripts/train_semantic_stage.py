import argparse
import os
import sys
from pathlib import Path
import torch
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer.songgen_trainer import SongGenTrainer
from models.Llambada.stages.semantic_stage import SemanticStage

def main():
    semantic_cfg = json.load(open("/workspace/singsong_upgrade/SG-SingSong-Internal/configs/model_config/llambada_tiny_cfg/semantic_stage.json"))
    print(semantic_cfg)
    semantic_transformer = SemanticStage(
        semantic_transformer=semantic_cfg["semantic_transformer"],
        clap=semantic_cfg["clap"],
        wav2vec=semantic_cfg["wav2vec"],
        neural_codec=semantic_cfg["neural_codec"],
        pad_id=semantic_cfg["pad_id"],
        unique_consecutive=semantic_cfg["unique_consecutive"],
        cross_entropy_loss_weights=semantic_cfg["cross_entropy_loss_weights"],
        mask_prob=semantic_cfg["mask_prob"]
    )
    trainer = S