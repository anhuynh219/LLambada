import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import (create_clap_quantized_from_config,
                                 create_coarse_transformer_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 create_single_stage_trainer_from_config,
                                 load_model_config, load_training_config)
from utils import load_checkpoint_from_args, validate_train_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train coarse stage')
    parser.add_argument('--results_folder', default='./results/coarse_singsong_new_sgs800_new_28000/')
    parser.add_argument('--continue_from_dir', default=None, type=str)
    parser.add_argument('--continue_from_step', default= 28000, type=int)
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_musiclm_fma.json')
    parser.add_argument('--rvq_path', default='/workspace/base_weight/base_weight/musiclm_large_small_context/clap.rvq.950_no_fusion.pt')
    parser.add_argument('--kmeans_path', default='/workspace/base_weight/base_weight/musiclm_large_small_context/kmeans_10s_no_fusion.joblib')
    parser.add_argument('--fine_tune_from', default="/workspace/continuation/songgen_singsong/results/coarse_singsong_new_sgs800/coarse.transformer.28000.pt", type=str)

    args = parser.parse_args()

    validate_train_args(args)

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_preprocessed_data = training_config.coarse_trainer_cfg.use_preprocessed_data

    if use_preprocessed_data:
        clap = None
        wav2vec = None
        print(f'training from preprocessed data {training_config.coarse_trainer_cfg.folder}')
    else:
        print('loading clap...')
        clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

        print('loading wav2vec...')
        wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)

    print('loading coarse stage...')
    coarse_transformer = create_coarse_transformer_from_config(model_config, args.fine_tune_from, device)

    trainer = create_single_stage_trainer_from_config(
        model_config=model_config,
        training_config=training_config,
        stage='coarse',
        results_folder=args.results_folder,
        transformer=coarse_transformer,
        clap=clap,
        wav2vec=wav2vec,
        encodec_wrapper=encodec_wrapper,
        device=device,
        accelerate_kwargs={
            'log_with': "tensorboard",
            'logging_dir': './logs/coarse'
        },
        config_paths=[args.model_config, args.training_config])

    load_checkpoint_from_args(trainer, args)

    trainer.train()