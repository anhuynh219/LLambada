import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import (create_clap_quantized_from_config,
                                 create_coarse_transformer_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config, create_singsong_transformer_from_config,
                                 create_single_stage_trainer_from_config,
                                 load_model_config, load_training_config)
from utils import load_checkpoint_from_args, validate_train_args
import os 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train singsong stage')
    parser.add_argument('--results_folder', default='./results/singsong_new')
    parser.add_argument('--continue_from_dir', default="/workspace/continuation/open-musiclm/results/singsong/", type=str)
    parser.add_argument('--continue_from_step', default=4000, type=int)
    parser.add_argument('--model_config', default='./configs/model_config/sg_singsong_med.json')
    parser.add_argument('--training_config', default='./configs/trainer_config/train_musiclm_lma.json')
    parser.add_argument('--rvq_path', default='/workspace/base_weight/base_weight/musiclm_large_small_context/clap.rvq.950_no_fusion.pt')
    parser.add_argument('--kmeans_path', default='/workspace/base_weight/base_weight/musiclm_large_small_context/kmeans_10s_no_fusion.joblib')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--fine_tune_from', default=None, type=str)

    args = parser.parse_args()
    validate_train_args(args)

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)
    device = 'cuda'
    use_preprocessed_data = training_config.singsong_trainer_cfg.use_preprocessed_data

    if use_preprocessed_data:
        clap = None
        wav2vec = None
        print(f'training from preprocessed data {training_config.singsong_trainer_cfg.folder}')
    else:
        print('loading clap...')
        clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

        print('loading wav2vec...')
        wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)

    print('loading singsong stage...')
    singsong_transformer = create_singsong_transformer_from_config(model_config, args.fine_tune_from, device)

    trainer = create_single_stage_trainer_from_config(
        model_config=model_config,
        training_config=training_config,
        stage='singsong',
        results_folder=args.results_folder,
        transformer=singsong_transformer,
        clap=clap,
        wav2vec=wav2vec,
        encodec_wrapper=encodec_wrapper,
        device=device,
        accelerate_kwargs={
            'log_with': "tensorboard",
            'logging_dir': './logs/singsong'
        },
        config_paths=[args.model_config, args.training_config])

    load_checkpoint_from_args(trainer, args)

    trainer.train()