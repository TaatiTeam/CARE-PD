import json
import os
import sys
from pathlib import Path
from const import path


def generate_config(param, f_name):
    data_params = {
        'skip': True,
        'dataset': 'BMCLABS',
        'LODO': False, # When LODO is turned on training will be done on all datasets merged EXCEPT the one specified with "dataset"
        'data_type': 'h36m',
        'in_data_dim': 3,
        'data_orient': 'preprocessed', # [original, preprocessed]
        'data_centered': True,
        'merge_last_dim': True,
        'simulate_confidence_score': False,
        'pretrained_dataset_name': 'NTU',
        'voting': False,
        'model_prefix': 'POTR_',
        'data_norm': 'zscore',  # [minmax, unnorm, zscore]
        # 'interpolate': True,
        # 'augmentation': [],
        'rotation_range': [-10, 10],
        'rotation_prob': 0.5,
        'mirror_prob': 0.5,
        'noise_prob': 0.0,
        'noise_std': 0.005,
        'axis_mask_prob': 0.5,
        'select_middle': False,
        'views': [''] # No views for 3D data
    }

    model_params = {
        'source_seq_len': 80,
        'model_dim': 128,
        'n_joints': 17,
        'input_dim': 17 * 3,
        'pose_dim': 17 * 3,
        'num_encoder_layers': 4,
        'num_heads': 4,
        'dim_ffn': 2048,
        'init_fn': 'xavier_init',
        'pose_embedding_type': 'gcn_enc',
        'pos_enc_alpha': 10,
        'pos_enc_beta': 500,
        'downstream_strategy': 'both',  # ['both', 'class', 'both_then_class'],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/potr/pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt",
        'pose_format':  None,
        'classifier_dropout': 0.0,
        'classifier_hidden_dim': [],
        'preclass_rem_T': True
    }
    learning_params = {
        'max_gradient_norm': 0.1, # Needed in potr model definition
        'learning_rate_fn': 'step', # Needed in potr model definition
        'warmup_epochs': 10, # Needed in potr model definition
        'dropout': 0.3, # Needed in potr model definition
        'experiment_name': None,
        'batch_size': None,
        'criterion': None,
        'optimizer': None,
        'lr_backbone': 0.0001, # We need it to be set to a number so it doesn't cause an error with the scheduler but if backbone is frozen it doesn't do anything
        'lr_head': None,
        'weight_decay': None,
        'weight_decay_backbone': 0.0035, # We need it to be set to a number so it doesn't cause an error with the scheduler but if backbone is frozen it doesn't do anything
        'epochs': None,
        'scheduler': "StepLR",
        'lr_decay': 0.99,
        'lr_step_size': 1
    }

    params = {**param, **data_params, **model_params, **learning_params}

    if not param['combine_views_preds']:
        f = open("./configs/potr/" + f_name, "rb")
        params['model_prefix'] = params['model_prefix'] + f_name.split('.json')[0].replace('knn/', '')
    else:
        f = open(f_name, "rb")
        params['model_prefix'] = params['model_prefix'] + f_name.split('originalfile_')[1][:-5]
        params['this_run_num'] = Path(f_name).parent.parent.name
    new_param = json.load(f)

    for p in new_param:
        if not p in params.keys() and not param['combine_views_preds']:
            print("Error: One of the config parameters in " + "./Configs/" + f_name + " does not match code!")
            print('Configuration mismatch at:' + p)
            sys.exit(1)
        params[p] = new_param[p]
        
    params['data_path'] = [path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_POSES']['3D'][params['data_orient']]]
    params['labels_path'] = path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_LABELS']

    if params['force_LODO']:
        params['LODO'] = True
        if 'LODO' not in params['model_prefix']: params['model_prefix'] += '_LODO'

    return params, new_param
