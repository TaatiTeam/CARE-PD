import json
import sys
from pathlib import Path
from const import path


def generate_config(param, f_name):
    data_params = {
        'skip': True,
        'dataset': 'BMCLABS',
        'LODO': False, # When LODO is turned on training will be done on all datasets merged EXCEPT the one specified with "dataset"
        'data_type': 'humanML3D',
        'in_data_dim': 263,
        'data_centered': False,
        'merge_last_dim': False,
        'simulate_confidence_score': False,
        'pretrained_dataset_name': 'humanML3D',
        'model_prefix': 'momask_',
        'data_norm': None,
        'select_middle': False,
        'humanML3D_normalization_data_path': f'{path.PROJECT_ROOT}/assets/stats/HumanML3D_norm_data',
        'views': [''] # No views for 3D data
    }
    model_params = {
        'source_seq_len': 196, # TODO: figure out what to put here
        'dim_rep': 512,
        'avg_over_time': True,
        'downsample_ratio': 4, # If input to model has X frames output encoding will have X / downsample_ratio tokens
        'classifier_dropout': 0,
        'classifier_hidden_dims': [],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/momask/net_best_fid.tar",
        'opt_file_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/momask/opt.txt"
    }
    learning_params = {
        'experiment_name': None,
        'batch_size': None,
        'criterion': None,
        'optimizer': None,
        'lr_backbone': 0.0001, # We need it to be set to a number so it doesn't cause an error with the scheduler but if backbone is frozen it doesn't do anything
        'lr_head': None,
        'weight_decay': None,
        'weight_decay_backbone': 0.0035, # We need it to be set to a number so it doesn't cause an error with the scheduler but if backbone is frozen it doesn't do anything
        'lambda_l1': None,
        'epochs': None,
        'scheduler': "StepLR",
        'lr_decay': 0.99,
        'lr_step_size': 1
    }

    params = {**param, **data_params, **model_params, **learning_params}

    if not param['combine_views_preds']:
        f = open("./configs/momask/" + f_name, "rb")
        params['model_prefix'] = params['model_prefix'] + f_name.split('.json')[0].replace('knn/', '')
    else:
        f = open(f_name, "rb")
        params['model_prefix'] = params['model_prefix'] + f_name.split('originalfile_')[1][:-5]
        params['this_run_num'] = Path(f_name).parent.parent.name
    new_param = json.load(f)

    for p in new_param:
        if not p in params.keys() and not param['combine_views_preds']:
            raise ValueError(
                "Error: One of the config parameters in " + "./Configs/" + f_name + " does not match code!")
        params[p] = new_param[p]

    params['data_path'] = [path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_POSES']]
    params['labels_path'] = path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_LABELS']

    if params['force_LODO']:
        params['LODO'] = True
        if 'LODO' not in params['model_prefix']: params['model_prefix'] += '_LODO'

    return params, new_param
