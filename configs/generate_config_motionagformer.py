import json
import sys
from pathlib import Path
from const import path


def generate_config(param, f_name):
    data_params = {
        'skip': True,
        'dataset': 'BMCLABS',
        'LODO': False, # When LODO is turned on training will be done on all datasets merged EXCEPT the one specified with "dataset"
        'data_type': 'h36m',
        'in_data_dim': 2,
        'data_centered': False,
        'merge_last_dim': False,
        'simulate_confidence_score': True,
        'pretrained_dataset_name': 'h36m',
        'model_prefix': 'motionagformer_',
        # options: mirror_reflection, random_rotation, random_translation
        # 'augmentation': [],
        'rotation_range': [-10, 10],
        'rotation_prob': 0.5,
        'mirror_prob': 0.5,
        'noise_prob': 0.0,
        'noise_std': 0.005,
        'axis_mask_prob': 0.0, # for 2D input we don't use axis mask
        'select_middle': False,
        'data_norm': False,
        'image_resolution': [1100, 1100],
        'views': ['backright', 'side_right'] #['back', 'front', 'side_right', 'side_left'], One side is enough as we have mirror augmentation
    }
    model_params = {
        'source_seq_len': 81,
        'num_joints': 17,
        'n_layers': 26,
        'dim_in': 3,
        'dim_feat': 64,
        'dim_rep': 512,
        'dim_out': 3,
        'mlp_ratio': 4,
        'attn_drop': 0.0,
        'drop': 0.0,
        "drop_path": 0.0,
        "use_layer_scale": True,
        "layer_scale_init_value": 0.00001,
        "use_adaptive_fusion": True,
        "num_heads": 8,
        "qkv_bias": False,
        "qkv_scale": None,
        "hierarchical": False,
        "use_temporal_similarity": True,
        "neighbour_num": 2,
        "temporal_connection_len": 1,
        "use_tcn": False,
        "graph_only": False,
        'classifier_dropout': 0.0,
        'merge_joints': False,
        'classifier_hidden_dims': [],
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/motionagformer/motionagformer-s-h36m.pth.tr"
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
        f = open("./configs/motionagformer/" + f_name, "rb")
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

    p_list = []
    for v in params['views']:
        assert v in ['backright' ,'front', 'side_right', 'side_left', 'back']
        # ==== Uncomment for 2D prespective projected poses
        # p_list.append(path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_POSES']['2D'][v])
        # ==== for projecting the 3D motion to 2D orthographically
        vie = f'camera_{v}'
        p_list.append(path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_POSES']['3D'][vie])
    params['data_path'] = p_list 
    params['labels_path'] = path.POSE_AND_LABEL[params['dataset']][params['data_type']]['PATH_LABELS']
    
    if params['force_LODO']:
        params['LODO'] = True
        if 'LODO' not in params['model_prefix']: params['model_prefix'] += '_LODO'

    return params, new_param
