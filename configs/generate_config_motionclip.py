import json
import sys
from pathlib import Path
from const import path


def generate_config(param, f_name):
    data_params = {
        'skip': True,
        'dataset': 'BMCLab',
        'LODO': False, # When LODO is turned on training will be done on all datasets merged EXCEPT the one specified with "dataset"
        'data_type': '6DSMPL',
        'in_data_dim': 6,
        'data_centered': False,
        'merge_last_dim': False,
        'simulate_confidence_score': False,
        'pretrained_dataset_name': 'AMASS',
        'model_prefix': 'motionclip_',
        'data_norm': None,
        'select_middle': False,
        'data_norm': False,
        'views': [''] # No views for 3D data
    }
    model_params = {
        'source_seq_len': 60,
        'dim_rep': 512,
        'classifier_dropout': None,
        'classifier_hidden_dims': None,
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/motionclip/motionclip_encoder_checkpoint_0100.pth.tar",
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
        f = open("./configs/motionclip/" + f_name, "rb")
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



# opt.yml for motionclip pretrained chekcpoint 

# activation: gelu
# align_pose_frontview: false
# archiname: transformer
# batch_size: 20
# clip_image_losses:
# - cosine
# clip_lambda_ce: 1.0
# clip_lambda_cosine: 1.0
# clip_lambda_mse: 1.0
# clip_lambdas:
#   image:
#     cosine: 1.0
#   text:
#     cosine: 1.0
# clip_map_images: false
# clip_map_text: false
# clip_mappers_type: no_mapper
# clip_text_losses:
# - cosine
# cuda: true
# datapath: ./data/amass_db/amass_30fps_db.pt
# dataset: amass
# debug: false
# device: 0
# expname: exps
# folder: ./exps/paper-model
# glob: true
# glob_rot:
# - 3.141592653589793
# - 0
# - 0
# jointstype: vertices
# lambda_kl: 0.0
# lambda_rc: 100.0
# lambda_rcxyz: 100.0
# lambda_vel: 100.0
# lambda_velxyz: 1.0
# lambdas:
#   rc: 100.0
#   rcxyz: 100.0
#   vel: 100.0
# latent_dim: 512
# losses:
# - rc
# - rcxyz
# - vel
# lr: 0.0001
# max_len: -1
# min_len: -1
# modelname: cvae_transformer_rc_rcxyz_vel
# modeltype: cvae
# normalize_encoder_output: false
# num_epochs: 5000
# num_frames: 60
# num_layers: 8
# num_seq_max: -1
# pose_rep: rot6d
# sampling: conseq
# sampling_step: 1
# snapshot: 10
# translation: true
# vertstrans: false
