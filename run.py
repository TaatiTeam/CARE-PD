import sys
import argparse
import glob
import os
import torch
import json
import numpy as np

import matplotlib.pyplot as plt
from torch import nn

import wandb
import joblib
import shutil
import optuna
import datetime

from configs import generate_config_motionbert, generate_config_poseformerv2, generate_config_mixste, generate_config_motionagformer, generate_config_momask, generate_config_motionclip, generate_config_potr 
from data.dataloaders import *
from model.motion_encoder import MotionEncoder
from model.backbone_loader import load_pretrained_backbone, count_parameters, load_pretrained_weights
from train import train_model, validate_model
from const import path
from const import const
from learning.utils import log_cfm_to_wandb
from utility import utils
from utility.utils import set_random_seed, override_dataset
from test import *


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SaveStudyCallback:
    def __init__(self, save_frequency, file_path):
        self.save_frequency = save_frequency
        self.file_path = file_path
        self.trial_counter = 0
        if not os.path.exists(self.file_path.split('study_mid.pkl')[0]):
            os.mkdir(self.file_path.split('study_mid.pkl')[0])

    def __call__(self, study, trial):
        self.trial_counter += 1
        if self.trial_counter % self.save_frequency == 0:
            joblib.dump(study, self.file_path)
        print('⭐️'*80)


def load_hyperparams_from_study(params, backbone_name, exp_path):
    try:
        if 'study.pkl' in os.listdir(exp_path):
            study_path =  os.path.join(exp_path, 'study.pkl')
        else:
            study_path =  os.path.join(exp_path, 'study_mid.pkl') # loading from mid study
        print(f"Loading study from {study_path}")
        study = joblib.load(study_path)
        best_params = study.best_trial.params
        best_params['best_trial_number'] = study.best_trial.number
        if 'num_epochs'  in best_params: best_params['epochs'] = best_params['num_epochs'] # Compatibility with older hypertuning studies that were conducted
        if 'avg_best_epoch' in study.best_trial.user_attrs:
            avg_best_epoch = study.best_trial.user_attrs['avg_best_epoch']
            print(f"Overriding number of epochs {best_params['epochs']} with avg_best_epoch found during this trial: {avg_best_epoch}")
            best_params['epochs'] = avg_best_epoch
        
        print(f'Testing the best params found with trial {study.best_trial.number}: {best_params}')
        
        if params['dataset'] != const.DATASET_FOR_TUNING[params['train_mode']] or params['LODO']: # only do epoch tunning:
            extra_params = study.best_trial.user_attrs
            best_params = {**best_params, **{k: v for k, v in extra_params.items() if k != 'val_f1_scores_all_folds'}}
            
        top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5] #change to get the best X trials
        for i, trial in enumerate(top_trials):
            avg_best_epoch = trial.user_attrs['avg_best_epoch'] if 'avg_best_epoch' in trial.user_attrs else None
            print(f"best{i}: Trial {trial.number}, Value: {trial.value}, Params: {trial.params}, avg_best_epoch: {avg_best_epoch}")
    except Exception as e:
        print(f"**STUDY NOT FOUND: Running test with some default params**, because of {e}")
        variations = []
        changes = ['']
        best_params = {
                "lr": 1e-05,
                "epochs": 20,
                "num_hidden_layers": 2,
                "layer_sizes":[256, 50, 16, 3],
                # "architecture_type": architecture_type,
                "optimizer": 'RMSprop',
                "use_weighted_loss": True,
                "batch_size": 128,
                # "use_batch_norm": False,
                "dropout_rate": 0.1,
                'weight_decay': 0.00057,
                'classifier_hidden_dims': 'no_hidden_layers',
                'momentum': 0.66
                }
        variations.append(best_params)

    if 'no_hidden_layers' in best_params['classifier_hidden_dims']:
        best_params['classifier_hidden_dims'] = []
    print(exp_path)
    print("==============================🏆 BEST MODEL 🏆===========================================")
    print("========================================================================================")
    print(f"Trial {best_params['best_trial_number']}, epochs: {best_params['epochs']}, batch_size: {best_params['batch_size']}")
    print(f"classifier_hidden_dims: {best_params['classifier_hidden_dims']}")
    print(f"optimizer_name: {best_params['optimizer']}, , weight_decay:{best_params['weight_decay']}, criterion: {best_params['criterion']}")
    print(f"classifier_dropout: {best_params['dropout_rate']}, lambda_l1:{best_params['lambda_l1']}")
    print(f"lr_head: {best_params['lr']}, lr_backbone: {best_params.get('lr_backbone')}")
    if best_params['criterion'] == 'FocalLoss': print(f"alpha: {best_params['alpha']}, gamma: {best_params['gamma']}")
    if params['train_mode'] == 'end2end': print(f"weight_decay_backbone:{best_params['weight_decay_backbone']}")
    print("========================================================================================")
    print("========================================================================================")
        
    params['classifier_dropout'] = best_params['dropout_rate']
    params['classifier_hidden_dims'] = best_params['classifier_hidden_dims']
    params['batch_size'] = best_params['batch_size']
    params['optimizer'] = best_params['optimizer']
    params['lr_head'] = best_params['lr']
    params['epochs'] = best_params['epochs']
    params['lambda_l1'] = best_params['lambda_l1']
    params['criterion'] = best_params['criterion']
    if params['criterion'] == 'FocalLoss':
        params['alpha'] = best_params['alpha']
        params['gamma'] = best_params['gamma']
    if params['optimizer'] in ['AdamW', 'Adam', 'RMSprop']:
        params['weight_decay'] =  best_params['weight_decay']
    if params['optimizer'] == 'SGD':
        params['momentum'] = best_params['momentum']
    if 'lr_backbone' in best_params:
        params['lr_backbone'] = best_params['lr_backbone']

    return params, best_params


def objective_hyperparam_opt_CV(trial, splits, backbone_name, run_name, params, search_space_key):
    if 'motionbertlite' in params['model_checkpoint_path']:  search_space_key = 'motionbertlite'
    search_space = const.MODEL_HYPERTUNING_CHOICES[params['train_mode']]
    for key in search_space_key:
        search_space = search_space[key]
    classifier_hidden_dims_choices = search_space['classifier_hidden_dims']
    classifier_op = list(classifier_hidden_dims_choices.keys())

    # Only tune epochs if not BMCLab
    if params['dataset'] != const.DATASET_FOR_TUNING[params['train_mode']] or params['LODO']: # only do epoch tunning
        epochs = trial.suggest_categorical('epochs', search_space['epochs'])
        params['epochs'] = epochs
        # Other hyperparams: use fixed values from search_space
        params['lr_head'] = search_space['lr'][0]
        params['lr_backbone'] = search_space.get('lr_backbone', [params['lr_backbone']])[0]
        params['batch_size'] = search_space['batch_size'][0]
        params['optimizer'] = search_space['optimizer'][0]
        params['lambda_l1'] = search_space['lambda_l1'][0]
        params['classifier_dropout'] = search_space['dropout_rate'][0]
        params['criterion'] = search_space['criterion'][0]
        chosen_option = list(search_space['classifier_hidden_dims'].keys())[0]
        params['classifier_hidden_dims'] = search_space['classifier_hidden_dims'][chosen_option]
        if params['criterion'] == 'FocalLoss':
            params['alpha'] = search_space['alpha'][0]
            params['gamma'] = search_space['gamma'][0]
        if params['optimizer'] in ['AdamW', 'Adam', 'RMSprop', 'SGD']:
            params['weight_decay'] = search_space['weight_decay'][0]
            if 'weight_decay_backbone' in search_space:
                params['weight_decay_backbone'] = search_space['weight_decay_backbone'][0]
        fixed_param_mapping = {
                'lr_head': 'lr',
                'lr_backbone': 'lr_backbone',
                'batch_size': 'batch_size',
                'optimizer': 'optimizer',
                'lambda_l1': 'lambda_l1',
                'classifier_dropout': 'dropout_rate',
                'criterion': 'criterion',
                'classifier_hidden_dims': 'classifier_hidden_dims'
            }
        for internal_key, external_key in fixed_param_mapping.items():
            if internal_key in params:
                trial.set_user_attr(external_key, params[internal_key])
        # Log alpha and gamma if using FocalLoss
        if params['criterion'] == 'FocalLoss':
            trial.set_user_attr('alpha', params['alpha'])
            trial.set_user_attr('gamma', params['gamma'])
        if 'weight_decay' in params:
            trial.set_user_attr('weight_decay', params['weight_decay'])
        if 'weight_decay_backbone' in params:
            trial.set_user_attr('weight_decay_backbone', params['weight_decay_backbone'])

    else:
        # If BMCLab full search space is needed
        epochs = trial.suggest_categorical('epochs', search_space['epochs']) 
        lr = trial.suggest_categorical('lr', search_space['lr'])
        lr_backbone = trial.suggest_categorical('lr_backbone', search_space.get('lr_backbone', [params['lr_backbone']]))
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])
        optimizer_name = trial.suggest_categorical('optimizer', search_space['optimizer'])
        lambda_l1 = trial.suggest_categorical('lambda_l1', search_space['lambda_l1']) #[0, 0.0001, 0.001] # Try this also
        dropout_rate = trial.suggest_categorical('dropout_rate', search_space['dropout_rate'])
        criterion = trial.suggest_categorical('criterion', search_space['criterion']) 
        chosen_option  = trial.suggest_categorical('classifier_hidden_dims', classifier_op) 
        classifier_hidden_dims = classifier_hidden_dims_choices[chosen_option]
        
        params['classifier_dropout'] = dropout_rate
        params['classifier_hidden_dims'] = classifier_hidden_dims
        params['batch_size'] = batch_size
        params['optimizer'] = optimizer_name
        params['lr_head'] = lr
        params['lr_backbone'] = lr_backbone
        params['epochs'] = epochs
        params['lambda_l1'] = lambda_l1
        params['criterion'] = criterion
        
        if params['criterion'] == 'FocalLoss':
            params['alpha'] = trial.suggest_categorical('alpha', search_space['alpha'])
            params['gamma'] = trial.suggest_categorical('gamma', search_space['gamma'])
            # a_min, a_max = search_space['alpha']
            # g_min, g_max = search_space['gamma']
            # params['alpha'] = trial.suggest_float('alpha', a_min, a_max)
            # params['gamma'] = trial.suggest_float('gamma', g_min, g_max)
            
        if params['optimizer'] in ['AdamW', 'Adam', 'RMSprop', 'SGD']:
            params['weight_decay'] = trial.suggest_categorical('weight_decay', search_space['weight_decay'])
            if 'weight_decay_backbone' in search_space:
                params['weight_decay_backbone'] = trial.suggest_categorical('weight_decay_backbone', search_space['weight_decay_backbone'])
    
    print("========================================================================================")
    print("========================================================================================")
    print(f"⭐️ Trial {trial.number}/{params['trials_left_to_do']} ⭐️: epochs: {epochs}, batch_size: {params['batch_size']}")
    print(f"Backbone: {backbone_name}, dataset: {params['dataset']}")
    print(f"optimizer_name: {params['optimizer']}, weight_decay:{params['weight_decay']}")
    print(f"lr_head: {params['lr_head']}, lr_backbone: {params['lr_backbone']}")
    print(f"criterion: {params['criterion']}")
    if params['criterion'] == 'FocalLoss': print(f"alpha: {params['alpha']}, gamma: {params['gamma']}")
    if params['train_mode'] == 'end2end': print(f"weight_decay_backbone:{params['weight_decay_backbone']}")
    print("========================================================================================")
    print("========================================================================================")
    tags = [
        f'{backbone_name}',
        f"trial:{format(trial.number, '05d')}",
        f"dataset:{params['dataset']}",
        f"train_mode:{params['train_mode']}",
        f"num_folds:{params['num_folds']}",
        f"run_num:{params['this_run_num']}",
        f"LODO:{params['LODO']}",
        f"hypertune:{params['hypertune']}",
        f"configname:{params['configname_thisrunnum'].split('_run')[0]}",
    ]
    wandb.init(project='MotionEncoderEvaluator', 
            group=params['experiment_name'], 
            job_type='hyperparam_trial', 
            name=f"{backbone_name}_{params['dataset']}{'LODO' if params['LODO'] else ''}_{params['configname_thisrunnum']}_{format(trial.number, '05d')}",
            tags=tags,
            reinit=True)
    
    val_f1_scores_all_folds = []  # Aggregate validation F1 score across all folds
    for fold, (train_dataset, eval_dataset) in enumerate(splits, start=1):
        train_dataset_fn, eval_dataset_fn, class_weights = dataloader_factory(params, train_dataset, eval_dataset)
        start_time = datetime.datetime.now()

        model_backbone = load_pretrained_backbone(params, backbone_name)
        model = MotionEncoder(backbone=model_backbone,
                                params=params,
                                num_classes=params['num_classes'],
                                train_mode=params['train_mode'])
        model = model.to(_DEVICE)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            
        if fold == 1:
            model_params = count_parameters(model)
            print(f"[INFO] Model has {model_params} parameters.")
        
        _, _, best_val_cfm, all_val_f1 = train_model(params, class_weights, train_dataset_fn, eval_dataset_fn, model, fold, backbone_name, mode="HYPERTUNE")

        _, _, _, cfm_train = validate_model(model, train_dataset_fn, params, class_weights)
        log_cfm_to_wandb(best_val_cfm, fold, params['num_classes'], kind='val')
        log_cfm_to_wandb(cfm_train, fold, params['num_classes'], kind='train')

        # all_val_f1 = [f1_epoch1, f1_epoch2, ..., f1_epochN]
        # val_f1_scores_all_folds: list of those all_val_f1 lists, one per fold
        val_f1_scores_all_folds.append(all_val_f1)
        end_time = datetime.datetime.now()
        
        duration = end_time - start_time
        print(f"Fold {fold} run time:", duration)
        torch.cuda.empty_cache()

    # Compute average performance across all folds
    # 1. Initialize with average F1 at epoch 0 across all folds
    best_avg_val_f1 = np.mean([f1_scores[0] for f1_scores in val_f1_scores_all_folds])
    best_epoch = 0
    # 2. Go through every later epoch and check if its average F1 is better
    for epoch in range(1, len(val_f1_scores_all_folds[0])):
        avg_f1 = np.mean([f1_scores[epoch] for f1_scores in val_f1_scores_all_folds])
        if avg_f1 > best_avg_val_f1:
            best_avg_val_f1 = avg_f1
            best_epoch = epoch

    trial.set_user_attr("avg_best_epoch", best_epoch)
    print(f"Trial {trial.number}, avg_best_epoch: {best_epoch}")
    trial.set_user_attr("val_f1_scores_all_folds", val_f1_scores_all_folds)
    trial.set_user_attr("best_val_f1", best_avg_val_f1)

    wandb.log({
            f'final/best_avg_val_f1': best_avg_val_f1,
            f'final/avg_best_epoch': best_epoch
        })

    wandb.config.update(params)
    wandb.config.update({'trial_name': f"{backbone_name}_{params['dataset']}{'LODO' if params['LODO'] else ''}_{format(trial.number, '05d')}"})
    wandb.finish()
    
    return best_avg_val_f1

def get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, fold, augmented_datasets=False):
    if not params['LODO']:
        train_dataset, eval_dataset = dataset_factory(params, backbone_name, fold)
    else:
        if params['AID'] and not augmented_datasets:
            # This should be LOSO as it is the in domain dataset
            assert params['num_folds'] == const.NUM_OF_PATIENTS_PER_DATASET[params['dataset']], "AID is only supported for LOSO"
            train_dataset, eval_dataset = dataset_factory(params, backbone_name, fold)
        else:
            if params['AID'] and augmented_datasets:
                nn_params = params.copy()
                nn_params['num_folds'] = 6
                other_datasets = [d for d in const.SUPPORTED_DATASETS if d != params['dataset']]
                other_datasets = [dataset_factory(override_dataset(nn_params, d), backbone_name, fold) for d in other_datasets]
                train_dataset = torch.utils.data.ConcatDataset([x[0] for x in other_datasets])
                eval_dataset  = torch.utils.data.ConcatDataset([x[1] for x in other_datasets])
            else:
                other_datasets = [d for d in const.SUPPORTED_DATASETS if d != params['dataset']]
                other_datasets = [dataset_factory(override_dataset(params, d), backbone_name, fold) for d in other_datasets]
                train_dataset = torch.utils.data.ConcatDataset([x[0] for x in other_datasets])
                eval_dataset  = torch.utils.data.ConcatDataset([x[1] for x in other_datasets])
                
    return train_dataset, eval_dataset
    
# python eval_encoder_hypertune --hypertune 0 --cross_dataset_test 0 
# not param['hypertune'] and not param['cross_dataset_test'] and not param['just_gen_dataset']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  

    parser.add_argument('--backbone', type=str, default='motionbert', help='model name (motionbert, potr, mixste, poseformerv2, motionagformer, momask, motionclip)')
    parser.add_argument('--config', type=str, default=None, help='if left as None all configs will be processed, if it is set to a specific config file (with extension) only it will be processed.')
    parser.add_argument('--train_mode', type=str, default='classifier_only', help='train mode( end2end, classifier_only )')
    parser.add_argument('--num_folds', type=int, default=6, help='Which cv variant to use. If -1, LOSO is performed')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int, help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--ntrials', default=30, type=int, help='number of hyper-param tuning trials')
    parser.add_argument('--this_run_num', type=str, help='Prefix for folder in which results specific to this run should be outputted.')
    parser.add_argument('--readstudyfrom', type=int)
    parser.add_argument('--hypertune', default=1, type=int, help='perform hyper parameter tuning [0 or 1]')
    parser.add_argument('--just_gen_dataset', default=0, type=int, help='When set to 1 only the generation of .pkl dataset files will be initiated.')
    parser.add_argument('--cross_dataset_test', type=int, help='perform testing on --dataset flag dataset (0), or perform testing on all other supported datasets (1) [0 or 1]. Only matters when --hypertune=0')
    parser.add_argument('--pretrained', default=0, type=int, help='If 1 training will be skipped during testing and already existing checkpoints will be used.')
    parser.add_argument('--overwrite_results', default=0, type=int, help='If 1 computed results during model evaluation will be computed again and overwrite existing results')
    parser.add_argument('--force_LODO', default=0, type=int, help='If 1 all config files will have LODO overriden to True and the model_prefix will have the _LODO suffix added if it is not already present')
    parser.add_argument('--AID', default=0, type=int, help='If 1 the model will be trained with Augmented-In-Domain set-up')
    
    parser.add_argument('--combine_views_preds', default=0, type=int, help='If 1 the predictions of all views will be combined **(Make sure you pass --views_configs as well)**')
    parser.add_argument('--views_path', default=None, type=str, nargs='+', help='List of view file paths, First give BACK path and then SIDE path')
    parser.add_argument('--exp_name_rigid', default=None, type=str, help='if you want to override the experiment name in the config file')
    parser.add_argument('--prefer_right', default=0, type=int, help='Prefer right side view for prediction aggregation [0 or 1]')
    
    parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    parser.add_argument('--metadata', default='', type=str, help="add metadata prob to the training 'gender,age,bmi,height,weight'")

    args = parser.parse_args()
    
    param = vars(args)
    print("\n" + "="*40)
    print("🔹 Parameters Configuration 🔹")
    print("="*40)
    print(json.dumps(param, indent=4))
    print("="*40 + "\n")
    
    param['metadata'] = param['metadata'].split(',') if param['metadata'] else []
    
    torch.backends.cudnn.benchmark = False
    
    backbone_name = param['backbone']
    conf_path = const.BACKBONE_CONFIGS.get(backbone_name)
    if conf_path is None:
        raise NotImplementedError(f"Backbone '{backbone_name}' is not supported")
    
    backbone_config_generators = {
        'potr': generate_config_potr.generate_config,
        'motionbert': generate_config_motionbert.generate_config,
        'poseformerv2': generate_config_poseformerv2.generate_config,
        'mixste': generate_config_mixste.generate_config,
        'motionagformer': generate_config_motionagformer.generate_config,
        'momask': generate_config_momask.generate_config,
        'motionclip': generate_config_motionclip.generate_config
    }
    
    config_list = sorted(os.listdir(conf_path))
    if param['combine_views_preds']:
        assert param['views_path'] is not None, "If --combine_views_preds is set to 1, --views_configs must be provided with the config files for each view."
        config_list = []
        config_list.append(glob.glob(os.path.join(path.OUT_PATH, param['views_path'][0], 'config', 'originalfile_*.json'))[0])
        config_list.append(glob.glob(os.path.join(path.OUT_PATH, param['views_path'][1], 'config', 'originalfile_*.json'))[0]) 
        view_out_pathes = []

    for fi in config_list:
        if param['config'] is not None and param['config'] != fi and not param['combine_views_preds']: continue # If config was specified only run that config
        
        generate_config_func = backbone_config_generators.get(backbone_name)
        if generate_config_func is None:
            raise NotImplementedError(f"Backbone '{backbone_name}' does not exist!")

        params, new_params = generate_config_func(param, fi)
        if param['exp_name_rigid'] is not None:
            params['experiment_name'] = param['exp_name_rigid']
        
        # == Determine number of folds ==
        if params['num_folds'] == -1: 
            if params['LODO'] and not params['AID']: raise NotImplementedError('num_folds is -1 (which means LOSO). This is not implemented when performing LODO but not AID.')
            params['num_folds'] = const.NUM_OF_PATIENTS_PER_DATASET[params['dataset']]
        
        # == Determine number of classes ==
        if params['dataset'] in const.SUPPORTED_DATASETS:
            if not params['LODO']:
                params['num_classes'] = const.NUM_CLASSES_PER_DATASET[params['dataset']]
            else:
                params['num_classes'] = int(np.max([n for d,n in const.NUM_CLASSES_PER_DATASET.items() if d != params['dataset']]))
        else:
            print(f"Dataset '{params['dataset']}' not found in SUPPORTED_DATASETS. Please check the dataset name.")
            print(f"Supported datasets are: {const.SUPPORTED_DATASETS}")
            raise NotImplementedError(f"dataset '{params['dataset']}' is not supported.")
        
        if params['skip'] and params['config'] is None: continue

        all_folds = range(1, params['num_folds'] + 1)
        set_random_seed(param['seed'])
        
        if param['hypertune'] and not param['just_gen_dataset']: 
            EXP = 'perform_hypertunning' 
            assert param['combine_views_preds'] == 0, "Hypertuning is not supported with --combine_views_preds. Please set it to 0."
        elif not param['hypertune'] and not param['cross_dataset_test'] and not param['just_gen_dataset']: 
            EXP = 'perform_intra_dataset_test'
        elif not param['hypertune'] and param['cross_dataset_test'] and not params['LODO'] and not param['just_gen_dataset']:
            EXP = 'perform_cross_dataset_test'     
        elif not param['hypertune'] and param['cross_dataset_test'] and params['LODO'] and not param['just_gen_dataset'] and not param['AID']:
            EXP = 'perform_cross_dataset_test_LODO'
        elif not param['hypertune'] and param['cross_dataset_test'] and params['LODO'] and not param['just_gen_dataset'] and param['AID']:
            EXP = 'perform_cross_dataset_test_AID'
        

        # == Hypertuning ==
        if  EXP == 'perform_hypertunning':
            this_run_num = param['this_run_num']
            utils.create_dir_tree2(os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix']), this_run_num)
            params['model_prefix'] = params['model_prefix'] + '/' + str(this_run_num)

            splits = []
            for fold in all_folds:
                train_dataset, eval_dataset = get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, fold)
                splits.append((train_dataset, eval_dataset))
            
            print("🔹 Hypertuning space 🔹")
            print("="*40)
            if params['dataset'] == const.DATASET_FOR_TUNING[params['train_mode']] and not params['LODO']: 
                search_space_key = ['initial_space']
            else: # only do epoch tunning:
                if params['views'][0] in const.SUPPORTED_VIEWS:
                    search_space_key = [backbone_name, params['views'][0]]
                else:
                    search_space_key = [backbone_name]
            space = const.MODEL_HYPERTUNING_CHOICES[params['train_mode']]
            for key in search_space_key:
                space = space[key]
            print(space)
            print("="*40 + "\n")
             
            read_from_prev_study_path = os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'].split('/')[0], str(param['readstudyfrom']),'study_mid.pkl')
            print(f'Read from study: {read_from_prev_study_path} exists: {os.path.exists(read_from_prev_study_path)}')
            if not param['tune_fresh'] and os.path.exists(read_from_prev_study_path):
                # Load the existing study
                study = joblib.load(read_from_prev_study_path)
                print(f"Resumed study with {len(study.trials)} trials.")
                trials_left_to_do = param['ntrials'] - len(study.trials)
                if trials_left_to_do < 1: trials_left_to_do = 1
                print(f"{trials_left_to_do} trials left.")
            else:
                # Create a new study
                if params['dataset'] != const.DATASET_FOR_TUNING[params['train_mode']] or params['LODO']: # only do epoch tunning
                    sampler = optuna.samplers.GridSampler({'epochs': space['epochs']})
                    study = optuna.create_study(direction='maximize', sampler=sampler)
                else:
                    study = optuna.create_study(direction='maximize')  # full search for BMCLab 
                print("[INFO]: 🔎 Created a new study.")
                trials_left_to_do = param['ntrials']
                print(f"[INFO]: {trials_left_to_do} trials left.")
                config_outpath = os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'], 'config')
                with open(os.path.join(config_outpath, 'config_excluding_study_params.json'), 'w') as f:
                    json.dump(params, f, indent=4)
                shutil.copy(f"./configs/{backbone_name}/{fi}", os.path.join(config_outpath, f'originalfile_{fi}'))
                params['configname_thisrunnum'] = fi.split('.json')[0] + '_run' + str(this_run_num)
            
            params['trials_left_to_do'] = trials_left_to_do
            print("\n" + "="*40)
            print(f"[INFO]: ⚙️ Hypertuning {params['backbone']} on {params['dataset']}{'_LODO' if params['LODO'] else ''} of type {params['data_type']} ⚙️")
            save_study_callback = SaveStudyCallback(save_frequency=1, file_path=os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'],'study_mid.pkl'))
            study.optimize(lambda trial: objective_hyperparam_opt_CV(trial, splits, backbone_name, params['experiment_name'] + '/' + params['model_prefix'], params, search_space_key), n_trials=trials_left_to_do, callbacks=[save_study_callback])
            
            print(f"Number of finished trials on {params['dataset']}{'_LODO' if params['LODO'] else ''} with {backbone_name}:", len(study.trials))
            print(f"Best trial on {params['dataset']}{'_LODO' if params['LODO'] else ''} with {backbone_name}:", study.best_trial.params)
            print(f"Best trial metadata: {study.best_trial.user_attrs}, 'avg_best_epoch' denotes the optimal number of epochs found")
            print(f"avg f1 score of best trial on {params['dataset']}{'_LODO' if params['LODO'] else ''} with {backbone_name}: {study.best_value}")
            print(f"best trial number on {params['dataset']}{'_LODO' if params['LODO'] else ''} with {backbone_name}: {study.best_trial.number}")
            best_params = study.best_trial.params
            best_params['best_trial_number'] = study.best_trial.number
            tags = [
                f"{backbone_name}",
                f"dataset:{params['dataset']}",
                f"LODO:{params['LODO']}",
                f"train_mode:{params['train_mode']}",
                f"num_folds:{params['num_folds']}",
                f"run_num:{params['this_run_num']}"
            ]
            wandb.init(
                project='MotionEncoderEvaluator',
                name=f"{backbone_name}_{params['dataset']}{'_LODO' if params['LODO'] else ''}_final",
                group=params['experiment_name'], 
                job_type='hyperparam_study',
                tags=tags,
                reinit=True
            )
            wandb.config.update({'params': params})
            wandb.config.update({'best_optuna_params': study.best_trial.params})
            wandb.config.update({'best_trial_number': study.best_trial.number})
            wandb.config.update({'best_val_f1_score': study.best_value})
            wandb.log({
                'best_val_f1_score': study.best_value,
                'best_trial_number': study.best_trial.number,
                'best_optuna_params': study.best_trial.params
            })
            wandb.finish()
            joblib.dump(study, os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'],'study.pkl'))
            print('[INFO] 🏁🏁🏁 Hypertuning Finished 🏁🏁🏁, Study saved at:', os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'],'study.pkl'))
            
        # == Intra-dataset testing ==    
        elif EXP == 'perform_intra_dataset_test':
            exp_path = os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'], str(params['this_run_num']))
            params['model_prefix'] = params['model_prefix'] + '/' + str(params['this_run_num'])
            with open(os.path.join(exp_path, 'config', 'config_excluding_study_params.json'), 'r') as f:
                saved_config = json.load(f)
            utils.compare_two_configs(saved_config, params)
            params, best_params = load_hyperparams_from_study(params, backbone_name, exp_path)
            lodo_suffix = '_LODO' if params['LODO'] else ''
            clarification_str = f"train_{params['dataset']}{lodo_suffix}_test_{params['dataset']}{lodo_suffix}_{params['num_folds']}fold"
            model_checkpoint_clarification_str = f"train_{params['dataset']}{lodo_suffix}_{params['num_folds']}fold"
            print(f"Testing intra-dataset performance of {backbone_name}: {clarification_str}")
            splits = []
            for fold in all_folds:
                train_dataset, eval_dataset = get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, fold)
                train_dataset_fn, eval_dataset_fn, class_weights = dataloader_factory(params, train_dataset, eval_dataset, eval_batch_size=1)
                splits.append((train_dataset_fn, eval_dataset_fn, class_weights))
            test__hypertune(params, best_params, new_params, splits, backbone_name, clarification_str, model_checkpoint_clarification_str, exp_path, _DEVICE)
            if params['combine_views_preds']:
                view_out_pathes.append(os.path.join(exp_path, clarification_str))
            
            if params['combine_views_preds'] and len(view_out_pathes) == 2: # now we have results for both views
                print(" [INFO] 🔥🔥🔥 Now Combining the predictions of all views 🔥🔥🔥")
                path_back = view_out_pathes[0]
                path_side = view_out_pathes[1]
                out_folder_name = backbone_name+'_'+Path(path_back).parent.parent.name.replace(f'{backbone_name}_', '') + '--'+Path(path_side).parent.parent.name.replace(f'{backbone_name}_', '')
                if params['prefer_right']:
                    out_folder_name = out_folder_name + '_PreferedRight'
                out_folder = os.path.join(Path(exp_path).parent.parent, out_folder_name, Path(exp_path).name, clarification_str)
                test_combine_view_predictions(path_back, path_side, all_folds[-1], out_folder, out_folder_name, backbone_name, params)

            
        # == Cross-dataset testing ==    
        elif EXP == 'perform_cross_dataset_test':
            exp_path = os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'], str(params['this_run_num']))
            params['model_prefix'] = params['model_prefix'] + '/' + str(params['this_run_num'])
            params, best_params = load_hyperparams_from_study(params, backbone_name, exp_path)
            train_dataset_name = params['dataset']
            all_other_datasets = [x for x in const.SUPPORTED_DATASETS if x != train_dataset_name]
            print(f"Testing cross-dataset performance of {backbone_name} trained on {train_dataset_name}. Testing on: {all_other_datasets}")
            train_dataset, eval_dataset = dataset_factory(params, backbone_name, 1)
            train_dataset_fn, _, class_weights = dataloader_factory(params, train_dataset, eval_dataset) # This will return entire dataset in train_dataset_fn
            for test_dataset_name in all_other_datasets:
                clarification_str = f'train_{train_dataset_name}_test_{test_dataset_name}'
                model_checkpoint_clarification_str = f'train_{train_dataset_name}_1fold_all_merged'
                print(f'Testing {backbone_name}: {clarification_str}')
                train_dataset, eval_dataset = dataset_factory(override_dataset(params, test_dataset_name), backbone_name, 1)
                _, eval_dataset_fn, _ = dataloader_factory(override_dataset(params, test_dataset_name), train_dataset, eval_dataset, eval_batch_size=1) # This will return entire dataset in eval_dataset_fn
                splits = [(train_dataset_fn, eval_dataset_fn, class_weights)]
                test__hypertune(params, best_params, new_params, splits, backbone_name, clarification_str, model_checkpoint_clarification_str, exp_path, _DEVICE)
                if params['combine_views_preds']:
                    view_out_pathes.append(os.path.join(exp_path, clarification_str))
                    
            if params['combine_views_preds'] and len(view_out_pathes) == len(all_other_datasets)*2: # now we have results for both views for all datasets
                print(" [INFO] 🔥🔥🔥 Now Combining the predictions of all views 🔥🔥🔥")
                for ii, test_dataset_name  in enumerate(all_other_datasets):
                    path_back = view_out_pathes[ii]
                    path_side = view_out_pathes[ii+len(all_other_datasets)]
                    out_folder_name = backbone_name+'_'+Path(path_back).parent.parent.name.replace(f'{backbone_name}_', '') + '--'+Path(path_side).parent.parent.name.replace(f'{backbone_name}_', '')
                    if params['prefer_right']:
                        out_folder_name = out_folder_name + '_PreferedRight'
                    clarification_str = Path(path_back).name
                    out_folder = os.path.join(Path(exp_path).parent.parent, out_folder_name, Path(exp_path).name, clarification_str)
                    test_combine_view_predictions(path_back, path_side, 1, out_folder, clarification_str, backbone_name, params)
                
        # == Cross-dataset testing with LODO ==       
        elif EXP == 'perform_cross_dataset_test_LODO':
            exp_path = os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'], str(params['this_run_num']))
            params['model_prefix'] = params['model_prefix'] + '/' + str(params['this_run_num'])
            params, best_params = load_hyperparams_from_study(params, backbone_name, exp_path)

            all_other_datasets = [x for x in const.SUPPORTED_DATASETS if x != params['dataset']]
            print(f"Testing cross-dataset performance of {backbone_name} trained on {all_other_datasets}. Testing on: {params['dataset']}")
            clarification_str = f"train_{'_'.join(all_other_datasets)}_test_{params['dataset']}"
            model_checkpoint_clarification_str = f"train_{'_'.join(all_other_datasets)}_1fold_all_merged"
            print(f'Testing {backbone_name}: {clarification_str}')

            # TEST DATA LOADER
            train_dataset, eval_dataset = dataset_factory(params, backbone_name, 1)
            _, eval_dataset_fn, _ = dataloader_factory(params, train_dataset, eval_dataset, eval_batch_size=1) # This will return entire params['dataset'] in eval_dataset_fn
            
            # TRAIN DATALOADER
            train_dataset, eval_dataset = get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, 1)
            train_dataset_fn, _, class_weights = dataloader_factory(params, train_dataset, eval_dataset) # This will return all datasets but params['dataset'] in train_dataset_fn
            
            splits = [(train_dataset_fn, eval_dataset_fn, class_weights)]
            test__hypertune(params, best_params, new_params, splits, backbone_name, clarification_str, model_checkpoint_clarification_str, exp_path, _DEVICE)
            if params['combine_views_preds']:
                view_out_pathes.append(os.path.join(exp_path, clarification_str))
                    
            if params['combine_views_preds'] and len(view_out_pathes) == 2: # now we have results for both views
                print(" [INFO] 🔥🔥🔥 Now Combining the predictions of all views 🔥🔥🔥")
                path_back = view_out_pathes[0]
                path_side = view_out_pathes[1]

                out_folder_name = backbone_name+'_'+Path(path_back).parent.parent.name.replace(f'{backbone_name}_', '') + '--'+Path(path_side).parent.parent.name.replace(f'{backbone_name}_', '')
                if params['prefer_right']:
                    out_folder_name = out_folder_name + '_PreferedRight'
                clarification_str = Path(path_back).name
                out_folder = os.path.join(Path(exp_path).parent.parent, out_folder_name, Path(exp_path).name, clarification_str)
                test_combine_view_predictions(path_back, path_side, 1, out_folder, clarification_str, backbone_name, params)
        
        # == Cross-dataset testing with AID ==
        elif EXP == 'perform_cross_dataset_test_AID':
            exp_path = os.path.join(path.OUT_PATH, params['experiment_name'], params['model_prefix'], str(params['this_run_num']))
            params['model_prefix'] = params['model_prefix'] + '/' + str(params['this_run_num'])
            print(exp_path)
            params, best_params = load_hyperparams_from_study(params, backbone_name, exp_path)
            all_other_datasets = [x for x in const.SUPPORTED_DATASETS if x != params['dataset']]
            print(f"Testing cross-dataset performance of {backbone_name} trained on {all_other_datasets} + train of {params['dataset']}. Testing on test of: {params['dataset']}")
        
            clarification_str = f"train_AID_{'_'.join(all_other_datasets)}_&_{params['dataset']}_test_{params['dataset']}_{params['num_folds']}fold"
            model_checkpoint_clarification_str = f"train_AID_{'_'.join(all_other_datasets)}_&_{params['dataset']}_{params['num_folds']}fold"

            # TRAIN DATALOADER
            train_dataset_aug, eval_dataset_aug = get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, 1, augmented_datasets=True) # This will return all datasets but params['dataset'] in train_dataset_fn
            splits = []
            for fold in all_folds:
                train_dataset_fold, eval_dataset_fold = get_train_and_eval_datasets_depending_on_LODO(params, backbone_name, fold)
                combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset_aug, train_dataset_fold])
                combined_train_loader, eval_loader, class_weights = dataloader_factory(params, combined_train_dataset, eval_dataset_fold, eval_batch_size=1)
                splits.append((combined_train_loader, eval_loader, class_weights))
                
            test__hypertune(params, best_params, new_params, splits, backbone_name, clarification_str, model_checkpoint_clarification_str, exp_path, _DEVICE)
            if params['combine_views_preds']:
                view_out_pathes.append(os.path.join(exp_path, clarification_str))
            
            if params['combine_views_preds'] and len(view_out_pathes) == 2: # now we have results for both views
                print(" [INFO] 🔥🔥🔥 Now Combining the predictions of all views 🔥🔥🔥")
                path_back = view_out_pathes[0]
                path_side = view_out_pathes[1]
                out_folder_name = backbone_name+'_'+Path(path_back).parent.parent.name.replace(f'{backbone_name}_', '') + '--'+Path(path_side).parent.parent.name.replace(f'{backbone_name}_', '')
                if params['prefer_right']:
                    out_folder_name = out_folder_name + '_PreferedRight'
                out_folder = os.path.join(Path(exp_path).parent.parent, out_folder_name, Path(exp_path).name, clarification_str)
                test_combine_view_predictions(path_back, path_side, all_folds[-1], out_folder, out_folder_name, backbone_name, params)
                      
        # == Just generate dataset ==    
        elif param['just_gen_dataset']:
            dataset_factory(params, backbone_name, 1)
            
        

            