import torch


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUPPORTED_DATASETS = ['BMCLab', 'T-SDU-PD', 'PD-GaM', '3DGait', 'DNE', 'E-LC', 'KUL-DT-T', 'T-LTC', 'T-SDU'] 
DATASET_FOR_TUNING = {'classifier_only': 'BMCLab', 'end2end': 'PD-GaM'}
SUPPORTED_VIEWS = ['backright', 'side_right']
DATA_TYPES_SUPPORTING_RUNTIME_TRANSFORMS = ['h36m']
DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS = ['humanML3D']
BLOCK_ALL_PRECOPMUTED_TRANSFORMS = True
BACKBONES_WITH_MIRRORED_JOINTS = ['potr', 'motionagformer']
LABELS_INCLUDED_IN_F1_CALCULATION = [0,1,2]
NUM_OF_PATIENTS_PER_DATASET = {
    'BMCLab': 23,
    'T-SDU-PD': 14,
    'PD-GaM': 30,
    '3DGait': 43,
    'DNE': 97,
    'KUL-DT-T': 29,
    'E-LC': 59,
    'T-LTC': 14,
    'T-SDU': 53,
}
NUM_CLASSES_PER_DATASET = {
    'BMCLab': 3,
    'T-SDU-PD': 3,
    'PD-GaM': 4,
    '3DGait': 4
}

DATASET_FPS = {
    'BMCLab': 30,
    'T-SDU-PD': 30,
    'PD-GaM': 25,
    '3DGait': 30,
    'DNE': 30,
    'KUL-DT-T': 30,
    'E-LC': 30,
    'T-LTC': 30,
    'T-SDU': 30,
}

DATASET_ORIGINAL_FPS = {
    'BMCLab': 150,
    'T-SDU-PD': 30,
    'PD-GaM': 25,
    '3DGait': 30,
    'DNE': 60,
    'KUL-DT-T': 100,
    'E-LC': 120,
    'T-LTC': 30,
    'T-SDU': 30,
}

BACKBONE_POSETYPE_MAPPER = {
    'potr': '3D_processed',
    'motionbert': '2D_orthographic',
    'poseformerv2': '2D_prespective',
    'mixste': '2D_prespective', 
    'motionagformer': '2D_orthographic',
    'momask': 'original_hml3d',
    'motionclip': 'original_6d'
}


BACKBONE_CONFIGS = {
    'potr': './configs/potr/',
    'motionbert': './configs/motionbert/',
    'poseformerv2': './configs/poseformerv2',
    'mixste': './configs/mixste',
    'motionagformer': './configs/motionagformer',
    'momask': './configs/momask',
    'motionclip': './configs/motionclip'
}


MODEL_HYPERTUNING_CHOICES = {
    'classifier_only': {
        'initial_space': { # Initial space explored for every encoder on the BMCLab dataset
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.01, 0.001, 0.0001, 0.00001],
            'batch_size': [64, 128, 256],
            'epochs': [10, 20, 30, 50],
            'optimizer': ['AdamW'],# 'SGD'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss', 'FocalLoss'],#, 'CrossEntropyLoss'],
            'weight_decay': [0, 0.01, 0.001], #, 0.00001],
            'alpha': [1],
            'gamma': [1, 2],
        },

        'motionclip': {
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.001],
            'batch_size': [128],
            'epochs': [10, 20, 30, 50, 70],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss'],
            'weight_decay': [0.01],
        },

        'potr': { 
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.001],
            'batch_size': [128],
            'epochs': [10, 20, 30, 50, 70],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss'],
            'weight_decay': [0.01],
        },

        'motionbert': {
            'backright': {  
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.0001],
                'batch_size': [128],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['FocalLoss'],
                'weight_decay': [0],
                'alpha': [1],
                'gamma': [2],
            },
            'side_right': {
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.001],
                'batch_size': [256],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['WCELoss'],
                'weight_decay': [0],
            },
        },

        'motionbertlite': {
            'backright': {  
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.0001],
                'batch_size': [128],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['FocalLoss'],
                'weight_decay': [0],
                'alpha': [1],
                'gamma': [2],
            },
            'side_right': {
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.001],
                'batch_size': [256],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['WCELoss'],
                'weight_decay': [0],
            },
        },

        'mixste': {
            'backright': {  
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.01],
                'batch_size': [256],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['FocalLoss'],
                'weight_decay': [0],
                'alpha': [1],
                'gamma': [2],
            },
            'side_right': {
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.001],
                'batch_size': [256],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['WCELoss'],
                'weight_decay': [0.001],
            },
        },

        'motionagformer': {
            'backright': {  
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.001],
                'batch_size': [128],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['WCELoss'],
                'weight_decay': [0.01],
            },
            'side_right': {
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.001],
                'batch_size': [64],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['FocalLoss'],
                'weight_decay': [0.001],
                'alpha': [1],
                'gamma': [1],
            },
        },

        'momask': { # momask_hypertune.sh-node117-12918418.out
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.001],
            'batch_size': [128],
            'epochs': [10, 20, 30, 50, 70],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss'],
            'weight_decay': [0],
        },

        'poseformerv2': {
            'backright': {  
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.01],
                'batch_size': [64],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['FocalLoss'],
                'weight_decay': [0],
                'alpha': [1],
                'gamma': [2],
            },
            'side_right': {
                'classifier_hidden_dims': {'no_hidden_layers': []},
                'lr': [0.01],
                'batch_size': [64],
                'epochs': [10, 20, 30, 50, 70],
                'optimizer': ['AdamW'],
                'lambda_l1': [0],
                'dropout_rate': [0],
                'criterion': ['WCELoss'],
                'weight_decay': [0.001],
            },
        }
    },

    'end2end': {
        'initial_space': { # Search space for hypertuning the finetuned model on PD-GaM
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.03, 0.01, 0.001],
            'lr_backbone': [0.0001, 0.00001, 0.000003],
            'batch_size': [32, 64],
            'epochs': [10, 20, 30, 50],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss', 'FocalLoss'],
            'weight_decay': [0, 0.01, 0.001],
            'weight_decay_backbone': [0, 0.0003, 0.03],
            'alpha': [1],
            'gamma': [1, 2],
        },

        'poseformerv2': { # 'avg_best_epoch': 22 poseformerv2_hypertune_finetune.sh-node117-12942908.out
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.001],
            'lr_backbone': [0.00001],
            'batch_size': [128],
            'epochs': [10, 20, 30, 50, 70],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss'],
            'weight_decay': [0.01],
            'weight_decay_backbone': [0.00957528, 0.00957529]
        },

        'momask': { # 'avg_best_epoch': 16
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.001],
            'lr_backbone': [1e-05],
            'batch_size': [128],
            'epochs': [10, 20, 30, 50, 70],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['FocalLoss'],
            'weight_decay': [0.0],
            'weight_decay_backbone': [0.0003],
            'alpha': [1],
            'gamma': [2],
        },

        'motionclip': { #'avg_best_epoch': 10 motionclip_hypertune_finetune.sh-node117-12940575.out
            'classifier_hidden_dims': {
                'no_hidden_layers': []
            },
            'lr': [0.001],
            'lr_backbone': [1e-05],
            'batch_size': [64],
            'epochs': [10, 20, 30, 50, 70],
            'optimizer': ['AdamW'],
            'lambda_l1': [0],
            'dropout_rate': [0],
            'criterion': ['WCELoss'],
            'weight_decay': [0.001],
            'weight_decay_backbone': [0.0]
        }
    }
}




