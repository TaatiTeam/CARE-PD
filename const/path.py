import platform
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

OUT_PATH = './experiment_outs'

PREPROCESSED_DATA_ROOT_PATH = f'{PROJECT_ROOT}/assets/preprocessed_data'

# DATASET PRE PROCESSING ------------------------------------------------------------
NEUTRAL_SMPL_MODEL_PATH=f'{PROJECT_ROOT}/body_models/SMPL_NEUTRAL.pkl'
H36M_FROM_SMPL_REGRESSOR_PATH=f'{PROJECT_ROOT}/joint_regressors/J_regressor_h36m_correct.npy'

# ----------------------------------------------------------------------------------

# Vida's custom fold splits
BMCLab_6FOLD_SPLIT = f'{PROJECT_ROOT}/assets/datasets/folds/UPDRS_Datasets/BMCLab_6fold_participants.pkl'
BMCLab_23FOLD_SPLIT = f'{PROJECT_ROOT}/assets/datasets/folds/UPDRS_Datasets/BMCLab_23fold_participants.pkl'
T_SDU_PD_14FOLD_LOSO_SPLIT = f'{PROJECT_ROOT}/assets/datasets/folds/UPDRS_Datasets/T-SDU-PD_14fold_participants.pkl'
PD_GaM_AUTHORS_TRAIN_TEST_SPLIT = f'{PROJECT_ROOT}/assets/datasets/folds/UPDRS_Datasets/PD-GaM_authors_fixed.pkl'
THREEDGAIT_6FOLD_SPLIT = f'{PROJECT_ROOT}/assets/datasets/folds/UPDRS_Datasets/3DGait_6fold_participants.pkl'
THREEDGAIT_43FOLD_SPLIT = f'{PROJECT_ROOT}/assets/datasets/folds/UPDRS_Datasets/3DGait_43fold_participants.pkl'

POSE_AND_LABEL = {
    'BMCLab': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_back': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world2cam_back_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/BMCLab/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz',
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/BMCLab.pkl'
        },
        'humanML3D': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/BMCLab/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/BMCLab.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets6D_SMPL//BMCLab/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/BMCLab.pkl'
        }
    },

    'T-SDU-PD': { 
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world_30f_or_longer_slopeCorrected.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU-PD/h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/h36m/datasets/T-SDU-PD/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/h36m/datasets/T-SDU-PD/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    }
            },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-SDU-PD.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasetsHumanML3D//T-SDU-PD/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-SDU-PD.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/T-SDU-PD/6D_SMPL_30f_or_longer_slopeCorrected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-SDU-PD.pkl'
        }
    },

    'PD-GaM': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz'
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/PD-GaM/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz',
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/PD-GaM.pkl'
        },
        'humanML3D': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/PD-GaM/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/PD-GaM.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/PD-GaM/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/PD-GaM.pkl'
        }
    },

    '3DGait': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/3DGait/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/3DGait.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/3DGait/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/3DGait.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/3DGait/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/3DGait.pkl'
        }
    },
    
    'E-LC': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/E-LC/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/E-LC.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/E-LC/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/E-LC.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/E-LC/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/E-LC.pkl'
        }
    },
    
    'DNE': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/DNE/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/DNE.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/DNE/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/DNE.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/DNE/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/DNE.pkl'
        }
    },
    
    'KUL-DT-T': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/KUL-DT-T/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/KUL-DT-T.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/KUL-DT-T/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/KUL-DT-T.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/KUL-DT-T/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/KUL-DT-T.pkl'
        }
    },
    
    'T-SDU': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/T-SDU/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-SDU.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/T-SDU/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-SDU.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/T-SDU/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-SDU.pkl'
        }
    },
    
    'T-LDU': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/assets/datasets/h36m/T-LDU/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-LDU.pkl'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/HumanML3D/T-LDU/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-LDU.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/assets/datasets/6D_SMPL/T-LDU/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/assets/datasets/T-LDU.pkl'
        }
    },
}

PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = f'{PROJECT_ROOT}/assets/Pretrained_checkpoints'
