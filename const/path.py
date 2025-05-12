import platform
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

OUT_PATH = './experiment_outs'

PREPROCESSED_DATA_ROOT_PATH = f'{PROJECT_ROOT}/data'

# DATASET PRE PROCESSING ------------------------------------------------------------
NEUTRAL_SMPL_MODEL_PATH=f'{PROJECT_ROOT}/body_models/SMPL_NEUTRAL.pkl'
H36M_FROM_SMPL_REGRESSOR_PATH=f'{PROJECT_ROOT}/joint_regressors/J_regressor_h36m_correct.npy'

# ----------------------------------------------------------------------------------

# Vida's custom fold splits
VIDAS_CUSTOM_PD_6FOLD_SPLIT = f'{PROJECT_ROOT}/data/BMCLABS_6fold_participants.pkl'
VIDAS_CUSTOM_PD_23FOLD_SPLIT = f'{PROJECT_ROOT}/data/BMCLABS_23fold_participants.pkl'
TRI_PD_14FOLD_LOSO_SPLIT = f'{PROJECT_ROOT}/data/TRI_PD_14fold_participants.pkl'
PDGAM_AUTHORS_TRAIN_TEST_SPLIT = f'{PROJECT_ROOT}/data/PD4T_authors_custom_1fold_participants.pkl'
THREEDGAIT_6FOLD_SPLIT = f'{PROJECT_ROOT}/data/3DGAIT_6fold_participants.pkl'
THREEDGAIT_43FOLD_SPLIT = f'{PROJECT_ROOT}/data/3DGAIT_43fold_participants.pkl'

POSE_AND_LABEL = {
    'BMCLABS': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'side_right': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_back': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world2cam_back_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/datasets/BMClab/h36m/30fps/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz',
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/BMClab/PDGinfo.xlsx'
        },
        'humanML3D': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/BMClab/HumanML3D/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/BMClab/PDGinfo.xlsx'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/BMClab/6D_SMPL/30fps/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/BMClab/PDGinfo.xlsx'
        }
    },

    'TRI_PD': { 
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'front': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'side_right': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'side_left': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'backright': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world_30f_or_longer_slopeCorrected.npz',
                    'preprocessed': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'camera_backright': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/h36m/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer_slopeCorrected.npz',
                    }
            },
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/TRI_PD/ALL_PD_SCORES_AND_CLINICAL_DATA.xlsx'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/HumanML3D/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/TRI_PD/ALL_PD_SCORES_AND_CLINICAL_DATA.xlsx'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/TRI_PD/fromWHAM/6D_SMPL/6D_SMPL_30f_or_longer_slopeCorrected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/TRI_PD/ALL_PD_SCORES_AND_CLINICAL_DATA.xlsx'
        }
    },

    'PDGAM': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz'
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/h36m/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz',
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/PDGAM/PDGAM_labels.csv'
        },
        'humanML3D': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/HumanML3D/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/PDGAM/PDGAM_labels.csv'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/PDGAM/fromWHAM/6D_SMPL/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/PDGAM/PDGAM_labels.csv'
        }
    },

    'KIEL': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'side_right': f'{PROJECT_ROOT}/datasets/Kiel/h36m/30fps/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/datasets/Kiel/h36m/30fps/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/datasets/Kiel/h36m/30fps/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/datasets/Kiel/h36m/30fps/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/datasets/Kiel/h36m/30fps/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/datasets/Kiel/h36m/30fps/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz',
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/Kiel/kiel_labels.pkl'
        },
        'humanML3D': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/Kiel/HumanML3D/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/Kiel/kiel_labels.pkl'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/Kiel/6D_SMPL/30fps/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/Kiel/kiel_labels.pkl'
        }
    },

    '3DGAIT': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/h36m/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/3DGait/label.xlsx'
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/HumanML3D/HumanML3D_collected.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/3DGait/label.xlsx'
        },
        '6DSMPL': {
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/6D_SMPL/6D_SMPL_30f_or_longer.npz',
            'PATH_LABELS': f'{PROJECT_ROOT}/datasets/3DGait/label.xlsx'
        }
    },
    
    'Emory': {
        'h36m': {
            'PATH_POSES': {
                '2D': { 
                    'back': f'{PROJECT_ROOT}/datasets/Emory/h36m/h36m_3d_world2cam2img_back_floorXZZplus_30f_or_longer.npz',
                    'front': f'{PROJECT_ROOT}/datasets/Emory/h36m/h36m_3d_world2cam2img_front_floorXZZplus_30f_or_longer.npz',
                    'side_right': f'{PROJECT_ROOT}/datasets/Emory/h36m/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz',
                    'side_left': f'{PROJECT_ROOT}/datasets/Emory//h36m/h36m_3d_world2cam2img_sideleft_floorXZZplus_30f_or_longer.npz',
                    'backright': f'{PROJECT_ROOT}/datasets/Emory//h36m/h36m_3d_world2cam2img_backright_floorXZZplus_30f_or_longer.npz',
                    },
                '3D': { 
                    'original': f'{PROJECT_ROOT}/datasets/3DGait//h36m/h36m_3d_world_30f_or_longer.npz',
                    'preprocessed': f'{PROJECT_ROOT}/datasets/3DGait//h36m/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
                    'camera_backright': f'{PROJECT_ROOT}/datasets/3DGait//h36m/h36m_3d_world2cam_backright_floorXZZplus_30f_or_longer.npz',
                    'camera_side_right': f'{PROJECT_ROOT}/datasets/3DGait//h36m/h36m_3d_world2cam_sideright_floorXZZplus_30f_or_longer.npz'
                    }
                },
            'PATH_LABELS': f''
        },
        'humanML3D': { 
            'PATH_POSES': f'{PROJECT_ROOT}/datasets/3DGait/fromWHAM/HumanML3D/HumanML3D_collected.npz',
            'PATH_LABELS': f''
        }
    }
}

PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = f'{PROJECT_ROOT}/Pretrained_checkpoints'

THESIS_RF = '/cluster/projects/taati/ivan/data/motion_encoders/Thesis_RF'
FEATURES_PATH = f"{THESIS_RF}/features/features_PD_TRI_PD_PDGAM_KIEL_2024_07_31.pkl"
FOLDS_PATH = f"{THESIS_RF}/folds/COLLECTED_FOLDS_PD_TRI_PD_PDGAM_KIEL_2024_07_31.pkl"
HYPERPARAM_OUTPUT_RES_DIR = f"{THESIS_RF}/hypertuning_results"