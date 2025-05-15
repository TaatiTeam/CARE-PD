import json

import torch.utils.data as data
from abc import ABC
from torchvision import transforms
import torch
import os
from pathlib import Path

from collections import Counter
import numpy as np
import pickle
import random
import copy
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from const import path
from const.const import DATA_TYPES_SUPPORTING_RUNTIME_TRANSFORMS, DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS, BACKBONES_WITH_MIRRORED_JOINTS, BLOCK_ALL_PRECOPMUTED_TRANSFORMS
from data.bmclab_datareader import BMCLABSReader
from data.tri_pd_datareader import TRI_PD_Reader
from data.pdgam_datareader import PDGAMReader
from data.kiel_datareader import KIELReader
from data.threedgait_datareader import GAIT3DReader
from data.augmentations import MirrorReflection, RandomRotation, RandomNoise, axis_mask
from learning.utils import compute_class_weights

_TOTAL_SCORES = 3
_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
_SMPL_6D_ELEMENTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_MAJOR_JOINTS_MIRRORED = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
_HUMAN_ML3D_POSE_ELEMENTS = list(range(263))

_ROOT = 0
_MIN_STD = 1e-4

METADATA_MAP = {'gender': 0, 'age': 1, 'height': 2, 'weight': 3, 'bmi': 4}


class DataPreprocessor(ABC):
    def __init__(self, raw_data, params=None):
        self.pose_dict = raw_data.pose_dict
        self.labels_dict = raw_data.labels_dict
        self.metadata_dict = raw_data.metadata_dict
        self.video_names = raw_data.video_names
        self.participant_ID = raw_data.participant_ID
        self.params = params
        self.data_dir = self.params['data_path']

    def __len__(self):
        return len(self.labels_dict)

    def center_poses(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            self.pose_dict[key] = joints3d - joints3d[:, _ROOT:_ROOT + 1, :]

    def normalize_poses(self):
        if self.params['data_norm'] == 'minmax':
            """
            computing the per-axis min and max — for x, y, z — over all joints and all frames in a video.
            standard min-max normalization to [0, 1]
            :param pose_dict: dictionary of poses
            :return: dictionary of normalized poses
            """
            normalized_pose_dict = {}
            for video_name in self.pose_dict:
                poses = self.pose_dict[video_name].copy()

                mins = np.min(np.min(poses, axis=0), axis=0)
                maxes = np.max(np.max(poses, axis=0), axis=0)

                poses = (poses - mins) / (maxes - mins)

                normalized_pose_dict[video_name] = poses
            self.pose_dict = normalized_pose_dict

        elif self.params['data_norm'] == 'rescaling':
            """
            computing the per-axis min and max — for x, y, z — over all joints and all frames in a video.
            min-max normalization to [-1, 1]
            """
            normalized_pose_dict = {}
            for video_name in self.pose_dict:
                poses = self.pose_dict[video_name].copy()

                mins = np.min(poses, axis=(0, 1))
                maxes = np.max(poses, axis=(0, 1))

                poses = (2 * (poses - mins) / (maxes - mins)) - 1

                normalized_pose_dict[video_name] = poses
            self.pose_dict = normalized_pose_dict

        elif self.params['data_norm'] == 'zscore':
            norm_stats = self.compute_norm_stats()
            pose_dict_norm = self.pose_dict.copy()
            for k in self.pose_dict.keys():
                tmp_data = self.pose_dict[k].copy()
                tmp_data = tmp_data - norm_stats['mean']
                tmp_data = np.divide(tmp_data, norm_stats['std'])
                pose_dict_norm[k] = tmp_data
            self.pose_dict = pose_dict_norm
    
    def crop_scale(self, motion, scale_range=[1, 1]):
        '''
            Motion: [(M), T, 17, 3].
            Normalize to [-1, 1]
        '''
        result = copy.deepcopy(motion)
        valid_coords = motion[motion[..., 2]!=0][:,:2] # array of all valid (x, y) joint positions across the full motion input.
        if len(valid_coords) < 4:
            return np.zeros(motion.shape)
        xmin = min(valid_coords[:,0])
        xmax = max(valid_coords[:,0])
        ymin = min(valid_coords[:,1])
        ymax = max(valid_coords[:,1])
        ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
        scale = max(xmax-xmin, ymax-ymin) * ratio
        if scale==0:
            return np.zeros(motion.shape)
        xs = (xmin+xmax-scale) / 2
        ys = (ymin+ymax-scale) / 2
        result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
        result[...,:2] = (result[..., :2] - 0.5) * 2
        result = np.clip(result, -1, 1)
        return result

    def compute_norm_stats(self):
        all_data = []
        for k in self.pose_dict.keys():
            all_data.append(self.pose_dict[k])
        all_data = np.vstack(all_data)
        print('[INFO] ({}) Computing normalization stats!')
        norm_stats = {}
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        std[np.where(std < _MIN_STD)] = 1

        norm_stats['mean'] = mean  # .ravel()
        norm_stats['std'] = std  # .ravel()
        return norm_stats
        
    def generate_cv_folds(self, clip_dict, pad_masks_dict, save_dir, labels_dict):
        """
        Generate folds test & eval folds for CV.
        :param num_folds: number of dataset splits to perform
        :param clip_dict: dictionary of clips for each video
        :param save_dir: save directory for folds
        """

        num_folds    = self.params['num_folds']
        dataset_name = self.params['dataset']

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        clip_dict_vids = set(clip_dict.keys())
        clip_dict_pats = set([self.participant_ID[i] for i,v in enumerate(self.video_names) if v in clip_dict_vids])
        #if self.params['data_type'] != 'humanML3D': # If it is humanML3D and we are doing the video clipping on the fly so we will tolerate different sets
        assert set(self.video_names) == clip_dict_vids, f'set(self.video_names) and set(clip_dict.keys()) are not the same!!'

        self.participant_ID_info = {pid: {
            'video_names': [self.video_names[i] for i, p in enumerate(self.participant_ID) if (p == pid and self.video_names[i] in clip_dict_vids)],
            'labels': [],
            'most_common_label': None
        } for pid in clip_dict_pats}
        for pid, info in self.participant_ID_info.items():
            info['labels'] = [self.labels_dict[v] for v in info['video_names']]
            info['most_common_label'] = Counter(info['labels']).most_common(1)[0][0] # Returns a list of tuples [(el, freq), ...]

        existing_folds_name = f"{self.params['dataset']}_{num_folds}fold_participants.pkl"
        existing_folds_path = os.path.join(Path(save_dir).parent.parent.parent.parent, 'folds', existing_folds_name)

        folds_already_exist = os.path.exists(existing_folds_path)

        if not folds_already_exist:
            if self.params['dataset'] == 'BMCLABS' and num_folds == 6:
                cv_folds = pickle.load(open(path.VIDAS_CUSTOM_PD_6FOLD_SPLIT, "rb"))
                print(f'Using vidas custom 6fold split for BMCLABS data {path.VIDAS_CUSTOM_PD_6FOLD_SPLIT}')
            elif self.params['dataset'] == 'BMCLABS' and num_folds == 23:
                cv_folds = pickle.load(open(path.VIDAS_CUSTOM_PD_23FOLD_SPLIT, "rb"))
                print(f'Using vidas custom 23fold (LOSO) split for BMCLABS data {path.VIDAS_CUSTOM_PD_23FOLD_SPLIT}')
            elif self.params['dataset'] == 'TRI_PD' and num_folds == 14:
                cv_folds = pickle.load(open(path.TRI_PD_14FOLD_LOSO_SPLIT, "rb"))
                print(f'Using 14fold (LOSO) split for TRI_PD data {path.TRI_PD_14FOLD_LOSO_SPLIT}')
            elif self.params['dataset'] == 'PDGAM' and num_folds == 1:
                cv_folds = pickle.load(open(path.PDGAM_AUTHORS_TRAIN_TEST_SPLIT, "rb"))
                print(f'Using 1fold authors custom split for PDGAMM data {path.PDGAM_AUTHORS_TRAIN_TEST_SPLIT}')
            elif self.params['dataset'] == '3DGAIT' and num_folds == 6:
                cv_folds = pickle.load(open(path.THREEDGAIT_6FOLD_SPLIT, "rb"))
                print(f'Using vidas custom 6fold split for BMCLABS data {path.THREEDGAIT_6FOLD_SPLIT}')
            elif self.params['dataset'] == '3DGAIT' and num_folds == 43:
                cv_folds = pickle.load(open(path.THREEDGAIT_43FOLD_SPLIT, "rb"))
                print(f'Using vidas custom 6fold split for BMCLABS data {path.THREEDGAIT_43FOLD_SPLIT}')
            else:
                cv_folds = dict()
                print(f'Selected {dataset_name} folds do not exist.')
                X = np.array(sorted([pid for pid in self.participant_ID_info]))
                y = np.array([self.participant_ID_info[pid]['most_common_label'] for pid in X])
                print(f'Generating folds according to patients: {X}, and their severity scores: {y}')
                fold_gen = LeaveOneOut().split(X) if num_folds == len(X) else StratifiedKFold(n_splits=num_folds).split(X, y)
                for fold_num, (train_index, eval_index) in enumerate(fold_gen, start=1):
                    cv_folds[fold_num] = {'train': list(X[train_index]), 'eval': list(X[eval_index])}
                pickle.dump(cv_folds, open(existing_folds_path, "wb"))
        else:
            cv_folds = pickle.load(open(existing_folds_path, "rb"))
        
        for fold_num, fold_pids in cv_folds.items():
            train_list = [v for pid in fold_pids['train'] for v in self.participant_ID_info[pid]['video_names']]
            eval_list = [v for pid in fold_pids['eval'] for v in self.participant_ID_info[pid]['video_names']]
            print("Fold: ", fold_num)
            train, evaluate = self.generate_pose_label_videoname(clip_dict, pad_masks_dict, train_list, eval_list)
            pickle.dump(train_list, open(os.path.join(save_dir, f"{dataset_name}_train_list_{fold_num}.pkl"), "wb"))
            pickle.dump(eval_list, open(os.path.join(save_dir, f"{dataset_name}_eval_list_{fold_num}.pkl"), "wb"))
            pickle.dump(train, open(os.path.join(save_dir, f"{dataset_name}_train_{fold_num}.pkl"), "wb"))
            pickle.dump(evaluate, open(os.path.join(save_dir, f"{dataset_name}_eval_{fold_num}.pkl"), "wb"))
        pickle.dump(self.labels_dict, open(os.path.join(save_dir, f"{dataset_name}_labels.pkl"), "wb"))


    def get_data_split(self, split_list, clip_dict, pad_masks_dict):
        split = {'pose': [], 'label': [], 'video_name': [], 'metadata': [], 'pad_mask': []}
        for video_name in split_list:
            clips = clip_dict[video_name]
            for i, clip in enumerate(clips):
                split['label'].append(self.labels_dict[video_name])
                split['pose'].append(clip)
                split['video_name'].append(video_name)
                split['metadata'].append(self.metadata_dict[video_name])
                split['pad_mask'].append(pad_masks_dict[video_name][i])
        return split

    def generate_pose_label_videoname(self, clip_dict, pad_masks_dict, train_list, eval_list):
        train = self.get_data_split(train_list, clip_dict, pad_masks_dict)
        evaluate = self.get_data_split(eval_list, clip_dict, pad_masks_dict)

        #print how many samples are in each split
        print(f"Train Length: {len(train['video_name'])}")
        print(f"Evaluation Length: {len(evaluate['video_name'])}")
        return train, evaluate


class POTRPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()
        if self.params['data_norm'] in ['minmax', 'zscore']:
            self.normalize_poses()
        clip_dict, pad_masks_dict = self.partition_videos()
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)

    def partition_videos(self):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], self.params['source_seq_len'])
            clip_dict[video_name] = clips
            pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict
    
    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            clips.append(np.pad(video_sequence, pad_width=((0,clip_length-video_length), (0,0), (0,0)))) # Pad to be of length clip_length
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips, pad_masks


class MotionBERTPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        # if self.params['data_centered']:
        #     self.center_poses()
        # else:
        #     self.place_depth_of_first_frame_to_zero()
        # self.normalize_screen_coordinates(params['image_resolution'][0], params['image_resolution'][1])
        # self.normalize_poses()
        
        clip_dict, pad_masks_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        for k, vi in clip_dict.items():
            for ii, clip in enumerate(vi):
                clip_dict[k][ii] = self.crop_scale(clip)
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)

    def normalize_screen_coordinates(self, w, h): 
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints2d = joints3d[..., :2]  # (n_frames, n_joints, 2)
            assert joints2d.shape[-1] == 2
            # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
            self.pose_dict[key] = (joints2d/w)*2 - [1, h/w]
            
    def place_depth_of_first_frame_to_zero(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints3d[..., 2] = joints3d[..., 2] - joints3d[0:1, _ROOT:_ROOT + 1, 2]

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
            pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict

    def get_clips(self, video_sequence, clip_length):
        data_stride = clip_length
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            clips.append(np.pad(video_sequence, pad_width=((0,clip_length-video_length), (0,0), (0,0)))) # Pad to be of length clip_length
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
        return clips, pad_masks

class MoMaskPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        clip_dict, pad_masks_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], clip_length)
            if clips is not None: 
                clip_dict[video_name] = clips
                pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict

    def get_clips(self, video_sequence, clip_length):
        data_stride = clip_length
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[1] # Augmentations for momask have to be precomputed so video_sequence.shape should be A x F x 263 A being the number of different augumentations and F being the number of frames
        if video_length < clip_length:
            #raise ValueError(f'Length of video {video_length} is shorter than the defined clip length {clip_length}.')
            #return None
            clips.append(np.pad(video_sequence, pad_width=((0,0), (0,clip_length-video_length), (0,0)))) # Pad to be of length clip_length
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
            # clips.append(video_sequence) # No pad needed Pad
            # # pad_len = clip_length - video_length
            # pad_mask = np.ones(video_length) # shape: (video_length,)
            # pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[:, start_frame: start_frame + clip_length, :]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[:, start_frame:start_frame + clip_length, :])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
        return clips, pad_masks
    

class MotionAGFormerPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        # if self.params['data_centered']:
        #     self.center_poses()
        # else:
        #     self.place_depth_of_first_frame_to_zero()
        # self.normalize_screen_coordinates(params['image_resolution'][0], params['image_resolution'][1])
        clip_dict, pad_masks_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        for k, vi in clip_dict.items():
            for ii, clip in enumerate(vi):
                clip_dict[k][ii] = self.crop_scale(clip)
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)

    def normalize_screen_coordinates(self, w, h): 
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints2d = joints3d[..., :2]  # (n_frames, n_joints, 2)
            assert joints2d.shape[-1] == 2
            # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
            self.pose_dict[key] = (joints2d/w)*2 - [1, h/w]
    
    def place_depth_of_first_frame_to_zero(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints3d[..., 2] = joints3d[..., 2] - joints3d[0:1, _ROOT:_ROOT + 1, 2]

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
            pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict

    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            clips.append(np.pad(video_sequence, pad_width=((0,clip_length-video_length), (0,0), (0,0)))) # Pad to be of length clip_length
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips, pad_masks


class PoseformerV2Preprocessor(DataPreprocessor):
            
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        self.normalize_screen_coordinates(params['image_resolution'][0], params['image_resolution'][1])
        clip_dict, pad_masks_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)
        
    def normalize_screen_coordinates(self, w, h): 
        for key in self.pose_dict.keys():
            joints2d = self.pose_dict[key]  # (n_frames, n_joints, 2)
            assert joints2d.shape[-1] == 2
            # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
            self.pose_dict[key] = (joints2d/w)*2 - [1, h/w]

    def remove_last_dim_of_pose(self):
        for video_name in self.pose_dict:
            self.pose_dict[video_name] = self.pose_dict[video_name][..., :2]  # Ignoring confidence score

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
            pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict

    def get_clips(self, video_sequence, clip_length):
        data_stride = clip_length
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            pad_total = clip_length - video_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            # clips.append(np.pad(video_sequence, pad_width=((0,clip_length-video_length), (0,0), (0,0)))) # Pad to be of length clip_length
            clips.append(np.pad(video_sequence, ((pad_left, pad_right), (0, 0), (0, 0)), mode='edge')) # symmetric padding around a clip to center it and use 'edge' padding
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([
                    np.zeros(pad_left),          # padding = 0
                    np.ones(video_length),       # real frames = 1
                    np.zeros(pad_right)          # padding = 0
                ])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
        return clips, pad_masks

class MixSTEPreprocessor(DataPreprocessor):
    
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        self.normalize_screen_coordinates(params['image_resolution'][0], params['image_resolution'][1])
        clip_dict, pad_masks_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)
        
    def normalize_screen_coordinates(self, w, h): 
        for key in self.pose_dict.keys():
            joints2d = self.pose_dict[key]  # (n_frames, n_joints, 2)
            assert joints2d.shape[-1] == 2
            # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
            self.pose_dict[key] = (joints2d/w)*2 - [1, h/w]

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
            pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict

    def get_clips(self, video_sequence, clip_length):
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[0]
        if video_length <= clip_length:
            clips.append(np.pad(video_sequence, pad_width=((0,clip_length-video_length), (0,0), (0,0)))) # Pad to be of length clip_length
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                data_stride = clip_length
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
        return clips, pad_masks

class MotionCLIPPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        clip_dict, pad_masks_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        self.generate_cv_folds(clip_dict, pad_masks_dict, save_dir, raw_data.labels_dict)

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        pad_masks_dict = {}
        for video_name in self.pose_dict.keys():
            clips, pad_masks = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
            pad_masks_dict[video_name] = pad_masks
        return clip_dict, pad_masks_dict

    def get_clips(self, video_sequence, clip_length):
        clips = []
        pad_masks = []
        video_length = video_sequence.shape[0]
        if video_length <= clip_length:
            clips.append(np.pad(video_sequence, pad_width=((0,clip_length-video_length), (0,0), (0,0)))) # Pad to be of length clip_length
            pad_len = clip_length - video_length
            pad_mask = np.concatenate([np.ones(video_length), np.zeros(pad_len)])  # shape: (clip_length,)
            pad_masks.append(pad_mask)
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
                pad_mask = np.ones(clip_length)  # fully real
                pad_masks.append(pad_mask)
            else:
                data_stride = clip_length
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                    pad_mask = np.ones(clip_length)  # fully real
                    pad_masks.append(pad_mask)
        return clips, pad_masks

class ProcessedDataset(data.Dataset):
    def __init__(self, data_dir, params=None, mode='train', fold=1, transform=None, precomputed_transforms=False, utilize_precomputed_transforms=False):
        super(ProcessedDataset, self).__init__()
        self._params = params
        self._mode = mode
        self.data_dir = data_dir

        if self._mode not in ['train', 'eval', 'all_folds_merged']:
            raise NotImplementedError(f"Dataset mode='{self._mode}' is not implemented.")

        self.fold = fold
        self.transform = transform
        self.precomputed_transforms = precomputed_transforms
        self.utilize_precomputed_transforms = utilize_precomputed_transforms

        if self._params['data_type'] == 'humanML3D':
            norm_data_path = Path(self._params['humanML3D_normalization_data_path']) / f"{'LODO_' if self._params['LODO'] else ''}{self._params['dataset']}"
            self.mean = np.load(norm_data_path / "Mean.npy")
            self.std = np.load(norm_data_path / "Std_FEAT_BIAS_included.npy")

        self.poses, self.labels, self.video_names, self.metadata, self.pad_masks = self.load_data()
        self.video_name_to_index = {name: index for index, name in enumerate(self.video_names)} # Last index that a certain video appears at

    def load_data(self):
        dataset_name = self._params['dataset']

        if self._mode in ['train', 'eval']:
            data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_{self._mode}_{self.fold}.pkl"), "rb"))
        elif self._mode == 'all_folds_merged':
            train_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_train_{self.fold}.pkl"), "rb"))
            eval_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_eval_{self.fold}.pkl"), "rb"))
            data = {
                'pose': [*train_data['pose'], *eval_data['pose'],],
                'label': [*train_data['label'], *eval_data['label']],
                'video_name': [*train_data['video_name'], *eval_data['video_name']],
                'metadata': [*train_data['metadata'], *eval_data['metadata']],
                'pad_mask': [*train_data['pad_mask'], *eval_data['pad_mask']]
            }

        poses, labels, video_names, metadatas, pad_masks = self.data_generator(data)
        return poses, labels, video_names, metadatas, pad_masks

    @staticmethod
    def data_generator(data):
        poses = []
        labels = []
        video_names = []
        metadatas = []
        pad_masks = []

        for i in range(len(data['pose'])):
            poses.append(np.copy(data['pose'][i]))
            labels.append(data['label'][i])
            video_names.append(data['video_name'][i])
            metadatas.append(data['metadata'][i])
            pad_masks.append(data['pad_mask'][i])
            
        # can't stack poses because not all have equal frames
        labels = np.stack(labels)
        video_names = np.stack(video_names)
        metadatas = np.stack(metadatas)
        pad_masks = np.stack(pad_masks)
        

        return poses, labels, video_names, metadatas, pad_masks
    
    def _get_joint_orders(self):
        if self._params['data_type'] == 'h36m':
            joints = _MAJOR_JOINTS
            if self._params['backbone'] in BACKBONES_WITH_MIRRORED_JOINTS:
                joints = _MAJOR_JOINTS_MIRRORED
        elif self._params['data_type'] == 'humanML3D':
            joints = _HUMAN_ML3D_POSE_ELEMENTS
        elif  self._params['data_type'] == '6DSMPL':
            joints = _SMPL_6D_ELEMENTS
        return joints

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item for the training mode."""
        x = self.poses[idx]
        if self.precomputed_transforms:
            pose_version_index = 0 # It is assumed that the first version of the pose has no augmentations applied
            if self.utilize_precomputed_transforms:
                num_transformations = len(x) # If we have 1 original file and 40 augmentations this will be 41
                pose_version_index = np.random.randint(0,num_transformations)
            x = x[pose_version_index]
        label = self.labels[idx]
        pad_mask = self.pad_masks[idx]
        video_idx = self.video_name_to_index[self.video_names[idx]] 

        joints = self._get_joint_orders()
        if self._params['data_type'] in ['h36m', '6DSMPL']:
            assert x.shape[1] == len(joints)
            x = x[:, joints, :]
        elif self._params['data_type'] in ['humanML3D']:
            assert x.shape[1] == len(joints)
            x = x[:, joints]
            x = (x - self.mean) / self.std


        if self._params['in_data_dim'] == 2:
            if self._params['simulate_confidence_score']:
                if x.shape[-1] == 2:
                    confidence = np.ones_like(x[..., :1])  # shape matches (F, J, 1)
                    x = np.concatenate([x, confidence], axis=-1)
                else:
                    x[..., 2] = 1  # Consider 3rd dimension as confidence score and set to be 1.
                ''' set the confidence to 0 for any fully zero-padded frames '''
                zero_frames = np.all(x[..., :2] == 0, axis=(1, 2))  # (F)
                x[zero_frames, :, 2] = 0
            else:
                x = x[..., :2]  # Make sure it's two-dimensional, we are assuming that z is along the direction of movement
        elif self._params['in_data_dim'] == 3:
            x = x[..., :3] # Make sure it's 3-dimensional
        elif self._params['in_data_dim'] == 263:
            x = x[..., :263] # Make sure it's 263-dimensional
        elif self._params['in_data_dim'] == 6:
            x = x[..., :6] # Make sure it's 263-dimensional
                
        if self._params['merge_last_dim']:
            N = np.shape(x)[0]
            x = x.reshape(N, -1)   # N x 17 x 3 -> N x 51

        x = np.array(x, dtype=np.float32)

        if x.shape[0] > self._params['source_seq_len']:
            # If we're reading a preprocessed pickle file that has more frames
            # than the expected frame length, we throw away the last few ones.
            x = x[:self._params['source_seq_len']]
        elif x.shape[0] < self._params['source_seq_len']:
            raise ValueError("Number of frames in tensor x is shorter than expected one.")
        
        if len(self._params['metadata']) > 0:
            metadata_idx = [METADATA_MAP[element] for element in self._params['metadata']]
            md = self.metadata[idx][0][metadata_idx].astype(np.float32)
        else:
            md = []

        sample = {
            'encoder_inputs': x,
            'label': label,
            'video_idx': video_idx,
            'metadata': md,
            'pad_mask': pad_mask,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def collate_fn(batch):
    """Collate function for data loaders."""
    e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
    labels = torch.from_numpy(np.stack([e['label'] for e in batch]))
    video_idxs = torch.from_numpy(np.stack([e['video_idx'] for e in batch]))
    metadata = torch.from_numpy(np.stack([e['metadata'] for e in batch]))
    pad_mask = torch.from_numpy(np.stack([e['pad_mask'] for e in batch]))

    return e_inp, labels, video_idxs, metadata, pad_mask

def assert_backbone_is_supported(backbone_data_location_mapper, backbone):
    if backbone not in backbone_data_location_mapper:
        raise NotImplementedError(f"Backbone '{backbone}' is not supported.")

def dataset_factory(params, backbone, fold):
    """Defines the datasets that will be used for training and validation."""
    root_dir = f'{path.PREPROCESSED_DATA_ROOT_PATH}/{backbone}_processing'

    backbone_data_location_mapper = {
        'potr': os.path.join(root_dir, params['experiment_name'],
                                   f"{params['dataset']}_center_{params['data_centered']}_{params['data_norm']}", f"{params['num_folds']}fold"),
        'motionbert': os.path.join(root_dir, params['experiment_name'],
                                   f"{params['dataset']}_center_{params['data_centered']}_{'-'.join(params['views'])}", f"{params['num_folds']}fold"),
        'motionagformer': os.path.join(root_dir, params['experiment_name'],
                                   f"{params['dataset']}_center_{params['data_centered']}_{'-'.join(params['views'])}", f"{params['num_folds']}fold"),
        'poseformerv2': os.path.join(root_dir, params['experiment_name'],
                                     f"{params['dataset']}_center_{params['data_centered']}_{'-'.join(params['views'])}", f"{params['num_folds']}fold"),
        'mixste': os.path.join(root_dir, params['experiment_name'],
                                     f"{params['dataset']}_center_{params['data_centered']}_{'-'.join(params['views'])}", f"{params['num_folds']}fold"),
        'momask': os.path.join(root_dir, params['experiment_name'],
                               f"{params['dataset']}_augment_False", f"{params['num_folds']}fold"),
        'motionclip': os.path.join(root_dir, params['experiment_name'],
                               f"{params['dataset']}", f"{params['num_folds']}fold"),
    }

    # backbone_data_location_mapper = {
    #     'motionagformer': "./checkpoint/motionagformer-xs-h36m.pth.tr"
    # }

    datareader_mapper = {
        'BMCLABS': {
            'h36m': BMCLABSReader,
            'humanML3D': BMCLABSReader,
            '6DSMPL': BMCLABSReader
        },
        'TRI_PD': {
            'h36m': TRI_PD_Reader,
            'humanML3D': TRI_PD_Reader,
            '6DSMPL': TRI_PD_Reader
        },
        'PDGAM': {
            'h36m': PDGAMReader,
            'humanML3D': PDGAMReader,
            '6DSMPL': PDGAMReader
        },
        'KIEL': {
            'h36m': KIELReader,
            'humanML3D': KIELReader,
            '6DSMPL': KIELReader
        },
        '3DGAIT': {
            'h36m': GAIT3DReader,
            'humanML3D': GAIT3DReader,
            '6DSMPL': GAIT3DReader
        }
    }

    backbone_preprocessor_mapper = {
        'potr': POTRPreprocessor,
        'motionbert': MotionBERTPreprocessor,
        'poseformerv2': PoseformerV2Preprocessor,
        'mixste': MixSTEPreprocessor,
        'motionagformer': MotionAGFormerPreprocessor,
        'momask': MoMaskPreprocessor,
        'motionclip': MotionCLIPPreprocessor
    }

    assert_backbone_is_supported(backbone_data_location_mapper, backbone)
    data_dir = backbone_data_location_mapper[backbone]

    if not os.path.exists(data_dir):
        if not params['dataset'] in datareader_mapper or \
           not params['data_type'] in datareader_mapper[params['dataset']]:
            raise NotImplementedError(f"dataset '{params['dataset']}' of type: {params['data_type']} is not supported.")
        DataReader = datareader_mapper[params['dataset']][params['data_type']]
        raw_data = DataReader(params['data_path'], params['labels_path'], params) 
        Preprocessor = backbone_preprocessor_mapper[backbone]
        Preprocessor(data_dir, raw_data, params)

    merge_dataset = params['cross_dataset_test'] and not params['hypertune'] # If we testing a model across various datasets one of them is going to be the train and the other will be the test. In this case we want to load th eentire datasets into a single dataloader.

    runtime_train_transform = transforms.Compose([
        PreserveKeysTransform(transforms.RandomApply([MirrorReflection(data_dim=params['in_data_dim'])], p=params['mirror_prob'])),
        PreserveKeysTransform(transforms.RandomApply([RandomRotation(*params['rotation_range'], data_dim=params['in_data_dim'])], p=params['rotation_prob'])),
        PreserveKeysTransform(transforms.RandomApply([RandomNoise(data_dim=params['in_data_dim'], std=params['noise_std'])], p=params['noise_prob'])),
        PreserveKeysTransform(transforms.RandomApply([axis_mask(data_dim=params['in_data_dim'])], p=params['axis_mask_prob']))
    ]) if params['data_type'] in DATA_TYPES_SUPPORTING_RUNTIME_TRANSFORMS else None

    train_dataset = ProcessedDataset(
        data_dir,
        fold=fold,
        params=params,
        mode='train' if not merge_dataset else 'all_folds_merged',
        transform=runtime_train_transform,
        precomputed_transforms=(params['data_type'] in DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS),
        utilize_precomputed_transforms=(not BLOCK_ALL_PRECOPMUTED_TRANSFORMS) # We ended up not using any precomputed transformers
    )
    eval_dataset = ProcessedDataset(
        data_dir,
        fold=fold,
        params=params,
        mode='eval' if not merge_dataset else 'all_folds_merged',
        precomputed_transforms=(params['data_type'] in DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS),
        utilize_precomputed_transforms=False
    )

    return train_dataset, eval_dataset


def dataloader_factory(params, train_dataset, eval_dataset, eval_batch_size='default'):
    train_dataset_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=bool(len(train_dataset) >= params['batch_size']),
        pin_memory=True,
    )

    eval_dataset_fn = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=params['batch_size'] if eval_batch_size == 'default' else eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    train_dataset_for_class_weights_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    class_weights = compute_class_weights(train_dataset_for_class_weights_fn, params)
    return train_dataset_fn, eval_dataset_fn, class_weights

class PreserveKeysTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        transformed_sample = self.transform(sample)

        # Ensure all original keys are preserved
        for key in sample.keys():
            if key not in transformed_sample:
                transformed_sample[key] = sample[key]

        return transformed_sample
