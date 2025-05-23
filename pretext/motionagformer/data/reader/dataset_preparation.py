import os
import glob
import numpy as np
import pandas as pd
import random
import re
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
# from visualize_skel_walk_func import visualize_sequence

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


class MotionDataset3D(Dataset):

    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split  # 'train' or 'test'
        self.flip = True

        self.data_list = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        self.data_2d, self.data_3d, self.lambda_opts = [], [], []
        self.frame_ids = []
        self.last_frame_id = -1
        self.video_names = []

        for dataset in self.data_list:
            dataset_dir = os.path.join(self.data_dir, dataset)

            # --- Load and filter split annotations ---
            UPDRS_list = ['3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD']
            fold_dir = "../../assets/datasets/folds"
            if dataset in UPDRS_list:
                fold_path = os.path.join(fold_dir, "UPDRS_Datasets")
            else:
                fold_path = os.path.join(fold_dir, "Other_Datasets")

            if dataset == "PD-GaM":
                fold_path = os.path.join(fold_path, "PD-GaM_authors_fixed.pkl")
            elif dataset == "T-SDU-PD":
                fold_path = os.path.join(fold_path, "T-SDU-PD_PD_fixed.pkl")
            else:
                fold_path = os.path.join(fold_path, dataset + "_fixed.pkl")

            with open(fold_path, 'rb') as f:
                fold_dict = pickle.load(f)
            walkIDs = fold_dict[1][split]

            # --- Load NPZ files (lazy-loading via NpzFile) ---
            pattern = os.path.join(dataset_dir, '*.npz')
            npz_paths = glob.glob(pattern)
            self.npz_3d = []  # for world2cam (3D)
            self.npz_2d = []  # for world2cam2img (2D)
            for path in npz_paths:
                fname = os.path.basename(path)

                data = np.load(path, allow_pickle=True)

                data_dict = {key.split('__', 1)[0]: data[key] for key in data.files if key.split('__', 1)[0] in walkIDs}
                
                if 'world2cam2img' in fname:
                    self.npz_2d.append(data_dict)
                elif 'world2cam' in fname:
                    self.npz_3d.append(data_dict)
            self.trim_longer_pair()
            self.lambdas = self.normalize_3d()
            poses_2d, poses_3d, frame_ids, lambda_opts, labels = self.prepare_data()

            self.data_2d.extend(poses_2d)
            self.data_3d.extend(poses_3d)
            self.frame_ids.extend(frame_ids)
            self.lambda_opts.extend(lambda_opts)
            self.video_names.extend(labels)
            
        
        ll = set()
        for ele in self.video_names:
            ll.add(ele)
        print(len(ll))
        print(np.asarray(self.data_2d).shape)
        print(np.asarray(self.data_3d).shape)
        print(len(self.lambda_opts))
        print(len(self.video_names))

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

    def normalize(self, sequence, width=1005, height=1005, is_3d=False):
        result = np.copy(np.asarray(sequence))
        result[..., :2] = sequence[..., :2] / width * 2 - [1, height / width] 
        if is_3d:   # This is only required for training to have normalized 3d groundtruth
            result[..., 2:] = sequence[..., 2:] / width * 2 
        return result

    def prepare_data(self):
        data_2d, data_3d, frame_ids, lambda_opts, labels = [], [], [], [], []
        for seq2d_dict, seq3d_dict, seq_lambdas in zip(self.npz_2d, self.npz_3d, self.lambdas):
            for key in seq2d_dict:
                seq2d = seq2d_dict[key]
                seq3d = seq3d_dict[key]
                seq_lambda_opts = seq_lambdas[key]
                seq_frame_ids = np.arange(seq2d.shape[0]) + (self.last_frame_id + 1)
                self.last_frame_id += 1

                seq2d_partitioned = self.partition(seq2d, stride=27 if self.split == 'train' else 81)
                seq3d_partitioned = self.partition(seq3d, stride=27 if self.split == 'train' else 81)
                frame_ids_partitioned = self.partition(seq_frame_ids, stride=27 if self.split == 'train' else 81)

                seq_lambda_opts_partitioned = [seq_lambda_opts] * np.asarray(seq2d_partitioned).shape[0]
                labels_partitioned = [key] * np.asarray(seq2d_partitioned).shape[0]
                

                data_2d.extend(seq2d_partitioned)
                data_3d.extend(seq3d_partitioned)
                frame_ids.extend(frame_ids_partitioned)
                lambda_opts.extend(seq_lambda_opts_partitioned)
                labels.extend(labels_partitioned)
                
        return data_2d, data_3d, frame_ids, lambda_opts, labels

    def trim_longer_pair(self):
        for seq2d_dict, seq3d_dict in zip(self.npz_2d, self.npz_3d):
            for key in seq2d_dict:
                seq2d = seq2d_dict[key]
                seq3d = seq3d_dict[key]

                if seq3d.shape[0] > seq2d.shape[0]:
                    n_frames = seq2d.shape[0]
                    seq3d_dict[key] = seq3d[:n_frames]

    def normalize_3d(self):
        lambdas = []
        for seq2d_dict, seq3d_dict in zip(self.npz_2d, self.npz_3d):
            seq_lambdas = {}
            for key in seq2d_dict:
                seq2d = seq2d_dict[key]
                seq3d = seq3d_dict[key]

                seq2d = self.normalize(seq2d)
                # seq3d = self.normalize(seq3d, is_3d=False)

                seq2d_centered = seq2d - seq2d[:, 0:1]
                seq3d_centered = seq3d - seq3d[:, 0:1]
                
                numerator = np.sum(seq3d_centered[..., :2] * seq2d_centered)
                denominator = np.sum(seq3d_centered[..., :2] * seq3d_centered[..., :2])

                lambda_opt = numerator / denominator
                
                normalized_seq3d = seq3d_centered * lambda_opt

                # print(normalized_seq3d.min(), normalized_seq3d.max())

                seq3d_dict[key] = normalized_seq3d
                seq2d_dict[key] = seq2d
                seq_lambdas[key] = lambda_opt
            lambdas.append(seq_lambdas)
        return lambdas

    @staticmethod
    def resample(original_length, target_length):
        """
        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result
        
    def partition(self, data, clip_length=81, stride=27):
        """Partitions data (n_frames, 17, 3) into list of (clip_length, 17, 3) data with given stride"""
        data_list, valid_list = [], []
        n_frames = data.shape[0]
        for i in range(0, n_frames, stride):
            sequence = data[i:i+clip_length]
            sequence_length = sequence.shape[0]
            if sequence_length == clip_length:
                data_list.append(sequence)
            else:
                new_indices = self.resample(sequence_length, clip_length)
                extrapolated_sequence = sequence[new_indices]
                data_list.append(extrapolated_sequence)
        return data_list

    def __len__(self):
        return len(self.data_2d)

    def __getitem__(self, idx):

        pose3d = self.data_3d[idx]

        pose2d = self.data_2d[idx]
        frame_ids = self.frame_ids[idx]

        T, J, _ = pose2d.shape
        ones = np.ones((T, J, 1), dtype=pose2d.dtype)  # 1 as confidence score
        pose2d = np.concatenate([pose2d, ones], axis=2)

        # Convert to torch tensors
        motion_2d = torch.from_numpy(pose2d).float()
        motion_3d = torch.from_numpy(pose3d).float()

        if self.split == 'train':
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)

        # return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), torch.FloatTensor(frame_ids), torch.FloatTensor([lambda_opts]), torch.FloatTensor([video_name])
        return (
            torch.FloatTensor(motion_2d),     # [N,T,2]
            torch.FloatTensor(motion_3d),     # [N,T,3]
            torch.FloatTensor(frame_ids)                    # leave as-is (string)
        )

        

if __name__ == '__main__':
    # Example usage:
    # -----------------------------------------------
    data_dir = './data'
    batch_size = 32
    num_workers = 4
    common_loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    test_dataset = MotionDataset3D(data_dir, 'test')
    test_loader  = DataLoader(test_dataset, shuffle=False,  **common_loader_params)

    # for x, y, f, l in test_loader:
    #     if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(f).any():
    #         print("NAN!")
    #         exit()

    train_dataset  = MotionDataset3D(data_dir, 'train')
    train_loader   = DataLoader(test_dataset,  shuffle=False, **common_loader_params)
