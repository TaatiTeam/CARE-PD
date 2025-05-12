import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import *
import joblib

from const.const import DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS

class GAIT3DReader():
    """
    Reads the data from the PD-GaM dataset
    """

    UPDRS_SCORE_COLUMN = 'score'
    SEQ_NAME_COLUMN = 'vidname'

    def __init__(self, joints_path_list, labels_path, params):
        self.joints_path_list = joints_path_list
        self.labels_path = labels_path
        self.params = params
        self.label_df = joblib.load(labels_path)
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict = self.read_keypoints_and_labels()
        print(f"There are {len(self.pose_dict)} sequences in the 3dGait dataset from {len(params['views'])} views.")
        print(f"There are {len(set(self.participant_ID))} different patients in the 3dGait dataset: {set(self.participant_ID)}")
        unique, counts = np.unique(list(self.labels_dict.values()), return_counts=True)
        print(f"Distribution of labels in 3dGait dataset:")
        print(np.asarray((unique, counts)).T)
    
    def extract_base_seq_name(self, seq_name):
        return seq_name

    def read_label(self, seq_name):
        subject_id, walkid = seq_name.split("__")
        walkid = walkid.split("_down")[0]
        label = self.label_df[int(subject_id)][walkid]['UPDRS_GAIT']
        return int(label)

    def read_metadata(self, seq_name):
        #If you change this function make sure to adjust the METADATA_MAP in the dataloaders.py accordingly
        return [[]]
    
    def read_keypoints_and_labels(self):
        """
        Read npz file in given directory into arrays of pose keypoints.
        :return: dictionary with <key=video name, value=keypoints>
        """
        pose_dict = {}
        labels_dict = {}
        metadata_dict = {}
        video_names_list = []
        participant_ID = []

        print('[INFO - GAIT3DReader] Reading body keypoints or pose from npz')

        print(self.joints_path_list)

        view_counter = 0
        for joints_path in self.joints_path_list:
            seqs = np.load(joints_path, allow_pickle=True)
            for seq_name in tqdm(seqs.keys()):
                joints = seqs[seq_name]
                if self.params['data_type'] in DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS: # Block loading of any precomputed transforms
                    if joints.ndim == 2:
                        joints = np.expand_dims(joints, axis=0)
                    else:
                        joints = joints[:1, ...]
                label = self.read_label(seq_name)
                metadata = self.read_metadata(seq_name)
                if joints is None:
                    print(f"[WARN - GAIT3DReader] {seq_name} is None.")

                dict_seq_name = seq_name + f'_view{view_counter}'
                pose_dict[dict_seq_name] = joints
                labels_dict[dict_seq_name] = label
                metadata_dict[dict_seq_name] = metadata
                video_names_list.append(dict_seq_name)
                participant_ID.append(seq_name.split("__")[0])
            view_counter += 1

        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict
