import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from datetime import *

from const.const import DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS

class BMCLabReader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    ON_LABEL_COLUMN = 'ON - UPDRS-III - walking'
    OFF_LABEL_COLUMN = 'OFF - UPDRS-III - walking'

    def __init__(self, joints_path_list, labels_path, params):
        self.joints_path_list = joints_path_list
        self.labels_path = labels_path
        self.params = params
        self.label_df = joblib.load(labels_path)
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict = self.read_keypoints_and_labels()
        print(f"There are {len(self.pose_dict)} sequences in the BMCLab dataset.")
        print(f"There are {len(set(self.participant_ID))} different patients in the BMCLab dataset: {set(self.participant_ID)}")
        unique, counts = np.unique(list(self.labels_dict.values()), return_counts=True)
        print(f"Distribution of labels in BMCLab dataset:")
        print(np.asarray((unique, counts)).T)

    def read_label(self, seq_name):
        subject_id, walkid = seq_name.split("__")
        walkid = walkid.split("_down")[0]
        label = self.label_df[subject_id][walkid]['UPDRS_GAIT']
        return int(label)
    
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

        print('[INFO - PDReader] Reading body keypoints or pose from npz')

        print(self.joints_path_list)

        view_counter = 0
        for joints_path in self.joints_path_list:
            seqs = np.load(joints_path, allow_pickle=True)
            trimmed_counter = 0
            for seq_name in tqdm(seqs.keys()):
                if 'trimmed' in seq_name.lower():
                    trimmed_counter += 1
                    continue
                joints = seqs[seq_name]
                if self.params['data_type'] in DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS: # Block loading of any precomputed transforms
                    if joints.ndim == 2:
                        joints = np.expand_dims(joints, axis=0)
                    else:
                        joints = joints[:1, ...]
                label = self.read_label(seq_name)
                if joints is None:
                    print(f"[WARN - PDReader] {seq_name} is None.")

                dict_seq_name = seq_name + f'_view{view_counter}'
                pose_dict[dict_seq_name] = joints
                labels_dict[dict_seq_name] = label
                metadata_dict[dict_seq_name] = None
                video_names_list.append(dict_seq_name)
                participant_ID.append(dict_seq_name.split("_")[0])
            print(f"[INFO]: #{trimmed_counter} Trimmed sequences - IGNORED.")
            view_counter += 1

        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict
