import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from datetime import *

from const.const import DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS

class ELCReader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    def __init__(self, joints_path_list, labels_path, params):
        self.joints_path_list = joints_path_list
        self.labels_path = labels_path
        self.params = params
        self.label_df = joblib.load(labels_path)
        self.pose_dict, self.FoG_labels_dict, self.video_names, self.participant_ID, self.metadata_dict, self.medication_dict = self.read_keypoints_and_labels()
        print(f"There are {len(self.pose_dict)} sequences in the E-LC dataset.")
        print(f"There are {len(set(self.participant_ID))} different patients in the E-LC dataset: {set(self.participant_ID)}")
        unique, counts = np.unique(list(self.FoG_labels_dict.values()), return_counts=True)
        print(f"Distribution of FoG labels in E-LC dataset:")
        print(np.asarray((unique, counts)).T)
        unique, counts = np.unique(list(self.medication_dict.values()), return_counts=True)
        print(f"Distribution of medications in E-LC dataset:")
        print(np.asarray((unique, counts)).T)

    def read_label_and_med(self, seq_name):
        subject_id, walkid = seq_name.split("__")
        walkid = walkid.split("_down")[0]
        FoG = self.label_df[subject_id][walkid]['other']
        med_stat = self.label_df[subject_id][walkid]['medication']
        return FoG, med_stat
    
    def read_keypoints_and_labels(self):
        """
        Read npz file in given directory into arrays of pose keypoints.
        :return: dictionary with <key=video name, value=keypoints>
        """
        pose_dict = {}
        fog_dict = {}
        metadata_dict = {}
        medication_dict = {}
        video_names_list = []
        participant_ID = []

        print('[INFO - E-LCReader] Reading body keypoints or pose from npz')

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
                FoG_label, med = self.read_label_and_med(seq_name)
                if joints is None:
                    print(f"[WARN - E-LCReader] {seq_name} is None.")

                dict_seq_name = seq_name + f'_view{view_counter}'
                pose_dict[dict_seq_name] = joints
                fog_dict[dict_seq_name] = FoG_label
                medication_dict[dict_seq_name] = med
                metadata_dict[dict_seq_name] = None
                video_names_list.append(dict_seq_name)
                participant_ID.append(seq_name.split("__")[0])
            print(f"[INFO]: #{trimmed_counter} Trimmed sequences - IGNORED.")
            view_counter += 1

        return pose_dict, fog_dict, video_names_list, participant_ID, metadata_dict, medication_dict
