import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import *

from const.const import DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS

class TRI_PD_Reader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    SEQ_NAME_COLUMN = 'File Number/Title '
    UPDRS_SCORE_COLUMN = 'UPDRS__gait'

    def __init__(self, joints_path_list, labels_path, params):
        self.joints_path_list = joints_path_list
        self.labels_path = labels_path
        self.params = params
        self.label_df = pd.read_excel(self.labels_path)
        self.label_df = self.label_df[[self.SEQ_NAME_COLUMN, self.UPDRS_SCORE_COLUMN]]
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict = self.read_keypoints_and_labels()
        print(f"There are {len(self.pose_dict)} sequences in the TRI_PD dataset.")
        print(f"There are {len(set(self.participant_ID))} different patients in the TRI_PD dataset: {set(self.participant_ID)}")
        unique, counts = np.unique(list(self.labels_dict.values()), return_counts=True)
        print(f"Distribution of labels in TRI_PD dataset:")
        print(np.asarray((unique, counts)).T)

    def read_label(self, seq_name):
        video_rows = self.label_df[self.label_df[self.SEQ_NAME_COLUMN] == seq_name]
        label = video_rows[self.UPDRS_SCORE_COLUMN].values[0]
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

        print('[INFO - TRI_PD_Reader] Reading body keypoints from npz')

        print(self.joints_path_list)

        view_counter = 0
        for joints_path in self.joints_path_list:
            seqs = np.load(joints_path, allow_pickle=True)
            for seq_name in tqdm([s for s in seqs.keys() if not s.endswith('_frame_ids')]):
                joints = seqs[seq_name]
                if self.params['data_type'] in DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS: # Block loading of any precomputed transforms
                    if joints.ndim == 2:
                        joints = np.expand_dims(joints, axis=0)
                    else:
                        joints = joints[:1, ...]
                seq_name_in_annot = seq_name.split('_subclip')[0].replace('forward_', '').replace('backward_', '')
                label = self.read_label(seq_name_in_annot)
                metadata = self.read_metadata(seq_name)
                if joints is None:
                    print(f"[WARN - TRI_PD_Reader] {seq_name} is None.")

                dict_seq_name = seq_name + f'_view{view_counter}'
                pose_dict[dict_seq_name] = joints
                labels_dict[dict_seq_name] = label
                metadata_dict[dict_seq_name] = metadata
                video_names_list.append(dict_seq_name)
                participant_ID.append(dict_seq_name.split('ID_')[1].split('_')[0])
            view_counter += 1

        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict
