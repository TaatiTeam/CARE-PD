import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import *
import joblib

from const.const import DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS

class TSDU_Reader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    def __init__(self, joints_path_list, smpls_path, params):
        self.joints_path_list = joints_path_list
        self.smpls_path = smpls_path
        self.params = params
        self.smpl_df = joblib.load(smpls_path)
        self.pose_dict, self.video_names, self.participant_ID, self.metadata_dict = self.read_keypoints()
        print(f"There are {len(self.pose_dict)} sequences in the T-SDU dataset.")
        print(f"There are {len(set(self.participant_ID))} different patients in the T-SDU dataset: {set(self.participant_ID)}")

    
    def read_metadata(self, seq_name):
        #If you change this function make sure to adjust the METADATA_MAP in the dataloaders.py accordingly
        return [[]]
    
    def read_keypoints(self):
        """
        Read npz file in given directory into arrays of pose keypoints.
        :return: dictionary with <key=video name, value=keypoints>
        """
        pose_dict = {}
        metadata_dict = {}
        video_names_list = []
        participant_ID = []

        print('[INFO - TSDU_Reader] Reading body keypoints from npz')

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
                metadata = self.read_metadata(seq_name)
                if joints is None:
                    print(f"[WARN - TSDU_Reader] {seq_name} is None.")

                dict_seq_name = seq_name + f'_view{view_counter}'
                pose_dict[dict_seq_name] = joints
                metadata_dict[dict_seq_name] = None
                video_names_list.append(dict_seq_name)
                participant_ID.append(seq_name.split("__")[0])
            view_counter += 1

        return pose_dict, video_names_list, participant_ID, metadata_dict
