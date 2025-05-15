import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import *

from const.const import DATA_TYPES_WITH_PRECOMPUTED_AUGMENTATIONS

class BMCLABSReader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    ON_LABEL_COLUMN = 'ON - UPDRS-III - walking'
    OFF_LABEL_COLUMN = 'OFF - UPDRS-III - walking'

    def __init__(self, joints_path_list, labels_path, params):
        self.joints_path_list = joints_path_list
        self.labels_path = labels_path
        self.params = params
        self.label_df = pd.read_excel(self.labels_path)
        self.label_df = self.label_df[['ID', self.ON_LABEL_COLUMN, self.OFF_LABEL_COLUMN]]
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict = self.read_keypoints_and_labels()
        print(f"There are {len(self.pose_dict)} sequences in the BMCLABS dataset.")
        print(f"There are {len(set(self.participant_ID))} different patients in the BMCLABS dataset: {set(self.participant_ID)}")
        unique, counts = np.unique(list(self.labels_dict.values()), return_counts=True)
        print(f"Distribution of labels in BMCLABS dataset:")
        print(np.asarray((unique, counts)).T)

    def read_label(self, seq_name):
        subject_id, on_or_off = seq_name.split("_")[:2]
        subject_rows = self.label_df[self.label_df['ID'] == subject_id]
        if on_or_off == "on":
            label = subject_rows[self.ON_LABEL_COLUMN].values[0]
        else:
            label = subject_rows[self.OFF_LABEL_COLUMN].values[0]
        return int(label)
    
    def read_metadata(self, seq_name):
        #If you change this function make sure to adjust the METADATA_MAP in the dataloaders.py accordingly
        subject_id = seq_name.split("_")[0]
        df = pd.read_excel(self.labels_path)
        df = df[['ID', 'Gender', 'Age', 'Height (cm)', 'Weight (kg)', 'BMI (kg/m2)']]
        df.rename(columns={
            "Gender": "gender",
            "Age": "age",
            "Height (cm)": "height",
            "Weight (kg)": "weight",
            "BMI (kg/m2)": "bmi"}, inplace=True)
        df.loc[:, 'gender'] = df['gender'].map({'M': 0, 'F': 1})
        
        # Using Min-Max normalization
        df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        df['height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
        df['weight'] = (df['weight'] - df['weight'].min()) / (df['weight'].max() - df['weight'].min())
        df['bmi'] = (df['bmi'] - df['bmi'].min()) / (df['bmi'].max() - df['bmi'].min())

        subject_rows = df[df['ID'] == subject_id]
        return subject_rows.values[:, 1:] 
    
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
                metadata = self.read_metadata(seq_name)
                if joints is None:
                    print(f"[WARN - PDReader] {seq_name} is None.")

                dict_seq_name = seq_name + f'_view{view_counter}'
                pose_dict[dict_seq_name] = joints
                labels_dict[dict_seq_name] = label
                metadata_dict[dict_seq_name] = metadata
                video_names_list.append(dict_seq_name)
                participant_ID.append(dict_seq_name.split("_")[0])
            print(f"[INFO]: #{trimmed_counter} Trimmed sequences - IGNORED.")
            view_counter += 1

        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict
