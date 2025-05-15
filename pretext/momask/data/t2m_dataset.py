from os.path import join
from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
import os
import re
import pandas as pd 

from utils.get_opt import get_opt
from options.vq_option import arg_parse


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

healthy_exclude_list = [
    "Healthy_HEA176_gait05",
    "fullbodystroke_SUBJ135 (3)",
    "Healthy_HEA172_gait02",
    "Healthy_HEA174_gait03",
    "Healthy_HEA121_gait10",
    "Healthy_HEA171_gait03",
    "Healthy_HEA142_gait06",
    "Healthy_HEA125_gait10",
    "Healthy_HEA142_gait04",
    "Healthy_HEA124_gait13",
    "Healthy_HEA192_gait07",
    "Healthy_HEA173_gait11",
    "fullbodystroke_SUBJ60 (4)",
    "fullbodystroke_SUBJ70 (3)",
    "Healthy_HEA194_gait01",
    "Healthy_HEA191_gait05",
    "Healthy_HEA194_gait15",
    "Healthy_HEA193_gait04",
    "Healthy_HEA194_gait07",
    "Healthy_HEA189_gait01",
    "Healthy_HEA147_gait08",
    "Healthy_HEA178_gait11",
    "Healthy_HEA190_gait05",
    "Healthy_HEA142_gait14",
    "Healthy_HEA179_gait01",
    "Healthy_HEA196_gait09",
    "Healthy_HEA176_gait07",
    "Healthy_HEA155_gait13",
    "Healthy_HEA199_gait08",
    "Healthy_HEA153_gait01",
    "Healthy_HEA194_gait17",
    "Healthy_HEA185_gait02",
    "Healthy_HEA121_gait18",
    "Healthy_HEA172_gait08",
    "fullbodystroke_SUBJ86 (4)",
    "Healthy_HEA180_gait11",
    "Healthy_HEA179_gait15",
    "Healthy_HEA184_gait10",
    "Healthy_HEA175_gait12",
    "Healthy_HEA153_gait21",
    "Healthy_HEA169_gait09",
    "Healthy_HEA123_gait19",
    "fullbodystroke_SUBJ74 (4)",
    "Healthy_HEA192_gait05",
    "Healthy_HEA193_gait10",
    "Healthy_HEA142_gait16",
    "Healthy_HEA149_gait04",
    "Healthy_HEA194_gait19",
    "Healthy_HEA134_gait23",
    "Healthy_HEA172_gait04",
    "Healthy_HEA128_gait01",
    "Healthy_HEA191_gait01",
    "Healthy_HEA176_gait13",
    "Healthy_HEA155_gait25",
    "Healthy_HEA196_gait06",
    "Healthy_HEA124_gait03",
    "Healthy_HEA174_gait04",
    "Healthy_HEA158_gait14",
    "Healthy_HEA178_gait14",
    "Healthy_HEA195_gait13",
    "Healthy_HEA171_gait17",
    "Healthy_HEA188_gait12",
    "fullbodystroke_TVC21_BWA6",
    "Healthy_HEA133_gait15",
    "Healthy_HEA129_gait18",
    "Healthy_HEA155_gait01",
    "Healthy_HEA180_gait06",
    "Healthy_HEA172_gait20",
    "Healthy_HEA147_gait06",
    "fullbodystroke_SUBJ32 (2)",
    "Healthy_HEA132_gait07",
    "Healthy_HEA123_gait17",
    "fullbodystroke_SUBJ126 (4)",
    "Healthy_HEA133_gait09",
    "Healthy_HEA179_gait12",
    "Healthy_HEA149_gait16",
    "Healthy_HEA180_gait15",
    "Healthy_HEA172_gait17",
    "Healthy_HEA149_gait12",
    "Healthy_HEA196_gait02",
    "Healthy_HEA195_gait09",
    "Healthy_HEA155_gait16",
    "Healthy_HEA194_gait09",
    "Healthy_HEA187_gait15",
    "Healthy_HEA192_gait10",
    "Healthy_HEA179_gait20",
    "Healthy_HEA173_gait12",
    "Healthy_HEA191_gait11",
    "Healthy_HEA175_gait02",
    "Healthy_HEA199_gait14",
    "Healthy_HEA169_gait10",
    "Healthy_HEA165_gait07",
    "Healthy_HEA165_gait03",
    "Healthy_HEA181_gait16",
    "Healthy_HEA124_gait05",
    "fullbodystroke_SUBJ39 (1)",
    "Healthy_HEA169_gait07",
    "Healthy_HEA184_gait04",
    "Healthy_HEA180_gait09",
    "Healthy_HEA123_gait01",
    "Healthy_HEA191_gait09",
    "fullbodystroke_SUBJ24 (2)",
    "Healthy_HEA155_gait11",
    "Healthy_HEA181_gait11",
    "Healthy_HEA196_gait05",
    "Healthy_HEA132_gait19",
    "Healthy_HEA175_gait10",
    "fullbodystroke_SUBJ79 (1)",
    "fullbodystroke_SUBJ65 (3)",
    "Healthy_HEA173_gait03",
    "Healthy_HEA125_gait12",
    "Healthy_HEA192_gait08",
    "fullbodystroke_SUBJ135 (6)",
    "Healthy_HEA191_gait13",
    "fullbodystroke_SUBJ85 (4)",
    "Healthy_HEA181_gait06",
    "Healthy_HEA194_gait13",
    "fullbodystroke_SUBJ102 (1)",
    "Healthy_HEA152_gait15",
    "Healthy_HEA150_gait05",
    "Healthy_HEA142_gait10",
    "Healthy_HEA177_gait01",
    "Healthy_HEA176_gait01",
    "Healthy_HEA151_gait05",
    "Healthy_HEA132_gait01",
    "Healthy_HEA181_gait15",
    "Healthy_HEA128_gait20",
    "Healthy_HEA166_gait02",
    "Healthy_HEA141_gait03",
    "Healthy_HEA147_gait18",
    "Healthy_HEA192_gait02",
    "Healthy_HEA172_gait06",
    "Healthy_HEA180_gait07",
    "Healthy_HEA180_gait16",
    "Healthy_HEA189_gait09",
    "Healthy_HEA194_gait05",
    "Healthy_HEA126_gait16",
    "Healthy_HEA190_gait12",
    "Healthy_HEA196_gait07",
    "Healthy_HEA184_gait08",
    "Healthy_HEA171_gait13",
    "Healthy_HEA194_gait11",
    "Healthy_HEA194_gait03",
    "Healthy_HEA141_gait13",
    "Healthy_HEA177_gait08",
    "Healthy_HEA123_gait09",
    "Healthy_HEA179_gait13",
    "fullbodystroke_SUBJ84 (2)",
    "Healthy_HEA192_gait17",
    "Healthy_HEA155_gait15",
    "Healthy_HEA189_gait11",
    "Healthy_HEA193_gait01",
    "Healthy_HEA123_gait05",
    "Healthy_HEA178_gait12",
    "Healthy_HEA169_gait02",
    "Healthy_HEA131_gait16",
    "Healthy_HEA155_gait26",
    "Healthy_HEA123_gait13",
    "Healthy_HEA199_gait02",
    "Healthy_HEA176_gait09",
    "Healthy_HEA121_gait14",
    "Healthy_HEA121_gait12",
    "Healthy_HEA177_gait07",
    "Healthy_HEA169_gait05",
    "Healthy_HEA142_gait12",
    "Healthy_HEA180_gait01",
    "fullbodystroke_SUBJ117 (1)",
    "Healthy_HEA152_gait13",
    "Healthy_HEA176_gait15",
    "Healthy_HEA125_gait18",
    "Healthy_HEA155_gait04",
    "Healthy_HEA125_gait27",
    "fullbodystroke_SUBJ74 (2)",
    "Healthy_HEA171_gait04",
    "Healthy_HEA193_gait07",
    "Healthy_HEA155_gait28",
    "Healthy_HEA192_gait09",
    "Healthy_HEA125_gait01",
    "Healthy_HEA185_gait03",
    "Healthy_HEA152_gait07",
    "Healthy_HEA149_gait18",
    "Healthy_HEA172_gait13",
    "Healthy_HEA184_gait12",
    "fullbodystroke_SUBJ69 (2)",
    "Healthy_HEA196_gait03",
    "Healthy_HEA188_gait01",
    "Healthy_HEA123_gait07",
    "Healthy_HEA132_gait03",
    "Healthy_HEA134_gait14",
    "Healthy_HEA123_gait03",
    "Healthy_HEA131_gait17",
    "Healthy_HEA142_gait11",
    "Healthy_HEA186_gait08",
    "Healthy_HEA183_gait07",
    "Healthy_HEA172_gait19",
    "fullbodystroke_SUBJ116 (2)",
    "Healthy_HEA199_gait06",
    "fullbodystroke_SUBJ16 (2)",
    "Healthy_HEA126_gait10",
    "Healthy_HEA165_gait08",
    "Healthy_HEA191_gait07",
    "Healthy_HEA177_gait09",
    "Healthy_HEA169_gait12",
    "Healthy_HEA181_gait08",
    "Healthy_HEA195_gait11",
    "Healthy_HEA165_gait11",
    "Healthy_HEA191_gait03",
    "Healthy_HEA195_gait10",
    "Healthy_HEA165_gait14",
    "Healthy_HEA183_gait03",
    "Healthy_HEA192_gait03",
    "Healthy_HEA172_gait09",
    "Healthy_HEA192_gait04",
    "Healthy_HEA142_gait13",
    "Healthy_HEA124_gait02",
    "Healthy_HEA143_gait07",
    "Healthy_HEA147_gait03",
    "Healthy_HEA128_gait04",
    "Healthy_HEA187_gait03",
    "fullbodystroke_SUBJ96 (3)",
    "Healthy_HEA124_gait01",
    "Healthy_HEA180_gait05",
    "Healthy_HEA176_gait11",
    "Healthy_HEA138_gait03",
    "Healthy_HEA177_gait06",
    "Healthy_HEA141_gait09",
]

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split):
        self.opt = opt
        joints_num = opt.joints_num


        self.data_dir = opt.data_root
        self.data_list = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d != 'metadata']

        if split == "test":
            self.data_dir = "./dataset/carepd/"
            self.data_list = ['DNE', '3DGait', 'BMClab', 'PDGAM', 'TRI_PD']

        print("data_list:", self.data_list)

        self.data = []
        self.lengths = []
        for dataset in self.data_list:
            dataset_dir = os.path.join(self.data_dir, dataset)

            # # --- Load and filter split annotations ---
            # split_csv = os.path.join(self.data_dir, "metadata")
            # csv_name = dataset + "_restructured_metadata.csv"
            # split_csv = os.path.join(split_csv, csv_name)
            # df = pd.read_csv(split_csv)
            # df_split = df[df['split'] == split]
            # walkIDs = df_split['walkID'].tolist()
            if split == "test" or "carepd" in self.data_dir:
                split_csv = os.path.join(self.data_dir, "metadata")
                csv_name = dataset + "_restructured_metadata.csv"
                split_csv = os.path.join(split_csv, csv_name)
                df = pd.read_csv(split_csv)
                df_split = df[df['split'] == split]
                walkIDs = df_split['walkID'].tolist()

            npz_path = os.path.join(dataset_dir, "HumanML3D/HumanML3D_collected.npz")
            data = np.load(npz_path, allow_pickle=True)

            

            for key in data.files:
                base = re.sub(r'_down.*$', '', key)
                motion = data[key]

                if split == "test" or "carepd" in self.data_dir:
                    if (base in walkIDs) and (motion.shape[0] >= opt.window_size):
                        self.lengths.append(motion.shape[0] - opt.window_size)
                        self.data.append(motion) 
                else:
                    if (base not in healthy_exclude_list) and (motion.shape[0] >= opt.window_size):
                        self.lengths.append(motion.shape[0] - opt.window_size)
                        self.data.append(motion) 

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        assert motion.shape == (64, 263)


        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.opt.max_text_len = 20
        self.max_motion_length = 196
        self.opt.unit_length = 4
        # self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []

        new_name_list = []
        length_list = []
        
        # self.data_dir = opt.data_root
        self.data_dir = "./dataset/carepd/" ##
        # self.data_list = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d != 'metadata']
        self.data_list = ['DNE', '3DGait', 'BMClab', 'PDGAM', 'TRI_PD']
        split = "test"

        self.data = []
        self.lengths = []
        for dataset in self.data_list:
            dataset_dir = os.path.join(self.data_dir, dataset)

            # --- Load and filter split annotations ---
            split_csv = os.path.join(self.data_dir, "metadata")
            csv_name = dataset + "_restructured_metadata.csv"
            split_csv = os.path.join(split_csv, csv_name)
            df = pd.read_csv(split_csv)
            df_split = df[df['split'] == split]
            walkIDs = df_split['walkID'].tolist()

            npz_path = os.path.join(dataset_dir, "HumanML3D/HumanML3D_collected.npz")
            data = np.load(npz_path, allow_pickle=True)

            # data_dict = {re.sub(r'_down\d*', '', key): data[key] for key in data.files if re.sub(r'_down\d*', '', key) in walkIDs}
            for key in data.files:
                name = re.sub(r'_down.*$', '', key)
                motion = data[key]

                if (name not in walkIDs) or (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                
                text_data = []
                text_dict = {}
                text_dict['caption'] = 'DUMMY'
                text_dict['tokens'] = ['DUMMY/NOUN']
                text_data.append(text_dict)
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)

if __name__ == '__main__':

    opt = arg_parse(True)

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
    test_dataset = MotionDataset(None, 0, 0, 'test')
    # test_loader  = DataLoader(test_dataset, shuffle=False,  **common_loader_params)

    train_dataset  = MotionDataset(None, 0, 0, 'train')
    # train_loader   = DataLoader(test_dataset,  shuffle=False, **common_loader_params)