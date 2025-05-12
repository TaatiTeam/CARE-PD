import numpy as np
import os
import shutil
import json
import random
import torch
from numpyencoder import NumpyEncoder
import re

from const import path
from const import const

def override_dataset(params, new_dataset):
    # raise NotImplementedError("This function is not implemented yet.")
    result_params = params.copy()
    result_params['dataset'] = new_dataset
    
    pose_type = const.BACKBONE_POSETYPE_MAPPER[params['backbone']]
    if pose_type == '2D_prespective':
        p_list = []
        for v in result_params['views']:
            p_list.append(path.POSE_AND_LABEL[new_dataset][params['data_type']]['PATH_POSES']['2D'][v])
        params['data_path'] = p_list 
    elif pose_type == '2D_orthographic':
        p_list = []
        for v in params['views']:
            vie = f'camera_{v}'
            p_list.append(path.POSE_AND_LABEL[new_dataset][params['data_type']]['PATH_POSES']['3D'][vie])
        params['data_path'] = p_list
    elif pose_type == '3D_processed':
        p_list = [path.POSE_AND_LABEL[new_dataset][params['data_type']]['PATH_POSES']['3D'][params['data_orient']]]
    elif pose_type == 'original_hml3d':
        p_list = [path.POSE_AND_LABEL[new_dataset][params['data_type']]['PATH_POSES']]
    elif pose_type == 'original_6d':
        p_list = [path.POSE_AND_LABEL[new_dataset][params['data_type']]['PATH_POSES']]
    else:
        raise ValueError(f"Unknown pose type: {pose_type}")
    
    result_params['data_path'] = p_list
    result_params['labels_path'] = path.POSE_AND_LABEL[new_dataset][params['data_type']]['PATH_LABELS']
    return result_params



def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dir_tree(base_dir, numfolds):
    dir_tree = ['models', 'config', 'std_log']
    last_run = 1
    for dir_ in dir_tree:
        if dir_ == dir_tree[0]:
            if not os.path.exists(os.path.join(base_dir)):
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
            else:
                last_run = np.max(list(map(int, os.listdir(base_dir))))
                last_run += 1
                if not os.path.exists(
                        os.path.join(base_dir, str(last_run - 1), 'classification_report_last.txt')):
                    last_run -= 1
                    shutil.rmtree(os.path.join(base_dir, str(last_run)))
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
        else:
            os.makedirs(os.path.join(base_dir, str(last_run), dir_))
    return last_run

def create_dir_tree2(base_dir, last_run):
    dir_tree = ['models', 'config']
    for dir_ in dir_tree:
        if dir_ == dir_tree[0]:
            if not os.path.exists(os.path.join(base_dir, str(last_run))):
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
            else:
                shutil.rmtree(os.path.join(base_dir, str(last_run)))
                os.makedirs(os.path.join(base_dir, str(last_run), dir_))
        else:
            os.makedirs(os.path.join(base_dir, str(last_run), dir_))
            
def save_json(filename, attributes, names):
    """
  Save training parameters and evaluation results to json file.
  :param filename: save filename
  :param attributes: attributes to save
  :param names: name of attributes to save in json file
  """
    with open(filename, "w", encoding="utf8") as outfile:
        d = {}
        for i in range(len(attributes)):
            name = names[i]
            attribute = attributes[i]
            d[name] = attribute
        json.dump(d, outfile, indent=4, cls=NumpyEncoder)
        
        
def is_substring(str1, str2):
    return str1.lower() in str2.lower()



def check_and_get_first_elements(list_of_lists):
    """
    check that all elements within each inner list are the same
    and also retrieve the first element of each inner list
    
    Parameters: list_of_lists (list of lists): A list containing inner lists to be checked.
    Returns: list: A list of the first elements from each uniform inner list.
    Raises: ValueError: If any inner list is empty or contains non-uniform elements.
    """
    first_elements = []

    for inner_list in list_of_lists:
        if not inner_list:
            raise ValueError("One of the inner lists is empty.")
        
        first_element = inner_list[0]
        if all(element == first_element for element in inner_list):
            first_elements.append(first_element)
        else:
            raise ValueError(f"Elements in the inner list {inner_list} differ.")

    return first_elements

def check_uniformity_and_get_first_elements(mainlist):
    try:
        mainlist = check_and_get_first_elements(mainlist)
        # print("First elements of each uniform inner list:", mainlist)
        return mainlist
    except ValueError as e:
        print(f"Error: {e}")
        


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def convert_ndarray_to_list(d):
    """recursively convert all numpy arrays in a dict to lists"""
    if isinstance(d, dict):
        return {k: convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_ndarray_to_list(i) for i in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d
    
    
def compare_two_configs(saved_config, params):
    for k in saved_config:
        if k not in params:
            print(f"[WARN]‼️‼️‼️ Key '{k}' is missing in current params.")
        elif saved_config[k] != params[k]:
            if k in ['model_checkpoint_path', 'data_path', 'labels_path', 'hypertune', 'ntrials']:
                continue
            print(f"[WARN]‼️‼️‼️ Value mismatch for key '{k}': saved = {saved_config[k]}, current = {params[k]}")

    for k in params:
        if k not in saved_config:
            print(f"[WARN]‼️‼️‼️ Key '{k}' is missing in saved config.")
            
def extract_base_name(name):
    # Removes `_viewX` suffix from name
    return re.sub(r'_view\d+$', '', name)
            
# Build dictionaries: base_name -> logits
def build_logit_map(data):
    logit_map = {}
    for name, logits in zip(data["video_names"], data["predicted_logits"]):
        base_name = extract_base_name(name)
        logit_map[base_name] = np.array(logits)
    return logit_map


def get_last_folder(base_path):
    largest_folder = '-1'
    try:
        folder_names = os.listdir(base_path)
        int_dirs = []
        for name in folder_names:
            full_path = os.path.join(base_path, name)
            if os.path.isdir(full_path):
                try:
                    int_dirs.append(int(name))
                except ValueError:
                    pass  # skip non-integer names

        if int_dirs:
            largest_folder = str(max(int_dirs))
    except FileNotFoundError:
        pass  # base_path doesn't exist, stick with '0'
    return str(int(largest_folder)+1)