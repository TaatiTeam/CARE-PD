#!/usr/bin/env python3
# analyze_dataset.py

import os
import pickle
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(dataset_name, data_dir="."):
    path = os.path.join(data_dir, f"{dataset_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_info(data):
    fps_list, updrs_list, med_list, other_list, pose_shapes = [], [], [], [], []
    for subj_id, walks in data.items():
        for walk_id, walk in walks.items():
            fps_list.append(walk.get("fps"))
            u = walk.get("UPDRS_GAIT")
            if u is not None:
                updrs_list.append(u)
            m = walk.get("medication")
            if m is not None:
                med_list.append(m)
            o = walk.get("other")
            if o is not None:
                other_list.append(o)
            pose = walk.get("pose")
            if hasattr(pose, "shape"):
                pose_shapes.append(pose.shape)
    return fps_list, updrs_list, med_list, other_list, pose_shapes

def print_summary(data, fps, updrs, meds, others, poses):
    n_subj = len(data)
    n_walks = sum(len(w) for w in data.values())
    print(f"Subjects: {n_subj}")
    print(f"Total walks: {n_walks}\n")

    print("FPS distribution:")
    for val, cnt in Counter(fps).items():
        print(f"  {val} Hz: ")
    print()

    if updrs:
        print("UPDRS_GAIT distribution:")
        for val, cnt in Counter(updrs).items():
            print(f"  score {val}: {cnt}")
        print()

    if meds:
        print("Medication status distribution:")
        for val, cnt in Counter(meds).items():
            print(f"  {val}: {cnt}")
        print()

    if others:
        print("Other-label distribution:")
        for val, cnt in Counter(others).items():
            print(f"  {val}: {cnt}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect one or more gait dataset pickle files")
    parser.add_argument(
        "--datasets", required=True, nargs="+",
        help="List of dataset names (without .pkl), e.g. 3DGait BMCLab PD-GaM"
    )
    parser.add_argument(
        "--data-dir", default="./assets/datasets/",
        help="Folder where the .pkl files live"
    )
    args = parser.parse_args()

    for ds in args.datasets:
        print(f"\n=== Dataset: {ds} ===")
        data = load_dataset(ds, args.data_dir)
        fps, updrs, meds, others, poses = extract_info(data)

        print_summary(data, fps, updrs, meds, others, poses)



if __name__ == "__main__":
    main()
