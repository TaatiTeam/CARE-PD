import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import os
import argparse
import pickle
from pathlib import Path
from visualize_skel_walk_func import visualize_sequence, h36m_joint_paths, SMPL_joint_paths, NTU_joint_paths, AMASS_joint_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npzf')
    parser.add_argument('-b', '--binary')
    parser.add_argument('-npy', '--npypath')
    parser.add_argument('-f', '--format')
    parser.add_argument('-fps', '--fps', default=30, type=int)
    parser.add_argument('-p', '--projection', default='3d', type=str)
    args = parser.parse_args()
    print(args)
    
    if args.npzf:
        seqs = np.load(args.npzf)
        fname = Path(args.npzf).name
    elif args.binary:
        pickle_data = pickle.load(open(args.binary, 'rb'))
        seqs = {seq_name: pickle_data['pose'][i] for i,seq_name in enumerate(pickle_data['video_name'])}
        fname = Path(args.binary).name
    elif args.npypath:
        seqs = {}
        for seq_name in os.listdir(args.npypath):
            seq = np.load(os.path.join(args.npypath, seq_name))
            seqs[seq_name] = seq
        fname = Path(args.npypath).name
    else:
        raise NotImplementedError('Must supply either -b or -n option as source file path')
    print(f'There are {len(seqs)} sequences.')
    print(f'The average number of frames per clip is {np.mean([len(seqs[x]) for x in seqs])}')


    for name in seqs.keys():
        if name.endswith("_frame_ids"): continue
        seq = seqs[name]
        print(name)
        joint_paths = {
            'h36m': h36m_joint_paths,
            'SMPL': SMPL_joint_paths,
            'NTU': NTU_joint_paths,
            'AMASS': AMASS_joint_paths
        }
        skel_format = joint_paths[args.format]
        if fname == 'h36m_3d_world_30f_or_longer.npz' and args.projection == '2d':
            seq = seq[:, :, :2]
        if args.projection == '2d':
            invert = True
            minmax = [0, 1000, 0, 1000]
        else:
            invert = None
            minmax = None
        visualize_sequence(seq, name + f'\n from {fname}', show_joint_indexes=True, joint_paths=skel_format, projection=args.projection, fps=args.fps, invert=invert, minmax=minmax, save_gif=False)
        
        
    
if __name__ == '__main__':
    main()

# python utility/viz_seqs.py -n assets/datasets/h36m/BMCLab/h36m_3d_world2cam2img_sideright_floorXZZplus_30f_or_longer.npz  -f h36m -p 2d

# python utility/viz_seqs.py -n assets/datasets/h36m/BMCLab/h36m_3d_world_floorXZZplus_30f_or_longer.npz  -f h36m