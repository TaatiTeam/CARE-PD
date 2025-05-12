import sys
import argparse
import os

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_path + "/../")

from configs import generate_config_potr
from const import path
from data.dataloaders import *
from model.backbone_loader import load_pretrained_backbone
from configs import generate_config_motionbert, generate_config_poseformerv2, generate_config_mixste, generate_config_motionagformer

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIEWS = {
    "pd": {
        "best": (45, 20, 100),
        "best2": (0, 0, 0),
        "side": (90, 0, 90),
    },
    "tmp": {
        "best": (45, 20, 100),
        "side": (90, 0, 90),
    }
}

H36M_FULL = {
    'B.TORSO': 0,
    'L.HIP': 1,
    'L.KNEE': 2,
    'L.FOOT': 3,
    'R.HIP': 4,
    'R.KNEE': 5,
    'R.FOOT': 6,
    'C.TORSO': 7,
    'U.TORSO': 8,
    'NECK': 9,
    'HEAD': 10,
    'R.SHOULDER': 11,
    'R.ELBOW': 12,
    'R.HAND': 13,
    'L.SHOULDER': 14,
    'L.ELBOW': 15,
    'L.HAND': 16
}

H36M_CONNECTIONS_FULL = {
    (H36M_FULL['B.TORSO'], H36M_FULL['L.HIP']),
    (H36M_FULL['B.TORSO'], H36M_FULL['R.HIP']),
    (H36M_FULL['R.HIP'], H36M_FULL['R.KNEE']),
    (H36M_FULL['R.KNEE'], H36M_FULL['R.FOOT']),
    (H36M_FULL['L.HIP'], H36M_FULL['L.KNEE']),
    (H36M_FULL['L.KNEE'], H36M_FULL['L.FOOT']),
    (H36M_FULL['B.TORSO'], H36M_FULL['C.TORSO']),
    (H36M_FULL['C.TORSO'], H36M_FULL['U.TORSO']),
    (H36M_FULL['U.TORSO'], H36M_FULL['L.SHOULDER']),
    (H36M_FULL['L.SHOULDER'], H36M_FULL['L.ELBOW']),
    (H36M_FULL['L.ELBOW'], H36M_FULL['L.HAND']),
    (H36M_FULL['U.TORSO'], H36M_FULL['R.SHOULDER']),
    (H36M_FULL['R.SHOULDER'], H36M_FULL['R.ELBOW']),
    (H36M_FULL['R.ELBOW'], H36M_FULL['R.HAND']),
    (H36M_FULL['U.TORSO'], H36M_FULL['NECK']),
    (H36M_FULL['NECK'], H36M_FULL['HEAD'])
}

def rotate_around_z_axis(points, theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(points, R.T)


def visualize_sequence(seq, name):
    VIEWS = {
    "pd": {
        "best": (45, 20, 100),
        "best2": (0, 0, 0),
        "side": (90, 0, 90),
    },
    "tmp": {
        "best": (45, 20, 100),
        "side": (90, 0, 90),
    }
}
    elev, azim, roll = VIEWS["pd"]["side"]
    # Apply the rotation to each point in the sequence
    for i in range(seq.shape[1]):
        seq[:, i, :] = rotate_around_z_axis(seq[:, i, :], roll)

    def update(frame):
        ax.clear()

        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])

        # print(VIEWS[data_type][view_type])
        # ax.view_init(*VIEWS[data_type][view_type])
        elev, azim, roll = VIEWS["pd"]["best"]
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(f'Frame: {frame}')

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        for connection in H36M_CONNECTIONS_FULL:
            start = seq[frame, connection[0], :]
            end = seq[frame, connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]

            ax.plot(xs, ys, zs)
        ax.scatter(x, y, z)

    
    print(f"Number of frames: {seq.shape[0]}")

    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create the animation
    ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=1)
    ani.save(f'{name}.gif', writer='pillow')
    
    plt.close(fig)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='motionbert', help='model name ( potr, '
                                                                           'motionbert )')
    parser.add_argument('--train_mode', type=str, default='classifier_only', help='train mode( end2end, classifier_only )')
    parser.add_argument('--dataset', type=str, default='PD',
                        help='dataset ( TRI, Lakeside, PD) **currently code only works for TRI')
    parser.add_argument('--data_path', type=str,
                        default=path.PD_PATH_POSES)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int, help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--ntrials', default=30, type=int, help='number of hyper-param tuning trials')
    parser.add_argument('--last_run_foldnum', type=str, default='1')
    parser.add_argument('--readstudyfrom', default=5, type=int)
    
    parser.add_argument('--hypertune', default=1, type=int, help='perform hyper parameter tuning [0 or 1]')

    args = parser.parse_args()

    param = vars(args)

    backbone_name = param['backbone']
    
    if backbone_name == 'potr':
        conf_path = './configs/potr/'
    elif backbone_name == 'motionbert':
        conf_path = './configs/motionbert/'
    elif backbone_name == 'poseformerv2':
        conf_path = "./configs/poseformerv2"
    elif backbone_name == 'mixste':
        conf_path = "./configs/mixste",
    elif backbone_name == 'motionagformer':
        conf_path = "./configs/motionagformer"
    else:
        raise NotImplementedError(f"Backbone '{backbone_name}' is not supported")
    
    for fi in sorted(os.listdir(conf_path)):
        if backbone_name == 'potr':
            params, new_params = generate_config_potr.generate_config(param, fi)
        elif backbone_name == 'motionbert':
            params, new_params = generate_config_motionbert.generate_config(param, fi)
        elif backbone_name == "poseformerv2":
            params, new_params = generate_config_poseformerv2.generate_config(param, fi)
        elif backbone_name == "mixste":
            params, new_params = generate_config_mixste.generate_config(param, fi)
        elif backbone_name == "motionagformer":
            params, new_params = generate_config_motionagformer.generate_config(param, fi)
        else:
            raise NotImplementedError(f"Backbone '{param['backbone']}' does not exist.")
            
        train_dataset, eval_dataset = dataset_factory(params, backbone_name, 1)
        train_dataset_fn, _, class_weights = dataloader_factory(params, train_dataset, eval_dataset)
        
        params['input_dim'] = train_dataset_fn.dataset._pose_dim
        params['pose_dim'] = train_dataset_fn.dataset._pose_dim
        params['num_joints'] = train_dataset_fn.dataset._NMAJOR_JOINTS


        model_backbone = load_pretrained_backbone(params, backbone_name)
        model_backbone = model_backbone.to(_DEVICE)
        for param in model_backbone.parameters():
            param.requires_grad = False
        print("[INFO - MotionEncoder] Backbone parameters are frozen")
        
        
        
        for x, _, video_idx in train_dataset_fn:
            x = x.to(_DEVICE)

            batch_size = x.shape[0]

            pose3D = model_backbone(x, return_rep=False)
            pose3D = pose3D.cpu().numpy()
            for b in range(batch_size):
                visualize_sequence(pose3D[b,:,:,:], f'./data/pd/pd_reconst/video{video_idx[b].cpu().numpy()}')
                ppp=1


                