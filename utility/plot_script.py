import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import torch
from textwrap import wrap
import imageio
import io


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def plot_2d_pose(pose, pose_tree, class_type, save_path=None, excluded_joints=None):
    def init():
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(class_type)

    fig = plt.figure()
    init()
    data = np.array(pose, dtype=float)

    if excluded_joints is None:
        plt.scatter(data[:, 0], data[:, 1], color='b', marker='h', s=15)
    else:
        plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
        plt.scatter(data[plot_joints, 0], data[plot_joints, 1], color='b', marker='h', s=15)

    for idx1, idx2 in pose_tree:
        plt.plot([data[idx1, 0], data[idx2, 0]],
                [data[idx1, 1], data[idx2, 1]], color='r', linewidth=2.0)

    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


# def draw_pose_from_cords(img_mat_size, pose_2d, kinematic_tree, radius=2):
#     img = np.zeros(shape=img_mat_size + (3,), dtype=np.uint8)
#     lw = 2
#     pose = pose_2d.astype(np.int32)
#     for i, (idx1, idx2) in enumerate(kinematic_tree):
#         cv2.line(img, (pose[idx1, 0], pose[idx1, 1]), (pose[idx2, 0], pose[idx2, 1]), (255, 255, 255), lw)
#
#     for i, uv in enumerate(pose_2d):
#         point = tuple(uv.astype(np.int32))
#         cv2.circle(img, point, radius, COLORS[i % len(COLORS)], -1)
#     return img

def plot_3d_pose_v2(savePath, kinematic_tree, joints, title=None):
    figure = plt.figure()
    # ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), projection='3d')
    ax = Axes3D(figure)
#     ax.set_ylim(-1, 1)
#     ax.set_xlim(-1, 1)
#     ax.set_zlim(-1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=110, azim=90)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    for chain, color in zip(kinematic_tree, colors):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=2.0, color=color)
#     ax.set_aspect(1)
# #     plt.axis('off')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#     plt.savefig(savePath)
    plt.show()

def plot_3d_motion_v2(motion, kinematic_tree, save_path, interval=50, dataset=None, title=None):
#     matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(0, 800)
        ax.set_xlim(0, 800)
        ax.set_zlim(0, 5000)
        # ax.set_ylim(-0.75, 0.75)
        # ax.set_xlim(-0.75, 0.75)
        # ax.set_zlim(-0.75, 0.75)
        if title is not None:
            ax.set_title(title)

    motion = motion.reshape(motion.shape[0], -1, 3)
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'green', 'blue','red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        ax.view_init(elev=110, azim=-90)
        ax.scatter(motion[index, :, 0], motion[index, :, 1], motion[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=2.0, color=color)
#         ax.set_aspect('equal')
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


# radius = 10*offsets
def plot_3d_motion_kit(save_path, kinematic_tree, joints, title, figsize=(5, 5), interval=100, radius=246 * 12):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'magenta', 'black', 'green', 'blue', 'red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=2.0, color=color)
        #         print(trajec[:index, 0].shape)
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)

    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_motion_gt_pred(save_path, kinematic_tree, gt_joints, pred_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        for ax in axs:
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            fig.suptitle(title, fontsize=20)
            ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz, ax):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    def update(index):
        for i, ax in enumerate(axs):
            for line in ax.lines:
                line.remove()
            for collection in ax.collections:
                collection.remove()
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5

            MINS = motions_min[i]
            MAXS = motions_max[i]
            trajec = motions_traj[i]
            data = motions_data[i]

            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                        MAXS[2] - trajec[index, 1], ax)

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                        trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                        color='blue')

            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                        color=color)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    # (seq_len, joints_num, 3)
    
    motions_data = []
    motions_traj = []
    motions_min = []
    motions_max = []
    colors = ['red', 'blue', 'black', 'red', 'blue',
                'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    for i, joints in enumerate((gt_joints, pred_joints)):
        data = joints.copy().reshape(len(joints), -1, 3)
        frame_number = data.shape[0]

        MINS = data.min(axis=0).min(axis=0)
        motions_min.append(MINS)
        MAXS = data.max(axis=0).max(axis=0)
        motions_max.append(MAXS)
        height_offset = MINS[1]

        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        motions_traj.append(trajec)

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        motions_data.append(data)

    axs = []
    fig = plt.figure(figsize=(20,10))
    axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
    axs.append(fig.add_subplot(1, 2, 2, projection='3d'))
    init()

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()
    
        
def plot_3d_motion_tmp(out_name, kinematic_chain, joints, title,footcontacts=None, figsize=(10, 10), fps=120, radius=4, foot_contacts = None):
    """
    Plot 3D motion data of SMPL-H model.

    Parameters:
    - joints (numpy array): 3D motion data of joints, shape (frame_number, joint_number, 3).
    - out_name (str or None): If specified, save the plot to this file path. If None, return the plot as a tensor.
    - title (str or None): Title of the plot.
    - kinematic_chain (list): List defining the kinematic chain of joints.
    - figsize (tuple): Size of the figure in inches.
    - fps (int): Frames per second for the animation.
    - radius (int): Radius of the joint markers.

    Returns:
    - If out_name is None, returns the plot as a tensor.
    - If out_name is specified, saves the plot and returns None.
    """
    matplotlib.use('Agg')
    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    smpl_kinetic_chain = kinematic_chain

    limits = 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    if footcontacts is not None:
        left_contact = footcontacts['left']
        right_contact = footcontacts['right']

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            # Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(10, 10), dpi=96)
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        init()

        # ax.lines = []
        ax.clear()
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        
        #plot foot contacts
        if index < frame_number - 1:  
            if footcontacts is not None:
                if left_contact[index][0]:
                    ax.scatter(data[index, 7, 0], data[index, 7, 1], data[index, 7, 2], c='red', s=100, marker='x')
                if left_contact[index][1]:
                    ax.scatter(data[index, 10, 0], data[index, 10, 1], data[index, 10, 2], c='red', s=100, marker='*')
                if right_contact[index][0]:
                    ax.scatter(data[index, 8, 0], data[index, 8, 1], data[index, 8, 2], c='green', s=100, marker='x')
                if right_contact[index][1]:
                    ax.scatter(data[index, 11, 0], data[index, 11, 1], data[index, 11, 2], c='green', s=100, marker='*')
                    
        aa1 = [data[index, 15, 0]-1, data[index, 15, 1], data[index, 15, 2]]
        ax.plot3D([aa1[0],aa1[0]], [aa1[1],aa1[1]], [aa1[2], aa1[2]+0.5], linewidth=2,color='g')
        ax.quiver(aa1[0], aa1[1], aa1[2]+0.5, 0, 0, 0.1, color='g', arrow_length_ratio=0.8)
        ax.text(aa1[0], aa1[1], aa1[2]+0.5, '+Z', color='g')
        
        ax.plot3D([aa1[0],aa1[0]], [aa1[1],aa1[1]+0.1], [aa1[2], aa1[2]], linewidth=2,color='b')
        # ax.quiver(aa1[0], aa1[1]+0.5, aa1[2], 0, 0.1, 0, color='b', arrow_length_ratio=0.8)
        ax.text(aa1[0], aa1[1]+0.1, aa1[2], '+Y', color='b')
        
        ax.plot3D([aa1[0],aa1[0]+0.1], [aa1[1],aa1[1]], [aa1[2], aa1[2]], linewidth=2,color='r')
        # ax.quiver(aa1[0]+0.5, aa1[1], aa1[2], 0.1, 0, 0, color='r', arrow_length_ratio=0.8)
        ax.text(aa1[0]+0.1, aa1[1], aa1[2], '+X', color='r')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    out = torch.from_numpy(out)
    imageio.mimsave(out_name, np.array(out), duration=1) 

    return 



def plot_3d_motion_prev(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    frames_num =joints.shape[0]

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    title = title + f' - {frames_num} frames'
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        # for line in ax.lines:
        #     line.remove()
        # for collection in ax.collections:
        #     collection.remove()
        ax.clear()

        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)

    plt.close()
    
def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')
    frames_num =joints.shape[0]

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    title = title + f' - {frames_num} frames'

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        
    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    min_x, min_y, min_z = np.min(data, axis=(0, 1))
    max_x, max_y, max_z = np.max(data, axis=(0, 1))
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    ax.set_xlim3d([min_x, max_x])
    ax.set_ylim3d([min_y, max_y])
    ax.set_zlim3d([min_z, max_z])
    
    x, y, z = data[..., 0], data[..., 1], data[..., 2]
    ax.set_box_aspect([np.ptp(a) for a in [x, y, z]]) 

    def update(index):
        ax.clear()
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])

        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]),
                    trajec[:index, 1], linewidth=1.0, color='blue')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()

def plot_3d_motion_overlaid(save_path, kinematic_tree, joints1, joints2, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = np.min([data1.min(axis=0).min(axis=0), data2.min(axis=0).min(axis=0)], axis=0)
    MAXS = np.max([data1.max(axis=0).max(axis=0), data2.max(axis=0).max(axis=0)], axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data1.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data1[:, :, 1] -= height_offset
    data2[:, :, 1] -= height_offset
    trajec1 = data1[:, 0, [0, 2]]
    trajec2 = data2[:, 0, [0, 2]]

    data1[..., 0] -= data1[:, 0:1, 0]
    data1[..., 2] -= data1[:, 0:1, 2]
    data2[..., 0] -= data2[:, 0:1, 0]
    data2[..., 2] -= data2[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        plot_xzPlane(MINS[0] - trajec1[index, 0], MAXS[0] - trajec1[index, 0], 0, MINS[2] - trajec1[index, 1],
                     MAXS[2] - trajec1[index, 1])
        plot_xzPlane(MINS[0] - trajec2[index, 0], MAXS[0] - trajec2[index, 0], 0, MINS[2] - trajec2[index, 1],
                     MAXS[2] - trajec2[index, 1])

        if index > 1:
            ax.plot3D(trajec1[:index, 0] - trajec1[index, 0], np.zeros_like(trajec1[:index, 0]),
                      trajec1[:index, 1] - trajec1[index, 1], linewidth=1.0, color='blue')
            ax.plot3D(trajec2[:index, 0] - trajec2[index, 0], np.zeros_like(trajec2[:index, 0]),
                      trajec2[:index, 1] - trajec2[index, 1], linewidth=1.0, color='red')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth,
                      color='blue')
            ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth,
                      color='red')


        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_overlaid2(save_path, kinematic_tree, joints1, joints2, title, figsize=(10, 10), fps=120, radius=4):
    """
    Plot 3D motion data of SMPL-H model.

    Parameters:
    - joints (numpy array): 3D motion data of joints, shape (frame_number, joint_number, 3).
    - out_name (str or None): If specified, save the plot to this file path. If None, return the plot as a tensor.
    - title (str or None): Title of the plot.
    - kinematic_chain (list): List defining the kinematic chain of joints.
    - figsize (tuple): Size of the figure in inches.
    - fps (int): Frames per second for the animation.
    - radius (int): Radius of the joint markers.

    Returns:
    - If out_name is None, returns the plot as a tensor.
    - If out_name is specified, saves the plot and returns None.
    """
    matplotlib.use('Agg')
    from textwrap import wrap
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)

    nb_joints = joints1.shape[1]
    smpl_kinetic_chain = kinematic_tree

    limits = 2
    MINS = data1.min(axis=0).min(axis=0)
    MAXS = data1.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data1.shape[0]

    height_offset = MINS[1]
    data1[:, :, 1] -= height_offset
    trajec1 = data1[:, 0, [0, 2]]
    data2[:, :, 1] -= height_offset
    trajec2 = data2[:, 0, [0, 2]]

    data1[..., 0] -= data1[:, 0:1, 0]
    data1[..., 2] -= data1[:, 0:1, 2]
    data2[..., 0] -= data2[:, 0:1, 0]
    data2[..., 2] -= data2[:, 0:1, 2]
    
    fig = plt.figure(figsize=(10, 10), dpi=96)
    if title is not None:
        wrapped_title = '\n'.join(wrap(title, 40))
        fig.suptitle(wrapped_title, fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    
    def init():
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(0, limits)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def update(index):
        ax.clear()
        init()


        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(MINS[0] - trajec1[index, 0], MAXS[0] - trajec1[index, 0], 0, MINS[2] - trajec1[index, 1],
                     MAXS[2] - trajec1[index, 1])
        plot_xzPlane(MINS[0] - trajec2[index, 0], MAXS[0] - trajec2[index, 0], 0, MINS[2] - trajec2[index, 1],
                     MAXS[2] - trajec2[index, 1])

        if index > 1:
            ax.plot3D(trajec1[:index, 0] - trajec1[index, 0], np.zeros_like(trajec1[:index, 0]),
                      trajec1[:index, 1] - trajec1[index, 1], linewidth=1.0,
                      color='blue')
            ax.plot3D(trajec2[:index, 0] - trajec2[index, 0], np.zeros_like(trajec2[:index, 0]),
                      trajec2[:index, 1] - trajec2[index, 1], linewidth=1.0,
                      color='red')

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth,
                      color='blue')
            ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth,
                      color='red')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_old(motion, pose_tree, class_type, save_path, interval=300, excluded_joints=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlim(-0.75, 0.75)
        ax.set_zlim(-0.75, 0.75)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)
        ax.set_title(class_type)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        if excluded_joints is None:
            ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='b', marker='h', s=15)
        else:
            plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
            ax.scatter(data[index, plot_joints, 0], data[index, plot_joints, 1], data[index, plot_joints, 2], color='b', marker='h', s=15)

        for idx1, idx2 in pose_tree:
            ax.plot([data[index, idx1, 0], data[index, idx2, 0]],
                    [data[index, idx1, 1], data[index, idx2, 1]], [data[index, idx1, 2], data[index, idx2, 2]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


def plot_2d_motion(motion, pose_tree, axis_0, axis_1, class_type, save_path, interval=300):
    matplotlib.use('Agg')

    fig = plt.figure()
    plt.title(class_type)
    # ax = fig.add_subplot(111, projection='3d')
    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        plt.clf()
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        plt.scatter(data[index, :, axis_0], data[index, :, axis_1], color='b', marker='h', s=15)
        for idx1, idx2 in pose_tree:
            plt.plot([data[index, idx1, axis_0], data[index, idx2, axis_0]],
                    [data[index, idx1, axis_1], data[index, idx2, axis_1]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_multi_motion(motion_list, kinematic_tree, save_path, interval=50, dataset=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    frame_number = motion_list[0].shape[0]
    # dim (frame, joints, xyz)
    # print(data.shape)
    print("Number of motions %d" % (len(motion_list)))
    def update(index):
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for motion in motion_list:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2],
                          linewidth=4.0, color=color)
        plt.axis('off')

#         ax.set_xticks([])
#         ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()



