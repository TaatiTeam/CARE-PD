import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches


h36m_joint_paths = [
    [10, 9, 8, 7, 0, 1, 2, 3],
    [0, 4, 5, 6],
    [8, 11, 12, 13],
    [8, 14, 15, 16]
]

SMPL_joint_paths = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21, 23],
    [9, 13, 16, 18, 20, 22]
]

NTU_joint_paths = [
    [18, 16, 0, 15, 17],
    [0,1,8],
    [9,8,12],
    [12,13,14,19],
    [14,20],
    [14,21],
    [9,10,11,22],
    [11,23],
    [11,24],
    [4,3,2,1,5,6,7]
]


AMASS_joint_paths = [
    [0,2,5,8,11],
    [0,1,4,7,10],
    [0,3,6,9,12,15],
    [9,13,16,18,20],
    [9,14,17,19,21]
]


class PauseAnimation:
    def __init__(self, update, fargs, save_gif, projection='3d', interval=1):
        self.projection = projection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection if projection=='3d' else None)

        fargs = (ax,) + fargs
        self.fargs = fargs
        seqs = self.fargs[10] if projection=='3d' else self.fargs[6]
        if isinstance(seqs, list):
            self.num_frames = np.max([s.shape[0] for s in seqs])
        else:
            self.num_frames = seqs.shape[0] if projection=='3d' else seqs.shape[0]
        self.paused = False
        self.animation = FuncAnimation(fig, update, frames=self.num_frames, fargs=self.fargs, interval=interval)
        if save_gif: 
            self.save_gif()
            return
        
        #fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        fig.canvas.mpl_connect('key_press_event', self.restart_animation_or_toggle_pause)
        plt.show()

    def save_gif(self):
        name = self.fargs[9].split('from')[0] if self.projection=='3d' else self.fargs[5].split('from')[0]
        print(f'Saving animation for {name}')
        self.animation.save(f'{name}.gif', writer='pillow')

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def restart_animation_or_toggle_pause(self, event):
        if event.key == 'r':
            #print('Anim restarted')
            if self.paused: self.paused = not self.paused
            self.animation.event_source.stop()
            self.animation.frame_seq = self.animation.new_frame_seq()  # Reset the frame sequence
            self.animation.event_source.start()
        elif event.key == ' ': # Space was pressed
            #print(f'P pressed, {self.paused}')
            if self.paused:
                self.animation.resume()
            else:
                self.animation.pause()
            self.paused = not self.paused
        elif event.key == 'x':  # Check if the 's' key was pressed
            if self.paused: self.paused = not self.paused
            current_frame = self.animation.frame_seq.__next__()
            new_frame = current_frame + 50
            if new_frame >= self.num_frames: new_frame = self.num_frames - 1
            self.animation.event_source.stop()
            self.animation.frame_seq = iter(range(new_frame, self.num_frames))  # Fast forward 50 frames
            self.animation.event_source.start()
        elif event.key == 'z':  # Check if the 's' key was pressed
            if self.paused: self.paused = not self.paused
            current_frame = self.animation.frame_seq.__next__()
            new_frame = current_frame - 50
            if new_frame < 0: new_frame = 0
            self.animation.event_source.stop()
            self.animation.frame_seq = iter(range(new_frame, self.num_frames))  # Fast forward 50 frames
            self.animation.event_source.start()
        
def visualize_sequence(seq, name, show_joint_indexes=False, joint_paths=None, frame_offset=0, save_gif=False, projection='3d', fps=None, invert=None, minmax = None):
    if projection=='3d':
        visualize_sequence_3d(seq, name, show_joint_indexes, joint_paths, frame_offset, save_gif, fps)
    elif projection=='2d':
        visualize_sequence_2d(seq, name, show_joint_indexes, joint_paths, frame_offset, save_gif, fps, invert, minmax=minmax)



def visualize_sequence_3d(seq, name, show_joint_indexes=False, joint_paths=None, frame_offset=0, save_gif=False, fps=None):
    print(f'Visualizing {name}, has {seq.shape[0]} frames in total')
    VIEWS = {
        "pd": {
            "best": (0, 0)
        }
    }

    def update(frame,
               ax,
               min_x,
               max_x,
               min_y,
               max_y,
               min_z,
               max_z,
               VIEWS,
               aspect_ratio,
               name,
               seq,
               joint_paths,
               show_joint_indexes,
               frame_offset):
        ax.clear()

        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        # print(VIEWS[data_type][view_type])
        # ax.view_init(*VIEWS[data_type][view_type])
        elev, azim = VIEWS["pd"]["best"]
        #ax.view_init(elev=elev, azim=azim)
        name_nopth = name.split('/')[-1]
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(f'Frame: {frame+frame_offset}/{seq.shape[0]}\nseq:{name_nopth}')

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        if joint_paths is None: ax.scatter(x, y, z)

        if show_joint_indexes:
            for i in range(seq.shape[1]):
                ax.text(x[i], y[i], z[i], i, None, fontsize='xx-small')

        if joint_paths:
            for joint_path in joint_paths:
                joint_path_coords = [seq[frame, joint, :] for joint in joint_path]
                x = [coord[0] for coord in joint_path_coords]
                y = [coord[1] for coord in joint_path_coords]
                z = [coord[2] for coord in joint_path_coords]
                ax.plot(x,y,z,color = 'g')

    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]

    fargs = (min_x, 
             max_x,
             min_y,
             max_y,
             min_z,
             max_z,
             VIEWS,
             aspect_ratio,
             name,
             seq,
             joint_paths,
             show_joint_indexes,
             frame_offset)
    
    if fps is not None:
        interval = int(1/fps*1000)
    else:
        interval = 1
        
    print(f'Interval: {interval}')

    # create the animation
    #ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=1)
    PauseAnimation(update, fargs, save_gif, interval=interval)


def visualize_overlaid_sequences_3d(seqs, name, show_joint_indexes=False, joint_paths=None, frame_offset=0, save_gif=False, elev=0, azim=0, roll=0):
    print(f'Visualizing {name}, sequenecs have {[seq.shape[0] for seq in seqs]} frames in total')
    VIEWS = {
        "best": (elev, azim, roll)
    }
    colors = [plt.cm.tab20(i) for i in range(20)]
    colors_dataset_consistency_experiemnt = {
        0: colors[0],
        1: colors[2]
    }
    def update(frame,
               ax,
               min_x,
               max_x,
               min_y,
               max_y,
               min_z,
               max_z,
               VIEWS,
               aspect_ratio,
               name,
               seqs,
               joint_paths,
               show_joint_indexes,
               frame_offset):
        ax.clear()

        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        if save_gif:
            elev, azim, roll = VIEWS["best"]
            ax.view_init(elev=elev, azim=azim,  roll=roll)
        name_nopth = name.split('/')[-1]
        ax.set_box_aspect(aspect_ratio)
        ax.set_title(f'Frame: {frame+frame_offset}\nseq:{name_nopth}')

        legend_patches = []
        labels = {0: 'H36M', 1: 'After canolicalization'}
        for i,seq in enumerate(seqs):

            joint_paths_i = joint_paths[i]

            x = seq[frame, :, 0]
            y = seq[frame, :, 1]
            z = seq[frame, :, 2]
    
            if joint_paths_i is None: ax.scatter(x, y, z, c=colors[i % 20])
    
            if show_joint_indexes:
                for i in range(seq.shape[1]):
                    ax.text(x[i], y[i], z[i], i, None, fontsize='xx-small')
    
            if joint_paths_i:
                for joint_path in joint_paths_i:
                    joint_path_coords = [seq[frame, joint, :] for joint in joint_path]
                    x = [coord[0] for coord in joint_path_coords]
                    y = [coord[1] for coord in joint_path_coords]
                    z = [coord[2] for coord in joint_path_coords]
                    ax.plot(x,y,z,color = colors_dataset_consistency_experiemnt[i])

                patch = mpatches.Patch(color=colors_dataset_consistency_experiemnt[i], label=labels[i])
                legend_patches.append(patch)
                
        plt.legend(handles=legend_patches)

    max_len = np.max([s.shape[0] for s in seqs])
    seqs = [np.pad(s, pad_width=((0, max_len-s.shape[0]), (0,0), (0,0))) for s in seqs]


    seqs_concat = np.concatenate([s.reshape(-1, 3) for s in seqs])

    min_x, min_y, min_z = np.min(seqs_concat, axis=(0,))
    max_x, max_y, max_z = np.max(seqs_concat, axis=(0,))

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]

    fargs = (min_x, 
             max_x,
             min_y,
             max_y,
             min_z,
             max_z,
             VIEWS,
             aspect_ratio,
             name,
             seqs,
             joint_paths,
             show_joint_indexes,
             frame_offset)

    # create the animation
    #ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=1)
    PauseAnimation(update, fargs, save_gif)


def visualize_sequence_2d(seq, name, show_joint_indexes=False, joint_paths=None, frame_offset=0, save_gif=False, fps=None, invert=None, minmax=None):
    print(f'Visualizing {name}, has {seq.shape[0]} frames in total')
    def update(frame,
               ax,
               min_x,
               max_x,
               min_y,
               max_y,
               name,
               seq,
               joint_paths,
               show_joint_indexes,
               frame_offset,
               aspect_ratio):
        ax.clear()

        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_box_aspect(aspect_ratio)

        name_nopth = name.split('/')[-1]
        ax.set_title(f'Frame: {frame+frame_offset}/{seq.shape[0]}\nseq:{name_nopth}')

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]

        if joint_paths is None: ax.scatter(x, y)

        if show_joint_indexes:
            for i in range(seq.shape[1]):
                ax.text(x[i], y[i], i, None, fontsize='xx-small')

        if joint_paths:
            for joint_path in joint_paths:
                joint_path_coords = [seq[frame, joint, :] for joint in joint_path]
                x = [coord[0] for coord in joint_path_coords]
                y = [coord[1] for coord in joint_path_coords]
                ax.plot(x,y,color = 'g')
        
        if invert:       
            ax.invert_yaxis()

    if minmax is not None:
        min_x = minmax[0]
        max_x = minmax[1]
        min_y = minmax[2]
        max_y = minmax[3]
        print(f'Range: {minmax}')
    else:
        min_x, min_y = np.min(seq, axis=(0, 1))
        max_x, max_y = np.max(seq, axis=(0, 1))

    x_range = max_x - min_x
    y_range = max_y - min_y
    aspect_ratio = y_range / x_range 

    fargs = (min_x, 
             max_x,
             min_y,
             max_y,
             name,
             seq,
             joint_paths,
             show_joint_indexes,
             frame_offset, 
             aspect_ratio)
    
    if fps is not None:
        interval = int(1/fps*1000)
    else:
        interval = 1

    # create the animation
    #ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=1)
    PauseAnimation(update, fargs, save_gif, projection='2d', interval=interval)
    