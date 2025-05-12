from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import numpy as np


def detect_standing_still(seq):
    sacrum_locs = seq[:, 0, :]
    sdiff = np.diff(sacrum_locs, axis=0)
    sdiff = np.pad(sdiff, ((0,1),(0,0)), mode='edge') # repeat last value of sdiff one more time so it's the same size as seq
    sdiff = np.linalg.norm(sdiff, axis=1)
    standing_still_mask = sdiff < 0.007 # I found this threshold from watching the values of real sequences
    return standing_still_mask

def get_height_vec_estimate_h36m(pose):
    lank = pose[6, :]
    rank = pose[3, :]
    lknee = pose[5, :]
    rknee = pose[2, :]
    height_vector = pose[0, :] - np.mean([lank, rank, lknee, rknee], axis=0)
    return height_vector

def get_height_vec_estimate_AMASS(pose):
    lank = np.mean([pose[7, :], pose[10, :]], axis=0)
    rank = np.mean([pose[8, :], pose[11, :]], axis=0)
    lknee = pose[4, :]
    rknee = pose[5, :]
    height_vector = pose[0, :] - np.mean([lank, rank, lknee, rknee], axis=0)
    return height_vector

def get_hip_vec_estimate_h36m(pose):
    return pose[1, :] - pose[4, :]

def get_hip_vec_estimate_AMASS(pose):
    return pose[2, :] - pose[1, :]

def get_perpendicular_in_dir_of_nose(vec1, vec2, pose):
    res = np.cross(vec1,vec2)
    dir_of_nose = pose[9, :] - np.mean([1.5*pose[8, :], 0.5*pose[10, :]], axis=0)
    if np.dot(res, dir_of_nose) < 0: res *= -1
    return res / np.linalg.norm(res) # vector of length 1

def get_perpendicular_in_dir_of_toes(vec1, vec2, pose):
    res = np.cross(vec1,vec2)
    dir_of_right_toes = pose[11, :] - pose[8, :]
    dir_of_left_toes = pose[10, :] - pose[7, :]
    dir_of_toes = np.mean([dir_of_right_toes, dir_of_left_toes], axis=0)
    if np.dot(res, dir_of_toes) < 0: res *= -1
    return res / np.linalg.norm(res) # vector of length 1


def get_rotation_matrix(vec1, vec2=np.array(([[1, 0, 0], [0, 0, 1], [0, 1, 0]])), weights=[0.8, 1, 0.6]):
    """get rotation matrix between two sets of vectors using scipy vector sets of shape N x 3"""
    r = R.align_vectors(vec2, vec1, weights=weights)
    return r[0].as_matrix()

def transform_seq_so_it_has_straight_trajectory_h36m(seq, n_frames_est_mov_dir, window_size, polynomial):
    """ Assumes y axis roughly corressponds to height (movement in xz plane is used for trajectory reconstruction """
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, 0, :]
    
    hip_vectors    = []
    height_vectors = []
    mov_dir_ests2  = []
    for pose in centered_seq:
        hip_vector     = get_hip_vec_estimate_h36m(pose)
        height_vector  = get_height_vec_estimate_h36m(pose)
        mov_dir_est2   = get_perpendicular_in_dir_of_nose(hip_vector, height_vector, pose)
        hip_vectors.append(hip_vector)
        height_vectors.append(height_vector)
        mov_dir_ests2.append(mov_dir_est2)
    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests2 = np.stack(mov_dir_ests2)
    
    standing_still_mask = detect_standing_still(seq)
    sacrum_locs = seq[:, 0, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0,n),(0,0)), mode='edge') # Reapeat last value n times so it's the same len as seq
    for i,m_dir_est in enumerate(m_dir):
        if standing_still_mask[i] or \
           np.linalg.norm(m_dir[i]) < n*0.004 or \
           i >= len(m_dir) - n: 
            m_dir[i] = mov_dir_ests2[i]
        m_dir[i] = m_dir[i] / np.linalg.norm(m_dir[i])

    xhat = m_dir[:,0]
    yhat = m_dir[:,1]
    zhat = m_dir[:,2]
    if window_size is not None and polynomial is not None:
        window_size = np.min([window_size, m_dir.shape[0]])
        xhat = savgol_filter(m_dir[:,0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:,1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:,2], window_size, polynomial)

    movement_direction = np.swapaxes(np.stack([xhat, yhat, zhat]), 0, 1)
    sacrum_movements = np.diff(sacrum_locs, axis=0)
    sacrum_movements_xz = sacrum_movements[:, [0,2]]
    horizontal_movement = np.linalg.norm(sacrum_movements_xz, axis=1)

    centered_seq_rotated = []
    for i,mov_dir_vec in enumerate(movement_direction):
        rot_mat = get_rotation_matrix(np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]))
        rotated = np.stack([rot_mat.dot(joint) for joint in centered_seq[i]])
        rotated += np.array([np.sum(horizontal_movement[:i]), 0, 0])
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        centered_seq_rotated.append(rotated)
        
    centered_seq_rotated = np.stack(centered_seq_rotated)
    return centered_seq_rotated

def transform_seq_so_it_has_straight_trajectory_AMASS(seq, n_frames_est_mov_dir, window_size, polynomial):
    """ Assumes y axis roughly corressponds to height (movement in xz plane is used for trajectory reconstruction """
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, 0, :]
    
    hip_vectors    = []
    height_vectors = []
    mov_dir_ests2  = []
    for pose in centered_seq:
        hip_vector     = get_hip_vec_estimate_AMASS(pose)
        height_vector  = get_height_vec_estimate_AMASS(pose)
        mov_dir_est2   = get_perpendicular_in_dir_of_toes(hip_vector, height_vector, pose)
        hip_vectors.append(hip_vector)
        height_vectors.append(height_vector)
        mov_dir_ests2.append(mov_dir_est2)
    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests2 = np.stack(mov_dir_ests2)
    
    standing_still_mask = detect_standing_still(seq)
    sacrum_locs = seq[:, 0, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0,n),(0,0)), mode='edge') # Reapeat last value n times so it's the same len as seq
    for i,m_dir_est in enumerate(m_dir):
        if standing_still_mask[i] or \
           np.linalg.norm(m_dir[i]) < n*0.004 or \
           i >= len(m_dir) - n: 
            m_dir[i] = mov_dir_ests2[i]
        m_dir[i] = m_dir[i] / np.linalg.norm(m_dir[i])

    xhat = m_dir[:,0]
    yhat = m_dir[:,1]
    zhat = m_dir[:,2]
    if window_size is not None and polynomial is not None:
        window_size = np.min([window_size, m_dir.shape[0]])
        xhat = savgol_filter(m_dir[:,0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:,1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:,2], window_size, polynomial)

    movement_direction = np.swapaxes(np.stack([xhat, yhat, zhat]), 0, 1)
    sacrum_movements = np.diff(sacrum_locs, axis=0)
    sacrum_movements_xz = sacrum_movements[:, [0,2]]
    horizontal_movement = np.linalg.norm(sacrum_movements_xz, axis=1)

    centered_seq_rotated = []
    for i,mov_dir_vec in enumerate(movement_direction):
        rot_mat = get_rotation_matrix(np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]))
        rotated = np.stack([rot_mat.dot(joint) for joint in centered_seq[i]])
        rotated += np.array([np.sum(horizontal_movement[:i]), 0, 0])
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        centered_seq_rotated.append(rotated)
        
    centered_seq_rotated = np.stack(centered_seq_rotated)
    return centered_seq_rotated


def transform_seq_so_it_has_straight_trajectory(seq, n_frames_est_mov_dir=15, window_size=90, polynomial = 4, seq_format='h36m'): # supported format is also 'AMASS'
    if seq_format == 'h36m':
        return transform_seq_so_it_has_straight_trajectory_h36m(seq, n_frames_est_mov_dir, window_size, polynomial)
    elif seq_format == 'AMASS':
        return transform_seq_so_it_has_straight_trajectory_AMASS(seq, n_frames_est_mov_dir, window_size, polynomial)
    else:
        raise NotImplementedError(f'Unimplemented seq_format type {seq_format}')


def transform_seq_so_it_has_no_slope_AMASS(seq, n_frames_est_mov_dir=15, window_size=90, polynomial=4):
    """ Assumes y axis roughly corressponds to height (movement in xz plane is used for trajectory reconstruction """
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, 0, :]
    
    hip_vectors    = []
    height_vectors = []
    mov_dir_ests2  = []
    for pose in centered_seq:
        hip_vector     = get_hip_vec_estimate_AMASS(pose)
        height_vector  = get_height_vec_estimate_AMASS(pose)
        mov_dir_est2   = get_perpendicular_in_dir_of_toes(hip_vector, height_vector, pose)
        hip_vectors.append(hip_vector)
        height_vectors.append(height_vector)
        mov_dir_ests2.append(mov_dir_est2)
    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests2 = np.stack(mov_dir_ests2)
    
    standing_still_mask = detect_standing_still(seq)
    sacrum_locs = seq[:, 0, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0,n),(0,0)), mode='edge') # Reapeat last value n times so it's the same len as seq
    for i,m_dir_est in enumerate(m_dir):
        if standing_still_mask[i] or \
           np.linalg.norm(m_dir[i]) < n*0.004 or \
           i >= len(m_dir) - n: 
            m_dir[i] = mov_dir_ests2[i]
        m_dir[i] = m_dir[i] / np.linalg.norm(m_dir[i])

    xhat = m_dir[:,0]
    yhat = m_dir[:,1]
    zhat = m_dir[:,2]
    if window_size is not None and polynomial is not None:
        window_size = np.min([window_size, m_dir.shape[0]])
        xhat = savgol_filter(m_dir[:,0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:,1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:,2], window_size, polynomial)

    movement_direction = np.swapaxes(np.stack([xhat, yhat, zhat]), 0, 1)
    centered_seq_rotated = []
    for i,mov_dir_vec in enumerate(movement_direction):
        mov_dir_vec_in_xz_plane = mov_dir_vec.copy()
        mov_dir_vec_in_xz_plane[1] = 0
        mov_dir_vec_in_xz_plane /= np.linalg.norm(mov_dir_vec_in_xz_plane)

        to_align_with = np.stack([mov_dir_vec_in_xz_plane, hip_vectors[i], [0, 1, 0]])
        rot_mat = get_rotation_matrix(
            vec1=np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]),
            vec2=to_align_with,
            weights=[1, np.inf, 1]
        )

        rotated = np.stack([rot_mat.dot(joint) for joint in centered_seq[i]])
        offset = seq[i, 0, :]
        offset[1] = 0
        rotated += offset
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        centered_seq_rotated.append(rotated)
        
    centered_seq_rotated = np.stack(centered_seq_rotated)
    return centered_seq_rotated


def transform_seq_so_it_has_no_slope_h36m(seq, n_frames_est_mov_dir=15, window_size=90, polynomial=4, return_rot_matrices=False):
    """ Assumes y axis roughly corressponds to height (movement in xz plane is used for trajectory reconstruction """
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, 0, :]
    
    hip_vectors    = []
    height_vectors = []
    mov_dir_ests2  = []
    for pose in centered_seq:
        hip_vector     = get_hip_vec_estimate_h36m(pose)
        height_vector  = get_height_vec_estimate_h36m(pose)
        mov_dir_est2   = get_perpendicular_in_dir_of_nose(hip_vector, height_vector, pose)
        hip_vectors.append(hip_vector)
        height_vectors.append(height_vector)
        mov_dir_ests2.append(mov_dir_est2)
    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests2 = np.stack(mov_dir_ests2)
    
    standing_still_mask = detect_standing_still(seq)
    sacrum_locs = seq[:, 0, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0,n),(0,0)), mode='edge') # Reapeat last value n times so it's the same len as seq
    for i,m_dir_est in enumerate(m_dir):
        if standing_still_mask[i] or \
           np.linalg.norm(m_dir[i]) < n*0.004 or \
           i >= len(m_dir) - n: 
            m_dir[i] = mov_dir_ests2[i]
        m_dir[i] = m_dir[i] / np.linalg.norm(m_dir[i])

    xhat = m_dir[:,0]
    yhat = m_dir[:,1]
    zhat = m_dir[:,2]
    if window_size is not None and polynomial is not None:
        window_size = np.min([window_size, m_dir.shape[0]])
        xhat = savgol_filter(m_dir[:,0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:,1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:,2], window_size, polynomial)

    movement_direction = np.swapaxes(np.stack([xhat, yhat, zhat]), 0, 1)
    centered_seq_rotated = []
    rotation_matrices = []
    for i,mov_dir_vec in enumerate(movement_direction):
        mov_dir_vec_in_xz_plane = mov_dir_vec.copy()
        mov_dir_vec_in_xz_plane[1] = 0
        mov_dir_vec_in_xz_plane /= np.linalg.norm(mov_dir_vec_in_xz_plane)

        to_align_with = np.stack([mov_dir_vec_in_xz_plane, hip_vectors[i], [0, 1, 0]])
        rot_mat = get_rotation_matrix(
            vec1=np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]),
            vec2=to_align_with,
            weights=[1, np.inf, 1]
        )

        rotated = np.stack([rot_mat.dot(joint) for joint in centered_seq[i]])
        offset = seq[i, 0, :]
        offset[1] = 0
        rotated += offset
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        centered_seq_rotated.append(rotated)
        rotation_matrices.append(rot_mat)
        
    centered_seq_rotated = np.stack(centered_seq_rotated)
    if return_rot_matrices:
        return centered_seq_rotated, np.stack(rotation_matrices)
    else:
        return centered_seq_rotated






def get_hip_vec_estimate_vertices(pose, left_hip_idx, right_hip_idx):
    return pose[left_hip_idx] - pose[right_hip_idx]

def get_height_vec_estimate_vertices(pose, head_idx, pelvis_idx):
    return pose[head_idx] - pose[pelvis_idx]

def get_perpendicular_in_dir_of_feet(hip_vec, height_vec, pose, left_foot_idx, right_foot_idx):
    forward_vec = pose[left_foot_idx] - pose[right_foot_idx]
    perp_vec = np.cross(height_vec, forward_vec)
    return perp_vec / (np.linalg.norm(perp_vec) + 1e-8)


def transform_seq_so_it_has_no_slope_vertices(seq, 
                                               sacrum_index, 
                                               left_hip_idx, right_hip_idx,
                                               head_idx=None, 
                                               pelvis_idx=None,
                                               left_foot_idx=None, right_foot_idx=None,
                                               n_frames_est_mov_dir=15, 
                                               window_size=90, 
                                               polynomial=4):
    centered_seq = seq.copy()
    for frame in range(centered_seq.shape[0]):
        centered_seq[frame, :, :] -= seq[frame, sacrum_index, :]

    hip_vectors = []
    height_vectors = []
    mov_dir_ests = []
    for pose in centered_seq:
        hip_vec = get_hip_vec_estimate_vertices(pose, left_hip_idx, right_hip_idx)
        height_vec = get_height_vec_estimate_vertices(pose, head_idx, pelvis_idx)
        mov_dir = get_perpendicular_in_dir_of_feet(hip_vec, height_vec, pose, left_foot_idx, right_foot_idx)
        hip_vectors.append(hip_vec)
        height_vectors.append(height_vec)
        mov_dir_ests.append(mov_dir)

    hip_vectors = np.stack(hip_vectors)
    height_vectors = np.stack(height_vectors)
    mov_dir_ests = np.stack(mov_dir_ests)

    sacrum_locs = seq[:, sacrum_index, :]
    n = n_frames_est_mov_dir
    m_dir = sacrum_locs[n:] - sacrum_locs[:-n]
    m_dir = np.pad(m_dir, ((0, n), (0, 0)), mode='edge')

    for i in range(len(m_dir)):
        if np.linalg.norm(m_dir[i]) < n * 0.004 or i >= len(m_dir) - n:
            m_dir[i] = mov_dir_ests[i]
        m_dir[i] /= np.linalg.norm(m_dir[i]) + 1e-8

    if window_size is not None and polynomial is not None:
        window_size = min(window_size, len(m_dir))
        xhat = savgol_filter(m_dir[:, 0], window_size, polynomial)
        yhat = savgol_filter(m_dir[:, 1], window_size, polynomial)
        zhat = savgol_filter(m_dir[:, 2], window_size, polynomial)
    else:
        xhat, yhat, zhat = m_dir[:, 0], m_dir[:, 1], m_dir[:, 2]

    movement_direction = np.stack([xhat, yhat, zhat], axis=1)
    rotated_seq = []

    for i, mov_dir_vec in enumerate(movement_direction):
        mov_dir_vec_in_xz = mov_dir_vec.copy()
        mov_dir_vec_in_xz[1] = 0
        mov_dir_vec_in_xz /= np.linalg.norm(mov_dir_vec_in_xz) + 1e-8

        to_align_with = np.stack([mov_dir_vec_in_xz, hip_vectors[i], [0, 1, 0]])
        rot_mat = get_rotation_matrix(
            vec1=np.stack([mov_dir_vec, hip_vectors[i], height_vectors[i]]),
            vec2=to_align_with,
            weights=[1, np.inf, 1]
        )

        rotated = np.stack([rot_mat @ joint for joint in centered_seq[i]])
        offset = seq[i, sacrum_index, :].copy()
        offset[1] = 0
        rotated += offset
        rotated -= np.array([0, np.min(rotated[:, 1]), 0])
        rotated_seq.append(rotated)

    return np.stack(rotated_seq)