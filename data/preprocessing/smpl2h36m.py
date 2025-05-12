import os
import torch
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
import argparse
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from const.const import _DEVICE, SUPPORTED_DATASETS, DATASET_ORIGINAL_FPS
from const import path
from data.preprocessing.trajectory_correction import transform_seq_so_it_has_no_slope_h36m

def compute_cam_intrinsics(fx, fy, cx, cy):
    # assert fx == fy, "fx and fy should be equal"
    focal_length = fx
    cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
    cam_intrinsics[:, 0, 0] = fx
    cam_intrinsics[:, 1, 1] = fy
    cam_intrinsics[:, 0, 2] = cx
    cam_intrinsics[:, 1, 2] = cy
    return cam_intrinsics

def full_perspective_projection_wham(
        points, 
        cam_intrinsics, 
        rotation=None,
        translation=None,
    ):
    """
    points: (Frames, num_points, 3)
    cam_intrinsics: (1, 3, 3)
    """
    K = cam_intrinsics

    if rotation is not None:
        points = (rotation @ points.transpose(-1, -2)).transpose(-1, -2)
    if translation is not None:
        points = points + translation.unsqueeze(-2)
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = (K @ projected_points.transpose(-1, -2)).transpose(-1, -2)
    return projected_points[..., :-1]

def full_perspective_projection(
        points, 
        cam_intrinsics, 
        rotation=None,
        translation=None,
        distortion_coeffs=None  # New!
    ):
    """
    points: (Frames, num_points, 3)
    cam_intrinsics: (1, 3, 3)
    distortion_coeffs: (1, 5) or None
    """
    K = cam_intrinsics

    # Apply rotation and translation
    if rotation is not None:
        points = (rotation @ points.transpose(-1, -2)).transpose(-1, -2)
    if translation is not None:
        points = points + translation.unsqueeze(-2)
    
    # Normalize by depth (Z)
    projected_points = points / points[..., -1].unsqueeze(-1)
    
    # Apply distortion if given
    if distortion_coeffs is not None:
        k1, k2, k3, p1, p2 = distortion_coeffs.squeeze()
        
        r2 = projected_points[..., 0]**2 + projected_points[..., 1]**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        tangential_x = 2 * p1 * projected_points[..., 0] * projected_points[..., 1] + p2 * (r2 + 2 * projected_points[..., 0]**2)
        tangential_y = p1 * (r2 + 2 * projected_points[..., 1]**2) + 2 * p2 * projected_points[..., 0] * projected_points[..., 1]
        
        projected_points[..., 0] = projected_points[..., 0] * radial + tangential_x
        projected_points[..., 1] = projected_points[..., 1] * radial + tangential_y

    # Apply intrinsic matrix
    projected_points = (K @ projected_points.transpose(-1, -2)).transpose(-1, -2)
    
    return projected_points[..., :-1]  # Return (u, v) only

def generate_smpl_in_world(smpl_model, sequence, down_sample_rate, down):
    frame_number = sequence['pose'].shape[0]
    
    pose_world    = sequence['pose'].reshape(-1, 24, 3)  # (num_frames, 24, 3)
    betas         = sequence['beta']  # (num_frames, 10)
    world_trans   = sequence['trans']  # (num_frames, 3)
    if betas.shape[0] != frame_number:
        betas = np.tile(betas, (frame_number, 1))
        

    # Extract global orientation (index 0) and body pose (indices 1-23)
    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32)  # (num_frames, 1, 3)
    body_pose     = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32)  # (num_frames, 23, 3)
    betas         = torch.tensor(betas, dtype=torch.float32)  # (num_frames, 10)
    
    if down_sample_rate > 1:
        global_orient = global_orient[down::down_sample_rate,...]  # (num_frames, 24, 3)  # start from down and they select every down_sample_rate
        body_pose     = body_pose[down::down_sample_rate,...]  # (num_frames, 10)
        betas         = betas[down::down_sample_rate,...] 
        world_trans         = world_trans[down::down_sample_rate,...]
        frame_number  = body_pose.shape[0]

    # Ensure everything is on the same device
    global_orient = global_orient.reshape(frame_number, -1).to(_DEVICE)
    body_pose = body_pose.reshape(frame_number, -1).to(_DEVICE)
    betas = betas.reshape(frame_number, -1).to(_DEVICE)
    world_trans = torch.tensor(world_trans, dtype=torch.float32).to(_DEVICE)  # Ensure on same device

    # Zero values for face, hands, and expression
    zero_pose = torch.zeros((frame_number, 3), dtype=torch.float32).to(_DEVICE)
    zero_hand_pose = torch.zeros((frame_number, 15, 3), dtype=torch.float32).to(_DEVICE)
    zero_expression = torch.zeros((frame_number, 10), dtype=torch.float32).to(_DEVICE)

    # Generate SMPL output
    out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, 
                     jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                     left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose,
                     expression=zero_expression)

    # Apply global translation (world_trans) to the output vertices
    out.vertices += world_trans[:, None, :]  # Broadcasting (num_frames, 1, 3) to (num_frames, num_vertices, 3)

    return out


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()

def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)

def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1, dim=-1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))

def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()

def world_to_camera(X_w, R, t):
    """
    Converts world coordinates to camera coordinates.
    
    Parameters:
    X_w : np.ndarray
        Shape (T, #joints, 3) - 3D points in world coordinates.
    R : np.ndarray
        Shape (3, 3) - Rotation matrix.
    t : np.ndarray
        Shape (3,) - Translation vector.
    
    Returns:
    X_c : np.ndarray
        Shape (T, #joints, 3) - 3D points in camera coordinates.
    """
    # Ensure t has shape (1, 1, 3) for proper broadcasting
    t = t.reshape(1, 1, 3)
    
    # Apply transformation
    # X_c = np.einsum('ij,tbj->tbi', R, X_w) + t
    X_rot = np.dot(X_w, R.T)
    X_c = X_rot + t

    return X_c

def remove_out_of_bounds(W, H, walk, h36m_joints_img):
    # Mark Out-of-Bounds Points as Invalid
    # valid_mask: shape (Frames, J) of boolean
    # print(h36m_joints_img)
    valid_mask = (h36m_joints_img[..., 0] >= 0) & \
            (h36m_joints_img[..., 0] < W) & \
            (h36m_joints_img[..., 1] >= 0) & \
            (h36m_joints_img[..., 1] < H)
    # Discard Frame if any joint is out
    frame_valid_mask = valid_mask.all(axis=-1)  # (Frames,)
    valid_indices = np.where(frame_valid_mask)[0]
    # print(valid_indices)
    runs = []
    start = None
    prev = None
    for idx in valid_indices:
        if start is None:
            # start a new run
            start = idx
            prev = idx
        elif idx == prev + 1:
            # continue the run
            prev = idx
        else:
            # we found a gap, so end the old run
            runs.append((start, prev))
            start = idx
            prev = idx
    # close the last run if not None
    if start is not None:
        runs.append((start, prev))
    # print(runs)
    # 'runs' is now a list of (start_frame, end_frame) for consecutive in-bounds frames
    # e.g. keep each run separately (for now I only keep the first run)
    for s, e in runs[0:1]:
        h36m_joints_img_inbounds = h36m_joints_img[s:e+1]  # frames s..e inclusive
    # print(h36m_joints_img_inbounds.shape)
    if len(runs):
        if h36m_joints_img_inbounds.shape[0]/h36m_joints_img.shape[0] < 0.85:
            # print(f"⚠️ {walk} has more than 15% out-of-bounds frames: ({h36m_joints_img_inbounds.shape[0]/h36m_joints_img.shape[0]})")
            # print(f"{walk} has {h36m_joints_img_inbounds.shape[0]}/{h36m_joints_img.shape[0]} in-bounds frames")
            flag = False
        else:
            # no out-of-bounds frame
            h36m_joints_img_inbounds = h36m_joints_img
            # print("All frames are fully in-bounds")
            flag = True
    else:
        # raise ValueError(f"⚠️ {walk} has no in-bounds frames")
        a = 1
        
    return h36m_joints_img_inbounds, flag

                
def main_world2cam2img_custom_cm(cfg, view):
    if cfg.slope_correction:
        ext = '_slopeCorrected'
    else:
        ext = ''
    cfg.OUT_PATH_world = cfg.OUT_PATH / f'h36m_3d_world_floorXZZplus_30f_or_longer{ext}.npz'
    cfg.OUT_PATH_cam = cfg.OUT_PATH / f'h36m_3d_world2cam_{view}_floorXZZplus_30f_or_longer{ext}.npz'
    cfg.OUT_PATH_image = cfg.OUT_PATH / f'h36m_3d_world2cam2img_{view}_floorXZZplus_30f_or_longer{ext}.npz'
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    
    seq_include_outframe_counter = 0
    all_sample_count = 0
    result_world = dict()
    result_cam = dict()
    result_img = dict()
    for subject_id in tqdm(all_smpls):
        for walk_id in all_smpls[subject_id]:
            all_sample_count += 1
            smpl_data = all_smpls[subject_id][walk_id]
            
            if 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                if down_sample_rate == 1: 
                    walk_name = str(subject_id) + '__' + str(walk_id)   
                else:
                    walk_name = str(subject_id) + '__' + str(walk_id) + f'_down{down}'
                if smpl_data['pose'].shape[0] < 30:
                    # print(f"Skipping {walk_name} with {smpl_data['pose'].shape[0]} frames")
                    continue
                out_world = generate_smpl_in_world(smpl_model, smpl_data, down_sample_rate, down)
                vertices_world = out_world.vertices # of shape n_frames x num_vert x 3 
                h36m_joints_world = vertices2joints(h36m_regressor, vertices_world).cpu().detach().numpy()
                # print(h36m_joints_world.shape)
                
                if cfg.slope_correction:
                    h36m_joints_world = transform_seq_so_it_has_no_slope_h36m(h36m_joints_world, n_frames_est_mov_dir=15, window_size=90, polynomial = 4)
                
                '''Put on Floor'''
                floor_height = h36m_joints_world.min(axis=0).min(axis=0)[1]
                h36m_joints_world[:, :, 1] -= floor_height
                
                '''XZ at origin'''
                root_pos_init = h36m_joints_world[0]
                root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
                h36m_joints_world = h36m_joints_world - root_pose_init_xz
                
                '''All initially face Z+'''
                r_hip, l_hip, sdr_r, sdr_l = cfg.face_joint_indx
                across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
                across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
                across = across1 + across2
                across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
                # forward (3,), rotate around y-axis
                forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
                # forward (3,)
                forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
                
                # == for processed world coordinates ==
                target_for_world = np.array([[0, 0, 1]])
                h36m_joints_world_orig = h36m_joints_world.copy()
                root_quat_init = qbetween_np(forward_init, target_for_world)
                root_quat_init = np.ones(h36m_joints_world_orig.shape[:-1] + (4,)) * root_quat_init
                h36m_joints_world_orig = qrot_np(root_quat_init, h36m_joints_world_orig)
                # ====
                
                if view in ['back', 'front', 'backright']:
                    #Z+ direction
                    target = np.array([[0, 0, 1]])
                elif view == 'sideleft':
                    # direct "face X+" rotation
                    target = np.array([[1, 0, 0]])
                elif view == 'sideright':
                    # direct "face X-" rotation
                    target = np.array([[-1, 0, 0]])
                
                root_quat_init = qbetween_np(forward_init, target)
                root_quat_init = np.ones(h36m_joints_world.shape[:-1] + (4,)) * root_quat_init
                h36m_joints_world = qrot_np(root_quat_init, h36m_joints_world)
                
                '''Correct for curved walking direction'''
                if cfg.db in ['T-SDU-PD', 'PD-GaM']:
                    # Get first and middle frame positions
                    first_frame = h36m_joints_world[0, 0]  # Root joint (hip) at frame 0
                    middle_frame_idx = h36m_joints_world.shape[0] // 2
                    middle_frame = h36m_joints_world[middle_frame_idx, 0]  # Root joint at middle frame
                    
                    # Compute movement direction from first to middle frame
                    walking_direction = middle_frame - first_frame
                    walking_direction[1] = 0  # Ignore Y (height) changes, only consider XZ plane
                    walking_direction = walking_direction / np.linalg.norm(walking_direction)  # Normalize
                    
                    # # Compute Rotation to Align with Z+
                    # target = np.array([[0, 0, 1]])  # We want the person to face Z+
                    correction_quat = qbetween_np(walking_direction[np.newaxis, :], target)
                    correction_quat = np.ones(h36m_joints_world.shape[:-1] + (4,)) * correction_quat
                    h36m_joints_world = qrot_np(correction_quat, h36m_joints_world)
                
                if view == 'back':
                    # Rotation matrix
                    # the y-axis to point “down” in the camera’s view: a 180° rotation about the z-axis -> Applying it to a point (x,y,z) yields (−x,−y,z).
                    R = np.array([[-1,  0,  0],
                                [ 0,  -1,  0], 
                                [ 0,  0, 1]])  
                    # Translation vector: Camera behind, slightly above
                    y_cam = 1  # Camera height (adjust based on human height)
                    d = 2  # Distance behind subject
                    t = np.array([[0, y_cam, d]])   # Camera position in world coordinates
                elif view == 'front':
                    R = np.array([[1,  0,  0],
                                [ 0,  -1,  0], 
                                [ 0,  0, -1]])  
                    y_cam = 1 
                    d = 8 #meters 
                    t = np.array([[0, y_cam, d]])
                elif view in ['sideleft', 'sideright']:
                    R = np.array([[-1,  0,  0],
                                [ 0,  -1,  0], 
                                [ 0,  0, 1]])  
                    y_cam = 1  # Camera height (adjust based on human height)
                    d = 5  # Distance
                    t = np.array([[0, y_cam, d]])
                elif view == 'backright':
                    R = np.array([[-1,  0,  0],
                                [ 0,  -1,  0], 
                                [ 0,  0, 1]])  
                    y_cam = 1 
                    d = 2 
                    t = np.array([[0, y_cam, d]]) 
                    angle_rad = np.deg2rad(40)
                    R_y = np.array([
                            [np.cos(angle_rad), 0, np.sin(angle_rad)],
                            [0, 1, 0],
                            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
                        ])
                    h36m_joints_world = np.einsum('ij,fkj->fki', R_y, h36m_joints_world)
                else:
                    raise ValueError(f'Unknown view: {view}')
                
                h36m_joints_cam = world_to_camera(h36m_joints_world, R, t)
                
                
                fx, fy, cx, cy = 700, 700, cfg.H/2, cfg.W/2
                cam_intrinsics = compute_cam_intrinsics(fx, fy, cx, cy).to(_DEVICE)
                if view in ['sideleft', 'sideright']:
                    if view == 'sideleft':
                        tt = np.array([[d/2, 0, 0]])
                    else:
                        tt = np.array([[-d/2, 0, 0]]) 
                    translation = torch.tensor(tt, dtype=torch.float32).to(_DEVICE)
                    h36m_joints_img = full_perspective_projection_wham(torch.tensor(h36m_joints_cam, dtype=torch.float32).to(_DEVICE), cam_intrinsics, translation=translation)
                else:
                    h36m_joints_img = full_perspective_projection_wham(torch.tensor(h36m_joints_cam, dtype=torch.float32).to(_DEVICE), cam_intrinsics)

                # if walk_id == 'vid0083_0072':
                #     print(h36m_joints_img.shape)
                    # print(h36m_joints_img_inbounds.shape)
                    
                h36m_joints_img = h36m_joints_img.cpu().detach().numpy() # shape (Frames, J, 2)
                h36m_joints_img_inbounds, flag = remove_out_of_bounds(cfg.W, cfg.H, walk_name, h36m_joints_img)
                # if walk_id == 'vid0083_0072':
                #     print(h36m_joints_img.shape)
                #     print(h36m_joints_img_inbounds.shape)

                if not flag and down == 0:
                    seq_include_outframe_counter += 1

                if h36m_joints_world.shape[0] >= 30:
                    # print(f"Saving {walk_name} with {h36m_joints_img_inbounds.shape} frames")
                    result_world[walk_name] = h36m_joints_world_orig
                    result_cam[walk_name] = h36m_joints_cam
                    result_img[walk_name] = h36m_joints_img_inbounds
                # else:
                #     print(f"Discarding {walk_name} because it is less than 30 frames {h36m_joints_world.shape[0]}")
           
    # print(f"Number of sequences with out-of-frame for {view} view: {seq_include_outframe_counter}")   
    np.savez(cfg.OUT_PATH_world, **result_world)
    np.savez(cfg.OUT_PATH_cam, **result_cam)
    np.savez(cfg.OUT_PATH_image, **result_img)
    print(all_sample_count)
    print(f"Number of sqeuences in world cooridnates: {len(result_world)}")
    print(f"Number of sqeuences in camera cooridnates: {len(result_cam)}")
    print(f"Number of sqeuences in image cooridnates: {len(result_img)}")

def main_world(cfg):
    if cfg.slope_correction:
        ext = '_slopeCorrected'
    else:
        ext = ''
    cfg.OUT_PATH_world = cfg.OUT_PATH / f'h36m_3d_world_30f_or_longer{ext}.npz'
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    smpl_wham_outputs = [str(path) for path in cfg.DATA_DIR.rglob('*.pkl')]
    print(f'Number of walks: {len(smpl_wham_outputs)}')
    
    result = dict()
    for output in tqdm(smpl_wham_outputs):
        walk = os.path.basename(os.path.dirname(os.path.normpath(output)))
        out = joblib.load(output) 

        if cfg.db == '3DGait':
            wham_id = list(out.keys())[0]
            smpl_data = out[wham_id]
        elif cfg.db in ['T-SDU-PD', 'PD-GaM', 'BMCLab', 'DNE']:
            smpl_data = out
            
        down_sample_rate = int(cfg.fps / cfg.exfps)
        
        for down in range(down_sample_rate):
            if down_sample_rate == 1: 
                walk_name = walk 
            else:
                walk_name = walk + f'_down{down}'

            out_world = generate_smpl_in_world(smpl_model, smpl_data, down_sample_rate, down)
            vertices_world = out_world.vertices # of shape n_frames x num_vert x 3 
            h36m_joints_world = vertices2joints(h36m_regressor, vertices_world).cpu().detach().numpy()
            
            if cfg.slope_correction:
                h36m_joints_world = transform_seq_so_it_has_no_slope_h36m(h36m_joints_world, n_frames_est_mov_dir=15, window_size=90, polynomial = 4)
            # # Rotation matrix for 180-degree around Y-axis
            # R = np.array([[-1,  0,  0], 
            #             [ 0,  1,  0], 
            #             [ 0,  0, -1]])  # Negates X and Z
            # # Apply rotation
            # h36m_joints = np.dot(h36m_joints, R.T)
            
            # vertices = torch.tensor(out[wham_id]['verts'], dtype=torch.float32).to(_DEVICE) # shape: n_frames x num_vert x 3 
            # h36m_joints = vertices2joints(h36m_regressor, vertices).cpu().detach().numpy()

            if h36m_joints_world.shape[0] >= 30:
                result[walk_name] = h36m_joints_world
            else:
                print(f"Discarding {walk_name} because it is less than 30 frames {h36m_joints_world.shape[0]}")
            

    np.savez(cfg.OUT_PATH_world, **result)
    print(f"Number of sqeuences in world cooridnates: {len(result)}")


 # SUPPORTED_DATASETS = ['BMCLab', 'T-SDU-PD', 'PD-GaM', '3DGait', 'DNE', 'E-LC', 'KUL-DT-T', 'T-LTC', 'T-SDU']    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-skl', '--skeleton_format', default='h36m')
    parser.add_argument('-db', '--dataset', default='3DGait')
    args = parser.parse_args()
    print(args) 

    assert args.dataset in SUPPORTED_DATASETS, f"Dataset is not supported!"
    
    cfg = SimpleNamespace()
    cfg.BASE_DIR = Path(path.PROJECT_ROOT)
    cfg.H36M_J_REG = Path('./data/preprocessing/common/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('./data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = cfg.BASE_DIR / 'assets' / 'datasets' / (args.dataset + '.pkl')
    cfg.OUT_PATH = Path(path.PROJECT_ROOT) / 'assets' / 'datasets'  / 'h36m' / args.dataset
    
    print(cfg.DATA_DIR)
    print(cfg.OUT_PATH)
    
    DB_Slope_Correction = ['T-LTC', 'T-SDU', 'T-SDU-PD']
    cfg.db = args.dataset
    cfg.slope_correction = args.dataset in DB_Slope_Correction
    cfg.exfps = 30
    cfg.fps = DATASET_ORIGINAL_FPS[args.dataset]
    

    if args.skeleton_format == 'h36m':
        cfg.face_joint_indx = [1, 4, 14, 11]
    elif args.skeleton_format == 'AMASS':
        cfg.face_joint_indx = [2, 1, 17, 16]
        
    cfg.H = 1000
    cfg.W = 1000
    
    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    # print('Performing SMPL to H36M conversion in WORLD Coordinate...')
    # main_world(cfg)
    print('Side view right')
    main_world2cam2img_custom_cm(cfg, view='sideright')
    print('Back view right')
    main_world2cam2img_custom_cm(cfg, view='backright')
    
