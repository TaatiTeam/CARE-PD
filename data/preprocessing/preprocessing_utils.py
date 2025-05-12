import numpy as np
import joblib
import torch
import torch.nn.functional as F

# from human_body_prior.body_model.lbs import vertices2joints
from smplx.lbs import vertices2joints

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def get_6D_rep_from_24x3_pose(pose):
    pose6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).detach().cpu().numpy()
    pose6d=np.pad(pose6d, ((0,0), (0,1), (0,0))) # Adding [0,0,0,0,0,0] for translation
    return pose6d

def read_pd_SMPL_6D(SMPL_pkl_path):
    """
    Read SMPL params in format MotionCLIP expects. (6D reprsentations of joint rotations + (0,0,0,0,0,0) as translation)

    Parameters:
    SMPL_pkl_path (str): The file path for the mesh_seq.pkl file.

    Returns:
    numpy.ndarray: An array containing the 6D SMPL skeleton sequence.

    """
    bdata = joblib.load(SMPL_pkl_path)
    pose = torch.from_numpy(bdata['pose']).reshape(-1, 24, 3)
    return get_6D_rep_from_24x3_pose(pose)


def read_pd_h36m_from_SMPL(SMPL_pkl_path, neutral_bm, h36m_regressor, comp_device):
    """
    Reggresses h36m joint sequence from given .pkl file with SMPL parameters.

    Parameters:
    SMPL_pkl_path (str): The file path for the mesh_seq.pkl file.
    neutral_bm : The SMPL body model.
    h36m_regressor : The 17 x 6890 torch regresssor that regresses h36m joints from SMPL vertacies.
    comp_device : cpu / gpu

    Returns:
    numpy.ndarray: An array containing the h36m skeleton sequence.

    """
    bdata = joblib.load(SMPL_pkl_path)
    frame_number = bdata['cam'].shape[0]
    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    bm = neutral_bm

    with torch.no_grad():
        for fId in range(frame_number):
            root_orient = torch.Tensor(bdata['pose'][fId:fId+1, :3]).to(comp_device) # controls the global root orientation
            pose_body = torch.Tensor(bdata['pose'][fId:fId+1, 3:]).to(comp_device) # controls the body
            # pose_hand = torch.Tensor(bdata['pose'][fId:fId+1, 66:]).to(comp_device) # controls the finger articulation
            betas = torch.Tensor(bdata['beta_mean'][:10][np.newaxis]).to(comp_device) # controls the body shape
            trans = torch.Tensor(bdata['cam'][fId:fId+1]).to(comp_device)    
            body = bm(pose_body=pose_body, pose_hand=None, betas=betas, root_orient=root_orient)
            h36m_joints = vertices2joints(h36m_regressor, body.v) + trans
            pose_seq.append(h36m_joints)
    pose_seq = torch.cat(pose_seq, dim=0)
    
    pose_seq_np = pose_seq.detach().cpu().numpy()
    #pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    return pose_seq_np