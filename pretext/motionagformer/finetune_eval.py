import argparse
import os

import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from tqdm import tqdm

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.dataset_preparation import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import Augmenter2D

from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err

from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-base.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts

def p_mpjpe_err(predicted, target):
    assert predicted.shape == target.shape

    muX = target.mean(dim=1, keepdim=True)
    muY = predicted.mean(dim=1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = torch.norm(X0, dim=(1, 2), keepdim=True)
    normY = torch.norm(Y0, dim=(1, 2), keepdim=True)

    X0 = X0 / normX
    Y0 = Y0 / normY

    H = torch.matmul(X0.transpose(1, 2), Y0)
    U, s, Vt = torch.linalg.svd(H, full_matrices=True)
    V = Vt.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))

    # Avoid improper rotations
    detR = torch.linalg.det(R).unsqueeze(1)
    V[:, :, -1] *= detR
    s[:, -1] *= detR.squeeze(1)
    R = torch.matmul(V, U.transpose(1, 2))  # Corrected rotation

    tr = s.sum(dim=1, keepdim=True).unsqueeze(2)
    a = tr * normX / normY  # Scale
    t = muX - a * torch.matmul(muY, R)  # Translation

    predicted_aligned = a * torch.matmul(predicted, R) + t

    return torch.mean(torch.norm(predicted_aligned - target, dim=-1), dim=1)

def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    presentation = defaultdict(list)
    for x, y, frame_ids in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0

        pred = model(x)  # (N, T, 17, 3)

        arg_first = find_arg_first_occurrence(frame_ids)
        batch_mask = torch.zeros_like(arg_first, dtype=bool)
        for i in range(batch_mask.shape[0]):
            batch_mask[i, arg_first[i][arg_first[i] >= 0]] = 1

        batch_mask = batch_mask.flatten()
        predicted_3d_pos = predicted_3d_pos.reshape(-1, 17, 3)[batch_mask]
        y = y.reshape(-1, 17, 3).contiguous()[batch_mask]

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()

def find_arg_first_occurrence(x):
    mask = torch.zeros_like(x, dtype=torch.int)
    for i, frame_ids in enumerate(x):
        seen = set()
        for j, value in enumerate(frame_ids):
            if value.item() not in seen:
                mask[i, j] = 1
                seen.add(value.item())
    return mask

def evaluate(args, model, test_loader, device):
    print("[INFO] Evaluation")
    mpjpe_meter = AverageMeter()
    p_mpjpe_meter = AverageMeter()
    acc_err_meter = AverageMeter()

    presentation = defaultdict(list)

    with torch.no_grad():
        for x, y, frame_ids in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            
            args.flip = False
            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)

            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            arg_first = find_arg_first_occurrence(frame_ids).long()
            batch_mask = torch.zeros_like(arg_first, dtype=bool)
            for i in range(batch_mask.shape[0]):
                batch_mask[i, arg_first[i][arg_first[i] >= 0]] = 1

            batch_mask = batch_mask.flatten()
            predicted_3d_pos = predicted_3d_pos.reshape(-1, 17, 3)[batch_mask]
            y = y.reshape(-1, 17, 3).contiguous()[batch_mask]

            mpjpe = torch.mean(torch.norm(predicted_3d_pos - y, dim=-1)) * 1000
            p_mpjpe = torch.mean(p_mpjpe_err(predicted_3d_pos, y), dim=-1) * 1000
            if y.shape[0] >=  3:
                accel_gt = y[:-2] - 2 * y[1:-1] + y[2:]
                accel_pred = predicted_3d_pos[:-2] - 2 * predicted_3d_pos[1:-1] + predicted_3d_pos[2:]
                acc_err = torch.mean(torch.norm(accel_pred - accel_gt, dim=-1)) * 1000
            else:
                acc_err = torch.tensor(0.0, device=p_mpjpe.device)

            mpjpe_meter.update(mpjpe.item(), predicted_3d_pos.shape[0])
            p_mpjpe_meter.update(p_mpjpe.item(), predicted_3d_pos.shape[0])
            acc_err_meter.update(acc_err.item(), predicted_3d_pos.shape[0])
    

    print('Protocol #1 Error (MPJPE):', mpjpe_meter.avg, 'mm')
    print('Acceleration error:', acc_err_meter.avg, 'mm/s^2')
    print('Protocol #2 Error (P-MPJPE):', p_mpjpe_meter.avg, 'mm')
    return mpjpe_meter.avg, p_mpjpe_meter.avg, acc_err_meter.avg


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args.data_root, 'train')
    test_dataset = MotionDataset3D(args.data_root, 'eval')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    # print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        project='MotionAGFormer',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        name=opts.wandb_name,
                        project='MotionAGFormer',
                        settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            mpjpe, p_mpjpe, acc_err = evaluate(args, model, test_loader, device)
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)
        mpjpe, p_mpjpe, acc_err = evaluate(args, model, test_loader, device)
        log = {
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/acceleration_error': acc_err,
                'eval/p-mpjpe': p_mpjpe,
                # 'eval/min_mpjpe': min_mpjpe,
                # 'eval_additional/upper_body_error': np.mean(joints_error[H36M_UPPER_BODY_JOINTS]),
                # 'eval_additional/lower_body_error': np.mean(joints_error[H36M_LOWER_BODY_JOINTS]),
                # 'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                # 'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                # 'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                # **joint_label_errors
            }
        print(log)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)

        # joint_label_errors = {}
        # for joint_idx in range(args.num_joints):
        #     joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]
        if opts.use_wandb:
            wandb.log(log, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    if opts.use_wandb:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)
    
    train(args, opts)


if __name__ == '__main__':
    main()

# ./maf/bin/python finetune.py --eval-only --checkpoint checkpoint --checkpoint-file carePD_fromscratch.pth.tr --config configs/h36m/MotionAGFormer-small.yaml
# ./maf/bin/python finetune.py --checkpoint checkpoint --checkpoint-file carePD_fromscratch.pth.tr --config configs/h36m/MotionAGFormer-small.yaml
