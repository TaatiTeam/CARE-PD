import sys
import os
from os.path import join as pjoin

import torch
from models.vq.model import RVQVAE
from options.vq_option import arg_parse
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
import utils.eval_t2m as eval_t2m
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils.word_vectorizer import WordVectorizer

def load_vq_model(vq_opt, which_epoch):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')

    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    

    if ("carepd" in vq_opt.name) or ("healthy" in vq_opt.name):
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.finetune_dataset_name, vq_opt.name, 'model', which_epoch), map_location='cpu')
    else:
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', which_epoch), map_location='cpu')
        
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

if __name__ == "__main__":
    ##### ---- Exp dirs ---- #####
    args = arg_parse(False)
    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'eval')
    os.makedirs(args.out_dir, exist_ok=True)

    f = open(pjoin(args.out_dir, '%s.log'%args.ext), 'w')

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataset_name == 'kit' \
                                                        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    args.nb_joints = 21 if args.dataset_name == 'kit' else 22
    dim_pose = 251 if args.dataset_name == 'kit' else 263

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=args.device)

    print(len(eval_val_loader))

    ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=args.device)
    # net = load_vq_model()

    # vq_opt.finetune_dataset_name = "healthy" ###
    
    if ("carepd" in vq_opt.name) or ("healthy" in vq_opt.name):
        model_dir = pjoin(args.checkpoints_dir, vq_opt.finetune_dataset_name, args.name, 'model')
    else:
        model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')

    print(model_dir)

    for file in os.listdir(model_dir):
        # if not file.endswith('tar'):
        #     continue

        if ("carepd" in vq_opt.name) or ("healthy" in vq_opt.name):
            if not file.startswith('latest'):
                continue
        else:
            if not file.startswith('net_best_fid'):
                continue
        # if args.which_epoch != "all" and args.which_epoch not in file:
        #     continue
        print(file)
        net, ep = load_vq_model(vq_opt, file)

        net.eval()
        net.cuda()

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mae = []
        repeat_time = 20

        fid, mpjpe, pamjpe, acc_err = eval_t2m.evaluation_vqvae_inference(eval_val_loader, net, eval_wrapper=eval_wrapper)

        print(f'{file} final result, epoch {ep}')
        print(f'{file} final result, epoch {ep}', file=f, flush=True)

        msg_final = f"\tFID: {fid:.3f}\n" \
                    f"\tMPJPE: {mpjpe:.3f}\n" \
                    f"\tPAMPJE: {pamjpe:.3f}\n" \
                    f"\tACC Error: {acc_err:.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()

