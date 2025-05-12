from functools import partial

import torch
from torch import nn
import numpy as np

from model.motionbert.DSTformer import DSTformer
from model.potr import PoseTransformer
from model.potr import PoseEncoderDecoder
from model.poseformerv2.model_poseformer import PoseTransformerV2
from model.mixste.model_cross import MixSTE2
from model.motionagformer.MotionAGFormer import MotionAGFormer
from model.motionclip.transformer import Encoder_TRANSFORMER 
from model.momask.model import RVQVAE
from model.momask.get_opt import get_opt


def count_parameters(model):
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    return model_params


def load_pretrained_weights(model, checkpoint):
    """
    Load pretrained weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - checkpoint (dict): the checkpoint
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    model_first_key = next(iter(model_dict))
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if not 'module.' in model_first_key:
            if k.startswith('module.'):
                k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print(f'[INFO] (load_pretrained_weights) {len(matched_layers)} layers are loaded')
    print(f'[INFO] (load_pretrained_weights) {len(discarded_layers)} layers are discared')
    if len(matched_layers) == 0:
        print ("--------------------------model_dict------------------")
        print (model_dict.keys())
        print ("--------------------------discarded_layers------------------")
        print (discarded_layers)
        raise NotImplementedError(f"Loading problem!!!!!!")



def load_pretrained_backbone(params, backbone_name):
    ck_path = params['model_checkpoint_path']
    print(f'[INFO] (load_pretrained_backbone) from: {ck_path}')
    if backbone_name == 'motionbert':
        model_backbone = DSTformer(dim_in=3,
                                   dim_out=3,
                                   dim_feat=params['dim_feat'],
                                   dim_rep=params['dim_rep'],
                                   depth=params['depth'],
                                   num_heads=params['num_heads'],
                                   mlp_ratio=params['mlp_ratio'],
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   maxlen=params['maxlen'],
                                   num_joints=params['num_joints'])
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage, weights_only=False)['model_pos']
    elif backbone_name == 'potr':
        pose_encoder_fn, _ = PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)
        model_backbone = PoseTransformer.model_factory(params, pose_encoder_fn)
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage, weights_only=False)
    elif backbone_name == "poseformerv2":
        model_backbone = PoseTransformerV2(num_joints=params['num_joints'],
                                           embed_dim_ratio=params['embed_dim_ratio'],
                                           depth=params['depth'],
                                           number_of_kept_frames=params['number_of_kept_frames'],
                                           number_of_kept_coeffs=params['number_of_kept_coeffs'],
                                           in_chans=2,
                                           num_heads=8,
                                           mlp_ratio=2,
                                           qkv_bias=True,
                                           qk_scale=None,
                                           drop_path_rate=0,
                                           )
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage, weights_only=False)['model_pos']
    elif backbone_name == 'mixste':
        model_backbone = MixSTE2(
                                num_frame=params['source_seq_len'], 
                                num_joints=params['num_joints'],
                                in_chans=2,
                                embed_dim_ratio=params['embed_dim_ratio'],
                                depth=params['depth'],
                                num_heads=8,
                                mlp_ratio=2.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_path_rate=0
                                )
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage, weights_only=False)['model_pos']
    elif backbone_name == "motionagformer":
        model_backbone = MotionAGFormer(n_layers=params['n_layers'],
                               dim_in=params['dim_in'],
                               dim_feat=params['dim_feat'],
                               dim_rep=params['dim_rep'],
                               dim_out=params['dim_out'],
                               mlp_ratio=params['mlp_ratio'],
                               act_layer=nn.GELU,
                               attn_drop=params['attn_drop'],
                               drop=params['drop'],
                               drop_path=params['drop_path'],
                               use_layer_scale=params['use_layer_scale'],
                               layer_scale_init_value=params['layer_scale_init_value'],
                               use_adaptive_fusion=params['use_adaptive_fusion'],
                               num_heads=params['num_heads'],
                               qkv_bias=params['qkv_bias'],
                               qkv_scale=params['qkv_scale'],
                               hierarchical=params['hierarchical'],
                               num_joints=params['num_joints'],
                               use_temporal_similarity=params['use_temporal_similarity'],
                               temporal_connection_len=params['temporal_connection_len'],
                               use_tcn=params['use_tcn'],
                               graph_only=params['graph_only'],
                               neighbour_num=params['neighbour_num'],
                               n_frames=params['source_seq_len'])
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage, weights_only=False)['model']
    elif backbone_name == "momask":
        vq_opt = get_opt(params['opt_file_path'], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        chkpoint_file = params['model_checkpoint_path']
        model_backbone = RVQVAE(args=vq_opt,
                                input_width=263,
                                nb_code=vq_opt.nb_code,
                                code_dim=vq_opt.code_dim,
                                output_emb_width=vq_opt.code_dim,
                                down_t=vq_opt.down_t,
                                stride_t=vq_opt.stride_t,
                                width=vq_opt.width,
                                depth=vq_opt.depth,
                                dilation_growth_rate=vq_opt.dilation_growth_rate,
                                activation=vq_opt.vq_act,
                                norm=vq_opt.vq_norm)
        checkpoint = torch.load(chkpoint_file, map_location=lambda storage, loc: storage, weights_only=False)['net']
    elif backbone_name == "motionclip":
        model_backbone = Encoder_TRANSFORMER(modeltype='cvae',
                                             njoints=25,
                                             nfeats=6,
                                             num_frames=60,
                                             num_classes=0,
                                             translation=True, 
                                             pose_rep='rot6d',
                                             glob=True,
                                             glob_rot=[3.141592653589793, 0, 0],
                                             latent_dim=512,
                                             ff_size=1024,
                                             num_layers=8)
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage, weights_only=False)
    else:
        raise Exception("Undefined backbone type.")

    load_pretrained_weights(model_backbone, checkpoint)
    return model_backbone

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vq_opt = get_opt('/cluster/projects/taati/ivan/data/motion_encoders/Pretrained_checkpoints/momask/opt.txt', device=device)
    chkpoint_file = '/cluster/projects/taati/ivan/data/motion_encoders/Pretrained_checkpoints/momask/net_best_fid.tar'
    print('vq_opt')
    print(vq_opt)
    model_backbone = RVQVAE(args=vq_opt,
                            input_width=263,
                            nb_code=vq_opt.nb_code,
                            code_dim=vq_opt.code_dim,
                            output_emb_width=vq_opt.code_dim,
                            down_t=vq_opt.down_t,
                            stride_t=vq_opt.stride_t,
                            width=vq_opt.width,
                            depth=vq_opt.depth,
                            dilation_growth_rate=vq_opt.dilation_growth_rate,
                            activation=vq_opt.vq_act,
                            norm=vq_opt.vq_norm)
    checkpoint = torch.load(chkpoint_file, map_location=device)['net']
    load_pretrained_weights(model_backbone, checkpoint)
    #print("Model's state_dict:")
    #for param_tensor in model_backbone.state_dict():
    #    print(param_tensor, "\t", model_backbone.state_dict()[param_tensor].size())

    print('Model loaded successfully, now testing inference call')
    example_seq_path = '/cluster/projects/taati/ivan/data/motion_encoders/Pretrained_checkpoints/momask/example_data/000612.npy'
    seq = np.load(example_seq_path)
    seq = np.concatenate((seq[:90],))
    print(f'seq.shape: {seq.shape}')
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    print(f'input tensor shape: {seq.shape}')
    #code_idx, all_codes = model_backbone(seq)
    x_quantized = model_backbone(seq)
    print(f'x_quantized.shape: {x_quantized.shape}')
    #print(f'code_idx.shape: {code_idx.shape}')
    #print(f'all_codes.shape: {all_codes.shape}')
    #print(code_idx)

    #final_motion_representation = torch.mean(torch.sum(all_codes, dim=0), dim=-1)
    #print(f'code_idx.shape: {code_idx.shape}')
    #print(f'all_codes.shape: {all_codes.shape}')
    #print(f'final_motion_representation.shape: {final_motion_representation.shape}')

