###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive
# Transformers
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by
# Angel Martinez <angel.martinez@idiap.ch>,
#
# This file is part of
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
#
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################
"""Definition of pose encoder and encoder embeddings and model factory."""

import numpy as np
import os
import sys

import torch
import torch.nn as nn

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath + "/../")

from model import utils
import model.potr.PoseGCN as GCN
import model.potr.Conv1DEncoder as Conv1DEncoder


def pose_encoder_mlp(params):
    # These two encoders should be experimented with a graph NN and
    # a prior based pose decoder using also the graph
    init_fn = utils.normal_init_ \
        if params['init_fn'] == 'normal_init' else utils.xavier_init_
    pose_embedding = nn.Sequential(
        nn.Linear(params['input_dim'], params['model_dim']),
        nn.Dropout(0.1)
    )
    utils.weight_init(pose_embedding, init_fn_=init_fn)
    return pose_embedding


def pose_decoder_mlp(params):
    init_fn = utils.normal_init_ \
        if params['init_fn'] == 'normal_init' else utils.xavier_init_
    pose_decoder = nn.Linear(params['model_dim'], params['pose_dim'])
    utils.weight_init(pose_decoder, init_fn_=init_fn)
    return pose_decoder


def pose_decoder_gcn(params):
    decoder = GCN.PoseGCN(
        input_features=params['model_dim'],
        output_features=9 if params['pose_format'] == 'rotmat' else 3,
        model_dim=params['model_dim'],
        output_nodes=params['n_joints'],
        p_dropout=params['dropout'],
        num_stage=1
    )

    return decoder


def pose_encoder_gcn(params):
    encoder = GCN.SimpleEncoder(
        n_nodes=params['n_joints'],
        input_features=9 if params['pose_format'] == 'rotmat' else 3,
        # n_nodes=params['pose_dim'],
        # input_features=1,
        model_dim=params['model_dim'],
        p_dropout=params['dropout']
    )

    return encoder


def pose_encoder_conv1d(params):
    encoder = Conv1DEncoder.Pose1DEncoder(
        input_channels=9 if params['pose_format'] == 'rotmat' else 3,
        output_channels=params['model_dim'],
        n_joints=params['n_joints']
    )
    return encoder


def pose_encoder_conv1dtemporal(params):
    dof = 9 if params['pose_format'] == 'rotmat' else 3
    encoder = Conv1DEncoder.Pose1DTemporalEncoder(
        input_channels=dof * params['n_joints'],
        output_channels=params['model_dim']
    )
    return encoder


def select_pose_encoder_decoder_fn(params):
    if params['pose_embedding_type'].lower() == 'simple':
        return pose_encoder_mlp, pose_decoder_mlp
    if params['pose_embedding_type'].lower() == 'conv1d_enc':
        return pose_encoder_conv1d, pose_decoder_mlp
    if params['pose_embedding_type'].lower() == 'convtemp_enc':
        return pose_encoder_conv1dtemporal, pose_decoder_mlp
    if params['pose_embedding_type'].lower() == 'gcn_dec':
        return pose_encoder_mlp, pose_decoder_gcn
    if params['pose_embedding_type'].lower() == 'gcn_enc':
        return pose_encoder_gcn, pose_decoder_mlp
    if params['pose_embedding_type'].lower() == 'gcn_full':
        return pose_encoder_gcn, pose_decoder_gcn
    elif params['pose_embedding_type'].lower() == 'vae':
        return pose_encoder_vae, pose_decoder_mlp
    else:
        raise ValueError('Unknown pose embedding {}'.format(params['pose_embedding_type']))