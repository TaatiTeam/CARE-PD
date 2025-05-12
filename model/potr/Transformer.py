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


"""Implementation of the Transformer for sequence-to-sequence decoding.

Implementation of the transformer for sequence to sequence prediction as in
[1] and [2].

[1] https://arxiv.org/pdf/1706.03762.pdf
[2] https://arxiv.org/pdf/2005.12872.pdf
"""


import numpy as np
import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

from model import utils
import model.potr.TransformerEncoder as Encoder


class Transformer(nn.Module):
  def __init__(self,
              num_encoder_layers=6,
              model_dim=256,
              num_heads=8,
              dim_ffn=2048,
              dropout=0.1,
              init_fn=utils.normal_init_,
              pre_normalization=False):
    """Implements the Transformer model for sequence-to-sequence modeling."""
    super(Transformer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout

    self._encoder = Encoder.TransformerEncoder(
        num_layers=num_encoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        pre_normalization=pre_normalization
    )


  def forward(self,
              source_seq,
              encoder_position_encodings=None):

    
    memory, enc_weights = self._encoder(source_seq, encoder_position_encodings)

    # Save encoder outputs
    # if fold is not None:
    #   encoder_output_dir = 'encoder_outputs'
    #   if not os.path.exists(f'{encoder_output_dir}f{fold}/'):
    #     os.makedirs(f'{encoder_output_dir}f{fold}/')
    #   outpath = f'{encoder_output_dir}f{fold}/{eval_step}.npy'
    #   encoder_output = memory.detach().cpu().numpy()
    #   np.save(outpath, encoder_output)

    return memory
