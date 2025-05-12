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

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

from model import utils
from model.potr import PositionEncodings
from model.potr.Transformer import Transformer


_SOURCE_LENGTH = 110
_POSE_DIM = 54
_PAD_LENGTH = _SOURCE_LENGTH


class PoseTransformer(nn.Module):
  """Implements the sequence-to-sequence Transformer .model for pose prediction."""
  def __init__(self,
               pose_dim=_POSE_DIM,
               source_seq_length=_SOURCE_LENGTH,
               model_dim=256,
               num_encoder_layers=6,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               input_dim=None,
               init_fn=utils.xavier_init_,
               pre_normalization=False,
               pose_embedding=None,
               copy_method='uniform_scan',
               pos_encoding_params=(10000, 1)):
    """Initialization of pose transformers."""
    super(PoseTransformer, self).__init__()
    self._source_seq_length = source_seq_length
    self._pose_dim = pose_dim
    self._input_dim = pose_dim if input_dim is None else input_dim
    self._model_dim = model_dim
    self._use_class_token = False

    self._mlp_dim = model_dim
    self._pose_embedding = pose_embedding
    thisname = self.__class__.__name__
    self._copy_method = copy_method
    self._pos_encoding_params = pos_encoding_params

    self._transformer = Transformer(
        num_encoder_layers=num_encoder_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        dim_ffn=dim_ffn,
        dropout=dropout,
        init_fn=init_fn,
        pre_normalization=pre_normalization,
    )

    self._pos_encoder = PositionEncodings.PositionEncodings1D(
        num_pos_feats=self._model_dim,
        temperature=self._pos_encoding_params[0],
        alpha=self._pos_encoding_params[1]
    )

    self.init_position_encodings()


  def init_position_encodings(self):
    src_len = self._source_seq_length
    # when using a token we need an extra element in the sequence
    if self._use_class_token:
      src_len = src_len + 1
    encoder_pos_encodings = self._pos_encoder(src_len).view(
            src_len, 1, self._model_dim)
    self._encoder_pos_encodings = nn.Parameter(
        encoder_pos_encodings, requires_grad=False)


  def forward(self,
              input_pose_seq,
              get_attn_weights=False,
              fold=None,
              eval_step=None):
    """Performs the forward pass of the pose transformers.

    Args:
      input_pose_seq: Shape [batch_size, src_sequence_length, dim_pose].
      target_pose_seq: Shape [batch_size, tgt_sequence_length, dim_pose].

    Returns:
      A tensor of the predicted sequence with shape [batch_size,
      tgt_sequence_length, dim_pose].
    """
    # 1) Encode the sequence with given pose encoder
    # [batch_size, sequence_length, model_dim]
    input_pose_seq = input_pose_seq
    if self._pose_embedding is not None:
        input_pose_seq = self._pose_embedding(input_pose_seq)

    # 2) compute the look-ahead mask and the positional encodings
    # [sequence_length, batch_size, model_dim]
    input_pose_seq = torch.transpose(input_pose_seq, 0, 1)

    # 3) compute the attention weights using the transformer
    # [target_sequence_length, batch_size, model_dim]
    memory = self._transformer(
        input_pose_seq,
        encoder_position_encodings=self._encoder_pos_encodings,
    )

    return memory



def model_factory(params, pose_embedding_fn):
  init_fn = utils.normal_init_ \
      if params['init_fn'] == 'normal_init' else utils.xavier_init_
  return PoseTransformer(
      pose_dim=params['pose_dim'],
      input_dim=params['input_dim'],
      source_seq_length=params['source_seq_len'],
      model_dim=params['model_dim'],
      num_encoder_layers=params['num_encoder_layers'],
      num_heads=params['num_heads'],
      dim_ffn=params['dim_ffn'],
      dropout=params['dropout'],
      init_fn=init_fn,
      pose_embedding=pose_embedding_fn(params),
      pos_encoding_params=(params['pos_enc_beta'], params['pos_enc_alpha'])
  )


if __name__ == '__main__':
  transformer = PoseTransformer(model_dim=_POSE_DIM, num_heads=6)
  transformer.eval()
  batch_size = 8
  model_dim = 256
  tgt_seq = torch.FloatTensor(batch_size, _TARGET_LENGTH, _POSE_DIM).fill_(1)
  src_seq = torch.FloatTensor(batch_size, _SOURCE_LENGTH-1, _POSE_DIM).fill_(1)

  outputs = transformer(src_seq, tgt_seq)
  print(outputs[-1].size())

