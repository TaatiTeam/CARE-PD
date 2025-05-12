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
"""Model of 1D convolutions for encoding pose sequences."""

import numpy as np
import os
import sys

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath + "/../")

import torch
import torch.nn as nn


class Pose1DEncoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=128, n_joints=21):
        super(Pose1DEncoder, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._n_joints = n_joints
        self.init_model()

    def init_model(self):
        self._model = nn.Sequential(
            nn.Conv1d(in_channels=self._input_channels, out_channels=32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=self._output_channels, kernel_size=3),
            nn.BatchNorm1d(self._output_channels),
            nn.ReLU(True),
            nn.Conv1d(in_channels=self._output_channels, out_channels=self._output_channels, kernel_size=3)
        )

    def forward(self, x):
        """
        Args:
          x: [batch_size, seq_len, skeleton_dim].
        """
        # inputs to model is [batch_size, channels, n_joints]
        # transform the batch to [batch_size*seq_len, dof, n_joints]
        bs, seq_len, dim = x.size()
        dof = dim // self._n_joints
        x = x.view(bs * seq_len, dof, self._n_joints)

        # [batch_size*seq_len, dof, n_joints]
        x = self._model(x)
        # [batch_size, seq_len, output_channels]
        x = x.view(bs, seq_len, self._output_channels)

        return x


class Pose1DTemporalEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Pose1DTemporalEncoder, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self.init_model()

    def init_model(self):
        self._model = nn.Sequential(
            nn.Conv1d(
                in_channels=self._input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=self._output_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self._output_channels),
            nn.ReLU(True),
            nn.Conv1d(in_channels=self._output_channels, out_channels=self._output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Args:
          x: [batch_size, seq_len, skeleton_dim].
        """
        # batch_size, skeleton_dim, seq_len
        x = torch.transpose(x, 1, 2)
        x = self._model(x)
        # batch_size, seq_len, skeleton_dim
        x = torch.transpose(x, 1, 2)
        return x


if __name__ == '__main__':
    dof = 9
    output_channels = 128
    n_joints = 21
    seq_len = 49

    model = Pose1DTemporalEncoder(input_channels=dof * n_joints, output_channels=output_channels)
    inputs = torch.FloatTensor(10, seq_len, dof * n_joints)
    X = model(inputs)
    print(X.size())

#  model = Pose1DEncoder(input_channels=dof, output_channels=output_channels)
#  inputs = torch.FloatTensor(10, seq_len, dof*n_joints)
#  X = model(inputs)
#  print(X.size())
