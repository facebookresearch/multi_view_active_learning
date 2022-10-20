# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch
from utils import get_logger


class PoseEstimator(abc.ABC, torch.nn.Module):
    """Abstract class for Pose Estimation Models."""

    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self._logger = get_logger(__name__)

    @abc.abstractmethod
    def forward(self, x):
        pass
