import abc

import torch
import torch.manifold.patch
from fblearner.flow.projects.nimble.multi_view_active_learning.utils import get_logger


class PoseEstimator(abc.ABC, torch.nn.Module):
    """Abstract class for Pose Estimation Models."""

    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self._logger = get_logger(__name__)

    @abc.abstractmethod
    def forward(self, x):
        pass
