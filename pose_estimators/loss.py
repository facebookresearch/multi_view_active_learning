# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class Pose2DMeanSquaredError:
    def __init__(self):
        self.loss = torch.nn.MSELoss(reduction="none")

    def pose_2d_mse(self, heatmaps, gt_heatmaps, joint_valid=None):
        loss = self.loss(heatmaps, gt_heatmaps)
        if joint_valid is not None:
            loss = torch.where(joint_valid, loss, torch.zeros_like(loss))
        return torch.sum(loss) / (
            heatmaps.shape[0] * heatmaps.shape[-1] * heatmaps.shape[-2]
        )

    def pose_2d_mse_single_batch(self, heatmap, gt_heatmap):
        loss = self.loss(heatmap, gt_heatmap)
        return torch.sum(loss) / (heatmap.shape[-1] * heatmap.shape[-2])
