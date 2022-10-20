# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from fblearner.flow.projects.nimble.multi_view_active_learning.config import (
    get_default_configs,
)
from fblearner.flow.projects.nimble.multi_view_active_learning.dataset.ih26m_dataset import (
    InterHand26MDataset,
)
from fblearner.flow.projects.nimble.multi_view_active_learning.dataset.panoptic_dataset import (
    CMUPanopticDataset,
)
from fblearner.flow.projects.nimble.multi_view_active_learning.pose_estimators.pose_resnet import (
    PoseResNet,
)
from fblearner.flow.projects.nimble.multi_view_active_learning.strategy import (
    ActiveLearningStrategy,
)
from torch.utils.data import DataLoader


class TestALStrategy(unittest.TestCase):
    def test_kmeans(self):
        al_cfg = get_default_configs()
        al_cfg.EXPR_TYPE = "SAL"
        al_cfg.SAL.CLUSTER_FILE_PATH = "manifold://oculus-nimble-tensorboard/tree/fung/panoptics-pose-clusters.json"
        strategy = ActiveLearningStrategy(al_cfg)
        cluster_id = strategy.kmeans.predict([np.zeros(19 * 3)])[0]
        self.assertEqual(cluster_id, 4)

    def test_sampling_strategy(self):
        al_cfg = get_default_configs()
        al_cfg.TYPE = "AL"
        al_cfg.AL.STRATEGY = "RANDOM"
        al_cfg.DATA.PANOPTIC.SAMPLE_RATE = 512
        strategy = ActiveLearningStrategy(al_cfg)
        if not torch.cuda.is_available():
            return
        pose_estimator = PoseResNet(num_joints=19).cuda()
        train_dataset = CMUPanopticDataset(
            al_cfg.DATA.PANOPTIC, al_cfg.POSE_ESTIMATOR.STRIDE, train=True
        )
        strategy._get_dataloader = MagicMock(
            side_effect=self._get_dataloader_for_testing
        )
        train_dataset = strategy.sample_next_batch(
            train_dataset, 50, 50, pose_estimator, rank=0
        )
        train_dataset = strategy.sample_next_batch(
            train_dataset, 50, 50, pose_estimator, rank=0
        )

    def test_sampling_strategy_ih26m(self):
        al_cfg = get_default_configs()
        al_cfg.TYPE = "AL"
        al_cfg.AL.STRATEGY = "RANDOM"
        al_cfg.DATA.PANOPTIC.SAMPLE_RATE = 512
        strategy = ActiveLearningStrategy(al_cfg)
        if not torch.cuda.is_available():
            return
        pose_estimator = PoseResNet(num_joints=19).cuda()
        train_dataset = InterHand26MDataset(
            al_cfg.DATA.IH26M, al_cfg.POSE_ESTIMATOR.STRIDE, split="train"
        )
        strategy._get_dataloader = MagicMock(
            side_effect=self._get_dataloader_for_testing
        )
        train_dataset = strategy.sample_next_batch(
            train_dataset, 50, 50, pose_estimator, rank=0
        )
        train_dataset = strategy.sample_next_batch(
            train_dataset, 50, 50, pose_estimator, rank=0
        )

    @staticmethod
    def _get_dataloader_for_testing(
        train_dataset, batch_size, num_workers, la_interval, la_size
    ):
        return DataLoader(train_dataset, batch_size=batch_size, num_workers=0)


if __name__ == "__main__":
    unittest.main()
