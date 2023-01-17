# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
from dataset.config import get_default_data_configs
from dataset.ih26m_dataset import InterHand26MDataset
from torch.utils.data import DataLoader


class TestInterHand26MDataset(unittest.TestCase):
    def setUp(self):
        self.data_cfg = get_default_data_configs()
        self.data_cfg.TYPE = "ih26m"
        self.data_cfg.NUM_JOINTS = 42
        self.dataset = InterHand26MDataset(self.data_cfg, 8, split="val")

    def test_load_unlabedled_dataset(self):
        self.dataset.resample_unlabeled_data()
        dp = self.dataset.__getitem__(20)
        self.assertIsInstance(dp["pose"], int)
        self.assertIsInstance(dp["frame_id"], int)
        self.assertListEqual(
            list(dp["gt_heatmap"].numpy().shape),
            [
                self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                self.data_cfg.NUM_JOINTS,
                self.data_cfg.INPUT_HEIGHT // 8,
                self.data_cfg.INPUT_WIDTH // 8,
            ],
        )
        self.assertListEqual(
            list(dp["images"].numpy().shape),
            [
                self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                3,
                self.data_cfg.INPUT_HEIGHT,
                self.data_cfg.INPUT_WIDTH,
            ],
        )

    def test_load_dataset(self):
        self.dataset.label_all()
        self.dataset.resample_frames(2)
        dp = self.dataset.__getitem__(0)
        self.assertListEqual(
            list(dp["gt_heatmap"].numpy().shape),
            [
                self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                self.data_cfg.NUM_JOINTS,
                self.data_cfg.INPUT_HEIGHT // 8,
                self.data_cfg.INPUT_WIDTH // 8,
            ],
        )
        self.assertListEqual(
            list(dp["images"].numpy().shape),
            [
                self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                3,
                self.data_cfg.INPUT_HEIGHT,
                self.data_cfg.INPUT_WIDTH,
            ],
        )

    def test_data_loader(self):
        self.dataset.label_all()
        self.dataset.resample_frames(10)
        dataloader = DataLoader(self.dataset, batch_size=2, num_workers=0)
        for dp in dataloader:
            self.assertListEqual(
                list(dp["gt_heatmap"].numpy().shape),
                [
                    2,
                    self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                    self.data_cfg.NUM_JOINTS,
                    self.data_cfg.INPUT_HEIGHT // 8,
                    self.data_cfg.INPUT_WIDTH // 8,
                ],
            )
            self.assertListEqual(
                list(dp["images"].numpy().shape),
                [
                    2,
                    self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                    3,
                    self.data_cfg.INPUT_HEIGHT,
                    self.data_cfg.INPUT_WIDTH,
                ],
            )
            pose = np.array(dp["pose"]).transpose()
            self.assertEqual(len(pose), 2)
            self.assertListEqual(
                list(dp["2d_after_crop"].numpy().shape),
                [
                    2,
                    self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
                    self.data_cfg.NUM_JOINTS,
                    2,
                ],
            )
            self.assertListEqual(
                list(dp["joint_valid"].numpy().shape),
                [2, self.data_cfg.NUM_JOINTS],
            )


if __name__ == "__main__":
    unittest.main()
