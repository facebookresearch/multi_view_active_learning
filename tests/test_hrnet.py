# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pose_estimators.hrnet import PoseHighResolutionNet


class TestHRNet(unittest.TestCase):
    def test_output_shape(self):
        hrnet = PoseHighResolutionNet(num_joints=19)
        hrnet.train()
        x = torch.randn([2, 3, 256, 256])
        output = hrnet.forward(x)
        self.assertListEqual(
            list(output.detach().numpy().shape),
            [2, 19, 64, 64],
        )


if __name__ == "__main__":
    unittest.main()
