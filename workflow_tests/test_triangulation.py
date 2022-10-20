# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from fblearner.flow.projects.nimble.multi_view_active_learning.utils.triangulation import (
    triangulation,
)


class TestTriangulation(unittest.TestCase):
    def test_triangulation(self):
        proj_matrices = torch.Tensor(
            [
                [
                    [-3.4295e02, 6.8706e-01, 2.7159e02, 2.3787e04],
                    [-4.6607e01, 4.3164e02, -2.0346e01, 7.8610e04],
                    [-7.8969e-01, 1.1833e-01, -6.0199e-01, 2.9049e02],
                ],
                [
                    [-3.3046e02, 6.6025e01, 4.2674e02, 3.2056e04],
                    [2.5996e02, 4.6107e02, 1.6285e02, 9.8670e04],
                    [-5.6597e-01, 7.5490e-01, -3.3136e-01, 3.7961e02],
                ],
                [
                    [-4.5085e02, 2.6791e01, 1.0916e02, 2.2007e04],
                    [5.7573e01, 4.2051e02, 1.9940e02, 8.7025e04],
                    [-2.9529e-01, 6.2227e-01, -7.2497e-01, 3.6151e02],
                ],
                [
                    [4.4427e02, 1.2182e02, 2.6364e02, 6.7640e04],
                    [5.1922e01, 4.4883e02, -2.4331e02, 8.9032e04],
                    [-2.5639e-01, 6.2335e-01, 7.3871e-01, 3.5767e02],
                ],
                [
                    [-3.9666e02, 3.8124e01, -7.0161e01, 2.4880e04],
                    [1.2124e01, 4.1112e02, 7.6234e01, 7.8864e04],
                    [8.3997e-02, 4.5241e-01, -8.8784e-01, 3.3700e02],
                ],
                [
                    [3.8276e02, 1.5859e02, -4.3537e02, 6.9321e04],
                    [-3.1722e02, 4.7013e02, -1.2831e02, 7.6926e04],
                    [6.2258e-01, 6.5737e-01, 4.2456e-01, 3.6126e02],
                ],
                [
                    [-2.6491e02, 4.5420e01, -4.2691e02, 3.0369e04],
                    [-2.0931e02, 4.2484e02, 1.5935e02, 7.6858e04],
                    [5.7460e-01, 6.4258e-01, -5.0688e-01, 3.6323e02],
                ],
                [
                    [-7.5390e01, 5.6447e01, 4.6836e02, 4.3622e04],
                    [1.3967e01, 4.6652e02, -3.0601e01, 8.7342e04],
                    [-9.7148e-01, 2.2094e-01, 8.6076e-02, 3.0445e02],
                ],
            ]
        )
        heatmaps = torch.zeros([8, 19, 64, 64])
        valid_joints = torch.ones([19]).bool()
        heatmaps[:, :, 11, 11] = 1.0
        heatmaps[:, :, 10, 11] = 0.5
        heatmaps[:, :, 11, 10] = 0.5
        heatmaps[:, :, 11, 12] = 0.5
        heatmaps[:, :, 12, 11] = 0.5
        heatmaps[:, :, 12, 12] = 0.3
        heatmaps[:, :, 10, 10] = 0.3
        heatmaps[:, :, 10, 12] = 0.3
        heatmaps[:, :, 12, 10] = 0.3

        results = triangulation(heatmaps, proj_matrices, 8, valid_joints)
        self.assertListEqual(
            list(results["keypoints_3d"].shape),
            [19, 3],
        )
        self.assertListEqual(
            list(results["keypoints_2d"].shape),
            [8, 19, 2],
        )
        print(results["inlier_count"])


if __name__ == "__main__":
    unittest.main()
