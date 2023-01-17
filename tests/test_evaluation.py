# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from utils.evaluation import get_pred_coordinates


class TestEvaluation(unittest.TestCase):
    def test_softargmax_evaluation(self):
        heatmaps = torch.randn([8, 19, 64, 64])
        boxes = torch.Tensor(8 * [[0, 0, 64, 64]])
        coords = get_pred_coordinates(heatmaps, boxes, 19)
        np.testing.assert_array_equal([8, 19, 2], np.array(coords).shape)


if __name__ == "__main__":
    unittest.main()
