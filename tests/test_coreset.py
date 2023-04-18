# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
	
import unittest
import uuid

from utils.coreset import CoreSet


class TestCoreSet(unittest.TestCase):
    def test_softargmax_evaluation(self):
        sal_dict = {uuid.uuid4(): [[0, 1, 2] for _ in range(19)] for i in range(20)}
        al_dict = {i: [[0, 1, 2] for _ in range(19)] for i in range(5)}
        coreset = CoreSet(sal_dict, al_dict, 2)
        batch = coreset.select_batch(5)
        for selected_id in batch:
            self.assertTrue(selected_id in sal_dict.keys())


if __name__ == "__main__":
    unittest.main()