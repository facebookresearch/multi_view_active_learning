# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import numpy as np
from sklearn.metrics import pairwise_distances


class CoreSet:
    def __init__(self, sal_dict, al_dict, joint_root_index, metric="euclidean"):
        self.sal_dict = OrderedDict(sal_dict)
        self.al_dict = OrderedDict(al_dict)

        print(list(self.sal_dict.values())[0])
        print(list(self.al_dict.values())[0])
        self.features = self._compute_stacked_features(joint_root_index)

        print("feature shape: ", self.features[0].shape)
        print("len sal_dict: ", len(self.sal_dict))
        print("len al_dict: ", len(self.al_dict))

        self.sal_keys = list(self.sal_dict.keys())
        self.name = "kcenter"
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = len(sal_dict) + len(al_dict)
        self.al_indices = list(range(len(sal_dict), len(sal_dict) + len(al_dict)))
        self.already_selected = []

    def _compute_stacked_features(self, root_idx):
        all_pose_list = list(self.sal_dict.values()) + list(self.al_dict.values())
        print("pose 0: ", all_pose_list[0])
        # panoptic, human pose, single root
        # TODO interhand, hand pose, 2 roots for left/right hand
        features = [
            (
                np.array(pose).transpose([1, 0])[0:3, :]
                - np.array(pose).transpose([1, 0])[0:3, root_idx : root_idx + 1]
            ).flatten()
            for pose in all_pose_list
        ]
        return np.stack(features)

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [
                d for d in cluster_centers if d not in self.already_selected
            ]
        if cluster_centers:
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """
        already_selected = self.al_indices
        self.update_distances(already_selected, only_new=True, reset_dist=False)
        new_batch = []
        for _ in range(N):
            if self.already_selected is None:
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            assert ind not in already_selected
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        self.already_selected = already_selected
        return [self.sal_keys[i] for i in new_batch]
