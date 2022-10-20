# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections import OrderedDict

import numpy as np
from fair_infra.data.prefetcher.add_prefetcher import Prefetch
from libfb.py import parutil

from . import dataset


class CMUPanopticDataset(dataset.ActiveLearningDataset):
    def __init__(self, data_cfg, gt_stride, split="val"):
        super().__init__(data_cfg, gt_stride, split)
        self._logger.info("Loading labels from %s." % self.data_cfg.PANOPTIC.LABEL_PATH)
        with self._pathmgr.open(self.data_cfg.PANOPTIC.LABEL_PATH) as f_labels:
            labels = json.load(f_labels)
        with self._pathmgr.open(self.data_cfg.PANOPTIC.GT_BOXES) as f_boxes:
            boxes = json.load(f_boxes)
        with open(parutil.get_file_path(self.data_cfg.PANOPTIC.TRAIN_VAL_SPLIT)) as f:
            self.train_val_split = json.load(f)
        self.list_of_cameras = list(self.train_val_split["cameras"][self.split])

        # Labels for test split is still stored under "val"
        label_split = "val" if self.split == "test" else self.split
        for pose in self.train_val_split[self.split]:
            for section in self.train_val_split[self.split][pose]:
                for frame_id in range(
                    section[0], section[1], self.data_cfg.PANOPTIC.SAMPLE_RATE
                ):
                    if str(frame_id) not in labels[label_split][pose]:
                        continue
                    frame = OrderedDict()
                    frame["views"] = OrderedDict()
                    for camera_name in self.list_of_cameras:
                        if camera_name not in boxes[label_split][pose][str(frame_id)]:
                            frame = None
                            break
                        view = dict()
                        view["camera"] = labels[label_split][pose]["cameras"][
                            camera_name
                        ]
                        view["path"] = os.path.join(
                            self.data_cfg.PANOPTIC.HOME,
                            pose,
                            "hdImgs",
                            camera_name,
                            "%s_%08d.jpg" % (camera_name, frame_id),
                        )
                        view["box"] = boxes[label_split][pose][str(frame_id)][
                            camera_name
                        ]
                        view["camera_name"] = camera_name
                        view["prepared"] = False
                        view["joint_valid"] = [[True]] * self.data_cfg.NUM_JOINTS
                        view["per_view_joint_valid"] = [
                            [True]
                        ] * self.data_cfg.NUM_JOINTS
                        frame["views"][camera_name] = view
                    guid = "%s%s-%d" % (pose[:6], pose[-1], frame_id)
                    if guid in self.unlabeled_data:
                        self._logger.warning("Duplicate GUID: %s." % guid)
                    elif frame is None:
                        self._logger.warning(
                            "GT Box not found for %s-%d" % (pose, frame_id)
                        )
                    else:
                        frame["3d_keypoints"] = np.array(
                            labels[label_split][pose][str(frame_id)]
                        )
                        frame["joint_valid"] = [[True]] * self.data_cfg.NUM_JOINTS
                        frame["pose"] = int(guid.split("-")[0])
                        frame["frame_id"] = frame_id
                        frame["guid"] = guid
                        self.unlabeled_data[guid] = frame
        self._logger.info(
            "Dataset # of views is %d."
            % (len(self.unlabeled_data) * self.get_num_view_per_frame())
        )


@Prefetch(memcache_key_prefix="oculus-nimble", seed=None)
class PrefetchCMUPanopticDataset(CMUPanopticDataset):
    def get_data_paths(self, idx):
        frame = self.data[idx]
        return [frame["views"][camera]["path"] for camera in frame["views"]]
