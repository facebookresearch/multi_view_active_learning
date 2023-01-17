# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np

from . import dataset
from .ih26m_utils.ih26m_common_cams import get_default_common_cams


class InterHand26MDataset(dataset.ActiveLearningDataset):
    def __init__(self, data_cfg, gt_stride, split="val"):
        super().__init__(data_cfg, gt_stride, split)
        path_to_annotations = os.path.join(
            self.data_cfg.IH26M.HOME,
            "annotations",
            self.split,
        )
        with self._pathmgr.open(
            os.path.join(
                path_to_annotations,
                "InterHand2.6M_%s_camera.json" % self.split,
            )
        ) as f:
            self.all_cam = json.load(f)
        with self._pathmgr.open(
            os.path.join(
                path_to_annotations,
                "InterHand2.6M_%s_data.json" % self.split,
            )
        ) as f:
            data = json.load(f)
        data = list(zip(data["images"], data["annotations"]))
        with self._pathmgr.open(
            os.path.join(
                path_to_annotations,
                "InterHand2.6M_%s_joint_3d.json" % self.split,
            )
        ) as f:
            joints = json.load(f)
        for meta, annotation in data:
            capture = str(meta["capture"])
            if self.split == "train" and int(capture) not in range(
                self.data_cfg.IH26M.CAPTURE_RANGE[0],
                self.data_cfg.IH26M.CAPTURE_RANGE[1],
            ):
                continue
            if self.split == "test" and int(capture) not in range(
                self.data_cfg.IH26M.TEST_CAPTURE_RANGE[0],
                self.data_cfg.IH26M.TEST_CAPTURE_RANGE[1],
            ):
                continue
            frame_id = str(meta["frame_idx"])
            guid = "%s-%s" % (capture, frame_id)
            view = dict()
            ih_joint = joints[capture][frame_id]
            view["3d_keypoints"] = ih_joint["world_coord"]
            view["joint_valid"] = ih_joint["joint_valid"]
            view["per_view_joint_valid"] = annotation["joint_valid"]
            if len(view["3d_keypoints"]) != self.data_cfg.NUM_JOINTS:
                self._logger.warning(
                    "Frame does not correct number of keypoints: %s" % guid
                )
                continue
            view["3d_keypoints"] = np.transpose(np.array(view["3d_keypoints"]), [1, 0])
            view["hand_type"] = ih_joint["hand_type"]
            if self._filter_view(ih_joint, annotation):
                continue
            view["camera"] = self._get_camera(capture, meta["camera"])
            view["path"] = os.path.join(
                self.data_cfg.IH26M.HOME,
                "images",
                self.split,
                meta["file_name"],
            )
            xywh_box = annotation["bbox"]
            left, top, width, height = xywh_box
            right = left + width
            bottom = top + height
            view["box"] = (left, top, right, bottom)
            view["camera_name"] = meta["camera"]
            if guid in self.unlabeled_data:
                if meta["camera"] in self.unlabeled_data[guid]:
                    self._logger.warning(
                        "Duplicate Camera (%s) Found in GUID: %s."
                        % (meta["camera"], guid)
                    )
                self.unlabeled_data[guid]["views"][meta["camera"]] = view
            else:
                frame = dict()
                frame["views"] = {meta["camera"]: view}
                ih_joint = joints[capture][frame_id]
                frame["3d_keypoints"] = np.transpose(
                    np.array(ih_joint["world_coord"]), [1, 0]
                )
                frame["joint_valid"] = ih_joint["joint_valid"]
                frame["hand_type"] = ih_joint["hand_type"]
                frame["pose"] = int(capture)
                frame["frame_id"] = int(frame_id)
                frame["guid"] = guid
                self.unlabeled_data[guid] = frame
        selected_cams = self._select_camera()
        for guid in list(self.unlabeled_data.keys()):
            self.unlabeled_data[guid]["views"] = {
                cam: self.unlabeled_data[guid]["views"][cam]
                for cam in selected_cams
                if cam in self.unlabeled_data[guid]["views"]
            }
            if len(self.unlabeled_data[guid]["views"].keys()) != len(selected_cams):
                self._logger.warning(
                    "Frame %s does not have %d views." % (guid, len(selected_cams))
                )
                del self.unlabeled_data[guid]
        self._logger.info("Dataset # of frames is %d." % (len(self.unlabeled_data)))

    def _filter_view(self, ih_joint, annotation):
        if (
            ih_joint["hand_type"] == "right"
            and not np.array(ih_joint["joint_valid"][:21]).all()
            and not np.array(annotation["joint_valid"][:21]).all()
        ):
            return True
        if (
            ih_joint["hand_type"] == "left"
            and not np.array(ih_joint["joint_valid"][21:]).all()
            and not np.array(annotation["joint_valid"][21:]).all()
        ):
            return True
        if (
            ih_joint["hand_type"] == "interacting"
            and not np.array(ih_joint["joint_valid"]).all()
            and not np.array(annotation["joint_valid"]).all()
        ):
            return True
        return False

    def _select_camera(self):
        if self.split == "train":
            selected_cams = get_default_common_cams(
                self.all_cam,
                list(
                    range(
                        self.data_cfg.IH26M.CAPTURE_RANGE[0],
                        self.data_cfg.IH26M.CAPTURE_RANGE[1],
                    )
                ),
                num_cams=self.data_cfg.IH26M.NUM_VIEW_PER_FRAME,
            )
        elif self.split == "val":
            selected_cams = get_default_common_cams(
                self.all_cam, ["0"], num_cams=self.data_cfg.IH26M.NUM_VIEW_PER_FRAME
            )
        elif self.split == "test":
            selected_cams = get_default_common_cams(
                self.all_cam,
                list(
                    range(
                        self.data_cfg.IH26M.TEST_CAPTURE_RANGE[0],
                        self.data_cfg.IH26M.TEST_CAPTURE_RANGE[1],
                    )
                ),
                num_cams=self.data_cfg.IH26M.TEST_NUM_VIEW_PER_FRAME,
            )
        else:
            raise NotImplementedError("Split type %s not defined." % self.split)
        return selected_cams

    def _get_camera(self, capture, camera_name):
        cam = dict()
        cam["R"] = np.array(self.all_cam[capture]["camrot"][camera_name])
        t = -np.array(self.all_cam[capture]["campos"][camera_name])
        cam["t"] = cam["R"] @ t
        f = np.array(self.all_cam[capture]["focal"][camera_name])
        p = np.array(self.all_cam[capture]["princpt"][camera_name])
        K = np.diag([f[0], f[1], 1])
        K[0, 2], K[1, 2] = p[0], p[1]
        cam["K"] = K
        cam["dist"] = None
        return cam
