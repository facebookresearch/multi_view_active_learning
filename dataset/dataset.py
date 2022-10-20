# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import io
import random
from collections import OrderedDict

import numpy as np
import torch
from utils import (
    get_logger,
    triangulation,
)
from iopath.common.file_io import PathManager
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import RandAugment


class ActiveLearningDataset(Dataset, abc.ABC):
    def __init__(self, data_cfg, gt_stride, split):
        if split not in ["train", "val", "test"]:
            raise NotImplementedError("%s is not implemented for Dataset." % split)
        self.data_cfg = data_cfg
        self.augmentation = RandAugment(
            self.data_cfg.NUM_AUG,
            self.data_cfg.AUG_MAGNITUDE,
            self.data_cfg.USE_ROTATION,
            self.data_cfg.USE_IMAGE_AUG,
            self.data_cfg.USE_CONST_AUG_MAGNITUDE,
        )
        self.gt_stride = gt_stride
        self.split = split
        self._logger = get_logger(__name__)
        self._pathmgr = PathManager()
        self.unlabeled_data = OrderedDict()
        self.labeled_data = list()
        self.pseudo_labeled_data = list()
        self.pseudo_label_guids = list()
        self.data = list()

    def get_al_dict_for_coreset(self):
        return {
            idx: np.array(self.labeled_data[idx]["3d_keypoints"]).transpose([1, 0])
            for idx in range(len(self.labeled_data))
        }

    def get_num_view_per_frame(self):
        return len(self.list_of_cameras)

    def label_all(self, preload=False):
        for guid in self.unlabeled_data:
            self.labeled_data.append(self.unlabeled_data[guid])
        self.unlabeled_data = OrderedDict()

    def label_by_frame_guids(self, guids):
        for guid in guids:
            self.labeled_data.append(self.unlabeled_data[guid])
            del self.unlabeled_data[guid]

    def pseudo_label_by_frame_guids(self, guids, pseudo_labels):
        self.pseudo_label_guids = guids
        self.pseudo_labeled_data = list()
        for guid in guids:
            frame = self.unlabeled_data[guid].copy()
            frame["pseudo_3d_keypoints"] = np.array(pseudo_labels[guid]).transpose(
                [1, 0]
            )
            self.pseudo_labeled_data.append(frame)

    def resample_frames(self, num_frames=-1, epoch_size=0):
        """
        For training, validation and viusalization etc.
        """
        if num_frames == -1:
            self.data = (self.labeled_data + self.pseudo_labeled_data).copy()
            if self.split == "train":
                copies = epoch_size // len(self.data)
                self._logger.info(
                    "Resampling. Duplicating labeled data by %d times." % (copies + 1)
                )
                for _ in range(copies):
                    self.data += (self.labeled_data + self.pseudo_labeled_data).copy()
            random.shuffle(self.data)
        else:
            if num_frames > len(self.labeled_data):
                raise ArithmeticError(
                    "Labeled data size is %d, while sampling size is %d."
                    % (len(self.labeled_data), num_frames)
                )
            self.data = random.sample(self.labeled_data, num_frames)

    def resample_unlabeled_data(self):
        unlabeled = []
        for guid in self.unlabeled_data:
            unlabeled.append(self.unlabeled_data[guid])
        self.data = unlabeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx].copy()
        batched_frame = self.prepare_frame(frame)
        return batched_frame

    def prepare_frame(self, frame):
        batched_frame = dict()
        images = []
        heatmaps = []
        kps = []
        kps_after_crop = []
        proj_matrices = []
        square_box = []
        cam_name = []
        joint_valid = []
        per_view_joint_valid = []
        for camera_name in frame["views"]:
            view = frame["views"][camera_name].copy()
            if "pseudo_3d_keypoints" in frame:
                view = self.prepare_single_view(
                    view, frame["pseudo_3d_keypoints"], self.data_cfg.PSEUDO_LABEL_SIGMA
                )
            else:
                view = self.prepare_single_view(
                    view, frame["3d_keypoints"], self.data_cfg.SIGMA
                )
            images.append(view["images"])
            heatmaps.append(view["gt_heatmap"])
            kps.append(view["2d_keypoints"])
            kps_after_crop.append(view["2d_after_crop"])
            proj_matrices.append(view["proj_matrices"])
            square_box.append(view["square_box"])
            cam_name.append(view["camera_name"])
            joint_valid.append(view["joint_valid"])
            per_view_joint_valid.append(view["per_view_joint_valid"])
        batched_frame["images"] = torch.stack(images)
        batched_frame["gt_heatmap"] = torch.stack(heatmaps)
        batched_frame["2d_keypoints"] = torch.stack(kps)
        batched_frame["2d_after_crop"] = torch.stack(kps_after_crop)
        batched_frame["proj_matrices"] = torch.stack(proj_matrices)
        batched_frame["3d_keypoints"] = torch.Tensor(frame["3d_keypoints"])
        batched_frame["square_box"] = torch.stack(square_box)
        batched_frame["pose"] = frame["pose"]
        batched_frame["frame_id"] = frame["frame_id"]
        batched_frame["camera_name"] = cam_name
        batched_frame["joint_valid"] = torch.Tensor(joint_valid[0]).squeeze(-1)
        batched_frame["per_view_joint_valid"] = torch.Tensor(
            per_view_joint_valid
        ).squeeze(-1)
        return batched_frame

    def prepare_single_view(self, view, kp_3d, sigma):
        f = self._pathmgr.open(view["path"], "rb")
        f = f.read()
        image = Image.open(io.BytesIO(f))
        image = np.array(image)[..., ::-1]

        left, top, right, bottom = view["box"]
        bbox = (left, top, right, bottom)

        if top - bottom == 0 or left - right == 0:
            self._logger.debug("GT box invalid. %s" % (view["path"]))

        bbox = triangulation.get_square_bbox(bbox)
        bbox = triangulation.scale_bbox(bbox, self.data_cfg.SCALE_BBOX)
        view["square_box"] = torch.Tensor(bbox)
        retval_camera = triangulation.Camera(
            view["camera"]["R"],
            view["camera"]["t"],
            view["camera"]["K"],
            view["camera"]["dist"],
            name=view["camera_name"],
        )
        image = triangulation.crop_image(image, bbox)
        retval_camera.update_after_crop(bbox)
        skel = np.array(kp_3d.transpose([1, 0]))[:, :3]
        pt = triangulation.project_3d_points_with_camera(retval_camera, skel)
        view["2d_after_crop"] = torch.Tensor(pt).to(dtype=torch.float32)
        image_shape_before_resize = image.shape[:2]

        retval_camera.update_after_resize(
            image_shape_before_resize,
            self.data_cfg.INPUT_WIDTH,
            self.data_cfg.INPUT_HEIGHT,
        )

        view["image_shapes_before_resize"] = image_shape_before_resize

        view["proj_matrices"] = torch.from_numpy(retval_camera.projection)
        pt = triangulation.project_3d_points_with_camera(retval_camera, skel)
        view["2d_keypoints"] = torch.Tensor(pt).to(dtype=torch.float32)
        pt = torch.from_numpy(pt) / self.gt_stride
        w = self.data_cfg.INPUT_WIDTH // self.gt_stride
        h = self.data_cfg.INPUT_HEIGHT // self.gt_stride
        grid = torch.zeros(size=(h, w, 2))
        grid[..., 0] = torch.from_numpy(np.arange(w)).unsqueeze(0)
        grid[..., 1] = torch.from_numpy(np.arange(h)).unsqueeze(1)
        grid = grid.unsqueeze(0)
        labels = pt.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((grid - labels) ** 2, dim=-1)
        gt_heatmap = torch.exp(-exponent / (2.0 * (sigma**2)))
        image = Image.fromarray(image).resize(
            (self.data_cfg.INPUT_WIDTH, self.data_cfg.INPUT_HEIGHT),
            resample=Image.LANCZOS,
        )
        if self.split == "train":
            image, gt_heatmap = self.augmentation(image, gt_heatmap)
        image = np.asarray(image)
        image = triangulation.normalize_image(image)
        view["images"] = (
            torch.from_numpy(image).permute([2, 0, 1]).to(dtype=torch.float32)
        )
        view["gt_heatmap"] = gt_heatmap.to(dtype=torch.float32)
        return view
