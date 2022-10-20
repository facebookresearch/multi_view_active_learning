# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN


_C = CN()
_C.PANOPTIC = CN()
_C.PANOPTIC.HOME = "memcache_manifold://oculus-nimble/tree/public_datasets/panoptic"
_C.PANOPTIC.TRAIN_VAL_SPLIT = "fblearner/flow/projects/nimble/multi_view_active_learning/dataset/train_val_split.json"
_C.PANOPTIC.LABEL_PATH = (
    "manifold://oculus-nimble/tree/public_datasets/active_learning/cmu_gt_lables.json"
)
_C.PANOPTIC.GT_BOXES = (
    "manifold://oculus-nimble/tree/public_datasets/active_learning/cmu_gt_boxes.json"
)
_C.PANOPTIC.SAMPLE_RATE = 25

_C.IH26M = CN()
_C.IH26M.HOME = (
    "memcache_manifold://oculus-nimble/tree/public_datasets/InterHand2.6M_5fps_batch1"
)
_C.IH26M.CAPTURE_RANGE = [0, 10]
_C.IH26M.TEST_CAPTURE_RANGE = [0, 1]
_C.IH26M.NUM_VIEW_PER_FRAME = 16
_C.IH26M.TEST_NUM_VIEW_PER_FRAME = 32

_C.INPUT_WIDTH = 256
_C.INPUT_HEIGHT = 256
_C.SCALE_BBOX = 1.0
_C.SIGMA = 1.0
_C.PSEUDO_LABEL_SIGMA = 1.0
# "panoptic" or "ih26m"
_C.TYPE = "panoptic"
_C.EPOCH_SIZE = 2000
# 19 for panoptic; 42 for ih26m
_C.NUM_JOINTS = 19

# Augmentation
_C.NUM_AUG = 0
_C.AUG_MAGNITUDE = 0
_C.USE_ROTATION = True
_C.USE_IMAGE_AUG = True
_C.USE_CONST_AUG_MAGNITUDE = True


def get_default_data_configs():
    return _C.clone()
