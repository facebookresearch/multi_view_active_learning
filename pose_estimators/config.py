# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN


_C = CN()
_C.TYPE = "POSE_RESNET"
_C.LOAD_CNN_WEIGHTS = True
_C.STRIDE = 4

_C.HRNET = CN()

_C.HRNET.PRETRAINED_LAYERS = [
    "conv1",
    "bn1",
    "conv2",
    "bn2",
    "layer1",
    "transition1",
    "stage2",
    "transition2",
    "stage3",
]
_C.HRNET.FINAL_CONV_KERNEL = 1

_C.HRNET.STAGE2 = CN()
_C.HRNET.STAGE2.NUM_MODULES = 1
_C.HRNET.STAGE2.NUM_BRANCHES = 2
_C.HRNET.STAGE2.BLOCK = "BASIC"
_C.HRNET.STAGE2.NUM_BLOCKS = [4, 4]
_C.HRNET.STAGE2.NUM_CHANNELS = [32, 64]
_C.HRNET.STAGE2.FUSE_METHOD = "SUM"

_C.HRNET.STAGE3 = CN()
_C.HRNET.STAGE3.NUM_MODULES = 4
_C.HRNET.STAGE3.NUM_BRANCHES = 3
_C.HRNET.STAGE3.BLOCK = "BASIC"
_C.HRNET.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.HRNET.STAGE3.NUM_CHANNELS = [32, 64, 128]
_C.HRNET.STAGE3.FUSE_METHOD = "SUM"

_C.HRNET.STAGE4 = CN()
_C.HRNET.STAGE4.NUM_MODULES = 3
_C.HRNET.STAGE4.NUM_BRANCHES = 4
_C.HRNET.STAGE4.BLOCK = "BASIC"
_C.HRNET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.HRNET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
_C.HRNET.STAGE4.FUSE_METHOD = "SUM"


def get_default_configs():
    return _C.clone()
