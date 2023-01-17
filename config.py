# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from yacs.config import CfgNode as CN

from dataset.config import get_default_data_configs
from pose_estimators.config import \
    get_default_configs as get_default_pose_estimators_configs

_C = CN()
_C.EXPR_NAME = "EXPR"
# Type of Experiments
# AL                - Active Learning Full Pipeline.
# SUPERVISED        - Fully supervised training.
# AL_EVAL           - Evaluate all AL Iterations.
# EVAL              - Evaluate the performance of a HPE model from a checkpoint.
# CLUSTER           - Cluster the dataset based on either "hardness" or "pose".
# SAL               - Self Active Learning Full Pipeline.
_C.EXPR_TYPE = "SUPERVISED"
_C.LOG_DIR = "manifold://oculus-nimble-tensorboard/tree/fung/exprs"
_C.COMMENT = "N/A"
_C.RANDOM_SEED = 1307

_C.SAL = CN()
_C.SAL.NUM_FRAMES = [0, 20, 20, 30, 30, 40, 40, 50, 50, 50]
_C.SAL.INLIER_THRESHOLD = 7
_C.SAL.CLUSTER_FILE_PATH = ""
_C.SAL.NUM_CLUSTERS = 10

_C.AL = CN()
# AL strategy supported:
# HP
# BSB
# RANDOM
# MPE
# TRIANGULATION
# CORESET
_C.AL.STRATEGY = "RANDOM"
# Number of frames.
_C.AL.INITIAL_AMOUNT = 200
_C.AL.ITER_AMOUNT = 100
_C.AL.START_ITER = 0
_C.AL.PREVIOUS_AL_LOG_DIR = ""
_C.AL.ITERATIONS = 10
# For Triangulation Strategy.
_C.AL.USE_SOFTARGMAX = False
_C.AL.USE_REPROJECTION_XE = False
_C.AL.REPROJECTION_SIGMA = 1.0
# For MPE Strategy, AVG or STD.
_C.AL.MPE_CONFIG = "AVG"
# For BSB Strategy, AVG or STD.
_C.AL.BSB_CONFIG = "AVG"
# For HP Strategy, AVG or STD.
_C.AL.HP_CONFIG = "AVG"
# Weights for epipolar reprojection based strategy. Divergence & Entropy
_C.AL.EPI_WEIGHTS = [0.5, 0.5]

_C.AL.INFERENCE = CN()
# For AL_EVAL experiments.
_C.AL.INFERENCE.TRAINING_EXPR_NAME = ""
# For EVAL experiments.
_C.AL.INFERENCE.RESTORE_FROM = ""
# Default. Do not change.
_C.AL.INFERENCE.BATCH_SIZE = 2
_C.AL.INFERENCE.NUM_WORKERS = 2

_C.AL.CLUSTER = CN()
# Cluster by LOSS or by POSE.
_C.AL.CLUSTER.TYPE = "LOSS"
_C.AL.CLUSTER.SAVE_PATH = ""

_C.TRAIN = CN()
# For FIRST_BATCH experiment type.
_C.TRAIN.INIT_WEIGHT = ""
# For AL experiment type.
_C.TRAIN.AL_FIRST_BATCH_WEIGHTS = ""
_C.TRAIN.LOSS_CLIP_VALUE = 10.0
_C.TRAIN.RESTORE_FROM = ""

_C.TRAIN.FULLY_SUPERVISED_FRAME_COUNT = -1
# Default. Do not change.
_C.TRAIN.SIZE = -1
_C.TRAIN.VAL_SIZE = 320
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_WORKERS = 2
_C.TRAIN.LOG_EVERY_ITER = 500

_C.TRAIN.OPTIM = CN()
_C.TRAIN.OPTIM.TOTAL_STEPS = 5000
_C.TRAIN.OPTIM.LR = 0.001
_C.TRAIN.OPTIM.LR_DECAY_STEP_SIZE = 3000

_C.EVAL = CN()
# "2DPCKH", "3DPCK", "3DPCKH", "MKPE"
_C.EVAL.METRIC = "3DPCK"

_C.POSE_ESTIMATOR = get_default_pose_estimators_configs()

_C.DATA = get_default_data_configs()


def get_default_configs():
    return _C.clone()
