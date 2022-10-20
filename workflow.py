#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
from datetime import datetime
from typing import List, NamedTuple

import fblearner.flow.api as flow
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from libfb.py import parutil

from . import config
from .dataset.ih26m_dataset import PrefetchInterHand26MDataset
from .dataset.panoptic_dataset import PrefetchCMUPanopticDataset
from .pose_estimators.cpm import ConvolutionalPoseMachines
from .pose_estimators.hrnet import PoseHighResolutionNet
from .pose_estimators.pose_resnet import PoseResNet
from .strategy import ActiveLearningStrategy
from .utils import get_logger


def main(rank, cfg):
    strategy = ActiveLearningStrategy(cfg)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=cfg.NUM_GPUS, init_method=cfg.INIT_METHOD
    )
    torch.cuda.set_device(rank)
    backbone_pose_estimator = _build_ddp_model(cfg, rank)
    if cfg.EXPR_TYPE == "AL" or cfg.EXPR_TYPE == "SAL":
        if rank == 0:
            strategy.prepare_al_experiments()
        if cfg.AL.CURRENT_ITER == 0:
            if cfg.AL.PREVIOUS_AL_LOG_DIR != "":
                if rank == 0:
                    strategy._copy_previous_checkpoints()
                dist.destroy_process_group()
                return
            else:
                amount = cfg.AL.INITIAL_AMOUNT
        else:
            restore_cfg = cfg.clone()
            restore_cfg.TRAIN.RESTORE_FROM = os.path.join(
                restore_cfg.LOG_DIR,
                restore_cfg.EXPR_NAME,
                "ITER-%d" % (restore_cfg.AL.CURRENT_ITER - 1),
                "checkpoints",
                "CKPT-FINAL.pth",
            )
            backbone_pose_estimator = strategy._load_weights(
                restore_cfg, backbone_pose_estimator
            )
            amount = cfg.AL.ITER_AMOUNT

        train_dataset, val_dataset = build_datasets(cfg)
        train_dataset = strategy.restore_dataset(train_dataset, cfg.AL.CURRENT_ITER)
        train_dataset = strategy.sample_next_batch(
            train_dataset,
            amount,
            cfg.SAL.NUM_FRAMES[cfg.AL.CURRENT_ITER],
            backbone_pose_estimator,
            cfg.AL.CURRENT_ITER,
            rank,
        )
        del backbone_pose_estimator
        backbone_pose_estimator = _build_ddp_model(cfg, rank)
        strategy.al_iteration(
            backbone_pose_estimator,
            train_dataset,
            val_dataset,
            cfg.AL.CURRENT_ITER,
            rank,
        )
    elif cfg.EXPR_TYPE == "SUPERVISED":
        train_dataset, val_dataset = build_datasets(cfg)
        strategy.run_fully_supervised_training(
            backbone_pose_estimator, train_dataset, val_dataset, rank
        )
    elif cfg.EXPR_TYPE == "AL_EVAL":
        test_dataset = build_datasets(cfg)
        result = strategy.run_al_eval(backbone_pose_estimator, test_dataset, rank)
        if rank == 0:
            with open(cfg.RESULTS, "w") as f:
                json.dump(result, f)
    elif cfg.EXPR_TYPE == "EVAL":
        test_dataset = build_datasets(cfg)
        result = strategy.run_eval(backbone_pose_estimator, test_dataset, rank)
        if rank == 0:
            with open(cfg.RESULTS, "w") as f:
                json.dump(result, f)
    elif cfg.EXPR_TYPE == "CLUSETER":
        train_dataset, _ = build_datasets(cfg)
        strategy.cluster(backbone_pose_estimator, train_dataset, rank)
    dist.destroy_process_group()


def build_datasets(cfg):
    datasets = {
        "panoptic": PrefetchCMUPanopticDataset,
        "ih26m": PrefetchInterHand26MDataset,
    }
    if cfg.EXPR_TYPE == "AL_EVAL" or cfg.EXPR_TYPE == "EVAL":
        test_dataset = datasets[cfg.DATA.TYPE](
            cfg.DATA, cfg.POSE_ESTIMATOR.STRIDE, split="test"
        )
        return test_dataset

    train_dataset = datasets[cfg.DATA.TYPE](
        cfg.DATA, cfg.POSE_ESTIMATOR.STRIDE, split="train"
    )
    val_dataset = datasets[cfg.DATA.TYPE](
        cfg.DATA, cfg.POSE_ESTIMATOR.STRIDE, split="val"
    )
    return train_dataset, val_dataset


def _build_ddp_model(cfg, rank):
    if cfg.POSE_ESTIMATOR.TYPE == "CPM":
        backbone_pose_estimator = ConvolutionalPoseMachines(cfg.DATA.NUM_JOINTS)
    elif cfg.POSE_ESTIMATOR.TYPE == "POSE_RESNET":
        backbone_pose_estimator = PoseResNet(cfg.DATA.NUM_JOINTS)
    elif cfg.POSE_ESTIMATOR.TYPE == "HRNET":
        backbone_pose_estimator = PoseHighResolutionNet(
            cfg.DATA.NUM_JOINTS, hrnet_cfg=cfg.POSE_ESTIMATOR.HRNET
        )
    backbone_pose_estimator.cuda()
    backbone_pose_estimator = torch.nn.parallel.DistributedDataParallel(
        backbone_pose_estimator,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=cfg.NUM_GPUS > 1,
    )
    return backbone_pose_estimator


class FlowOutput(NamedTuple):
    sizes: List[int]
    mkpes: List[float]
    pck_thresholds: List[float]
    pcks: List[List[float]]
    pckh_thresholds: List[float]
    pckhs: List[List[float]]
    avg_mkpe: float
    avg_pcks: List[float]
    avg_pckhs: List[float]
    strategy: str
    num_aug: int
    magnitude_aug: int
    pose_estimator: str
    dataset: str
    sigma: float
    input_cfg: str
    expr_name: str


def prepare_output(cfg, eval_dict):
    mkpes = list()
    pck_thresholds = list()
    pcks = list()
    avg_pcks = list()
    pckh_thresholds = list()
    pckhs = list()
    avg_pckhs = list()
    sizes = list()
    for size in eval_dict.keys():
        sizes.append(int(size))
        mkpes.append(eval_dict[size]["mkpe"])
        pcks.append(eval_dict[size]["pcks"])
        avg_pck = np.mean(eval_dict[size]["pcks"])
        avg_pcks.append(avg_pck)
        pck_thresholds = eval_dict[size]["thresholds"]
        if "pckh_thresholds" in eval_dict[size]:
            pckh_thresholds = eval_dict[size]["pckh_thresholds"]
            pckhs.append(eval_dict[size]["pckh_pcks"])
            avg_pckh = np.mean(eval_dict[size]["pckh_pcks"])
            avg_pckhs.append(avg_pckh)
        else:
            pckh_thresholds = [0.0]
            pckhs.append([0.0])
            avg_pckhs.append(0.0)

    return FlowOutput(
        sizes=sizes,
        mkpes=mkpes,
        pck_thresholds=pck_thresholds,
        pcks=pcks,
        pckh_thresholds=pckh_thresholds,
        pckhs=pckhs,
        avg_mkpe=np.mean(mkpes),
        avg_pcks=avg_pcks,
        avg_pckhs=avg_pckhs,
        strategy=cfg.AL.STRATEGY,
        num_aug=cfg.DATA.NUM_AUG,
        magnitude_aug=cfg.DATA.AUG_MAGNITUDE,
        pose_estimator=cfg.POSE_ESTIMATOR.TYPE,
        dataset=cfg.DATA.TYPE,
        sigma=cfg.DATA.SIGMA,
        input_cfg=cfg.dump(),
        expr_name=cfg.EXPR_NAME,
    )


@flow.flow_async(
    use_resource_prediction=False,
    resource_requirements=flow.ResourceRequirements(gpu=8, memory="200g", cpu=40),
    use_forkserver=True,
)
@flow.registered(owners=["kunhe"])
@flow.typed()
def run(config_file: str):
    cfg = config.get_default_configs()
    cfg.merge_from_file(parutil.get_file_path(config_file))

    logger = get_logger("active_learning")
    logger.info("Merge configurations from %s." % str(config_file))
    logger.info("Experiment Type is " + cfg.EXPR_TYPE)

    cfg.NUM_GPUS = torch.cuda.device_count()
    if cfg.EXPR_TYPE == "CLUSETER":
        cfg.NUM_GPUS = 1
    logger.info("Number of GPUs: %d." % cfg.NUM_GPUS)

    cfg.EXPR_NAME = cfg.EXPR_NAME + "-" + datetime.now().strftime("%m.%d.%Y:%H:%M.%f")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dist_sync") as sync_file:
        cfg.INIT_METHOD = "file://" + sync_file.name
        logger.info("Init method: " + cfg.INIT_METHOD)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        cfg.RESULTS = f.name
        logger.info("Results are saved in: " + cfg.RESULTS)

    if cfg.EXPR_TYPE == "AL" or cfg.EXPR_TYPE == "SAL":
        if cfg.AL.START_ITER != 0:
            if cfg.AL.PREVIOUS_AL_LOG_DIR != "":
                cfg.AL.CURRENT_ITER = 0
                mp.spawn(main, args=(cfg,), nprocs=cfg.NUM_GPUS, join=True)
            else:
                raise ValueError(
                    "Restore from prior trials. "
                    + "Need to make sure that cfg.AL.PREVIOUS_AL_LOG_DIR is set."
                )
        for iteration in range(cfg.AL.START_ITER, cfg.AL.ITERATIONS):
            if iteration > 0:
                # For AL random strategies.
                cfg.RANDOM_SEED = str(datetime.now())
            cfg.AL.CURRENT_ITER = iteration
            mp.spawn(main, args=(cfg,), nprocs=cfg.NUM_GPUS, join=True)
        eval_cfg = cfg.clone()
        eval_cfg.EXPR_TYPE = "AL_EVAL"
        eval_cfg.AL.INFERENCE.TRAINING_EXPR_NAME = eval_cfg.EXPR_NAME
        mp.spawn(main, args=(eval_cfg,), nprocs=eval_cfg.NUM_GPUS, join=True)
    elif cfg.EXPR_TYPE == "SUPERVISED":
        mp.spawn(main, args=(cfg,), nprocs=cfg.NUM_GPUS, join=True)
        eval_cfg = cfg.clone()
        eval_cfg.EXPR_TYPE = "EVAL"
        eval_cfg.AL.INFERENCE.RESTORE_FROM = os.path.join(
            eval_cfg.LOG_DIR,
            eval_cfg.EXPR_NAME,
            "FULLY_SUPERVISED/checkpoints/CKPT-FINAL.pth",
        )
        mp.spawn(main, args=(eval_cfg,), nprocs=eval_cfg.NUM_GPUS, join=True)
    else:
        mp.spawn(main, args=(cfg,), nprocs=cfg.NUM_GPUS, join=True)

    try:
        with open(cfg.RESULTS) as f:
            result = json.load(f)
        return prepare_output(cfg, result)
    except FileNotFoundError:
        return "Experiment type %s does not have output." % cfg.EXPR_TYPE
