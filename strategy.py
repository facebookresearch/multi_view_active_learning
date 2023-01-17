# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import math
import os
import random
from collections import OrderedDict
from heapq import nlargest

import numpy as np
import torch
from torch.utils.tensorboard import summary_writer
from iopath.common.file_io import PathManager
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from pose_estimators.loss import Pose2DMeanSquaredError
from utils import TqdmToLogger, coreset, evaluation, get_logger, triangulation


class ActiveLearningStrategy:
    def __init__(self, al_cfg):
        self._logger = get_logger(__name__)
        self.al_cfg = al_cfg
        self.num_joints = al_cfg.DATA.NUM_JOINTS
        self._pathmgr = PathManager()
        if self.al_cfg.DATA.TYPE == "panoptic":
            self.joint_root_index = 2
        else:
            self.joint_root_index = 21
        if self.al_cfg.EXPR_TYPE == "SAL":
            if self.al_cfg.SAL.CLUSTER_FILE_PATH != "":
                with self._pathmgr.open(self.al_cfg.SAL.CLUSTER_FILE_PATH) as f:
                    clusters = json.load(f)
                kp_values = []
                for guid in clusters:
                    kp = np.array(clusters[guid])
                    kp = (
                        kp[0:3, :]
                        - kp[0:3, self.joint_root_index: self.joint_root_index + 1]
                    )
                    kp_values.append(kp.flatten())
                self.kmeans = KMeans(
                    self.al_cfg.SAL.NUM_CLUSTERS, random_state=self.al_cfg.RANDOM_SEED
                ).fit(kp_values)

    def sample_next_batch(
        self,
        train_dataset,
        al_num_frames,
        sal_num_frames,
        pose_estimator,
        iteration,
        rank=-1,
    ):
        if iteration == 0:
            self._logger.info(
                "%s Strategy, randomly sampling first/initial batch."
                % self.al_cfg.AL.STRATEGY
            )
            train_dataset, al_guids = self._random_sample_frames(
                train_dataset, al_num_frames
            )
        else:
            self._logger.info(
                "%s strategy sampling %d frames."
                % (self.al_cfg.AL.STRATEGY, al_num_frames)
            )
            train_dataset, al_guids, sal_guids, sal_dict = self._sal_pseudo_labeling(
                train_dataset, al_num_frames, sal_num_frames, pose_estimator
            )
            if rank == 0:
                if len(sal_guids) != 0:
                    try:
                        mkpe = [sal_dict["mkpe"][guid] for guid in sal_guids]
                        self.al_writer.add_histogram(
                            "sal/mkpe", np.array(mkpe), iteration
                        )
                        inlier_count = [
                            sal_dict["inlier_count"][guid] for guid in sal_guids
                        ]
                        self.al_writer.add_histogram(
                            "sal/inlier_count", np.array(
                                inlier_count), iteration
                        )
                        sal_metric = [
                            sal_dict["sal_metric"][guid] for guid in sal_guids
                        ]
                        self.al_writer.add_histogram(
                            "sal/sal_metric", np.array(sal_metric), iteration
                        )
                        al_metric = [sal_dict["al_metric"][guid]
                                     for guid in al_guids]
                        self.al_writer.add_histogram(
                            "sal/al_metric", np.array(al_metric), iteration
                        )
                        self.al_writer.add_scalar(
                            "sal/al_num_frames", len(al_metric), iteration
                        )
                        self.al_writer.add_scalar(
                            "sal/sal_num_frames", len(sal_metric), iteration
                        )
                    except Exception as ex:
                        self._logger.warning("TensorBoard Error: %s" % ex)
                    log_file = os.path.join(
                        self.al_cfg.LOG_DIR,
                        self.al_cfg.EXPR_NAME,
                        "SAL-GUID-ITER-%d" % iteration,
                    )
                    with self._pathmgr.open(log_file, "w") as f:
                        f.write(json.dumps(sal_guids))
                log_file = os.path.join(
                    self.al_cfg.LOG_DIR,
                    self.al_cfg.EXPR_NAME,
                    "SAL-DICT-ITER-%d" % iteration,
                )
                with self._pathmgr.open(log_file, "w") as f:
                    f.write(json.dumps(sal_dict))

        if rank == 0:
            log_file = os.path.join(
                self.al_cfg.LOG_DIR,
                self.al_cfg.EXPR_NAME,
                "SAMPLED-GUID-ITER-%d" % iteration,
            )
            with self._pathmgr.open(log_file, "w") as f:
                f.write(json.dumps(al_guids))
        return train_dataset

    def cluster(self, pose_estimator, train_dataset, rank):
        pose_estimator.eval()
        train_dataset.label_all()
        train_dataset.resample_frames(-1)
        train_dataloader = self._get_dataloader(
            train_dataset,
            self.al_cfg.TRAIN.BATCH_SIZE,
            self.al_cfg.TRAIN.NUM_WORKERS,
        )
        if self.al_cfg.AL.CLUSTER.TYPE == "LOSS":
            checkpoint_file = self.al_cfg.AL.CLUSTER.RESTORE_FROM
            self._logger.info("Loading weights from %s" % (checkpoint_file))
            with self._pathmgr.open(checkpoint_file, "rb") as ckpt_file:
                ckpt = torch.load(io.BytesIO(ckpt_file.read()))
            pose_estimator.load_state_dict(ckpt["state_dict"], strict=True)
            loss = Pose2DMeanSquaredError()
        pbar = tqdm(
            total=len(train_dataloader),
            leave=False,
            desc="Clustering",
            file=TqdmToLogger(),
            mininterval=5,
            maxinterval=100,
        )
        cluster_dict = {}
        for data in train_dataloader:
            with torch.autograd.no_grad():
                if self.al_cfg.AL.CLUSTER.TYPE == "POSE":
                    poses = np.array(data["pose"]).transpose()
                    frame_ids = np.array(data["frame_id"]).transpose()
                    batch_size = poses.shape[0]
                    for idx in range(batch_size):
                        guid = "%s-%s" % (poses[idx], frame_ids[idx][0])
                        cluster_dict[guid] = (
                            data["3d_keypoints"][idx].cpu().numpy().tolist()
                        )
                elif self.al_cfg.AL.CLUSTER.TYPE == "LOSS":
                    heatmaps = self._compute_batch_heatmap(
                        pose_estimator, data)
                    _, kp, w, h = heatmaps.shape
                    poses = np.array(data["pose"]).transpose()
                    frame_ids = np.array(data["frame_id"]).transpose()
                    batch_size = poses.shape[0]
                    heatmaps = heatmaps.view([batch_size, -1, kp, w, h])
                    gt_heatmaps = data["gt_heatmap"].cuda()
                    for idx in range(batch_size):
                        batch_loss = loss.pose_2d_mse_single_batch(
                            heatmaps[idx], gt_heatmaps[idx]
                        )
                        guid = "%s-%s" % (poses[idx][0], frame_ids[idx][0])
                        cluster_dict[guid] = batch_loss.data.item()
            pbar.update()
        pbar.close()
        with self._pathmgr.open(self.al_cfg.AL.CLUSTER.SAVE_PATH, "w") as f:
            json.dump(cluster_dict, f)

    def run_al_eval(self, pose_estimator, val_dataset, rank):
        pose_estimator.eval()
        val_dataloader = self._prepare_val_dataset_for_eval(val_dataset)
        if self.al_cfg.AL.INFERENCE.TRAINING_EXPR_NAME != "":
            list_of_sizes = []
            list_of_checkpoints = []
            for iteration in range(0, self.al_cfg.AL.ITERATIONS):
                size = (1 + iteration) * self.al_cfg.AL.ITER_AMOUNT
                path_to_final_ckpt = os.path.join(
                    self.al_cfg.LOG_DIR,
                    self.al_cfg.AL.INFERENCE.TRAINING_EXPR_NAME,
                    "ITER-%d" % iteration,
                    "checkpoints",
                    "CKPT-FINAL.pth",
                )
                if self._pathmgr.isfile(path_to_final_ckpt):
                    list_of_sizes.append(size)
                    list_of_checkpoints.append(path_to_final_ckpt)
        eval_dict = dict()
        for iteration, size in enumerate(list_of_sizes):
            checkpoint_file = list_of_checkpoints[iteration]
            eval_dict[size] = self._evaluate_on_checkpoint(
                iteration, pose_estimator, checkpoint_file, val_dataloader
            )
        line_size = "Size\n"
        line_mkpe = "MKPE\n"
        for size in eval_dict.keys():
            line_size += f"{size}\n"
            line_mkpe += "%.2f\n" % (eval_dict[size]["mkpe"])
        self._logger.info("Size of Training dataset:")
        self._logger.info(line_size)
        self._logger.info("Eval Metric:")
        self._logger.info(line_mkpe)
        return eval_dict

    def run_eval(self, pose_estimator, val_dataset, rank):
        """
        Called by workflow.py when cfg.EXPR_TYPE == "EVAL"
        """
        pose_estimator.eval()
        val_dataloader = self._prepare_val_dataset_for_eval(val_dataset)
        checkpoint_file = self.al_cfg.AL.INFERENCE.RESTORE_FROM
        result = self._evaluate_on_checkpoint(
            -1, pose_estimator, checkpoint_file, val_dataloader
        )
        return {-1: result}

    def al_iteration(self, pose_estimator, train_dataset, val_dataset, iteration, rank):
        train_cfg = self.al_cfg.clone()
        # Train the pose estimator.
        train_cfg.EXPR_NAME = train_cfg.EXPR_NAME + "/ITER-%d" % (iteration)
        (
            pose_estimator,
            optimizer,
            lr_scheduler,
            val_dataloader,
            loss,
            checkpoints_dir,
            writer,
        ) = self._prepare_for_training(pose_estimator, train_cfg, val_dataset, rank)
        self._train_pose_estimator(
            iteration,
            train_cfg,
            train_dataset,
            val_dataloader,
            optimizer,
            lr_scheduler,
            loss,
            rank,
            checkpoints_dir,
            pose_estimator,
            writer,
        )

    def prepare_al_experiments(self):
        self._logger.info("Prepare AL Experiments.")
        checkpoints_dir = os.path.join(
            self.al_cfg.LOG_DIR, self.al_cfg.EXPR_NAME)
        if not self._pathmgr.isdir(checkpoints_dir):
            self._pathmgr.mkdirs(checkpoints_dir)
        self.al_writer = summary_writer(
            log_dir=os.path.join(self.al_cfg.LOG_DIR,
                                 self.al_cfg.EXPR_NAME, "AL")
        )
        self.al_writer.add_text("comment", self.al_cfg.COMMENT, 0)

    def run_fully_supervised_training(
        self, pose_estimator, train_dataset, val_dataset, rank
    ):
        if self.al_cfg.TRAIN.FULLY_SUPERVISED_FRAME_COUNT != -1:
            train_dataset, guids = self._random_sample_frames(
                train_dataset,
                self.al_cfg.TRAIN.FULLY_SUPERVISED_FRAME_COUNT,
            )
        else:
            train_dataset.label_all()
        train_cfg = self.al_cfg.clone()
        train_cfg.EXPR_NAME = train_cfg.EXPR_NAME + "/FULLY_SUPERVISED"
        pose_estimator.train(True)
        (
            pose_estimator,
            optimizer,
            lr_scheduler,
            val_dataloader,
            loss,
            checkpoints_dir,
            writer,
        ) = self._prepare_for_training(pose_estimator, train_cfg, val_dataset, rank)
        self._train_pose_estimator(
            -1,
            train_cfg,
            train_dataset,
            val_dataloader,
            optimizer,
            lr_scheduler,
            loss,
            rank,
            checkpoints_dir,
            pose_estimator,
            writer,
        )

    def restore_dataset(self, train_dataset, iteration):
        for i in range(0, iteration):
            path_to_guids = os.path.join(
                self.al_cfg.LOG_DIR, self.al_cfg.EXPR_NAME, "SAMPLED-GUID-ITER-%d" % i
            )
            with self._pathmgr.open(path_to_guids, "r") as guids_file:
                guids = json.loads(guids_file.readline())
            train_dataset.label_by_frame_guids(guids)
        if self.al_cfg.EXPR_TYPE == "SAL" and iteration > 1:
            path_to_sal_guids = os.path.join(
                self.al_cfg.LOG_DIR,
                self.al_cfg.EXPR_NAME,
                "SAL-GUID-ITER-%d" % (iteration - 1),
            )
            with self._pathmgr.open(path_to_sal_guids, "r") as guids_file:
                sal_guids = json.loads(guids_file.readline())
            train_dataset.pseudo_label_guids = sal_guids
        self._logger.info(
            "Restored dataset upto iteration %d. Dataset size is now: %d"
            % (iteration - 1, len(train_dataset.labeled_data))
        )
        return train_dataset

    def _copy_previous_checkpoints(self):
        for i in range(0, self.al_cfg.AL.START_ITER):
            path_to_checkpoints = os.path.join(
                self.al_cfg.AL.PREVIOUS_AL_LOG_DIR,
                "ITER-%d" % i,
                "checkpoints",
                "CKPT-FINAL.pth",
            )
            path_to_copy_to = os.path.join(
                self.al_cfg.LOG_DIR,
                self.al_cfg.EXPR_NAME,
                "ITER-%d" % i,
                "checkpoints",
                "CKPT-FINAL.pth",
            )
            self._pathmgr.mkdirs(
                os.path.join(
                    self.al_cfg.LOG_DIR,
                    self.al_cfg.EXPR_NAME,
                    "ITER-%d" % i,
                    "checkpoints",
                )
            )
            if self._pathmgr.isfile(path_to_checkpoints):
                self._pathmgr.copy(path_to_checkpoints, path_to_copy_to)
            else:
                self._logger.info(
                    "Path to checkpoints is not found: %s" % path_to_checkpoints
                )
            path_to_guids = os.path.join(
                self.al_cfg.AL.PREVIOUS_AL_LOG_DIR, "SAMPLED-GUID-ITER-%d" % i
            )
            path_to_copy_to = os.path.join(
                self.al_cfg.LOG_DIR, self.al_cfg.EXPR_NAME, "SAMPLED-GUID-ITER-%d" % i
            )
            if self._pathmgr.isfile(path_to_guids):
                self._pathmgr.copy(path_to_guids, path_to_copy_to)
            else:
                self._logger.warning(
                    "Path to GUIDs is not found: %s" % path_to_guids)
            path_to_sal_guids = os.path.join(
                self.al_cfg.AL.PREVIOUS_AL_LOG_DIR, "SAL-GUID-ITER-%d" % i
            )
            sal_guid_path_to_copy_to = os.path.join(
                self.al_cfg.LOG_DIR, self.al_cfg.EXPR_NAME, "SAL-GUID-ITER-%d" % i
            )
            if self._pathmgr.isfile(path_to_sal_guids):
                self._pathmgr.copy(path_to_sal_guids, sal_guid_path_to_copy_to)
            else:
                self._logger.warning(
                    "Path to SAL GUIDs is not found: %s" % path_to_sal_guids
                )

    def _prepare_for_training(self, pose_estimator, train_cfg, val_dataset, rank):
        if rank == 0:
            checkpoints_dir, writer = self._prepare_experiment(train_cfg)
        else:
            checkpoints_dir, writer = None, None

        val_dataset.label_all()
        val_dataset.resample_frames(train_cfg.TRAIN.VAL_SIZE)
        val_dataloader = self._get_dataloader(
            val_dataset,
            train_cfg.TRAIN.BATCH_SIZE,
            train_cfg.TRAIN.NUM_WORKERS,
        )
        loss = Pose2DMeanSquaredError()
        optimizer = torch.optim.Adam(
            [{"params": pose_estimator.parameters(), "lr": train_cfg.TRAIN.OPTIM.LR}],
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_cfg.TRAIN.OPTIM.LR_DECAY_STEP_SIZE
        )
        pose_estimator = self._load_weights(train_cfg, pose_estimator)
        return (
            pose_estimator,
            optimizer,
            lr_scheduler,
            val_dataloader,
            loss,
            checkpoints_dir,
            writer,
        )

    def _train_pose_estimator(
        self,
        iteration_idx,  # <0 for fully-supervised, >=0 for AL/SAL
        train_cfg,
        train_dataset,
        val_dataloader,
        optimizer,
        lr_scheduler,
        loss,
        rank,
        checkpoints_dir,
        pose_estimator,
        writer,
    ):
        epoch = 0
        global_step = 0
        early_stopping = True
        train_dataset.resample_frames(-1, train_cfg.DATA.EPOCH_SIZE)
        train_dataloader = self._get_dataloader(
            train_dataset,
            train_cfg.TRAIN.BATCH_SIZE,
            train_cfg.TRAIN.NUM_WORKERS,
        )
        while early_stopping:
            epoch += 1
            desc = (
                f"AL_iter {iteration_idx} Training Epoch {epoch}"
                if iteration_idx >= 0
                else f"Fully-Supervised Training Epoch {epoch}"
            )
            pbar = tqdm(
                total=len(train_dataloader),
                leave=False,
                desc=desc,
                file=TqdmToLogger(),
                mininterval=5,
                maxinterval=100,
            )
            for data in train_dataloader:
                with torch.autograd.enable_grad():
                    optimizer.zero_grad()
                    heatmaps = self._compute_batch_heatmap(
                        pose_estimator, data)
                    gt_heatmap = data["gt_heatmap"].cuda()
                    per_view_joint_valid = (
                        data["per_view_joint_valid"].type(
                            torch.ByteTensor).cuda()
                    )
                    batch_loss = self._compute_batch_loss(
                        gt_heatmap, heatmaps, per_view_joint_valid, loss
                    )
                    if not (
                        math.isnan(batch_loss.data.item())
                        or math.isinf(batch_loss.data.item())
                        or batch_loss.data.item() > train_cfg.TRAIN.LOSS_CLIP_VALUE
                    ):
                        batch_loss.backward()
                        optimizer.step()
                    else:
                        self._logger.warning(
                            "Current Training Loss: %.4f. Not Valid. Skipped."
                            % batch_loss.data.item()
                        )
                global_step += 1
                lr_scheduler.step()
                pbar.update()
                if global_step % train_cfg.TRAIN.LOG_EVERY_ITER == 0:
                    avg_loss = batch_loss.data.item()
                    eval_results = self._evaluate_all(
                        iteration_idx, pose_estimator, val_dataloader
                    )
                    if rank == 0:
                        self._save_checkpoints(
                            checkpoints_dir,
                            epoch,
                            global_step,
                            pose_estimator,
                            optimizer,
                            ckpt_name="CKPT-E%d-MKPE%.2f.pth"
                            % (global_step, eval_results["mkpe"]),
                        )
                        self._save_checkpoints(
                            checkpoints_dir,
                            epoch,
                            global_step,
                            pose_estimator,
                            optimizer,
                            ckpt_name="CKPT-FINAL.pth",
                        )
                    current_lr = optimizer.param_groups[0]["lr"]
                    if rank == 0:
                        writer.add_scalar("lr", current_lr, global_step)
                        writer.add_scalar(
                            "3D MKPE", eval_results["mkpe"], global_step)
                        self._log_loss_info(
                            writer, epoch, global_step, avg_loss)
                        self._log_pck_info(
                            writer,
                            "3DPCK",
                            global_step,
                            eval_results["thresholds"],
                            eval_results["pcks"],
                        )
                        if "pckh_thresholds" in eval_results:
                            self._log_pck_info(
                                writer,
                                "3DPCKH",
                                global_step,
                                eval_results["pckh_thresholds"],
                                eval_results["pckh_pcks"],
                            )
                    self._logger.info(
                        "GPU [%d] MEMORY USAGE: %.2f / %.2f GB."
                        % (
                            rank,
                            torch.cuda.memory_reserved(
                                rank) / (1024 * 1024 * 1024),
                            torch.cuda.get_device_properties(rank).total_memory
                            / (1024 * 1024 * 1024),
                        )
                    )
            pbar.close()
            early_stopping = global_step < max(
                train_cfg.TRAIN.OPTIM.TOTAL_STEPS, train_cfg.TRAIN.LOG_EVERY_ITER
            )

    def _evaluate_2d_pckh(self, pose_estimator, val_dataloader, rank=-1):
        all_pred_labels = []
        gt_labels = []
        pbar = tqdm(
            total=len(val_dataloader),
            leave=False,
            desc="Eval 2D PCKh",
            file=TqdmToLogger(),
            mininterval=5,
            maxinterval=100,
        )
        for val_data in val_dataloader:
            with torch.no_grad():
                heatmaps = self._compute_batch_heatmap(
                    pose_estimator, val_data)
                pred_labels = torch.Tensor(
                    self._compute_pred_labels(
                        pose_estimator, val_data, heatmaps)
                ).cuda()
            gt_label = val_data["2d_after_crop"].reshape([-1, 19, 2]).cuda()
            pred_list = [
                torch.zeros_like(pred_labels) for _ in range(self.al_cfg.NUM_GPUS)
            ]
            gt_list = [torch.zeros_like(gt_label)
                       for _ in range(self.al_cfg.NUM_GPUS)]
            torch.distributed.all_gather(pred_list, pred_labels)
            torch.distributed.all_gather(gt_list, gt_label)
            all_pred_labels += pred_list
            gt_labels += gt_list
            pbar.update()
        pbar.close()
        thresholds, pcks = evaluation.compute_pckh_figure(
            all_pred_labels, gt_labels, self.num_joints
        )
        return thresholds, pcks

    def _evaluate_all(self, iteration_idx, pose_estimator, val_dataloader):
        pred_3d_labels = []
        gt_3d_labels = []
        valid_joints = []
        self._logger.info(f"AL_iter {iteration_idx}: Evaluate 3D PCK & MKPE.")
        pbar = tqdm(
            total=len(val_dataloader),
            leave=False,
            desc=f"AL_iter {iteration_idx} Eval 3D PCK/MKPE",
            file=TqdmToLogger(),
            mininterval=5,
            maxinterval=100,
        )
        for val_data in val_dataloader:
            batch_3d_label = val_data["3d_keypoints"].cuda()
            batch_valid_kp = val_data["joint_valid"].cuda()
            with torch.no_grad():
                batch_size = len(batch_3d_label)
                heatmaps = self._compute_batch_heatmap(
                    pose_estimator, val_data)
                _, kp, w, h = heatmaps.shape
                heatmaps = heatmaps.view([batch_size, -1, kp, w, h])
                for idx in range(batch_size):
                    results = triangulation.triangulation(
                        heatmaps[idx],
                        val_data["proj_matrices"][idx],
                        self.al_cfg.POSE_ESTIMATOR.STRIDE,
                        val_data["joint_valid"][idx],
                    )
                    pred_labels = torch.Tensor(results["keypoints_3d"]).cuda()
                    gt_3d_label = batch_3d_label[idx]
                    valid_joint = batch_valid_kp[idx]
                    pred_list = [
                        torch.zeros_like(pred_labels)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    gt_list = [
                        torch.zeros_like(gt_3d_label)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    valid_list = [
                        torch.zeros_like(valid_joint)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    torch.distributed.all_gather(pred_list, pred_labels)
                    torch.distributed.all_gather(gt_list, gt_3d_label)
                    torch.distributed.all_gather(valid_list, valid_joint)
                    pred_3d_labels += pred_list
                    gt_3d_labels += gt_list
                    valid_joints += valid_list
            pbar.update()
        pbar.close()
        mkpe = evaluation.compute_mkpe(
            pred_3d_labels, gt_3d_labels, valid_joints)
        thresholds, pcks = evaluation.compute_3d_pck_figure(
            pred_3d_labels, gt_3d_labels, valid_joints, self.num_joints
        )
        results = {"mkpe": mkpe.data.item(), "thresholds": thresholds,
                   "pcks": pcks}
        if self.al_cfg.DATA.TYPE == "panoptic":
            pckh_thresholds, pckh_pcks = evaluation.compute_3d_pckh_figure(
                pred_3d_labels, gt_3d_labels, self.num_joints
            )
            results["pckh_thresholds"] = pckh_thresholds
            results["pckh_pcks"] = pckh_pcks
        return results

    def _prepare_experiment(self, cfg):
        self._logger.info("Experiment name: {}".format(cfg.EXPR_NAME))
        experiment_dir = os.path.join(cfg.LOG_DIR, cfg.EXPR_NAME)
        if not self._pathmgr.isdir(experiment_dir):
            self._pathmgr.mkdirs(experiment_dir)
        else:
            self._logger.warning(
                "Experiment directory already exists. Double check: %s."
                % experiment_dir
            )
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        if not self._pathmgr.isdir(checkpoints_dir):
            self._pathmgr.mkdirs(checkpoints_dir)
        else:
            self._logger.warning(
                "Checkpoint directory already exists. Double check: %s."
                % checkpoints_dir
            )
        with self._pathmgr.open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(cfg.dump())
        log_dir = os.path.join(experiment_dir, "summary")
        if not self._pathmgr.isdir(log_dir):
            self._pathmgr.mkdirs(log_dir)
        else:
            self._logger.warning(
                "TensorBoard directory already exists. Double check: %s." % log_dir
            )
        writer = summary_writer(log_dir=log_dir)
        return checkpoints_dir, writer

    def _save_checkpoints(
        self,
        checkpoints_dir,
        epoch,
        global_step,
        pose_estimator,
        optimizer,
        ckpt_name=None,
    ):
        if ckpt_name is None:
            ckpt_name = "CKPT-E%d-S%d.pth" % (epoch, global_step)
        checkpoint_file = os.path.join(checkpoints_dir, ckpt_name)
        if self._pathmgr.isfile(checkpoint_file):
            self._pathmgr.rm(checkpoint_file)
            self._logger.info("Overwriting checkpoint file: %s" %
                              checkpoint_file)
        with self._pathmgr.open(checkpoint_file, "wb") as ckpt_file:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": pose_estimator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_file,
            )
        self._logger.info(
            "[EPOCH %d][STEP %d] Checkpoint Saved at %s."
            % (epoch, global_step, str(checkpoint_file))
        )
        return str(checkpoint_file)

    def _load_weights(self, cfg, pose_estimator):
        if cfg.TRAIN.RESTORE_FROM:
            self._logger.info("Loading weights from %s" %
                              cfg.TRAIN.RESTORE_FROM)
            with self._pathmgr.open(cfg.TRAIN.RESTORE_FROM, "rb") as ckpt_file:
                ckpt = torch.load(io.BytesIO(ckpt_file.read()))
            pose_estimator.load_state_dict(ckpt["state_dict"], strict=True)
            self._logger.info("Loaded weights from %s." %
                              cfg.TRAIN.RESTORE_FROM)
        elif cfg.TRAIN.INIT_WEIGHT:
            self._logger.info("Initializing weights from %s." %
                              cfg.TRAIN.INIT_WEIGHT)
            if cfg.POSE_ESTIMATOR.TYPE == "POSE_RESNET":
                pretrained_state_dict = torch.load(cfg.TRAIN.INIT_WEIGHT)
                del pretrained_state_dict["final_layer.weight"]
                del pretrained_state_dict["final_layer.bias"]
                pose_estimator.load_state_dict(
                    pretrained_state_dict, strict=False)
            elif cfg.POSE_ESTIMATOR.TYPE == "HRNET":
                pretrained_state_dict = torch.load(cfg.TRAIN.INIT_WEIGHT)
                need_init_state_dict = {}
                for name, m in pretrained_state_dict.items():
                    if (
                        name.split(".")[
                            0] in pose_estimator.module.pretrained_layers
                        or pose_estimator.module.pretrained_layers[0] == "*"
                    ):
                        need_init_state_dict[name] = m
                pose_estimator.load_state_dict(
                    need_init_state_dict, strict=False)
        else:
            self._logger.info("Training from scratch.")
        return pose_estimator

    def _get_dataloader(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )
        return dataloader

    @staticmethod
    def _compute_batch_loss(gt_heatmap, heatmaps, per_view_joint_valid, loss):
        batch_size, num_view, num_joints, w, h = gt_heatmap.shape
        gt_heatmap = gt_heatmap.reshape([-1, num_joints, w, h])
        per_view_joint_valid = per_view_joint_valid.reshape(
            [-1, num_joints, 1, 1])
        batch_loss = loss.pose_2d_mse(
            heatmaps, gt_heatmap, per_view_joint_valid)
        return batch_loss

    @staticmethod
    def _compute_batch_heatmap(pose_estimator, data):
        images_batch = data["images"].cuda()
        channel, w, h = (
            images_batch.shape[2],
            images_batch.shape[3],
            images_batch.shape[4],
        )
        images_batch = images_batch.reshape([-1, channel, w, h])
        heatmaps = pose_estimator(images_batch)
        return heatmaps

    def _compute_pred_labels(self, pose_estimator, data, heatmaps):
        boxes = data["square_box"].cuda()
        boxes = boxes.reshape([-1, 4])
        pred_labels = evaluation.get_pred_coordinates(
            heatmaps, boxes, self.num_joints)
        return pred_labels

    def _log_loss_info(self, writer, epoch, global_step, avg_loss):
        self._logger.info(
            "[EPOCH %d][STEP %d] AVG TRAIN LOSS: %.4f." % (
                epoch, global_step, avg_loss)
        )
        writer.add_scalar("loss/train", avg_loss, global_step)
        writer.add_scalar(
            "gpu/reserved_memory",
            torch.cuda.memory_reserved(0) / (1024 * 1024 * 1024),
            global_step,
        )
        writer.add_scalar(
            "gpu/allocated_memory",
            torch.cuda.memory_allocated(0) / (1024 * 1024 * 1024),
            global_step,
        )

    def _log_pck_info(self, writer, eval_metric, global_step, thresholds, pcks):
        avg_pcks = []
        per_joint_pcks = [[] for _ in range(self.num_joints)]
        for idx, pck_dict in enumerate(pcks):
            threshold = thresholds[idx]
            avg_pck = sum(pck_dict) / len(pck_dict)
            writer.add_scalar(
                "%s@%.1f/average" % (eval_metric, threshold),
                avg_pck,
                global_step,
            )
            avg_pcks.append(avg_pck)
            for kp_id, pck in enumerate(pck_dict):
                writer.add_scalar(
                    "%s@%.1f/keypoint-%d" % (eval_metric, threshold, kp_id),
                    pck,
                    global_step,
                )
                per_joint_pcks[kp_id].append(pck)
        for kp_id in range(self.num_joints):
            avg_pck = sum(per_joint_pcks[kp_id]) / len(per_joint_pcks[kp_id])
            writer.add_scalar(
                "%s-AVG/keypoint-%d" % (eval_metric, kp_id),
                avg_pck,
                global_step,
            )
            figure = evaluation.plot_pckh_figure(
                thresholds, per_joint_pcks[kp_id])
            writer.add_image(
                "%s/keypoint-%d" % (eval_metric, kp_id),
                figure,
                global_step,
                dataformats="HWC",
            )
        figure = evaluation.plot_pckh_figure(thresholds, avg_pcks)
        writer.add_image(
            "%s/average" % eval_metric,
            figure,
            global_step,
            dataformats="HWC",
        )

    def _save_init_weight_for_al(self, pose_estimator, rank):
        pose_estimator = self._load_weights(self.al_cfg, pose_estimator)
        checkpoints_dir = os.path.join(
            self.al_cfg.LOG_DIR, self.al_cfg.EXPR_NAME)
        init_checkpoint_file = os.path.join(checkpoints_dir, "INIT-WEIGHT.pth")
        if rank == 0:
            self._logger.info("Prepare initial weights for AL Experiments.")
            if not self._pathmgr.isdir(checkpoints_dir):
                self._pathmgr.mkdirs(checkpoints_dir)
            with self._pathmgr.open(init_checkpoint_file, "wb") as ckpt_file:
                torch.save(
                    {"state_dict": pose_estimator.state_dict()}, ckpt_file)
            self._logger.info(
                "Initial Weights Checkpoint Saved at %s." % str(
                    init_checkpoint_file)
            )
        return init_checkpoint_file

    def _random_sample_frames(self, train_dataset, num_frames, seed=None):
        """
        Sampling N *frames* from unlabeled data to labeled data.
        """
        if seed is None:
            seed = self.al_cfg.RANDOM_SEED
        random.seed(seed)
        guids = random.sample(
            list(train_dataset.unlabeled_data.keys()), num_frames)
        train_dataset.label_by_frame_guids(guids)
        return train_dataset, guids

    def _evaluate_on_checkpoint(
        self, iteration_idx, pose_estimator, checkpoint_file, val_dataloader
    ):
        self._logger.info(
            f"AL_iter {iteration_idx}: Loading weights from {checkpoint_file}"
        )
        with self._pathmgr.open(checkpoint_file, "rb") as ckpt_file:
            ckpt = torch.load(io.BytesIO(ckpt_file.read()))
        pose_estimator.load_state_dict(ckpt["state_dict"], strict=True)
        result = self._evaluate_all(
            iteration_idx, pose_estimator, val_dataloader)

        avg_pck_per_threshold = []
        for pck_dict in result["pcks"]:
            avg_pck = sum(pck_dict) / len(pck_dict)
            avg_pck_per_threshold.append(avg_pck)
        result["pcks"] = avg_pck_per_threshold
        if "pckh_pcks" in result:
            avg_pckh_per_threshold = []
            for pck_dict in result["pckh_pcks"]:
                avg_pck = sum(pck_dict) / len(pck_dict)
                avg_pckh_per_threshold.append(avg_pck)
            result["pckh_pcks"] = avg_pckh_per_threshold
        return result

    def _prepare_val_dataset_for_eval(self, val_dataset):
        val_dataset.label_all()
        val_dataset.resample_frames(-1)
        val_dataloader = self._get_dataloader(
            val_dataset,
            self.al_cfg.AL.INFERENCE.BATCH_SIZE,
            self.al_cfg.AL.INFERENCE.NUM_WORKERS,
        )
        return val_dataloader

    def _sal_pseudo_labeling(
        self, train_dataset, al_num_frames, pseudo_num_frames, pose_estimator
    ):
        if self.al_cfg.AL.STRATEGY == "RANDOM" and self.al_cfg.EXPR_TYPE == "AL":
            # Random Active Learning is treated separately as no inference is needed.
            train_dataset, al_guids = self._random_sample_frames(
                train_dataset, al_num_frames, seed=self.al_cfg.RANDOM_SEED
            )
            return train_dataset, al_guids, [], {}

        train_dataset.resample_unlabeled_data()
        data_loader = self._get_dataloader(
            train_dataset,
            self.al_cfg.AL.INFERENCE.BATCH_SIZE,
            self.al_cfg.AL.INFERENCE.NUM_WORKERS,
        )
        sal_dict = self._compute_sal_dict(data_loader, pose_estimator)
        al_metric_dict = {
            guid: sal_dict["al_metric"][guid]
            for guid in sal_dict["al_metric"].keys()
            if not math.isnan(sal_dict["al_metric"][guid])
        }
        if self.al_cfg.AL.STRATEGY == "CORESET":
            cs = coreset.CoreSet(
                sal_dict["pred_3d_keypoints"],
                train_dataset.get_al_dict_for_coreset(),
                self.joint_root_index,
            )
            al_guids = cs.select_batch(al_num_frames)
        else:
            al_guids = nlargest(
                al_num_frames,
                al_metric_dict,
                key=al_metric_dict.get,
            )
        train_dataset.label_by_frame_guids(al_guids)
        sal_sampled_guids = []
        if self.al_cfg.EXPR_TYPE == "SAL":
            # Filter the pseudo-labels:
            # 1. Not sampled by AL.
            # 2. Not already pseudo-labeled.
            # 3. > # of inliers.
            sal_metric_dict = {
                guid: sal_dict["sal_metric"][guid]
                for guid in sal_dict["sal_metric"].keys()
                if (
                    guid not in al_guids
                    and not math.isnan(sal_dict["sal_metric"][guid])
                    and guid not in train_dataset.pseudo_label_guids
                    and sal_dict["inlier_count"][guid]
                    > self.al_cfg.SAL.INLIER_THRESHOLD
                )
            }
            # Sort by SAL metric.
            sal_guids = sorted(
                sal_metric_dict,
                key=sal_metric_dict.get,
            )
            if self.al_cfg.SAL.CLUSTER_FILE_PATH != "":
                self._logger.info(
                    "Pseudo-labeling frames by %d clusters."
                    % self.al_cfg.SAL.NUM_CLUSTERS
                )
                counter = [0 for _ in range(self.al_cfg.SAL.NUM_CLUSTERS)]
                sal_sampled_guids = []
                per_cluster_count = pseudo_num_frames // self.al_cfg.SAL.NUM_CLUSTERS
                for guid in sal_guids:
                    kp = np.array(sal_dict["pred_3d_keypoints"][guid]).T
                    kp = (
                        kp[0:3, :]
                        - kp[0:3, self.joint_root_index: self.joint_root_index + 1]
                    )
                    kp = kp.flatten()
                    cluster_id = self.kmeans.predict([kp])[0]
                    if counter[cluster_id] < per_cluster_count:
                        counter[cluster_id] += 1
                        sal_sampled_guids.append(guid)
            else:
                sal_sampled_guids = random.sample(
                    sal_guids[: 2 * pseudo_num_frames], pseudo_num_frames
                )

            self._logger.info("Pseudo-labeling %d frames." %
                              len(sal_sampled_guids))
            train_dataset.pseudo_label_by_frame_guids(
                sal_sampled_guids, sal_dict["pred_3d_keypoints"]
            )
        return train_dataset, al_guids, sal_sampled_guids, sal_dict

    def _compute_sal_dict(self, data_loader, pose_estimator):
        """
        Run inference on unlabeled training data to compute AL metrics,
        SAL metrics, and pseudo labels dictionary.
        """
        sal_dict = {
            "al_metric": OrderedDict(),
            "sal_metric": OrderedDict(),
            "inlier_count": OrderedDict(),
            "pred_3d_keypoints": OrderedDict(),
            "mkpe": OrderedDict(),
        }
        pbar = tqdm(
            total=len(data_loader),
            leave=False,
            desc="Inference on Unlabeled Data",
            file=TqdmToLogger(),
            mininterval=5,
            maxinterval=100,
        )
        for dp in data_loader:
            pbar.update()
            with torch.no_grad():
                heatmaps = self._compute_batch_heatmap(pose_estimator, dp)
                _, kp, w, h = heatmaps.shape
                poses = np.array(dp["pose"]).transpose()
                proj_matrices = dp["proj_matrices"]
                joint_valid = dp["joint_valid"]
                batch_3d_label = dp["3d_keypoints"].cuda()
                batch_valid_kp = dp["joint_valid"].cuda()
                batch_size = poses.shape[0]
                heatmaps = heatmaps.view([batch_size, -1, kp, w, h])
                for idx in range(batch_size):
                    results = triangulation.triangulation(
                        heatmaps[idx],
                        proj_matrices[idx],
                        self.al_cfg.POSE_ESTIMATOR.STRIDE,
                        joint_valid[idx],
                        self.al_cfg.AL.USE_SOFTARGMAX,
                        self.al_cfg.AL.USE_REPROJECTION_XE,
                        self.al_cfg.AL.REPROJECTION_SIGMA,
                    )
                    pred_labels = torch.Tensor(results["keypoints_3d"]).cuda()
                    pred_list = [
                        torch.zeros_like(pred_labels)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    gt_3d_label = batch_3d_label[idx]
                    gt_list = [
                        torch.zeros_like(gt_3d_label)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    valid_joint = batch_valid_kp[idx]
                    valid_list = [
                        torch.zeros_like(valid_joint)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    sal_metric = torch.Tensor([results["metric"]]).cuda()
                    sal_inlier_count = torch.Tensor(
                        [results["inlier_count"]]).cuda()
                    sal_metric_list = [
                        torch.zeros_like(sal_metric)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    sal_inlier_list = [
                        torch.zeros_like(sal_inlier_count)
                        for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    if self.al_cfg.AL.STRATEGY == "RANDOM":
                        al_metric = torch.rand(1).cuda()
                    elif self.al_cfg.AL.STRATEGY == "TRIANGULATION":
                        al_metric = torch.tensor([results["metric"]]).cuda()
                    elif self.al_cfg.AL.STRATEGY == "MPE":
                        al_metric = torch.tensor(
                            self._compute_mpe(
                                heatmaps[idx], dp["joint_valid"][idx])
                        ).cuda()
                    elif self.al_cfg.AL.STRATEGY == "HP":
                        al_metric = torch.tensor(
                            self._compute_hp(
                                heatmaps[idx], dp["joint_valid"][idx])
                        ).cuda()
                    elif self.al_cfg.AL.STRATEGY == "BSB":
                        al_metric = torch.tensor(
                            self._compute_bsb(
                                heatmaps[idx], dp["joint_valid"][idx])
                        ).cuda()
                    elif self.al_cfg.AL.STRATEGY == "CORESET":
                        al_metric = torch.tensor(0.0).cuda()
                    else:
                        raise NotImplementedError()
                    al_metric_list = [
                        torch.zeros_like(al_metric) for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    pose = dp["pose"][idx].cuda()
                    frame = dp["frame_id"][idx].cuda()
                    poses_list = [
                        torch.zeros_like(pose) for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    frame_id_list = [
                        torch.zeros_like(frame) for _ in range(self.al_cfg.NUM_GPUS)
                    ]
                    torch.distributed.all_gather(sal_metric_list, sal_metric)
                    torch.distributed.all_gather(
                        sal_inlier_list, sal_inlier_count)
                    torch.distributed.all_gather(al_metric_list, al_metric)
                    torch.distributed.all_gather(pred_list, pred_labels)
                    torch.distributed.all_gather(gt_list, gt_3d_label)
                    torch.distributed.all_gather(valid_list, valid_joint)
                    torch.distributed.all_gather(poses_list, pose)
                    torch.distributed.all_gather(frame_id_list, frame)
                    for (
                        pose,
                        frame_id,
                        pred_3d_kp,
                        gt_3d_kp,
                        valid_joint,
                        sal_metric,
                        sal_inlier,
                        al_metric,
                    ) in zip(
                        poses_list,
                        frame_id_list,
                        pred_list,
                        gt_list,
                        valid_list,
                        sal_metric_list,
                        sal_inlier_list,
                        al_metric_list,
                    ):
                        mkpe = evaluation.compute_mkpe(
                            [pred_3d_kp], [gt_3d_kp], [valid_joint]
                        )
                        guid = "%s-%s" % (pose.data.item(),
                                          frame_id.data.item())
                        sal_dict["sal_metric"][guid] = sal_metric.data.item()
                        sal_dict["inlier_count"][guid] = sal_inlier.data.item()
                        sal_dict["pred_3d_keypoints"][guid] = (
                            pred_3d_kp.cpu().numpy().tolist()
                        )
                        sal_dict["al_metric"][guid] = al_metric.data.item()
                        sal_dict["mkpe"][guid] = mkpe.data.item()
        pbar.close()
        return sal_dict

    def _compute_mpe(self, heatmaps, joint_valid):
        ents = self._compute_mpes(heatmaps, joint_valid)
        if self.al_cfg.AL.MPE_CONFIG == "AVG":
            return sum(ents) / len(ents)
        elif self.al_cfg.AL.MPE_CONFIG == "STD":
            ents = np.array(ents)
            return np.std(ents)
        else:
            raise NotImplementedError(
                "AL.MPE_CONFIG should be either AVG or STD.")

    def _compute_mpes(self, heatmaps, joint_valid):
        heatmaps = heatmaps.cpu().numpy()
        num_views, num_kp, w, h = heatmaps.shape
        ents = []
        for view in range(num_views):
            for kp in range(num_kp):
                if not joint_valid[kp]:
                    continue
                coordinates = peak_local_max(
                    heatmaps[view][kp], min_distance=2, indices=True
                )
                peaks = [heatmaps[view][kp][cood[0]][cood[1]]
                         for cood in coordinates]
                probs = np.exp(peaks) / sum(np.exp(peaks))
                ent = sum(-prob * math.log(prob) for prob in probs)
                ents.append(ent)
        return ents

    def _compute_hp(self, heatmaps, joint_valid):
        num_views, num_kp, w, h = heatmaps.shape
        hps = []
        for view in range(num_views):
            for kp in range(num_kp):
                if not joint_valid[kp]:
                    continue
                joint_hm = torch.nn.functional.softmax(heatmaps[view][kp])
                highest_prob = 1 - torch.max(joint_hm)
                hps.append(highest_prob.data.item())
        if self.al_cfg.AL.HP_CONFIG == "AVG":
            avg_hp = sum(hps) / len(hps)
            return avg_hp
        if self.al_cfg.AL.HP_CONFIG == "STD":
            hps = np.array(hps)
            return np.std(hps)

    def _compute_bsb(self, heatmaps, joint_valid):
        num_views, num_kp, w, h = heatmaps.shape
        bsbs = []
        for view in range(num_views):
            for kp in range(num_kp):
                if not joint_valid[kp]:
                    continue
                joint_hm = torch.nn.functional.softmax(
                    heatmaps[view][kp]).cpu().numpy()
                coordinates = peak_local_max(
                    joint_hm, min_distance=2, indices=True, num_peaks=2
                )
                probs = [joint_hm[cood[0]][cood[1]] for cood in coordinates]
                bsb = abs(probs[0] - probs[1])
                bsbs.append(bsb)
        if self.al_cfg.AL.BSB_CONFIG == "AVG":
            avg_bsb = sum(bsbs) / len(bsbs)
            return avg_bsb
        if self.al_cfg.AL.BSB_CONFIG == "STD":
            bsbs = np.array(bsbs)
            return np.std(bsbs)
