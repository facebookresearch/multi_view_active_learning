# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_scaled_pred_corrdinates(pred_map, stride, num_keypoints, valid_joints):
    """
    Returns the predicted locations of highest score on the heatmap multiplied by the stride.
    """
    all_pred_labels = []
    for b in range(pred_map.shape[0]):  # for image in one batch
        label_list = []
        for k in range(num_keypoints):
            if not valid_joints[k]:
                label_list.append([0, 0])
                continue
            corr = torch.argmax(pred_map[b, k, :, :])
            x = (corr % pred_map.shape[2]) * stride
            y = (corr // pred_map.shape[2]) * stride
            label_list.append([x.data.item(), y.data.item()])
        all_pred_labels.append(label_list)
    all_pred_labels = np.asarray(all_pred_labels)
    return all_pred_labels


def get_pred_coordinates(pred_map, bbox, num_keypoints, use_softargmax=False):
    """
    Returns the predicted locations of highest score on the heatmap scaled by the bounding box.
    """
    if use_softargmax:
        coords = kornia.spatial_soft_argmax2d(pred_map, normalized_coordinates=False)
        for b in range(pred_map.shape[0]):  # for image in one batch
            # TODO(fung@): This implementation only works with square bounding boxes.
            scale = (bbox[b][3] - bbox[b][1]) / (1.0 * pred_map.shape[3])
            coords[b, :, :] = scale * coords[b, :, :]
        return coords
    else:
        all_pred_labels = []
        for b in range(pred_map.shape[0]):  # for image in one batch
            label_list = []
            for k in range(num_keypoints):
                corr = torch.argmax(pred_map[b, k, :, :])
                y = (corr // pred_map.shape[2]) * (
                    (bbox[b][2] - bbox[b][0]) / (1.0 * pred_map.shape[2])
                )
                x = (corr % pred_map.shape[2]) * (
                    (bbox[b][3] - bbox[b][1]) / (1.0 * pred_map.shape[3])
                )
                label_list.append([x, y])
            all_pred_labels.append(label_list)
        return all_pred_labels


def compute_pckh_0_5(predict_labels_dict, gt_labels, num_keypoints):
    return compute_pckh(predict_labels_dict, gt_labels, 0.5, num_keypoints)


def compute_pckh(
    predict_labels_dict, gt_labels, threshold, num_keypoints, kp0=0, kp1=1
):
    pck = [0] * num_keypoints
    count = 0
    for batch in range(len(predict_labels_dict)):
        for frame in range(len(predict_labels_dict[batch])):
            d = (
                torch.sqrt(
                    (gt_labels[batch][frame][kp0][0] - gt_labels[batch][frame][kp1][0])
                    ** 2
                    + (
                        gt_labels[batch][frame][kp0][1]
                        - gt_labels[batch][frame][kp1][1]
                    )
                    ** 2
                )
                * threshold
            )
            count += 1
            for i in range(num_keypoints):
                pre = predict_labels_dict[batch][frame][i]
                tar = gt_labels[batch][frame][i]
                dis = torch.sqrt((pre[0] - tar[0]) ** 2 + (pre[1] - tar[1]) ** 2)

                if dis.data.item() < d.data.item():
                    pck[i] += 1
    pck = [k / count for k in pck]
    return pck


def compute_pckh_figure(
    predict_labels_dict,
    gt_labels,
    num_keypoints,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
):
    pcks = []
    for threshold in thresholds:
        pck = compute_pckh(predict_labels_dict, gt_labels, threshold, num_keypoints)
        pcks.append(pck)
    return thresholds, pcks


def plot_pckh_figure(thresholds, pck):
    fig, axis = plt.subplots(figsize=(4, 3))
    axis.set_ylim((0.0, 1.0))
    axis.plot(thresholds, pck, "r+")
    axis.plot(thresholds, pck)
    axis.grid(True)
    fig.canvas.draw()
    figure = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return figure


def compute_3d_pckh_figure(
    pred_3d_labels,
    gt_3d_labels,
    num_keypoints,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
):
    pcks = []
    for threshold in thresholds:
        pck = compute_3d_pckh(pred_3d_labels, gt_3d_labels, threshold, num_keypoints)
        pcks.append(pck)
    return thresholds, pcks


def compute_3d_pck_figure(
    pred_3d_labels,
    gt_3d_labels,
    valid_joints,
    num_keypoints,
    thresholds=(1, 2, 3, 4, 5),
):
    pcks = []
    for threshold in thresholds:
        pck = compute_3d_pck(
            pred_3d_labels, gt_3d_labels, valid_joints, threshold, num_keypoints
        )
        pcks.append(pck)
    return thresholds, pcks


def compute_3d_pckh(pred_3d_labels, gt_3d_labels, threshold, num_keypoints):
    pck = [0] * num_keypoints
    count = 0
    for idx, pred_3d_label in enumerate(pred_3d_labels):
        gt_3d_label = gt_3d_labels[idx]
        d = (
            torch.sqrt(
                (gt_3d_label[0][0] - gt_3d_label[0][1]) ** 2
                + (gt_3d_label[1][0] - gt_3d_label[1][1]) ** 2
                + (gt_3d_label[2][0] - gt_3d_label[2][1]) ** 2
            )
            * threshold
        )
        count += 1
        for i in range(num_keypoints):
            dis = torch.sqrt(
                (pred_3d_label[i][0] - gt_3d_label[0][i]) ** 2
                + (pred_3d_label[i][1] - gt_3d_label[1][i]) ** 2
                + (pred_3d_label[i][2] - gt_3d_label[2][i]) ** 2
            )

            if dis.data.item() < d.data.item():
                pck[i] += 1
    pck = [k / count for k in pck]
    return pck


def compute_3d_pck(
    pred_3d_labels, gt_3d_labels, valid_joints, threshold_mm, num_keypoints
):
    pck = [0] * num_keypoints
    count = [0] * num_keypoints
    for pred, gt, valid in zip(pred_3d_labels, gt_3d_labels, valid_joints):
        for i in range(num_keypoints):
            if not valid[i]:
                continue
            dis = torch.sqrt(
                (pred[i][0] - gt[0][i]) ** 2
                + (pred[i][1] - gt[1][i]) ** 2
                + (pred[i][2] - gt[2][i]) ** 2
            )
            count[i] += 1
            if dis.data.item() < threshold_mm:
                pck[i] += 1
    pck = [k / c for k, c in zip(pck, count)]
    return pck


def compute_mkpe(pred_3d_labels, gt_3d_labels, valid_joints):
    kpe = torch.zeros_like(valid_joints[0]).float()
    count = torch.zeros_like(valid_joints[0])
    for pred, gt, valid in zip(pred_3d_labels, gt_3d_labels, valid_joints):
        d = torch.square(pred.permute([1, 0]) - gt[:3, :])
        d = torch.where(valid.bool(), d, torch.zeros_like(d))
        d = torch.sqrt(torch.sum(d, dim=0))
        kpe = kpe + d
        count = count + valid
    mkpe = kpe / count
    return torch.mean(mkpe)
