# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
from fblearner.flow.util.visualization_utils import summary_writer

from .triangulation import denormalize_image


def save_heatmaps_to_tb(writer: summary_writer, heatmaps, step=0, name="prediction"):
    b, num_kp = heatmaps.shape[0], heatmaps.shape[1]
    for idx in range(b):
        for kp in range(num_kp):
            fig, axis = plt.subplots(figsize=(15, 15))
            normalized_heatmap = heatmaps[idx][kp]
            axis.matshow(normalized_heatmap)
            fig.canvas.draw()
            normalized_heatmap = np.fromstring(
                fig.canvas.tostring_rgb(), dtype=np.uint8, sep=""
            )
            normalized_heatmap = normalized_heatmap.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            writer.add_image(
                name + "/view-" + str(idx) + "/kp-" + str(kp),
                normalized_heatmap,
                global_step=step,
                dataformats="HWC",
            )


def save_2d_keypoints_to_tb(
    writer: summary_writer, cropped_image, keypoints_2d, step=0, name="prediction"
):
    b = cropped_image.shape[0]
    for idx in range(b):
        view = cropped_image[idx].permute([1, 2, 0]).cpu().numpy()
        view = denormalize_image(view)[..., ::-1]
        fig, axis = plt.subplots(figsize=(15, 15))
        axis.imshow(view)
        axis.plot(keypoints_2d[idx][:, 0], keypoints_2d[idx][:, 1], ".", color="red")
        fig.canvas.draw()
        view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        view = view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.add_image(name + "/2d-" + str(idx), view, step, dataformats="HWC")


def view_heatmap_vs_gt_heatmap(heatmaps, gt_heatmaps):
    fig, list_of_axis = plt.subplots(19, 10, figsize=(15, 15))
    heatmaps = heatmaps.cpu().numpy()
    gt_heatmaps = gt_heatmaps.cpu().numpy()
    for view in range(0, 10, 2):
        for keypoint in range(19):
            gt_heatmap = gt_heatmaps[view // 2][keypoint]
            list_of_axis[keypoint][view].imshow(gt_heatmap)
            list_of_axis[keypoint][view].axis("off")
            pred_heatmap = heatmaps[view // 2][keypoint]
            list_of_axis[keypoint][view + 1].imshow(pred_heatmap)
            list_of_axis[keypoint][view + 1].axis("off")
    plt.show()
