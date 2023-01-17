# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import json
import os

import numpy as np
from utils import (
    triangulation,
)
from tqdm import tqdm


def get_gt_box_for_frame(person_data_filename, retval_camera, idx, dilation_amount=0.1):
    with open(person_data_filename) as info_file:
        person_data_arr = json.load(info_file)["bodies"]
    skel = np.array(person_data_arr[0]["joints19"]).reshape((-1, 4)).transpose()
    pts = triangulation.projectPoints(
        skel[0:3, :],
        retval_camera.K,
        retval_camera.R,
        retval_camera.t,
        retval_camera.dist,
    )
    x_min = np.min(pts[0])
    x_max = np.max(pts[0])
    y_min = np.min(pts[1])
    y_max = np.max(pts[1])
    width = x_max - x_min
    height = y_max - y_min
    x_min = int(max(0, x_min - dilation_amount * width))
    x_max = int(min(1919, x_max + dilation_amount * width))
    y_min = int(max(0, y_min - dilation_amount * height))
    y_max = int(min(1079, y_max + dilation_amount * height))
    box = (x_min, y_min, x_max, y_max)
    return idx, box


def compute(train_val_split_file, max_workers):
    with open(train_val_split_file, "r") as train_val_file:
        split = json.load(train_val_file)
    cmu_home = "/mnt/shared/fung/raw_cmu"
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    gt_boxes = {}
    for s in split:
        if s == "cameras":
            continue
        gt_boxes[s] = {}
        for pose in split[s]:
            gt_boxes[s][pose] = {}
            calibration_file = os.path.join(
                cmu_home, pose, "calibration_" + pose + ".json"
            )
            with open(calibration_file) as info_file:
                info_array = json.load(info_file)["cameras"]
            cams = {}
            for camera_params in info_array:
                # make it a number
                if camera_params["type"] == "hd":
                    name = camera_params["name"]
                    cams[name] = {}
                    cams[name]["R"] = camera_params["R"]
                    cams[name]["t"] = camera_params["t"]
                    cams[name]["K"] = camera_params["K"]
                    cams[name]["dist"] = camera_params["distCoef"]

            for frames in split[s][pose]:
                for frame in range(frames[0], frames[1]):
                    gt_boxes[s][pose][frame] = {}
                    for camera in cams.keys():
                        cam = cams[camera]
                        retval_camera = triangulation.Camera(
                            cam["R"], cam["t"], cam["K"], cam["dist"], camera
                        )

                        person_data_filename = os.path.join(
                            cmu_home,
                            pose,
                            "hdPose3d_stage1_coco19",
                            "body3DScene_%08d.json" % frame,
                        )
                        f = executor.submit(
                            fn=get_gt_box_for_frame,
                            person_data_filename=person_data_filename,
                            retval_camera=retval_camera,
                            idx=[s, pose, frame, camera],
                        )
                        futures.append(f)

    pbar = tqdm(total=len(futures), desc="Computing GT Boxes")
    for f in concurrent.futures.as_completed(futures):
        try:
            r = f.result()
            idx = r[0]
            gt_boxes[idx[0]][idx[1]][idx[2]][idx[3]] = r[1]
            pbar.update()
        except Exception as ex:
            print(ex)
    pbar.close
    return gt_boxes


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--train_val_split",
        help="Path to train validation split json file",
        default="train_val_split.json",
    )
    argparser.add_argument(
        "--output_json", help="Path to output json file.", default="gt_boxes.json"
    )
    argparser.add_argument("--max_workers", help="Number of workers.", default=20)
    args = argparser.parse_args()

    gt_boxes = compute(args.train_val_split, args.max_workers)
    with open("gt_boxes.json", "w") as gt_boxes_file:
        json.dump(gt_boxes, gt_boxes_file)
