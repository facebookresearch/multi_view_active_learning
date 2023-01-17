# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import json
import os

import numpy as np
from tqdm import tqdm


def get_gt_keypoints(person_data_filename, idx):
    with open(person_data_filename) as info_file:
        person_data_arr = json.load(info_file)["bodies"]
    skel = (
        np.array(person_data_arr[0]["joints19"]).reshape((-1, 4)).transpose().tolist()
    )
    return idx, skel


def aggregate_labels(train_val_split_file, max_workers):
    with open(train_val_split_file, "r") as train_val_file:
        split = json.load(train_val_file)
    cmu_home = "/mnt/shared/data/raw_cmu"
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    gt_keypoints = {}
    for s in split:
        if s == "cameras":
            continue
        gt_keypoints[s] = {}
        for pose in split[s]:
            gt_keypoints[s][pose] = {}
            calibration_file = os.path.join(
                cmu_home, pose, "calibration_" + pose + ".json"
            )
            with open(calibration_file) as info_file:
                info_array = json.load(info_file)["cameras"]
            cams = {}
            for camera_params in info_array:
                if camera_params["type"] == "hd":
                    name = camera_params["name"]
                    cams[name] = {}
                    cams[name]["R"] = camera_params["R"]
                    cams[name]["t"] = camera_params["t"]
                    cams[name]["K"] = camera_params["K"]
                    cams[name]["dist"] = camera_params["distCoef"]
            gt_keypoints[s][pose]["cameras"] = cams
            for frames in split[s][pose]:
                for frame in range(frames[0], frames[1]):
                    gt_keypoints[s][pose][frame] = {}
                    person_data_filename = os.path.join(
                        cmu_home,
                        pose,
                        "hdPose3d_stage1_coco19",
                        "body3DScene_%08d.json" % frame,
                    )
                    f = executor.submit(
                        fn=get_gt_keypoints,
                        person_data_filename=person_data_filename,
                        idx=[s, pose, frame],
                    )
                    futures.append(f)

    pbar = tqdm(total=len(futures), desc="Aggregating Labels")
    for f in concurrent.futures.as_completed(futures):
        try:
            r = f.result()
            idx = r[0]
            gt_keypoints[idx[0]][idx[1]][idx[2]] = r[1]
            pbar.update()
        except Exception as ex:
            print(ex)
    pbar.close
    return gt_keypoints


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--train_val_split",
        help="Path to train validation split json file",
        default="train_val_split.json",
    )
    argparser.add_argument(
        "--output_json", help="Path to output json file.", default="cmu_gt_labels.json"
    )
    argparser.add_argument("--max_workers", help="Number of workers.", default=20)
    args = argparser.parse_args()

    gt_keypoints = aggregate_labels(args.train_val_split, args.max_workers)
    with open(args.output_json, "w") as gt_boxes_file:
        json.dump(gt_keypoints, gt_boxes_file)
