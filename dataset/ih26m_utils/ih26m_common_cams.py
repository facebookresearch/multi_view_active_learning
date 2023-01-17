# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import os

import numpy as np
from iopath.common.file_io import PathManager


def get_default_common_cams(all_cams, captures, num_cams=-1):
    common_cams = list(all_cams[str(captures[0])]["campos"].keys())
    for capture in captures:
        capture = str(capture)
        common_cams = [
            cam for cam in common_cams if cam in all_cams[capture]["campos"].keys()
        ]

    all_cams = {cam: all_cams[capture]["campos"][cam] for cam in common_cams}
    if num_cams != -1:
        if num_cams > len(common_cams):
            raise ArithmeticError(
                "Not enough common cameras among the selected captures."
            )
        selected_cams = {common_cams[0]: all_cams[common_cams[0]]}
        del all_cams[common_cams[0]]
        for _ in range(1, num_cams):
            selected_cam = _get_furtherest_cam(selected_cams, all_cams)
            selected_cams[selected_cam] = all_cams[selected_cam]
            del all_cams[selected_cam]
        return list(selected_cams.keys())
    return common_cams


def _get_furtherest_cam(sampled_cams, other_cams):
    furthurest_dis = 0.0
    for cam in other_cams:
        nearest_dis = math.inf
        for nearest_cam in sampled_cams:
            distance = _distance(other_cams[cam], sampled_cams[nearest_cam])
            if distance < nearest_dis:
                nearest_dis
        if nearest_dis > furthurest_dis:
            furthurest_dis = nearest_dis
            selected_cam = cam
    return selected_cam


def _distance(pt1, pt2):
    d = np.sqrt(np.sum(np.square(np.array(pt1) - np.array(pt2))))
    return d


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ih26m_home",
        help="Path to IH26M home.",
        default="InterHand2.6M_5fps_batch1",
    )
    argparser.add_argument(
        "--split", help="Split of IH26M to use. train/val/test", default="test"
    )
    argparser.add_argument(
        "--captures", help="Captures to use.", default=list(range(2))
    )
    argparser.add_argument(
        "--num_cams", help="# of cameras to select based on distance.", default=-1
    )
    args = argparser.parse_args()

    _pathmgr = PathManager()

    path_to_annotations = os.path.join(
        args.ih26m_home,
        "annotations",
        args.split,
    )
    with _pathmgr.open(
        os.path.join(
            path_to_annotations,
            "InterHand2.6M_%s_camera.json" % args.split,
        )
    ) as f:
        all_cams = json.load(f)

    default_cams = get_default_common_cams(all_cams, args.captures, args.num_cams)

    print(default_cams)
    print(len(default_cams))
