# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import concurrent.futures
import json
import os

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def process_and_upload_to_manifold(s, pose, frame):
    """
    "format": "CHW_BGR",
    "note": "views should be of size [31x1080]x1920x3",
    """

    cmu_home = "/mnt/shared/fung/raw_cmu/"
    views = []
    for camera in range(31):
        camera = "00_%02d" % camera
        image_path = os.path.join(
            cmu_home, pose, "hdImgs", camera, "%s_%08d.jpg" % (camera, frame)
        )
        try:
            image = Image.open(image_path)
            image = np.array(image)[..., ::-1]
        except Exception:
            image = np.zeros([1080, 1920, 3], dtype=np.uint8)
        views.append(image)
    views = np.array(views)
    path_on_tmp = os.path.join("/mnt/shared/fung/", "upload", pose, "%08d.jpg" % frame)
    imageio.imwrite(path_on_tmp, views.reshape([31 * 1080, 1920, 3]))
    return True


def upload_all(max_workers):
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    with open("train_val_split.json") as train_val_file:
        split = json.load(train_val_file)
    for s in split:
        if s == "cameras":
            continue
        for pose in split[s]:
            os.makedirs(
                os.path.join("/mnt/shared/fung/", "upload", pose), exist_ok=True
            )
            for frames in split[s][pose]:
                for frame in range(frames[0], frames[1]):
                    f = executor.submit(
                        fn=process_and_upload_to_manifold, s=s, pose=pose, frame=frame
                    )
                    futures.append(f)
    pbar = tqdm(total=len(futures), desc="Creating stacked images")
    for _ in concurrent.futures.as_completed(futures):
        pbar.update()
    pbar.close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--max_workers", type=int, help="Number of workers to upload to manifold"
    )
    args = parser.parse_args()
    upload_all(max_workers=args.max_workers)
