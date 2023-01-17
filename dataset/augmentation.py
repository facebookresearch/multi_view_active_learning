# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import PIL
import torch
from PIL import Image, ImageEnhance, ImageOps


def Rotate(img, heatmap, v):
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    rotated_heatmap = []
    for kp in range(heatmap.shape[2]):
        h = PIL.Image.fromarray(heatmap[:, :, kp])
        h.rotate(v, resample=Image.BICUBIC)
        rotated_heatmap.append(h)
    rotated_heatmap = np.stack(rotated_heatmap, axis=-1)
    return img.rotate(v, resample=Image.BICUBIC), rotated_heatmap


def AutoContrast(img, heatmap, _):
    return ImageOps.autocontrast(img), heatmap


def Invert(img, heatmap, _):
    return ImageOps.invert(img), heatmap


def Equalize(img, heatmap, _):
    return ImageOps.equalize(img), heatmap


def Solarize(img, heatmap, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), heatmap


def Posterize(img, heatmap, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v), heatmap


def Contrast(img, heatmap, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v), heatmap


def Color(img, heatmap, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v), heatmap


def Brightness(img, heatmap, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v), heatmap


def Sharpness(img, heatmap, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v), heatmap


class RandAugment:
    def __init__(
        self,
        num_aug,
        magnitude,
        rotation: bool,
        image_aug: bool,
        const_magnitude: bool = True,
    ):
        self.num_aug = num_aug
        self.magnitude = magnitude
        self.const_magnitude = const_magnitude
        self.augment_list = []
        if rotation:
            self.augment_list.append((Rotate, 0, 30))
        if image_aug:
            self.augment_list += [
                (AutoContrast, 0, 1),
                (Equalize, 0, 1),
                (Invert, 0, 1),
                (Posterize, 0, 4),
                (Solarize, 0, 256),
                (Color, 0.1, 1.9),
                (Contrast, 0.1, 1.9),
                (Brightness, 0.1, 1.9),
                (Sharpness, 0.1, 1.9),
            ]

    def __call__(self, img, heatmap):
        """
        img: Pillow Image.
        heatmap: Tensor, shape should be [num_keypoints, w, h].
        """
        heatmap = heatmap.permute([1, 2, 0]).numpy()
        ops = random.choices(self.augment_list, k=self.num_aug)
        for op, minval, maxval in ops:
            if self.const_magnitude:
                # constant mangnitude for all ops, 30 - max
                val = (float(self.magnitude) / 30) * float(maxval - minval) + minval
            else:
                # random magnitudes, self.magnitude controls the maximum
                val = np.random.rand() * float(self.magnitude) / 30
                val = val * float(maxval - minval) + minval
            img, heatmap = op(img, heatmap, val)
        heatmap = torch.from_numpy(heatmap).permute([2, 0, 1])
        return img, heatmap
