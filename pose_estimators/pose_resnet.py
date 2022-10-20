# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from utils import get_logger

from .pose_estimator import PoseEstimator


BN_MOMENTUM = 0.1


class PoseResNet(PoseEstimator):
    def __init__(self, num_joints, num_layers=50):
        resnet_spec = {
            18: (BasicBlock, [2, 2, 2, 2]),
            34: (BasicBlock, [3, 4, 6, 3]),
            50: (Bottleneck, [3, 4, 6, 3]),
            101: (Bottleneck, [3, 4, 23, 3]),
            152: (Bottleneck, [3, 8, 36, 3]),
        }
        block, layers = resnet_spec[num_layers]

        self.inplanes = 64
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__(num_joints=num_joints)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])

        self.final_layer = nn.Conv2d(
            in_channels=256, out_channels=num_joints, kernel_size=1, stride=1, padding=0
        )
        self._logger = get_logger(__name__)
        self._logger.info("Init deconv weights from normal distribution")
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, torch.nn.ConvTranspose2d):
                self._logger.info("Init {}.weight as normal(0, 0.001)".format(name))
                self._logger.info("Init {}.bias as 0".format(name))
                torch.nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                self._logger.info("Init {}.weight as 1".format(name))
                self._logger.info("Init {}.bias as 0".format(name))
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        self._logger.info("Init final conv weights from normal distribution")
        for m in self.final_layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                self._logger.info("Init {}.weight as normal(0, 0.001)".format(m))
                self._logger.info("Init {}.bias as 0".format(m))
                torch.nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise NotImplementedError()

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        if num_layers != len(num_filters):
            self._logger.fatal(
                "ERROR: num_deconv_layers is different len(num_deconv_filters)"
            )
        if num_layers != len(num_kernels):
            self._logger.fatal(
                "ERROR: num_deconv_layers is different len(num_deconv_filters)"
            )

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                )
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
