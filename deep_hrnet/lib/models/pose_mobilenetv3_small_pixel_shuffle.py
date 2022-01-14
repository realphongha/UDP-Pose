# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from .backbones.mobilenetv3 import get_mobilenet_v3
from .decoders.pixelshuffle import PixelShuffleDecoder

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class PoseMobileNetV3PixelShuffle(nn.Module):

    def __init__(self, cfg, pretrained, **kwargs):
        self.inplanes = 576
        extra = cfg.MODEL.EXTRA
        super(PoseMobileNetV3PixelShuffle, self).__init__()

        self.backbone = get_mobilenet_v3(size=extra.MODEL_SIZE, pretrained=pretrained)

        self.decoder = PixelShuffleDecoder(self.inplanes,
                                           extra.START_CHANNELS,
                                           extra.ARCHITECTURE)

        if not cfg.MODEL.TARGET_TYPE=='offset':
            factor=1
        else:
            factor=3
        self.final_layer = nn.Conv2d(
            in_channels=self.decoder.out_channels,
            out_channels=cfg.MODEL.NUM_JOINTS*factor,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


def get_pose_net(cfg, is_train, **kwargs):

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        pretrained = cfg.MODEL.PRETRAINED
    else:
        pretrained = None

    model = PoseMobileNetV3PixelShuffle(cfg, pretrained, **kwargs)

    return model
