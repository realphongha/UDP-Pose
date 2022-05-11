# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pose_resnet import get_pose_net as pr
from .pose_hrnet  import get_pose_net as ph
from .pose_resnet_psa import get_pose_net as prp
from .pose_hrnet_psa import get_pose_net as php
from .pose_shufflenetv2_plus import get_pose_net as ps2p
from .pose_shufflenetv2_10x import get_pose_net as ps2_10
from .pose_mobilenetv3_small import get_pose_net as pm3_s
from .pose_shufflenetv2_plus_pixel_shuffle import get_pose_net as ps2p_ps
from .pose_shufflenetv2_10x_pixel_shuffle import get_pose_net as ps2_10_ps
from .pose_mobilenetv3_small_pixel_shuffle import get_pose_net as pm3_s_ps
from .pose_mobilevit_pixel_shuffle import  get_pose_net as pmv_ps

MODELS = {
    "pose_resnet": pr,
    "pose_hrnet": ph,
    "pose_resnet_psa": prp,
    "pose_hrnet_psa": php,
    "pose_shufflenetv2_plus": ps2p,
    "pose_shufflenetv2_10x": ps2_10,
    "pose_mobilenetv3_small": pm3_s,
    "pose_shufflenetv2_plus_pixel_shuffle": ps2p_ps,
    "pose_shufflenetv2_10x_pixel_shuffle": ps2_10_ps,
    "pose_mobilenetv3_small_pixel_shuffle": pm3_s_ps,
    "pose_mobilevit_pixel_shuffle": pmv_ps,
}
