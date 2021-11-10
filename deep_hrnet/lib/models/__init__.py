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
from .pose_hrnet_psa2 import get_pose_net as php2

MODELS = {
    "pose_resnet": pr,
    "pose_hrnet": ph,
    "pose_resnet_psa": prp,
    "pose_hrnet_psa": php,
    "pose_hrnet_psa2": php2,
}
