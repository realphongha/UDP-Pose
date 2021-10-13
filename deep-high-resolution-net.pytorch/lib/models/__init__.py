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

import lib.models.pose_resnet as pr
import lib.models.pose_hrnet as ph
import lib.models.pose_resnet_psa as prp
import lib.models.pose_hrnet_psa as php

MODELS = {
    "pose_resnet": pr.get_pose_net,
    "pose_hrnet": ph.get_pose_net,
    "pose_resnet_psa": prp.get_pose_net,
    "pose_hrnet_psa": php.get_pose_net,
}
