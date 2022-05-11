# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped

def flip_back_offset(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]
    shape_ori = output_flipped.shape
    output_flipped[:,1::3,:,:] = -output_flipped[:,1::3,:,:]
    output_flipped = output_flipped.reshape(shape_ori[0],-1,3,shape_ori[2],shape_ori[3])
    for pair in matched_parts:
        tmp = output_flipped[:, pair[0],:, :, :].copy()
        output_flipped[:, pair[0],:, :, :] = output_flipped[:, pair[1],:, :, :]
        output_flipped[:, pair[1],:, :, :] = tmp
    output_flipped = output_flipped.reshape(shape_ori)
    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    scale = scale * 200.0
    scale_x = scale[0]/(output_size[0]-1.0)
    scale_y = scale[1]/(output_size[1]-1.0)
    target_coords = np.zeros(coords.shape)
    target_coords[:,0] = coords[:,0]*scale_x + center[0]-scale[0]*0.5
    target_coords[:,1] = coords[:,1]*scale_y + center[1]-scale[1]*0.5
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


class HideAndSeek:
    """Augmentation by informantion dropping in Hide-and-Seek paradigm. Paper
    ref: Huang et al. AID: Pushing the Performance Boundary of Human Pose
    Estimation with Information Dropping Augmentation (arXiv:2008.07139 2020).
    Args:
        prob (float): Probability of performing hide-and-seek.
        prob_hiding_patches (float): Probability of hiding patches.
        grid_sizes (list): List of optional grid sizes.
    """

    def __init__(self,
                 prob=1.0,
                 prob_hiding_patches=0.5,
                 grid_sizes=(0, 16, 32, 44, 56)):
        self.prob = prob
        self.prob_hiding_patches = prob_hiding_patches
        self.grid_sizes = grid_sizes

    def _hide_and_seek(self, img):
        # get width and height of the image
        height, width, _ = img.shape

        # randomly choose one grid size
        index = np.random.randint(0, len(self.grid_sizes) - 1)
        grid_size = self.grid_sizes[index]

        # hide the patches
        if grid_size != 0:
            for x in range(0, width, grid_size):
                for y in range(0, height, grid_size):
                    x_end = min(width, x + grid_size)
                    y_end = min(height, y + grid_size)
                    if np.random.rand() <= self.prob_hiding_patches:
                        img[x:x_end, y:y_end, :] = 0
        return img

    def __call__(self, img):
        if np.random.rand() < self.prob:
            img = self._hide_and_seek(img)
        return img


class Cutout:
    """Augmentation by informantion dropping in Cutout paradigm. Paper ref:
    Huang et al. AID: Pushing the Performance Boundary of Human Pose Estimation
    with Information Dropping Augmentation (arXiv:2008.07139 2020).
    Args:
        prob (float): Probability of performing cutout.
        radius_factor (float): Size factor of cutout area.
        num_patch (float): Number of patches to be cutout.
    """

    def __init__(self, prob=1.0, radius_factor=0.2, num_patch=1):

        self.prob = prob
        self.radius_factor = radius_factor
        self.num_patch = num_patch

    def _cutout(self, img):
        height, width, _ = img.shape
        img = img.reshape(height * width, -1)
        feat_x_int = np.arange(0, width)
        feat_y_int = np.arange(0, height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.flatten()
        feat_y_int = feat_y_int.flatten()
        for _ in range(self.num_patch):
            center = [np.random.rand() * width, np.random.rand() * height]
            radius = self.radius_factor * (1 + np.random.rand(2)) * width
            x_offset = (center[0] - feat_x_int) / radius[0]
            y_offset = (center[1] - feat_y_int) / radius[1]
            dis = x_offset**2 + y_offset**2
            indexes = np.where(dis <= 1)[0]
            img[indexes, :] = 0
        img = img.reshape(height, width, -1)
        return img

    def __call__(self, img):
        if np.random.rand() < self.prob:
            img = self._cutout(img)
        return img