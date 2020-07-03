# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Simple utility functions.

Authors
-------
Jeff Mahler, Vishal Satish, Lucas Manuelli
"""
from functools import reduce
import os
import sys

import numpy as np
import skimage.transform as skt

from autolab_core import Logger
from .enums import GripperMode

# Set up logger.
logger = Logger.get_logger("gqcnn/utils/utils.py")


def is_py2():
    return sys.version[0] == "2"


def set_cuda_visible_devices(gpu_list):
    """Sets CUDA_VISIBLE_DEVICES environment variable to only show certain
    gpus.

    Note
    ----
    If gpu_list is empty does nothing.

    Parameters
    ----------
    gpu_list : list
        List of gpus to set as visible.
    """
    if len(gpu_list) == 0:
        return

    cuda_visible_devices = ""
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ","

    logger.info(
        "Setting CUDA_VISIBLE_DEVICES = {}".format(cuda_visible_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def pose_dim(gripper_mode):
    """Returns the dimensions of the pose vector for the given
    gripper mode.

    Parameters
    ----------
    gripper_mode: :obj:`GripperMode`
        Enum for gripper mode, see optimizer_constants.py for all possible
        gripper modes.

    Returns
    -------
    :obj:`numpy.ndarray`
        Sliced pose_data corresponding to gripper mode.
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.SUCTION:
        return 2
    elif gripper_mode == GripperMode.MULTI_SUCTION:
        return 1
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.LEGACY_SUCTION:
        return 2
    else:
        raise ValueError(
            "Gripper mode '{}' not supported.".format(gripper_mode))


def read_pose_data(pose_arr, gripper_mode):
    """Read the pose data and slice it according to the specified gripper mode.

    Parameters
    ----------
    pose_arr: :obj:`numpy.ndarray`
        Full pose data array read in from file.
    gripper_mode: :obj:`GripperMode`
        Enum for gripper mode, see optimizer_constants.py for all possible
        gripper modes.

    Returns
    -------
    :obj:`numpy.ndarray`
        Sliced pose_data corresponding to input data mode.
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:, 2:3]
    elif gripper_mode == GripperMode.SUCTION:
        if pose_arr.ndim == 1:
            return np.r_[pose_arr[2], pose_arr[4]]
        else:
            return np.c_[pose_arr[:, 2], pose_arr[:, 4]]
    elif gripper_mode == GripperMode.MULTI_SUCTION:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:, 2:3]
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:, 2:3]
    elif gripper_mode == GripperMode.LEGACY_SUCTION:
        if pose_arr.ndim == 1:
            return pose_arr[2:4]
        else:
            return pose_arr[:, 2:4]
    else:
        raise ValueError(
            "Gripper mode '{}' not supported.".format(gripper_mode))


def reduce_shape(shape):
    """Get shape of a layer for flattening."""
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y  # noqa: E731
    return reduce(f, shape, 1)


def weight_name_to_layer_name(weight_name):
    """Convert the name of weights to the layer name."""
    tokens = weight_name.split("_")
    type_name = tokens[-1]

    # Modern naming convention.
    if type_name == "weights" or type_name == "bias":
        if len(tokens) >= 3 and tokens[-3] == "input":
            return weight_name[:weight_name.rfind("input") - 1]
        return weight_name[:weight_name.rfind(type_name) - 1]
    # Legacy.
    if type_name == "im":
        return weight_name[:-4]
    if type_name == "pose":
        return weight_name[:-6]
    return weight_name[:-1]


def imresize(image, size, interp="nearest"):
    """Wrapper over `skimage.transform.resize` to mimic `scipy.misc.imresize`.
    Copied from https://github.com/BerkeleyAutomation/perception/blob/master/perception/image.py#L38.  # noqa: E501

    Since `scipy.misc.imresize` has been removed in version 1.3.*, instead use
    `skimage.transform.resize`. The "lanczos" and "cubic" interpolation methods
    are not supported by `skimage.transform.resize`, however there is now
    "biquadratic", "biquartic", and "biquintic".

    Parameters
    ----------
    image : :obj:`numpy.ndarray`
        The image to resize.

    size : int, float, or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : :obj:`str`, optional
        Interpolation to use for re-sizing ("neartest", "bilinear",
        "biquadratic", "bicubic", "biquartic", "biquintic"). Default is
        "nearest".

    Returns
    -------
    :obj:`np.ndarray`
        The resized image.
    """
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")
