# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Simple utility functions.
Authors: Jeff Mahler, Vishal Satish, Lucas Manuelli
"""
import os
import logging

import colorlog
import numpy as np

from enums import GripperMode

def clear_root_logger():
    # clear the root logger's handlers so we have full control over logging
    for hdlr in logging.getLogger().handlers:
        logging.getLogger().removeHandler(hdlr)

def get_logger(name, log_file=None, log_stream=None):
    # clear the root logger's handlers
    clear_root_logger()

    # create the logger and set the logging level
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # set up handlers    
    if log_file is not None:
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(name)-10s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
    if log_stream is not None:
        hdlr = logging.StreamHandler(log_stream)
        formatter = colorlog.ColoredFormatter(
                            '%(purple)s%(name)-10s %(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
                            reset=True,
                            log_colors={
                                'DEBUG': 'cyan',
                                'INFO': 'green',
                                'WARNING': 'yellow',
                                'ERROR': 'red',
                                'CRITICAL': 'red,bg_white',
                            },
                                            )
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    if len(logger.handlers) == 0:
        # if no handlers were added, add the NullHandler to squelch output complaining about no handlers
        logger.addHandler(logging.NullHandler())

    return logger

def set_cuda_visible_devices(gpu_list):
    """
    Sets CUDA_VISIBLE_DEVICES environment variable to only show certain gpus.
    If gpu_list is empty does nothing.

    Parameters
    ----------
    gpu_list : list
        list of gpus to set as visible
    """

    if len(gpu_list) == 0:
        return

    cuda_visible_devices = ''
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ','

    logging.info('Setting CUDA_VISIBLE_DEVICES = {}'.format(cuda_visible_devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

def pose_dim(gripper_mode):
    """ Returns the dimensions of the pose vector for the given
    gripper mode.
    
    Parameters
    ----------
    gripper_mode: :obj:`GripperMode`
        enum for gripper mode, see enums.py for all
        possible gripper modes 

    Returns
    -------
    :obj:`numpy.ndarray`
        sliced pose_data corresponding to gripper mode
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.SUCTION:
        return 2
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.LEGACY_SUCTION:
        return 2
    else:
        raise ValueError('Gripper mode %s not supported.' %(gripper_mode))
    
def read_pose_data(pose_arr, gripper_mode):
    """ Read the pose data and slice it according to the specified gripper_mode.
    
    Parameters
    ----------
    pose_arr: :obj:`numpy.ndarray`
        full pose data array
    gripper_mode: :obj:`GripperMode`
        enum for gripper mode, see enums.py for all
        possible gripper modes 

    Returns
    -------
    :obj:`numpy.ndarray`
        sliced pose_data corresponding to input data mode
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:,2:3]
    elif gripper_mode == GripperMode.SUCTION:
        if pose_arr.ndim == 1:
            return np.r_[pose_arr[2], pose_arr[4]]
        else:
            return np.c_[pose_arr[:,2], pose_arr[:,4]]
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        if pose_arr.ndim == 1:
            return pose_arr[2:3]
        else:
            return pose_arr[:,2:3]
    elif gripper_mode == GripperMode.LEGACY_SUCTION:
        if pose_arr.ndim == 1:
            return pose_arr[2:4]
        else:
            return pose_arr[:,2:4]
    else:
        raise ValueError('Gripper mode %s not supported.' %(gripper_mode))

def reduce_shape(shape):
    """Reduce shape."""
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape, 1)

def weight_name_to_layer_name(weight_name):
    """Extract the layer name from a weight name."""
    tokens = weight_name.split('_')
    type_name = tokens[-1]

    # modern naming convention
    if type_name == 'weights' or type_name == 'bias':
        if len(tokens) >= 3 and tokens[-3] == 'input':
            return weight_name[:weight_name.rfind('input')-1]            
        return weight_name[:weight_name.rfind(type_name)-1]
    # legacy support
    if type_name == 'im':
        return weight_name[:-4]
    if type_name == 'pose':
        return weight_name[:-6]
    return weight_name[:-1]
