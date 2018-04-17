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
Simple utility functions
Author: Lucas Manuelli
"""

import os

from . import InputDataMode

def set_cuda_visible_devices(gpu_list):
    """
    Sets CUDA_VISIBLE_DEVICES environment variable to only show certain gpus
    If gpu_list is empty does nothing
    :param gpu_list: list of gpus to set as visible
    :return: None
    """

    if len(gpu_list) == 0:
        return

    cuda_visible_devices = ""
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ","

    print "setting CUDA_VISIBLE_DEVICES = ", cuda_visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def pose_dim(self, pose_arr, input_data_mode):
    """ Returns the dimensions of the pose vector for the given
    input data mode.
    
    Parameters
    ----------
    input_data_mode: :obj:`InputDataMode`
        enum for input data mode, see optimizer_constants.py for all
        possible input data modes 

    Returns
    -------
    :obj:`ndArray`
        sliced pose_data corresponding to input data mode
    """
    if input_data_mode == InputDataMode.PARALLEL_JAW:
        return 1
    elif input_data_mode == InputDataMode.SUCTION:
        return 2
    elif input_data_mode == InputDataMode.TF_IMAGE:
        return 1
    elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        return 3
    elif input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
        return 2
    else:
        raise ValueError('Input data mode %s not supported. The RAW_* input data modes have been deprecated.' %(input_data_mode))
    
def read_pose_data(self, pose_arr, input_data_mode):
    """ Read the pose data and slice it according to the specified input_data_mode
    
    Parameters
    ----------
    pose_arr: :obj:`ndArray`
        full pose data array read in from file
    input_data_mode: :obj:`InputDataMode`
        enum for input data mode, see optimizer_constants.py for all
        possible input data modes 

    Returns
    -------
    :obj:`ndArray`
        sliced pose_data corresponding to input data mode
    """
    if input_data_mode == InputDataMode.PARALLEL_JAW:
        return pose_arr[:,2:3]
    elif input_data_mode == InputDataMode.SUCTION:
        return np.c_[pose_arr[:,2], pose_arr[:,4]]
    elif input_data_mode == InputDataMode.TF_IMAGE:
        return pose_arr[:,2:3]
    elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
    elif input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
        return pose_arr[:,2:4]
    else:
        raise ValueError('Input data mode %s not supported. The RAW_* input data modes have been deprecated.' %(input_data_mode))
