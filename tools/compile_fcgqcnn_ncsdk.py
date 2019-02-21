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
Script for compiling an FC-GQ-CNN for the Intel Myriad X VPU using the Intel NCSDK 2.
Author: Vishal Satish
"""
import argparse
import os
import subprocess

from autolab_core import Logger
from gqcnn import get_nc_fc_gqcnn_model

# setup logger
logger = Logger.get_logger('tools/compile_fcgqcnn_ncsdk.py')

if __name__=='__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Compile an FC-GQ-CNN model for the Movidius Myriad X VPU using the Intel NCSDK 2.')
    parser.add_argument('model_dir', type=str, default=None, help='path to the GQ-CNN model')
    parser.add_argument('input_h', type=int, default=None, help='input height for the FC-GQ-CNN')
    parser.add_argument('input_w', type=int, default=None, help='input width for the FC-GQ-CNN')
    parser.add_argument('--shave_cores', type=int, default=1, help='number of SHAVE cores to compile network for')
    parser.add_argument('--hardware_version', type=str, default=None, help='specific hardware version to compile for')

    args = parser.parse_args()
    model_dir = args.model_dir
    input_h = args.input_h
    input_w = args.input_w
    shave_cores = args.shave_cores
    hardware_version = args.hardware_version

    # load the FC-GQ-CNN
    logger.info('Loading FC-GQ-CNN...')
    fc_config = {'im_height': input_h, 'im_width': input_w}
    fcgqcnn = get_nc_fc_gqcnn_model().load(model_dir, fc_config)

    # save a clean inference graph for the FC-GQ-CNN
    logger.info('Saving clean inference graph...')
    fcgqcnn.save()

    # compile the model using NCSDK 2 
    logger.info('Compiling model...')
    command = ['mvNCCompile', os.path.join(model_dir, 'model.pb'), '-s', str(shave_cores), '-in', 'input_im', '-on', 'softmax/output', '-o', os.path.join(model_dir, 'model.graph')]
    if hardware_version is not None:
        # add the hardware version flag
        command.append('--{}'.format(hardware_version))
    p = subprocess.Popen(command)
    p.wait()
