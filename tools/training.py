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
Script for training a Grasp Quality Neural Network (GQ-CNN) from scratch.

Author
------
Vishal Satish & Jeff Mahler
"""
import argparse
import logging
import os
import time
import os

import autolab_core.utils as utils
from autolab_core import YamlConfig
from gqcnn import GQCNN, GQCNNOptimizer, GQCNNAnalyzer
from gqcnn import utils as gqcnn_utils

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Train a Grasp Quality Convolutional Neural Network from scratch with TensorFlow')
    parser.add_argument('dataset_dir', type=str, default=None,
                        help='path to the dataset to use for training and validation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='path to store the model')
    parser.add_argument('--config_filename', type=str, default=None,
                        help='path to the configuration file to use')
    parser.add_argument('--unique_name', type=bool, default=False,
                        help='add a unique name to dataset path which encodes the current date')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    config_filename = args.config_filename

    # set default output dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  '../models')
    
    # set default config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/training.yaml')

    # turn relative paths absolute
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # create output dir if necessary
    utils.mkdir_safe(output_dir)
        
    # open train config
    train_config = YamlConfig(config_filename)
    gqcnn_params = train_config['gqcnn']

    # create a unique output folder based on the date and time
    if args.unique_name:
        # create output dir
        unique_name = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, unique_name)
        utils.mkdir_safe(output_dir)

    # set visible devices
    if 'gpu_list' in train_config:
        gqcnn_utils.set_cuda_visible_devices(train_config['gpu_list'])

    # train the network
    start_time = time.time()
    gqcnn = GQCNN(gqcnn_params)
    optimizer = GQCNNOptimizer(gqcnn,
                               dataset_dir,
                               output_dir,
                               train_config)
    with gqcnn.tf_graph.as_default():
        optimizer.optimize()
    logging.info('Total Training Time:' + str(utils.get_elapsed_time(time.time() - start_time))) 
