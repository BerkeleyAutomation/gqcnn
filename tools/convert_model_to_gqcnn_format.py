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
Converts a GQ-CNN model trained using the Dex-Net train_grasp_quality_cnn.py script to a model that the GQ-CNN package can import.
Author: Jeff Mahler
"""
import argparse
import collections
import logging
import json
import os
import sys

from autolab_core import YamlConfig

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)
    
    # parse args
    parser = argparse.ArgumentParser(description='Train a CNN for grasp quality prediction')
    parser.add_argument('model_dir', type=str, help='directory of GQ-CNN model')
    args = parser.parse_args()
    model_dir = args.model_dir
    
    # read config
    if not os.path.exists(model_dir):
        raise ValueError('Model dir %s does not exist!')
        exit(0)
    config_filename = os.path.join(model_dir, 'config.yaml')
    config = YamlConfig(config_filename)
    
    # update config

    # update architecture
    for layer_name in config['architecture'].keys():
        if layer_name.find('conv') > -1:
            config['architecture'][layer_name]['norm_type'] = 'local_response'

    # update learning params
    config['loss'] = 'sparse'
    config['fine_tune'] = 0

    # create gqcnn config
    config['gqcnn_config'] = {}
    config['gqcnn_config']['architecture'] = config['architecture']
    config['architecture'] = 0
    config['gqcnn_config']['im_height'] = 32
    config['gqcnn_config']['im_width'] = 32
    config['gqcnn_config']['im_channels'] = 1
    config['gqcnn_config']['input_data_mode'] = 'tf_image'
    config['gqcnn_config']['batch_size'] = config['train_batch_size']
    config['gqcnn_config']['radius'] = 2
    config['gqcnn_config']['alpha'] = 2.0e-05
    config['gqcnn_config']['beta'] = 0.75
    config['gqcnn_config']['bias'] = 1.0

    # convert to ordered dict and write
    ordered_dict = collections.OrderedDict()
    for key in config.keys():
        ordered_dict[key] = config[key]

    # write to file
    json_config_filename = os.path.join(model_dir, 'config.json')
    with open(json_config_filename, 'w') as outfile:
        json.dump(ordered_dict, outfile)
  
