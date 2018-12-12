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
Creates a dataset for multi-gripper training
Author: Jeff Mahler
"""
import argparse
import copy
import IPython
import logging
import numpy as np
import os
import shutil
import sys

from autolab_core import TensorDataset, YamlConfig
import autolab_core.utils as utils

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)
    
    # parse args
    parser = argparse.ArgumentParser(description='Merges a set of tensor datasets')
    parser.add_argument('output_dataset_name', type=str, default=None, help='name of output dataset')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/create_multi_gripper_training_dataset.yaml', help='configuration file to use')
    args = parser.parse_args()
    output_dataset_name = args.output_dataset_name
    config_filename = args.config_filename

    # open config file
    cfg = YamlConfig(config_filename)
    input_datasets = cfg['input_datasets']
    display_rate = cfg['display_rate']

    input_dataset_names = []
    for gripper_id, gripper_name in enumerate(input_datasets.keys()):
        dataset_config = input_datasets[gripper_name]
        dataset_name = dataset_config['dataset']
        input_dataset_names.append(dataset_name)
        
    # open tensor dataset
    dataset = TensorDataset.open(input_dataset_names[0])
    tensor_config = copy.deepcopy(dataset.config)
    for field_name in cfg['exclude_fields']:
        if field_name in tensor_config['fields'].keys():
            del tensor_config['fields'][field_name]
    field_names = tensor_config['fields'].keys()
    tensor_config['fields']['gripper_ids'] = {'dtype': 'uint8'}        

    # init tensor dataset
    output_dataset = TensorDataset(output_dataset_name, tensor_config)

    # copy config
    out_config_filename = os.path.join(output_dataset_name, 'merge_config.yaml')
    shutil.copyfile(config_filename, out_config_filename)
    
    # incrementally add points to the new dataset
    gripper_name_map = {}
    gripper_type_map = {}
    gripper_id_map = {}
    for gripper_id, gripper_name in enumerate(input_datasets.keys()):
        dataset_config = input_datasets[gripper_name]
        dataset_name = dataset_config['dataset']
        gripper_type  = dataset_config['type']

        dataset = TensorDataset.open(dataset_name)
        gripper_name_map[gripper_id] = gripper_name
        gripper_type_map[gripper_id] = gripper_type
        gripper_id_map[dataset_name] = gripper_id

        logging.info('Aggregating data from dataset %s' %(dataset_name))        
        for i in range(dataset.num_datapoints):
            # read a datapoint
            datapoint = dataset.datapoint(i, field_names=field_names)

            # display rate
            if i % display_rate == 0:
                logging.info('Datapoint: %d of %d' %(i+1, dataset.num_datapoints))

            # add gripper id
            datapoint['gripper_ids'] = gripper_id
            
            # add datapoint    
            output_dataset.add(datapoint)

    # set metadata
    gripper_id_map = utils.reverse_dictionary(gripper_id_map)
    output_dataset.add_metadata('gripper_names', gripper_name_map)
    output_dataset.add_metadata('gripper_ids', gripper_id_map)
    output_dataset.add_metadata('gripper_types', gripper_type_map)
    for field_name, field_data in dataset.metadata.iteritems():
        if field_name not in ['obj_ids']:
            output_dataset.add_metadata(field_name, field_data)

    # flush to disk
    output_dataset.flush()    
