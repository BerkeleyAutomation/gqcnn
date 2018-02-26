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
'''
Class to handle experiment logging.
WARNING: This is a READ-ONLY script and is not guaranteed to work. It is meant as an example along with the yumi_control_node script.
Authors: Jeff, Jacky
'''
from abc import ABCMeta, abstractmethod
import csv
from datetime import datetime
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
from time import time

from autolab_core import CSVModel, ExperimentLogger, YamlConfig
from autolab_core.utils import gen_experiment_id

from perception import BinaryImage, DepthImage, Image

class LoggerField(object):
    """ Encapsulates fields saved during experiments """

    # default values
    default_values = {'str':'',
                      'int':0,
                      'float':0.0,
                      'bool':False}                      

    def __init__(self, name, data_type='', value=None, file_ext=None, output_filename_prefix=None):
        self.name = name
        self.data_type = data_type
        self.value = value
        self.file_ext = file_ext
        self.output_filename_prefix = output_filename_prefix

        # set to default
        if self.value is None:
            try:
                self.value = LoggerField.default_values[self.data_type]
            except KeyError as e:
                logging.error('Data type %s not supported. Please create an entry in the defaults file' %(LoggerField.default_values))
                raise e

class GraspIsolatedObjectExperimentLogger(ExperimentLogger):

    def __init__(self, experiment_root_path, supervisor, camera_intr, T_camera_world, cfg_filename, planner_type='default'):
        self.cfg = YamlConfig(cfg_filename)
        self.camera_intr = camera_intr
        self.T_camera_world = T_camera_world
        self.supervisor = supervisor
        self.planner_type = planner_type
        
        super(GraspIsolatedObjectExperimentLogger, self).__init__(experiment_root_path, experiment_tag=planner_type)

        # open csv for data
        data_path = os.path.join(self.experiment_path, 'data.csv')
        self._data_csv = CSVModel.get_or_create(data_path, self.experiment_headers)
        self._cur_uid = None

        # create raw data directory
        if not os.path.exists(self.raw_data_path):
            os.mkdir(self.raw_data_path)

        # write key files
        dst_cfg_filename = os.path.join(self.experiment_path, 'config.yaml')
        shutil.copyfile(cfg_filename, dst_cfg_filename)

        dst_intr_filename = os.path.join(self.experiment_path, '%s.intr' %(camera_intr.frame))
        self.camera_intr.save(dst_intr_filename)

        dst_reg_filename = os.path.join(self.experiment_path,
                                        '%s_to_%s.tf' %(T_camera_world.from_frame,
                                                        T_camera_world.to_frame))
        T_camera_world.save(dst_reg_filename)

    @property
    def raw_data_path(self):
        return os.path.join(self.experiment_path, 'raw')

    @property
    def compressed_data_path(self):
        return os.path.join(self.experiment_path, 'compressed')
        
    @property
    def experiment_meta_headers(self):
        return {
            'experiment_id':'str',
            'use':'bool',
            'time_started':'str',
            'time_stopped':'str',
            'duration':'float',
            'planner_type':'str',
            'supervisor':'str'
        }

    @property
    def experiment_meta_data(self):
        return {
            'experiment_id': self.id, 
            'use': True,
            'planner_type': self.planner_type,
            'supervisor': self.supervisor
        }

    @property
    def experiment_headers(self):
        return {
            'object_key': 'str',
            'trial_num': 'int',
            'test_case_num': 'int',
            'color_im': 'str',
            'depth_im': 'str',
            'seg_binary_im': 'str',
            'seg_color_im': 'str',
            'seg_depth_im': 'str',
            'seg_camera_intrinsics': 'str',
            'input_color_im': 'str',
            'input_depth_im': 'str',
            'input_binary_im': 'str',
            'input_pose': 'str',
            'planning_time': 'float',
            'gripper_pose': 'str',
            'gripper_width': 'float',
            'gripper_torque': 'float',
            'dropped_object': 'bool',
            'lifted_object': 'bool',
            'table_clear': 'bool',
            'pred_robustness': 'float',
            'human_label': 'int',
            'found_grasp': 'int',
            'completed': 'bool'
        }

    @property
    def default_experiment_data(self):
        return {
            'object_key': '',
            'trial_num': 0,
            'test_case_num': 0,
            'color_im': '',
            'depth_im': '',
            'seg_binary_im': '',
            'seg_color_im': '',
            'seg_depth_im': '',
            'seg_camera_intrinsics': '',
            'input_color_im': '',
            'input_depth_im': '',
            'input_binary_im': '',
            'input_pose': '',
            'planning_time': 0.0,
            'gripper_pose': '',
            'gripper_width': 0.0,
            'gripper_torque': 0.0,
            'dropped_object': False,
            'lifted_object': False,
            'table_clear': False,
            'pred_robustness': 0.0,
            'human_label': 0,
            'found_grasp': 0,
            'completed': False
        }

    @property
    def experiment_data_file_exts(self):
        return {
            'object_key': None,
            'trial_num': None,
            'test_case_num': None,
            'color_im': 'png',
            'depth_im': 'npy',
            'seg_binary_im': 'png',
            'seg_color_im': 'png',
            'seg_depth_im': 'npy',
            'seg_camera_intrinsics': 'intr',
            'input_color_im': 'png',
            'input_depth_im': 'npy',
            'input_binary_im': 'png',
            'input_pose': 'npy',
            'planning_time': None,
            'gripper_pose': 'tf',
            'gripper_width': None,
            'gripper_torque': None,
            'dropped_object': None,
            'lifted_object': None,
            'table_clear': None,
            'pred_robustness': None,
            'human_label': None,
            'found_grasp': None,
            'completed': None
        }

    @property
    def experiment_data_output_names(self):
        return {
            'object_key': None,
            'trial_num': None,
            'test_case_num': None,
            'color_im': None,
            'depth_im': None,
            'seg_binary_im': None,
            'seg_color_im': None,
            'seg_depth_im': None,
            'seg_camera_intrinsics': None,
            'input_color_im': None,
            'input_depth_im': 'depth_ims_tf',
            'input_binary_im': 'binary_ims_tf',
            'input_pose': 'hand_poses',
            'planning_time': None,
            'gripper_pose': None,
            'gripper_width': None,
            'gripper_torque': None,
            'dropped_object': None,
            'lifted_object': None,
            'table_clear': None,
            'pred_robustness': None,
            'human_label': None,
            'found_grasp': None,
            'completed': None
        }
        
    @property
    def cur_trial_data(self):
        if self._cur_uid is None:
            raise ValueError('No trials started yet')
        return self._data_csv.get_by_uid(self._cur_uid)

    def start_trial(self):
        """ Starts a new trial (row in the output csv """
        self._cur_uid = self._data_csv.insert(self.default_experiment_data)
        return self._cur_uid

    def update_trial_attribute(self, name, value, uid=None):
        if name not in self.experiment_headers.keys():
            raise ValueError('Attribute %s not supported' %(name))
        if uid is None:
            uid = self._cur_uid
        if isinstance(value, unicode):
            value = str(value)

        if self.experiment_headers[name] == 'str' and not isinstance(value, str):
            # assume auto-save if the passed value was not a string
            file_ext = self.experiment_data_file_exts[name]
            if file_ext is None:
                raise ValueError('Saving raw data for attribute %s is not supported' %(name))
            filename = os.path.join(self.raw_data_path,
                                    '%s_%d.%s' %(name, self._cur_uid, file_ext))
            if file_ext == 'npy' and not isinstance(value, Image):
                np.save(filename, value)
            else:
                value.save(filename)

            # save the filename to the csv
            data = self.cur_trial_data
            data[name] = filename
            self._data_csv.update_by_uid(uid, data)            
        else:
            # save all non-string data types to the CSV
            data = self.cur_trial_data
            data[name] = value
            self._data_csv.update_by_uid(uid, data)

    def record_trial(self, data_dict, uid=None):
        """ Record a full trial """
        # check valid headers
        for name in data_dict.keys():
            if name not in self.experiment_headers.keys():
                raise ValueError('Attribute %s not supported' %(name))

        # write each entry individually
        for name, value in data_dict:
            self.update_trial_data(name, value, uid)

    def create_dataset(self, datapoints_per_file=100):
        """ Compress all known data into a numpy dataset """
        # create compressed data dir
        if not os.path.exists(self.compressed_data_path):
            os.mkdir(self.compressed_data_path)

        # write all data
        file_num = 0
        num_recorded = 0
        data_tensors = {}
        for row in self._data_csv:
            # add all attributes of the current row to the tensor
            if not row['completed']:
                continue
            for name, value in row.iteritems():
                # read raw data
                data = None
                if name == 'input_depth_im':
                    if os.path.exists(value):
                        data = DepthImage.open(value).raw_data[np.newaxis,...]
                elif name == 'input_binary_im':
                    if os.path.exists(value):
                        data = BinaryImage.open(value).raw_data[np.newaxis,...]
                elif name == 'input_pose':
                    if os.path.exists(value):
                        data = np.load(value)[np.newaxis,...]
                elif name == 'lifted_object' or name == 'human_label' \
                     or name == 'gripper_width' or name == 'found_grasp' \
                     or name == 'gripper_torque':
                    data = float(value)

                if data is None:
                    continue

                # update tensor
                if name not in data_tensors or data_tensors[name] is None:
                    data_tensors[name] = data
                else:
                    data_tensors[name] = np.r_[data_tensors[name], data]
            num_recorded += 1

            # write to file if necessary
            if num_recorded >= datapoints_per_file:
                # save each tensor
                for name, tensor in data_tensors.iteritems():
                    output_name = self.experiment_data_output_names[name]
                    if output_name is None:
                        output_name = name
                    filename = os.path.join(self.compressed_data_path, '%s_%05d.npz' %(output_name, file_num))
                    np.savez_compressed(filename, tensor)
                
                # update for next round
                for name in data_tensors.keys():
                    del data_tensors[name]
                    data_tensors[name] = None
                file_num += 1
                num_recorded = 0

        # save final tensor
        if num_recorded > 0:
            for name, tensor in data_tensors.iteritems():
                output_name = self.experiment_data_output_names[name]
                if output_name is None:
                    output_name = name
                filename = os.path.join(self.compressed_data_path, '%s_%05d.npz' %(output_name, file_num))
                np.savez_compressed(filename, tensor)

