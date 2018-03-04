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
Quick wrapper for grasp quality neural network
Author: Jeff Mahler
"""

import copy
import json
import logging
import matplotlib.pyplot as plt
import numbers
import numpy as np
import os
import scipy.misc as sm
import sys
import tensorflow as tf

from autolab_core import YamlConfig

import optimizer_constants
from optimizer_constants import InputDataMode
def reduce_shape(shape):
    """ Get shape of a layer for flattening """
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape, 1)


class GQCnnWeights(object):
    """ Struct helper for storing weights """

    def __init__(self):
        pass


class GQCnnDenoisingWeights(object):
    """ Struct helper for storing weights """

    def __init__(self):
        pass


class GQCNN(object):
    """ Wrapper for grasp quality CNN """

    def __init__(self, config):
        """
        Parameters
        ----------
        config :obj: dict
            python dictionary of configuration parameters such as architecure and basic data params such as batch_size for prediction,
            im_height, im_width, ...
        """
        self._sess = None
        self._graph = tf.Graph()
        self._feature_tensors = {}
        self._parse_config(config)

    @staticmethod
    def load(model_dir):
        """ Instantiates a GQCNN object using the model found in model_dir 

        Parameters
        ----------
        model_dir :obj: str
            path to model directory where weights and architecture are stored

        Returns
        -------
        :obj:`GQCNN`
            GQCNN object initialized with the weights and architecture found in the specified model directory
        """
        # get config dict with architecture and other basic configurations for GQCNN from config.json in model directory
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file) as data_file:    
            train_config = json.load(data_file)

        gqcnn_config = train_config['gqcnn_config']

        # create GQCNN object and initialize weights and network
        gqcnn = GQCNN(gqcnn_config)
        gqcnn.init_weights_file(os.path.join(model_dir, 'model.ckpt'))
        gqcnn.initialize_network()
        gqcnn.init_mean_and_std(model_dir)

        return gqcnn

    @property
    def feature_tensors(self):
        """ Dictionary containing the tensors for intermediate GQ-CNN layers. """
        return self._feature_tensors
    
    def get_tf_graph(self):
        """ Returns the graph for this tf session 

        Returns
        -------
        :obj:`tf Graph`
            TensorFlow Graph 
        """
        return self._graph

    def get_weights(self):
        """ Returns the weights for this network 

        Returns
        -------
        :obj:`GQCnnWeights`
            network weights
        """
        return self._weights

    def init_mean_and_std(self, model_dir):
        """ Initializes the mean and std to use for data normalization during prediction 

        Parameters
        ----------
        model_dir :obj: str
            path to model directory where means and standard deviations are stored
        """
        # load in means and stds for all 7 possible pose variables
        # grasp center row, grasp center col, gripper depth, grasp theta, crop center row, crop center col, grip width
        self._im_mean = np.load(os.path.join(model_dir, 'mean.npy'))
        self._im_std = np.load(os.path.join(model_dir, 'std.npy'))
        self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))

        # slice out the variables we want based on the input pose_dim, which
        # is dependent on the input data mode used to train the model
        if len(self._pose_mean.shape) > 0 and self._pose_mean.shape[0] == 7:
            if self._input_data_mode == InputDataMode.TF_IMAGE:
                # depth
                if isinstance(self.pose_mean, numbers.Number) \
                   or len(self.pose_mean.shape) == 0 \
                   or self.pose_mean.shape[0] == 1:
                    self._pose_mean = self.pose_mean
                else:
                    self._pose_mean = self._pose_mean[2]
                    self._pose_std = self._pose_std[2]
            elif self._input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
                # depth, cx, cy
                self._pose_mean = np.concatenate([self._pose_mean[2:3], self._pose_mean[4:6]])
                self._pose_std = np.concatenate([self._pose_std[2:3], self._pose_std[4:6]])
            elif self._input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
                # depth, phi
                self._pose_mean = self._pose_mean[2:4]
                self._pose_std = self._pose_std[2:4]
            elif self._input_data_mode == InputDataMode.RAW_IMAGE:
                # u, v, depth, theta
                self._pose_mean = self._pose_mean[:4]
                self._pose_std = self._pose_std[:4]
            elif self._input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
                # u, v, depth, theta, cx, cy
                self._pose_mean = self._pose_mean[:6]
                self._pose_std = self._pose_std[:6]

        elif len(self._pose_mean.shape) > 0 and self._pose_mean.shape[0] == 4 and self._input_data_mode == InputDataMode.TF_IMAGE:
            # depth
            if isinstance(self.pose_mean, numbers.Number) \
               or len(self.pose_mean.shape) == 0 \
               or self.pose_mean.shape[0] == 1:
                self._pose_mean = self.pose_mean
            else:
                self._pose_mean = self._pose_mean[2]
                self._pose_std = self._pose_std[2]
        elif len(self._pose_mean.shape) > 0 and (self._pose_mean.shape[0] == 5 or self._pose_mean.shape[0] == 6):
            if self._input_data_mode == InputDataMode.PARALLEL_JAW:
                # depth
                if isinstance(self.pose_mean, numbers.Number) \
                   or len(self.pose_mean.shape) == 0 \
                   or self.pose_mean.shape[0] == 1:
                    self._pose_mean = self.pose_mean
                else:
                    self._pose_mean = self._pose_mean[2]
                    self._pose_std = self._pose_std[2]
            elif self._input_data_mode == InputDataMode.SUCTION:
                self._pose_mean = np.r_[self._pose_mean[2], self._pose_mean[4]]
                self._pose_std = np.r_[self._pose_std[2], self._pose_std[4]]
                
    def init_weights_file(self, model_filename):
        """ Initialize network weights from the specified model 

        Parameters
        ----------
        model_filename :obj: str
            path to model to be loaded into weights
        """

        # read the input image
        with self._graph.as_default():

            # read in filenames
            reader = tf.train.NewCheckpointReader(model_filename)

            # create empty weight object
            self._weights = GQCnnWeights()

            # read in conv1 & conv2
            try:
                self._weights.conv1_1W = tf.Variable(reader.get_tensor("conv1_1W"), name="conv1_1W")
                self._weights.conv1_1b = tf.Variable(reader.get_tensor("conv1_1b"), name="conv1_1b")
                self._weights.conv1_2W = tf.Variable(reader.get_tensor("conv1_2W"), name="conv1_2W")
                self._weights.conv1_2b = tf.Variable(reader.get_tensor("conv1_2b"), name="conv1_2b")

                # if conv2 is to be used, read in conv2
                if self._use_conv2:
                    self._weights.conv2_1W = tf.Variable(reader.get_tensor("conv2_1W"), name="conv2_1W")
                    self._weights.conv2_1b = tf.Variable(reader.get_tensor("conv2_1b"), name="conv2_1b")
                    self._weights.conv2_2W = tf.Variable(reader.get_tensor("conv2_2W"), name="conv2_2W")
                    self._weights.conv2_2b = tf.Variable(reader.get_tensor("conv2_2b"), name="conv2_2b")

                # if conv3 is to be used, read in conv3
                if self._use_conv3:
                    self._weights.conv3_1W = tf.Variable(reader.get_tensor("conv3_1W"), name="conv3_1W")
                    self._weights.conv3_1b = tf.Variable(reader.get_tensor("conv3_1b"), name="conv3_1b")
                    self._weights.conv3_2W = tf.Variable(reader.get_tensor("conv3_2W"), name="conv3_2W")
                    self._weights.conv3_2b = tf.Variable(reader.get_tensor("conv3_2b"), name="conv3_2b")

                # read in pc1
                self._weights.pc1W = tf.Variable(reader.get_tensor("pc1W"), name="pc1W")
                self._weights.pc1b = tf.Variable(reader.get_tensor("pc1b"), name="pc1b")
                
                # if pc2 is to be used, read in pc2
                if self._use_pc2:
                    self._weights.pc2W = tf.Variable(reader.get_tensor("pc2W"), name="pc2W")
                    self._weights.pc2b = tf.Variable(reader.get_tensor("pc2b"), name="pc2b")

                self._weights.fc3W = tf.Variable(reader.get_tensor("fc3W"), name="fc3W")
                self._weights.fc3b = tf.Variable(reader.get_tensor("fc3b"), name="fc3b")
                self._weights.fc4W_im = tf.Variable(reader.get_tensor("fc4W_im"), name="fc4W_im")
                self._weights.fc4W_pose = tf.Variable(reader.get_tensor("fc4W_pose"), name="fc4W_pose")
                self._weights.fc4b = tf.Variable(reader.get_tensor("fc4b"), name="fc4b")
                self._weights.fc5W = tf.Variable(reader.get_tensor("fc5W"), name="fc5W")
                self._weights.fc5b = tf.Variable(reader.get_tensor("fc5b"), name="fc5b")
            except:
                self._weights.conv1_1W = tf.Variable(reader.get_tensor("validation_network/im_stream/conv1_1/conv1_1_weights"), name="conv1_1W")
                self._weights.conv1_1b = tf.Variable(reader.get_tensor("validation_network/im_stream/conv1_1/conv1_1_bias"), name="conv1_1b")
                self._weights.conv1_2W = tf.Variable(reader.get_tensor("validation_network/im_stream/conv1_2/conv1_2_weights"), name="conv1_2W")
                self._weights.conv1_2b = tf.Variable(reader.get_tensor("validation_network/im_stream/conv1_2/conv1_2_bias"), name="conv1_2b")

                # if conv3 is to be used, read in conv3
                if self._use_conv2:
                    self._weights.conv2_1W = tf.Variable(reader.get_tensor("validation_network/im_stream/conv2_1/conv2_1_weights"), name="conv2_1W")
                    self._weights.conv2_1b = tf.Variable(reader.get_tensor("validation_network/im_stream/conv2_1/conv2_1_bias"), name="conv2_1b")
                    self._weights.conv2_2W = tf.Variable(reader.get_tensor("validation_network/im_stream/conv2_2/conv2_2_weights"), name="conv2_2W")
                    self._weights.conv2_2b = tf.Variable(reader.get_tensor("validation_network/im_stream/conv2_2/conv2_2_bias"), name="conv2_2b")

                # if conv3 is to be used, read in conv3
                if self._use_conv3:
                    self._weights.conv3_1W = tf.Variable(reader.get_tensor("validation_network/im_stream/conv3_1/conv3_1_weights"), name="conv3_1W")
                    self._weights.conv3_1b = tf.Variable(reader.get_tensor("validation_network/im_stream/conv3_1/conv3_1_bias"), name="conv3_1b")
                    self._weights.conv3_2W = tf.Variable(reader.get_tensor("validation_network/im_stream/conv3_1/conv3_2_weights"), name="conv3_2W")
                    self._weights.conv3_2b = tf.Variable(reader.get_tensor("validation_network/im_stream/conv3_1/conv3_2_bias"), name="conv3_2b")

                # read in pc1
                self._weights.pc1W = tf.Variable(reader.get_tensor("validation_network/pose_stream/pc1_weights"), name="pc1W")
                self._weights.pc1b = tf.Variable(reader.get_tensor("validation_network/pose_stream/pc1_bias"), name="pc1b")
                
                # if pc2 is to be used, read in pc2
                if self._use_pc2:
                    self._weights.pc2W = tf.Variable(reader.get_tensor("validation_network/pose_stream/pc2_weights"), name="pc2W")
                    self._weights.pc2b = tf.Variable(reader.get_tensor("validation_network/pose_stream/pc2_bias"), name="pc2b")

                self._weights.fc3W = tf.Variable(reader.get_tensor("validation_network/im_stream/fc3_weights"), name="fc3W")
                self._weights.fc3b = tf.Variable(reader.get_tensor("validation_network/im_stream/fc3_bias"), name="fc3b")
                self._weights.fc4W_im = tf.Variable(reader.get_tensor("validation_network/merge_stream/fc4_input_1_weights"), name="fc4W_im")
                self._weights.fc4W_pose = tf.Variable(reader.get_tensor("validation_network/merge_stream/fc4_input_2_weights"), name="fc4W_pose")
                self._weights.fc4b = tf.Variable(reader.get_tensor("validation_network/merge_stream/fc4_bias"), name="fc4b")
                self._weights.fc5W = tf.Variable(reader.get_tensor("validation_network/merge_stream/fc5_weights"), name="fc5W")
                self._weights.fc5b = tf.Variable(reader.get_tensor("validation_network/merge_stream/fc5_bias"), name="fc5b")

    def reinitialize_layers(self, reinit_fc3, reinit_fc4, reinit_fc5, reinit_pc1=False, fc5_out_size=1):
        """ Re-initializes final fully-connected layers for fine-tuning 

        Parameters
        ----------
        reinit_fc3 : bool
            whether to re-initialize fc3
        reinit_fc4 : bool
            whether to re-initialize fc4
        reinit_fc5 : bool
            whether to re-initialize fc5
        reinit_pc1 : bool
            whether to re-initiazlize pc1
        """
        with self._graph.as_default():
            if reinit_pc1:
                pc1_std = np.sqrt(2.0 / self.pc1_in_size)
                self._weights.pc1W = tf.Variable(tf.truncated_normal([self.pc1_in_size, self.pc1_out_size],
                                                       stddev=pc1_std), name='pc1W')
                self._weights.pc1b = tf.Variable(tf.truncated_normal([self.pc1_out_size],
                                                       stddev=pc1_std), name='pc1b')

            if reinit_fc3:
                fc3_std = np.sqrt(2.0 / (self.fc3_in_size))
                self._weights.fc3W = tf.Variable(tf.truncated_normal([self.fc3_in_size, self.fc3_out_size], stddev=fc3_std), name='fc3W')
                self._weights.fc3b = tf.Variable(tf.truncated_normal([self.fc3_out_size], stddev=fc3_std), name='fc3b')  
            if reinit_fc4:
                fc4_std = np.sqrt(2.0 / (self.fc4_in_size))
                self._weights.fc4W_im = tf.Variable(tf.truncated_normal([self.fc4_in_size, self.fc4_out_size], stddev=fc4_std), name='fc4W_im')
                self._weights.fc4W_pose = tf.Variable(tf.truncated_normal([self.fc4_pose_in_size, self.fc4_out_size], stddev=fc4_std), name='fc4W_pose')
                self._weights.fc4b = tf.Variable(tf.truncated_normal([self.fc4_out_size], stddev=fc4_std), name='fc4b')
            if reinit_fc5:
                self.fc5_out_size = fc5_out_size
                fc5_std = np.sqrt(2.0 / (self.fc5_in_size))
                self._weights.fc5W = tf.Variable(tf.truncated_normal([self.fc5_in_size, self.fc5_out_size], stddev=fc5_std), name='fc5W')
                self._weights.fc5b = tf.Variable(tf.constant(0.0, shape=[self.fc5_out_size]), name='fc5b')
    
    def init_weights_gaussian(self):
        """ Initializes weights for network from scratch using Gaussian Distribution """

        # init pool size variables
        cfg = self._architecture
        layer_height = self._im_height
        layer_width = self._im_width
        layer_channels = self._num_channels

        # conv1_1
        conv1_1_filt_dim = cfg['conv1_1']['filt_dim']
        conv1_1_num_filt = cfg['conv1_1']['num_filt']
        conv1_1_size = layer_height * layer_width * conv1_1_num_filt
        conv1_1_shape = [conv1_1_filt_dim, conv1_1_filt_dim, layer_channels, conv1_1_num_filt]

        conv1_1_num_inputs = conv1_1_filt_dim**2 * layer_channels
        conv1_1_std = np.sqrt(2.0 / (conv1_1_num_inputs))
        conv1_1W = tf.Variable(tf.truncated_normal(conv1_1_shape, stddev=conv1_1_std), name='conv1_1W')
        conv1_1b = tf.Variable(tf.truncated_normal([conv1_1_num_filt], stddev=conv1_1_std), name='conv1_1b')

        conv1_1_padding = 'SAME'
        if 'padding' in self._architecture['conv1_1'].keys():
            conv1_1_padding = self._architecture['conv1_1']['padding']
        if conv1_1_padding == 'VALID':
            border_dim = (conv1_1_filt_dim - 1) / 2
            layer_height = layer_height - 2 * border_dim
            layer_width = layer_width - 2 * border_dim
        layer_height = int(np.ceil(float(layer_height) / cfg['conv1_1']['pool_stride']))
        layer_width = int(np.ceil(float(layer_width) / cfg['conv1_1']['pool_stride']))
        layer_channels = conv1_1_num_filt
        
        # conv1_2
        conv1_2_filt_dim = cfg['conv1_2']['filt_dim']
        conv1_2_num_filt = cfg['conv1_2']['num_filt']
        conv1_2_size = layer_height * layer_width * conv1_2_num_filt
        conv1_2_shape = [conv1_2_filt_dim, conv1_2_filt_dim, layer_channels, conv1_2_num_filt]

        conv1_2_num_inputs = conv1_2_filt_dim**2 * layer_channels
        conv1_2_std = np.sqrt(2.0 / (conv1_2_num_inputs))
        conv1_2W = tf.Variable(tf.truncated_normal(conv1_2_shape, stddev=conv1_2_std), name='conv1_2W')
        conv1_2b = tf.Variable(tf.truncated_normal([conv1_2_num_filt], stddev=conv1_2_std), name='conv1_2b')

        conv1_2_padding = 'SAME'
        if 'padding' in self._architecture['conv1_2'].keys():
            conv1_2_padding = self._architecture['conv1_2']['padding']
        if conv1_2_padding == 'VALID':
            border_dim = (conv1_2_filt_dim - 1) / 2
            layer_height = layer_height - 2 * border_dim
            layer_width = layer_width - 2 * border_dim                    
        layer_height = int(np.ceil(float(layer_height) / cfg['conv1_2']['pool_stride']))
        layer_width = int(np.ceil(float(layer_width) / cfg['conv1_2']['pool_stride']))
        layer_channels = conv1_2_num_filt
        conv_output_size = layer_height * layer_width * layer_channels
        logging.info('Conv1 output size: %d' %(conv_output_size))

        use_conv2 = False
        if 'conv2_1' in cfg.keys():
            use_conv2 = True

        if use_conv2:
            # conv2_1
            conv2_1_filt_dim = cfg['conv2_1']['filt_dim']
            conv2_1_num_filt = cfg['conv2_1']['num_filt']
            conv2_1_size = layer_height * layer_width * conv2_1_num_filt
            conv2_1_shape = [conv2_1_filt_dim, conv2_1_filt_dim, layer_channels, conv2_1_num_filt]
            
            conv2_1_num_inputs = conv2_1_filt_dim**2 * layer_channels
            conv2_1_std = np.sqrt(2.0 / (conv2_1_num_inputs))
            conv2_1W = tf.Variable(tf.truncated_normal(conv2_1_shape, stddev=conv2_1_std), name='conv2_1W')
            conv2_1b = tf.Variable(tf.truncated_normal([conv2_1_num_filt], stddev=conv2_1_std), name='conv2_1b')

            conv2_1_padding = 'SAME'
            if 'padding' in self._architecture['conv2_1'].keys():
                conv2_1_padding = self._architecture['conv2_1']['padding']
            if conv2_1_padding == 'VALID':
                border_dim = (conv2_1_filt_dim - 1) / 2
                layer_height = layer_height - 2 * border_dim
                layer_width = layer_width - 2 * border_dim            
            layer_height = int(np.ceil(float(layer_height) / cfg['conv2_1']['pool_stride']))
            layer_width = int(np.ceil(float(layer_width) / cfg['conv2_1']['pool_stride']))
            layer_channels = conv2_1_num_filt

            conv_output_size = layer_height * layer_width * layer_channels
            logging.info('Conv2_1 output size: %d' %(conv_output_size))
            
            # conv2_2
            conv2_2_filt_dim = cfg['conv2_2']['filt_dim']
            conv2_2_num_filt = cfg['conv2_2']['num_filt']
            conv2_2_size = layer_height * layer_width * conv2_2_num_filt
            conv2_2_shape = [conv2_2_filt_dim, conv2_2_filt_dim, layer_channels, conv2_2_num_filt]
            
            conv2_2_num_inputs = conv2_2_filt_dim**2 * layer_channels
            conv2_2_std = np.sqrt(2.0 / (conv2_2_num_inputs))
            conv2_2W = tf.Variable(tf.truncated_normal(conv2_2_shape, stddev=conv2_2_std), name='conv2_2W')
            conv2_2b = tf.Variable(tf.truncated_normal([conv2_2_num_filt], stddev=conv2_2_std), name='conv2_2b')
            
            conv2_2_padding = 'SAME'
            if 'padding' in self._architecture['conv2_2'].keys():
                conv2_2_padding = self._architecture['conv2_2']['padding']
            if conv2_2_padding == 'VALID':
                border_dim = (conv2_2_filt_dim - 1) / 2
                layer_height = layer_height - 2 * border_dim
                layer_width = layer_width - 2 * border_dim            
            layer_height = int(np.ceil(float(layer_height) / cfg['conv2_2']['pool_stride']))
            layer_width = int(np.ceil(float(layer_width) / cfg['conv2_2']['pool_stride']))
            layer_channels = conv2_2_num_filt

            conv_output_size = layer_height * layer_width * layer_channels
            logging.info('Conv2_2 output size: %d' %(conv_output_size))

        use_conv3 = False
        if 'conv3_1' in cfg.keys():
            use_conv3 = True

        if use_conv3:
            # conv3_1
            conv3_1_filt_dim = cfg['conv3_1']['filt_dim']
            conv3_1_num_filt = cfg['conv3_1']['num_filt']
            conv3_1_size = layer_height * layer_width * conv3_1_num_filt
            conv3_1_shape = [conv3_1_filt_dim, conv3_1_filt_dim, layer_channels, conv3_1_num_filt]
            
            conv3_1_num_inputs = conv3_1_filt_dim**2 * layer_channels
            conv3_1_std = np.sqrt(2.0 / (conv3_1_num_inputs))
            conv3_1W = tf.Variable(tf.truncated_normal(conv3_1_shape, stddev=conv3_1_std), name='conv3_1W')
            conv3_1b = tf.Variable(tf.truncated_normal([conv3_1_num_filt], stddev=conv3_1_std), name='conv3_1b')
            
            conv3_1_padding = 'SAME'
            if 'padding' in self._architecture['conv3_1'].keys():
                conv3_1_padding = self._architecture['conv3_1']['padding']
            if conv3_1_padding == 'VALID':
                border_dim = (conv3_1_filt_dim - 1) / 2
                layer_height = layer_height - 2 * border_dim
                layer_width = layer_width - 2 * border_dim            
            layer_height = int(np.ceil(float(layer_height) / cfg['conv3_1']['pool_stride']))
            layer_width = int(np.ceil(float(layer_width) / cfg['conv3_1']['pool_stride']))
            layer_channels = conv3_1_num_filt

            # conv3_2
            conv3_2_filt_dim = cfg['conv3_2']['filt_dim']
            conv3_2_num_filt = cfg['conv3_2']['num_filt']
            conv3_2_size = layer_height * layer_width * conv3_2_num_filt
            conv3_2_shape = [conv3_2_filt_dim, conv3_2_filt_dim, layer_channels, conv3_2_num_filt]
            
            conv3_2_num_inputs = conv3_2_filt_dim**2 * layer_channels
            conv3_2_std = np.sqrt(2.0 / (conv3_2_num_inputs))
            conv3_2W = tf.Variable(tf.truncated_normal(conv3_2_shape, stddev=conv3_2_std), name='conv3_2W')
            conv3_2b = tf.Variable(tf.truncated_normal([conv3_2_num_filt], stddev=conv3_2_std), name='conv3_2b')
            
            conv3_2_padding = 'SAME'
            if 'padding' in self._architecture['conv3_2'].keys():
                conv3_2_padding = self._architecture['conv3_2']['padding']
            if conv3_2_padding == 'VALID':
                border_dim = (conv3_2_filt_dim - 1) / 2
                layer_height = layer_height - 2 * border_dim
                layer_width = layer_width - 2 * border_dim            
            layer_height = int(np.ceil(float(layer_height) / cfg['conv3_2']['pool_stride']))
            layer_width = int(np.ceil(float(layer_width) / cfg['conv3_2']['pool_stride']))
            layer_channels = conv3_2_num_filt

            conv_output_size = layer_height * layer_width * layer_channels

        # fc3
        fc3_in_size = conv_output_size
        fc3_out_size = cfg['fc3']['out_size']
        fc3_std = np.sqrt(2.0 / fc3_in_size)
        fc3W = tf.Variable(tf.truncated_normal([fc3_in_size, fc3_out_size], stddev=fc3_std), name='fc3W')
        fc3b = tf.Variable(tf.truncated_normal([fc3_out_size], stddev=fc3_std), name='fc3b')

        # pc1
        pc1_in_size = self._pose_dim
        pc1_out_size = cfg['pc1']['out_size']

        pc1_std = np.sqrt(2.0 / pc1_in_size)
        pc1W = tf.Variable(tf.truncated_normal([pc1_in_size, pc1_out_size],
                                               stddev=pc1_std), name='pc1W')
        pc1b = tf.Variable(tf.truncated_normal([pc1_out_size],
                                               stddev=pc1_std), name='pc1b')

        # pc2
        pc2_in_size = pc1_out_size
        pc2_out_size = cfg['pc2']['out_size']

        if pc2_out_size > 0:
            pc2_std = np.sqrt(2.0 / pc2_in_size)
            pc2W = tf.Variable(tf.truncated_normal([pc2_in_size, pc2_out_size],
                                                   stddev=pc2_std), name='pc2W')
            pc2b = tf.Variable(tf.truncated_normal([pc2_out_size],
                                                   stddev=pc2_std), name='pc2b')

        # fc4
        fc4_im_in_size = fc3_out_size
        if pc2_out_size == 0:
            fc4_pose_in_size = pc1_out_size
        else:
            fc4_pose_in_size = pc2_out_size
        fc4_out_size = cfg['fc4']['out_size']
        fc4_std = np.sqrt(2.0 / (fc4_im_in_size + fc4_pose_in_size))
        fc4W_im = tf.Variable(tf.truncated_normal([fc4_im_in_size, fc4_out_size], stddev=fc4_std), name='fc4W_im')
        fc4W_pose = tf.Variable(tf.truncated_normal([fc4_pose_in_size, fc4_out_size], stddev=fc4_std), name='fc4W_pose')
        fc4b = tf.Variable(tf.truncated_normal([fc4_out_size], stddev=fc4_std), name='fc4b')

        # fc5
        fc5_in_size = fc4_out_size
        fc5_out_size = cfg['fc5']['out_size']
        fc5_std = np.sqrt(2.0 / (fc5_in_size))
        fc5W = tf.Variable(tf.truncated_normal([fc5_in_size, fc5_out_size], stddev=fc5_std), name='fc5W')
        fc5b = tf.Variable(tf.constant(0.0, shape=[fc5_out_size]), name='fc5b')

        # create empty weight object and fill it up
        self._weights = GQCnnWeights()

        self._weights.conv1_1W = conv1_1W
        self._weights.conv1_1b = conv1_1b
        self._weights.conv1_2W = conv1_2W
        self._weights.conv1_2b = conv1_2b
        if use_conv2:
            self._weights.conv2_1W = conv2_1W
            self._weights.conv2_1b = conv2_1b
            self._weights.conv2_2W = conv2_2W
            self._weights.conv2_2b = conv2_2b
        
        if use_conv3:
            self._weights.conv3_1W = conv3_1W
            self._weights.conv3_1b = conv3_1b
            self._weights.conv3_2W = conv3_2W
            self._weights.conv3_2b = conv3_2b

        self._weights.fc3W = fc3W
        self._weights.fc3b = fc3b
        self._weights.fc4W_im = fc4W_im
        self._weights.fc4W_pose = fc4W_pose
        self._weights.fc4b = fc4b
        self._weights.fc5W = fc5W
        self._weights.fc5b = fc5b
        self._weights.pc1W = pc1W
        self._weights.pc1b = pc1b

        if pc2_out_size > 0:
            self._weights.pc2W = pc2W
            self._weights.pc2b = pc2b

    def _parse_config(self, config):
        """ Parses configuration file for this GQCNN 

        Parameters
        ----------
        config : dict
            python dictionary of configuration parameters such as architecure and basic data params such as batch_size for prediction,
            im_height, im_width, ... 
        """

        # load tensor params
        self._batch_size = config['batch_size']
        self._im_height = config['im_height']
        self._im_width = config['im_width']
        self._num_channels = config['im_channels']
        self._input_data_mode = config['input_data_mode']

        # setup correct pose dimensions 
        if self._input_data_mode == InputDataMode.PARALLEL_JAW:
            self._pose_dim = 1
        elif self._input_data_mode == InputDataMode.SUCTION:
            self._pose_dim = 2
        elif self._input_data_mode == InputDataMode.TF_IMAGE:
            # depth
            self._pose_dim = 1
        elif self._input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
            # depth, cx, cy
            self._pose_dim = 3
        elif self._input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
            # depth, phi
            self._pose_dim = 2
        elif self._input_data_mode == InputDataMode.RAW_IMAGE:
            # u, v, depth, theta
            self._pose_dim = 4
        elif self._input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
            # u, v, depth, theta, cx, cy
            self._pose_dim = 6

        # create feed tensors for prediction
        self._input_im_arr = np.zeros([self._batch_size, self._im_height,
                                       self._im_width, self._num_channels])
        self._input_pose_arr = np.zeros([self._batch_size, self._pose_dim])

        # load architecture
        self._architecture = config['architecture']
        self._use_conv2 = False
        if 'conv2_1' in self._architecture.keys():
            self._use_conv2 = True
        self._use_conv3 = False
        if 'conv3_1' in self._architecture.keys():
            self._use_conv3 = True
        self._use_pc2 = False
        if 'pc2' in self._architecture.keys() and self._architecture['pc2']['out_size'] > 0:
            self._use_pc2 = True

        # get in and out sizes of fully-connected layer for possible re-initialization
        self.pc2_out_size = 0
        if self._use_pc2:
            self.pc2_out_size = self._architecture['pc2']['out_size']
        self.pc1_in_size = self._pose_dim
        self.pc1_out_size = self._architecture['pc1']['out_size']
        self.fc3_in_size = self._architecture['pc1']['out_size']
        if self._use_pc2:        
            self.fc3_in_size = self._architecture['pc2']['out_size']
        self.fc3_out_size = self._architecture['fc3']['out_size']
        self.fc4_in_size = self._architecture['fc3']['out_size']
        self.fc4_out_size = self._architecture['fc4']['out_size'] 
        self.fc5_in_size = self._architecture['fc4']['out_size']
        self.fc5_out_size = self._architecture['fc5']['out_size']

        if self.pc2_out_size == 0:
            self.fc4_pose_in_size = self.pc1_out_size
        else:
            self.fc4_pose_in_size = self.pc2_out_size

        # load normalization constants
        self.normalization_radius = config['radius']
        self.normalization_alpha = config['alpha']
        self.normalization_beta = config['beta']
        self.normalization_bias = config['bias']

        # initialize means and standard deviation to be 0 and 1, respectively
        self._im_mean = 0
        self._im_std = 1
        self._pose_mean = np.zeros(self._pose_dim)
        self._pose_std = np.ones(self._pose_dim)

    def initialize_network(self, add_softmax=True, add_sigmoid=False):
        """ Set up input nodes and builds network.

        Parameters
        ----------
        add_softmax : float
            whether or not to add a softmax layer
        """

        with self._graph.as_default():
            # setup tf input placeholders and build network
            logging.info("batch size during initialize_network is %d" % (self._batch_size))
            self._input_im_node = tf.placeholder(
                tf.float32, (self._batch_size, self._im_height, self._im_width, self._num_channels))
            self._input_pose_node = tf.placeholder(
                tf.float32, (self._batch_size, self._pose_dim))

            # build network
            self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node)
            if add_softmax:
                self.add_softmax_to_predict()
            if add_sigmoid:
                self.add_sigmoid_to_predict()

    def open_session(self):
        """ Open tensorflow session """
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            # create custom config that tells tensorflow to allocate GPU memory 
            # as needed so it is possible to run multiple tf sessions on the same GPU
            self.tf_config = tf.ConfigProto()
            self.tf_config.gpu_options.allow_growth = True
            self._sess = tf.Session(config = self.tf_config)
            self._sess.run(init)
        return self._sess

    def close_session(self):
        """ Close tensorflow session """
        with self._graph.as_default():
            self._sess.close()
            self._sess = None

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def im_height(self):
        return self._im_height

    @property
    def im_width(self):
        return self._im_width

    @property
    def im_mean(self):
        return self._im_mean

    @property
    def im_std(self):
        return self._im_std

    @property
    def pose_mean(self):
        return self._pose_mean

    @property
    def pose_std(self):
        return self._pose_std

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def pose_dim(self):
        return self._pose_dim

    @property
    def input_data_mode(self):
        return self._input_data_mode

    @property
    def input_im_node(self):
        return self._input_im_node

    @property
    def input_pose_node(self):
        return self._input_pose_node

    @property
    def output(self):
        return self._output_tensor

    @property
    def weights(self):
        return self._weights

    @property
    def graph(self):
        return self._graph

    def update_im_mean(self, im_mean):
        """ Updates image mean to be used for normalization when predicting 
        
        Parameters
        ----------
        im_mean : float
            image mean to be used
        """
        self._im_mean = im_mean
    
    def get_im_mean(self):
        """ Get the current image mean to be used for normalization when predicting

        Returns
        -------
        : float
            image mean
        """
        return self.im_mean

    def update_im_std(self, im_std):
        """ Updates image standard deviation to be used for normalization when predicting 
        
        Parameters
        ----------
        im_std : float
            image standard deviation to be used
        """
        self._im_std = im_std

    def get_im_std(self):
        """ Get the current image standard deviation to be used for normalization when predicting

        Returns
        -------
        : float
            image standard deviation
        """
        return self.im_std

    def update_pose_mean(self, pose_mean):
        """ Updates pose mean to be used for normalization when predicting 
        
        Parameters
        ----------
        pose_mean :obj:`tensorflow Tensor`
            pose mean to be used
        """
        self._pose_mean = pose_mean

    def get_pose_mean(self):
        """ Get the current pose mean to be used for normalization when predicting

        Returns
        -------
        :obj:`tensorflow Tensor`
            pose mean
        """
        return self._pose_mean

    def update_pose_std(self, pose_std):
        """ Updates pose standard deviation to be used for normalization when predicting 
        
        Parameters
        ----------
        pose_std :obj:`tensorflow Tensor`
            pose standard deviation to be used
        """
        self._pose_std = pose_std

    def get_pose_std(self):
        """ Get the current pose standard deviation to be used for normalization when predicting

        Returns
        -------
        :obj:`tensorflow Tensor`
            pose standard deviation
        """
        return self._pose_std
        
    def add_softmax_to_predict(self):
        """ Adds softmax to output tensor of prediction network """
        self._output_tensor = tf.nn.softmax(self._output_tensor)

    def add_sigmoid_to_predict(self):
        """ Adds sigmoid to output tensor of prediction network """
        self._output_tensor = tf.nn.sigmoid(self._output_tensor)

    def update_batch_size(self, batch_size):
        """ Updates the prediction batch size 

        Parameters
        ----------
        batch_size : float
            batch size to be used for prediction
        """
        self._batch_size = batch_size

    def predict(self, image_arr, pose_arr):
        """ Predict a set of images in batches 

        Parameters
        ----------
        image_arr : :obj:`tensorflow Tensor`
            4D Tensor of images to be predicted
        pose_arr : :obj:`tensorflow Tensor`
            4D Tensor of poses to be predicted
        """

        # setup prediction
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]
        output_arr = np.zeros([num_images, self.fc5_out_size])
        if num_images != num_poses:
            raise ValueError('Must provide same number of images and poses')

        # resize if necessary
        if image_arr.shape[1] != self._im_height or image_arr.shape[2] != self._im_width:
            rescale_factor = float(self._im_height) / image_arr.shape[1]
            new_image_arr = np.zeros([num_images, self._im_height,
                                      self._im_width, image_arr.shape[3]])
            for i in range(num_images):
                for c in range(image_arr.shape[3]):
                    new_image_arr[i,:,:,c] = sm.imresize(image_arr[i,:,:,c],
                                                         rescale_factor,
                                                         interp='bicubic', mode='F')
            image_arr = new_image_arr

        # predict by filling in image array in batches
        close_sess = False
        with self._graph.as_default():
            if self._sess is None:
                close_sess = True
                self.open_session()
            i = 0
            while i < num_images:
                logging.debug('Predicting file %d' % (i))
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim
                self._input_im_arr[:dim, :, :, :] = (
                    image_arr[cur_ind:end_ind, :, :, :] - self._im_mean) / self._im_std
                self._input_pose_arr[:dim, :] = (
                    pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std
                gqcnn_output = self._sess.run(self._output_tensor,
                                              feed_dict={self._input_im_node: self._input_im_arr,
                                                         self._input_pose_node: self._input_pose_arr})
                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]

                i = end_ind
            if close_sess:
                self.close_session()
        return output_arr

    def featurize(self, image_arr, pose_arr, feature_layer='conv2_2'):
        """ Predict a set of images in batches 

        Parameters
        ----------
        image_arr : :obj:`tensorflow Tensor`
            4D Tensor of images to be predicted
        pose_arr : :obj:`tensorflow Tensor`
            4D Tensor of poses to be predicted
        feature_layer : str
            name of layer to use in featurization
        """
        # setup prediction
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]
        output_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images and poses')

        # predict by filling in image array in batches
        close_sess = False
        with self._graph.as_default():
            if self._sess is None:
                close_sess = True
                self.open_session()
            i = 0
            while i < num_images:
                logging.debug('Predicting file %d' % (i))
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim
                self._input_im_arr[:dim, :, :, :] = (
                    image_arr[cur_ind:end_ind, :, :, :] - self._im_mean) / self._im_std
                self._input_pose_arr[:dim, :] = (
                    pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

                gqcnn_output = self._sess.run(self.feature_tensors[feature_layer],
                                              feed_dict={self._input_im_node: self._input_im_arr,
                                                         self._input_pose_node: self._input_pose_arr})
                if output_arr is None:
                    output_arr = np.zeros([num_images,
                                           gqcnn_output.shape[1],
                                           gqcnn_output.shape[2],
                                           gqcnn_output.shape[3]])
                output_arr[cur_ind:end_ind, ...] = gqcnn_output[:dim, ...]

                i = end_ind
            if close_sess:
                self.close_session()
        return output_arr

    @property
    def filters(self):
        """ Returns the set of conv1_1 filters 

        Returns
        -------
        :obj:`tensorflow Tensor`
            filters(weights) from conv1_1 of the network
        """

        close_sess = False
        if self._sess is None:
            close_sess = True
            self.open_session()

        filters = self._sess.run(self._weights.conv1_1W)

        if close_sess:
            self.close_session()
        return filters

    def _build_network(self, input_im_node, input_pose_node,  drop_fc3=False, drop_fc4=False, fc3_drop_rate=0, fc4_drop_rate=0):
        """ Builds neural network 

        Parameters
        ----------
        input_im_node : :obj:`tensorflow Placeholder`
            network input image placeholder
        input_pose_node : :obj:`tensorflow Placeholder`
            network input pose placeholder
        drop_fc3 : bool
            boolean value whether to drop third fully-connected layer or not to reduce over_fitting
        drop_fc4 : bool
            boolean value whether to drop fourth fully-connected layer or not to reduce over_fitting
        fc3_drop_rate : float
            drop rate for third fully-connected layer
        fc4_drop_rate : float
            drop rate for fourth fully-connected layer

        Returns
        -------
        :obj:`tensorflow Tensor`
            output of network
        """
        # conv1_1
        conv1_1_padding = 'SAME'
        if 'padding' in self._architecture['conv1_1'].keys():
            conv1_1_padding = self._architecture['conv1_1']['padding']
        conv1_1h = tf.nn.relu(tf.nn.conv2d(input_im_node, self._weights.conv1_1W, strides=[
                                1, 1, 1, 1], padding=conv1_1_padding) + self._weights.conv1_1b)
        if self._architecture['conv1_1']['norm']:
            conv1_1h = tf.nn.local_response_normalization(conv1_1h,
                                                          depth_radius=self.normalization_radius,
                                                          alpha=self.normalization_alpha,
                                                          beta=self.normalization_beta,
                                                          bias=self.normalization_bias)
        pool1_1_size = self._architecture['conv1_1']['pool_size']
        pool1_1_stride = self._architecture['conv1_1']['pool_stride']
        pool1_1 = tf.nn.max_pool(conv1_1h,
                                ksize=[1, pool1_1_size, pool1_1_size, 1],
                                strides=[1, pool1_1_stride,
                                        pool1_1_stride, 1],
                                padding='SAME')
        conv1_1_num_nodes = reduce_shape(pool1_1.get_shape())
        conv1_1_flat = tf.reshape(pool1_1, [-1, conv1_1_num_nodes])

        # conv1_2
        conv1_2_padding = 'SAME'
        if 'padding' in self._architecture['conv1_2'].keys():
            conv1_2_padding = self._architecture['conv1_2']['padding']
        conv1_2h = tf.nn.relu(tf.nn.conv2d(pool1_1, self._weights.conv1_2W, strides=[
                                1, 1, 1, 1], padding=conv1_2_padding) + self._weights.conv1_2b)
        if self._architecture['conv1_2']['norm']:
            conv1_2h = tf.nn.local_response_normalization(conv1_2h,
                                                          depth_radius=self.normalization_radius,
                                                          alpha=self.normalization_alpha,
                                                          beta=self.normalization_beta,
                                                          bias=self.normalization_bias)
        pool1_2_size = self._architecture['conv1_2']['pool_size']
        pool1_2_stride = self._architecture['conv1_2']['pool_stride']
        pool1_2 = tf.nn.max_pool(conv1_2h,
                                ksize=[1, pool1_2_size, pool1_2_size, 1],
                                strides=[1, pool1_2_stride,
                                        pool1_2_stride, 1],
                                padding='SAME')
        conv1_2_num_nodes = reduce_shape(pool1_2.get_shape())
        conv1_2_flat = tf.reshape(pool1_2, [-1, conv1_2_num_nodes])

        if self._use_conv2:
            # conv2_1
            conv2_1_padding = 'SAME'
            if 'padding' in self._architecture['conv2_1'].keys():
                conv2_1_padding = self._architecture['conv2_1']['padding']
            conv2_1h = tf.nn.relu(tf.nn.conv2d(pool1_2, self._weights.conv2_1W, strides=[
                1, 1, 1, 1], padding=conv2_1_padding) + self._weights.conv2_1b)
            if self._architecture['conv2_1']['norm']:
                conv2_1h = tf.nn.local_response_normalization(conv2_1h,
                                                              depth_radius=self.normalization_radius,
                                                              alpha=self.normalization_alpha,
                                                              beta=self.normalization_beta,
                                                              bias=self.normalization_bias)
            pool2_1_size = self._architecture['conv2_1']['pool_size']
            pool2_1_stride = self._architecture['conv2_1']['pool_stride']
            pool2_1 = tf.nn.max_pool(conv2_1h,
                                     ksize=[1, pool2_1_size, pool2_1_size, 1],
                                     strides=[1, pool2_1_stride,
                                              pool2_1_stride, 1],
                                     padding='SAME')
            conv2_1_num_nodes = reduce_shape(pool2_1.get_shape())
            conv2_1_flat = tf.reshape(pool2_1, [-1, conv2_1_num_nodes])

            # conv2_2
            conv2_2_padding = 'SAME'
            if 'padding' in self._architecture['conv2_2'].keys():
                conv2_2_padding = self._architecture['conv2_2']['padding']
            conv2_2h = tf.nn.relu(tf.nn.conv2d(pool2_1, self._weights.conv2_2W, strides=[
                1, 1, 1, 1], padding=conv2_2_padding) + self._weights.conv2_2b)
            if self._architecture['conv2_2']['norm']:
                conv2_2h = tf.nn.local_response_normalization(conv2_2h,
                                                              depth_radius=self.normalization_radius,
                                                              alpha=self.normalization_alpha,
                                                              beta=self.normalization_beta,
                                                              bias=self.normalization_bias)
            pool2_2_size = self._architecture['conv2_2']['pool_size']
            pool2_2_stride = self._architecture['conv2_2']['pool_stride']
            pool2_2 = tf.nn.max_pool(conv2_2h,
                                     ksize=[1, pool2_2_size, pool2_2_size, 1],
                                     strides=[1, pool2_2_stride,
                                              pool2_2_stride, 1],
                                    padding='SAME')
            conv2_2_num_nodes = reduce_shape(pool2_2.get_shape())
            conv2_2_flat = tf.reshape(pool2_2, [-1, conv2_2_num_nodes])
        else:
            pool2_2 = pool1_2

        if self._use_conv3:
            # conv3_1
            conv3_1_padding = 'SAME'
            if 'padding' in self._architecture['conv3_1'].keys():
                conv3_1_padding = self._architecture['conv3_1']['padding']
            conv3_1h = tf.nn.relu(tf.nn.conv2d(pool2_2, self._weights.conv3_1W, strides=[
                1, 1, 1, 1], padding=conv3_1_padding) + self._weights.conv3_1b)
            if self._architecture['conv3_1']['norm']:
                conv3_1h = tf.nn.local_response_normalization(conv3_1h,
                                                              depth_radius=self.normalization_radius,
                                                              alpha=self.normalization_alpha,
                                                              beta=self.normalization_beta,
                                                            bias=self.normalization_bias)
            pool3_1_size = self._architecture['conv3_1']['pool_size']
            pool3_1_stride = self._architecture['conv3_1']['pool_stride']
            pool3_1 = tf.nn.max_pool(conv3_1h,
                                     ksize=[1, pool3_1_size, pool3_1_size, 1],
                                     strides=[1, pool3_1_stride,
                                              pool3_1_stride, 1],
                                     padding='SAME')
            conv3_1_num_nodes = reduce_shape(pool3_1.get_shape())
            conv3_1_flat = tf.reshape(pool3_1, [-1, conv3_1_num_nodes])

            # conv3_2
            conv3_2_padding = 'SAME'
            if 'padding' in self._architecture['conv3_2'].keys():
                conv3_2_padding = self._architecture['conv3_2']['padding']
            conv3_2h = tf.nn.relu(tf.nn.conv2d(pool3_1, self._weights.conv3_2W, strides=[
                1, 1, 1, 1], padding=conv3_2_padding) + self._weights.conv3_2b)
            if self._architecture['conv3_2']['norm']:
                conv3_2h = tf.nn.local_response_normalization(conv3_2h,
                                                              depth_radius=self.normalization_radius,
                                                              alpha=self.normalization_alpha,
                                                              beta=self.normalization_beta,
                                                              bias=self.normalization_bias)
            pool3_2_size = self._architecture['conv3_2']['pool_size']
            pool3_2_stride = self._architecture['conv3_2']['pool_stride']
            pool3_2 = tf.nn.max_pool(conv3_2h,
                                     ksize=[1, pool3_2_size, pool3_2_size, 1],
                                     strides=[1, pool3_2_stride,
                                              pool3_2_stride, 1],
                                     padding='SAME')
            conv3_2_num_nodes = reduce_shape(pool3_2.get_shape())
            conv3_2_flat = tf.reshape(pool3_2, [-1, conv3_2_num_nodes])

        # fc3
        if self._use_conv3:
            fc3 = tf.nn.relu(tf.matmul(conv3_2_flat, self._weights.fc3W) +
                             self._weights.fc3b)
        elif self._use_conv2:
            fc3 = tf.nn.relu(tf.matmul(conv2_2_flat, self._weights.fc3W) +
                             self._weights.fc3b)
        else:
            fc3 = tf.nn.relu(tf.matmul(conv1_2_flat, self._weights.fc3W) +
                             self._weights.fc3b)            
            
        # drop fc3 if necessary
        if drop_fc3:
            fc3 = tf.nn.dropout(fc3, fc3_drop_rate)

        # pc1
        pc1 = tf.nn.relu(tf.matmul(input_pose_node, self._weights.pc1W) +
                         self._weights.pc1b)

        if self._use_pc2:
            # pc2
            pc2 = tf.nn.relu(tf.matmul(pc1, self._weights.pc2W) +
                             self._weights.pc2b)
            # fc4
            fc4 = tf.nn.relu(tf.matmul(fc3, self._weights.fc4W_im) +
                             tf.matmul(pc2, self._weights.fc4W_pose) +
                             self._weights.fc4b)
        else:
            # fc4
            fc4 = tf.nn.relu(tf.matmul(fc3, self._weights.fc4W_im) +
                             tf.matmul(pc1, self._weights.fc4W_pose) +
                             self._weights.fc4b)

        # drop fc4 if necessary
        if drop_fc4:
            fc4 = tf.nn.dropout(fc4, fc4_drop_rate)

        # fc5
        fc5 = tf.matmul(fc4, self._weights.fc5W) + self._weights.fc5b

        # setup feature tensors
        self._feature_tensors['conv1_1'] = pool1_1
        self._feature_tensors['conv1_2'] = pool1_2
        if self._use_conv2:
            self._feature_tensors['conv2_1'] = pool2_1
            self._feature_tensors['conv2_2'] = pool2_2
        if self._use_conv3:
            self._feature_tensors['conv3_1'] = pool3_1
            self._feature_tensors['conv3_2'] = pool3_2
        self._feature_tensors['fc3'] = fc3
        self._feature_tensors['fc4'] = fc4
        self._feature_tensors['pc1'] = pc1
        if self._use_pc2:
            self._feature_tensors['pc2'] = pc2
            
        return fc5
