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
FC-GQ-CNN network implemented in Tensorflow
Author: Vishal Satish
"""
import os
import json
from collections import OrderedDict
import sys

import tensorflow as tf

from network_tf import GQCNNTF
from gqcnn.utils import TrainingMode, InputDepthMode

class FCGQCNNTF(GQCNNTF):
    """FC-GQ-CNN network implemented in Tensorflow. Note that FC-GQ-CNNs are never directly trained, but instead a pre-trained GQ-CNN is converted to an FC-GQ-CNN at inference time."""

    def __init__(self, gqcnn_config, fc_config, verbose=True, log_file=None):
        """
        Parameters
        ----------
        gqcnn_config : dict
            python dictionary of pre-trained GQ-CNN model configuration parameters
        fc_config : dict
            python dictionary of FC-GQ-CNN model configuration parameters
        verbose : bool
            whether or not to log model output to stdout
        log_file : str
            if provided, model output will also be logged to this file
        """

        super(FCGQCNNTF, self).__init__(gqcnn_config, log_file=log_file)
        super(FCGQCNNTF, self)._parse_config(gqcnn_config) 
        self._parse_config(fc_config) # we call this again(even though it gets called in the parent constructor on line 42) because the call to the parent _parse_config() on line 43 overwrites our first call

        # check that conv layers of GQ-CNN were trained with VALID padding
        for layer_name, layer_config in self._architecture['im_stream'].iteritems():
            if layer_config['type'] == 'conv':
                assert layer_config['pad'] == 'VALID', 'GQ-CNN used for FC-GQ-CNN must have VALID padding for conv layers. Found layer: {} with padding: {}'.format(layer_name, layer_config['pad'])

    @staticmethod
    def load(model_dir, fc_config, log_file=None):
        """Load an FC-GQ-CNN from a pre-trained GQ-CNN. 

        Parameters
        ----------
        model_dir : str
            path to pre-trained GQ-CNN model
        fc_config : dict
            python dictionary of FC-GQ-CNN model configuration parameters
        log_file : str
            if provided, model output will also be logged to this file

        Returns
        -------
        :obj:`FCGQCNNTF`
            initialized FC-GQ-CNN 
        """
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file) as data_file:    
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)

        gqcnn_config = train_config['gqcnn']
        
        # initialize weights and Tensorflow network
        fcgqcnn = FCGQCNNTF(gqcnn_config, fc_config, log_file=log_file)
        fcgqcnn.init_weights_file(os.path.join(model_dir, 'model.ckpt'))
        fcgqcnn.init_mean_and_std(model_dir)
        training_mode = train_config['training_mode']
        if training_mode == TrainingMode.CLASSIFICATION:
            fcgqcnn.initialize_network(add_softmax=True)
        elif training_mode == TrainingMode.REGRESSION:
            fcgqcnn.initialize_network()
        else:
            raise ValueError('Invalid training mode: {}'.format(training_mode))
        return fcgqcnn

    def _parse_config(self, cfg):
        # override GQ-CNN image height and width
        self._im_width = cfg['im_width']
        self._im_height = cfg['im_height']

    def _pack(self, dim_h, dim_w, data, vector=False):
        if vector:
            # first reshape vector into 3-dimensional tensor
            reshaped = tf.reshape(data, tf.concat([[1, 1], tf.shape(data)], 0))
         
            # then tile into tensor of shape dim x dim x data.dim0
            packed = tf.tile(reshaped, [dim_h, dim_w, 1])
        else:
            # first reshape second dimension of tensor into 3-dimensional tensor
            reshaped = tf.reshape(data, tf.concat([tf.shape(data)[0:1], [1, 1], tf.shape(data)[1:]], 0))

            # then tile into tensor of shape bsize x dim_h x dim_w x data.dim1
            packed = tf.tile(reshaped, [1, dim_h, dim_w, 1])
        return packed

    def _build_fully_conv_layer(self, input_node, filter_dim, fc_name, final_fc_layer=False):
        self._logger.info('Converting fc layer {} to fully convolutional...'.format(fc_name))
        
        # create new set of weights by reshaping fully connected layer weights
        fcW = self._weights.weights['{}_weights'.format(fc_name)]
        convW = tf.Variable(tf.reshape(fcW, tf.concat([[filter_dim, filter_dim], [tf.shape(fcW)[0] / (filter_dim * filter_dim)], tf.shape(fcW)[1:]], 0)), name='{}_fully_conv_weights'.format(fc_name))
        self._weights.weights['{}_fully_conv_weights'.format(fc_name)] = convW
        
        # get bias
        convb = self._weights.weights['{}_bias'.format(fc_name)]

        # compute conv out(note that we use padding='VALID' here because we want an output size of 1x1xnum_filts for the original input size)
        convh = tf.nn.conv2d(input_node, convW, strides=[1, 1, 1, 1], padding='VALID')

        # pack bias into tensor of shape=tf.shape(convh)
        bias_packed = self._pack(tf.shape(convh)[1], tf.shape(convh)[2], convb, vector=True)

        # add bias term
        convh = convh + bias_packed

        # apply activation
        if not final_fc_layer:
            convh = self._leaky_relu(convh, alpha=self._relu_coeff)

        # add output to feature_dict
        self._feature_tensors[fc_name] = convh

        return convh

    def _build_fully_conv_merge_layer(self, input_node_im, input_node_pose, filter_dim, fc_name):
        self._logger.info('Converting fc merge layer {} to fully convolutional...'.format(fc_name))

        # create new set of weights for image stream by reshaping fully-connected layer weights
        fcW_im = self._weights.weights['{}_input_1_weights'.format(fc_name)]
        convW = tf.Variable(tf.reshape(fcW_im, tf.concat([[filter_dim, filter_dim], [tf.shape(fcW_im)[0] / (filter_dim * filter_dim)], tf.shape(fcW_im)[1:]], 0)), name='{}_im_fully_conv_weights'.format(fc_name))
        self._weights.weights['{}_im_fully_conv_weights'.format(fc_name)] = convW

        # compute im stream conv out(note that we use padding='VALID' here because we want an output size of 1x1xnum_filts for the original input size)
        convh_im = tf.nn.conv2d(input_node_im, convW, strides=[1, 1, 1, 1], padding='VALID')

        # get pose stream fully-connected weights
        fcW_pose = self._weights.weights['{}_input_2_weights'.format(fc_name)]

        # compute matmul for pose stream           
        pose_out = tf.matmul(input_node_pose, fcW_pose)

        # pack pose_out into a tensor of shape=tf.shape(convh_im)
        pose_packed = self._pack(tf.shape(convh_im)[1], tf.shape(convh_im)[2], pose_out)

        # add the im and pose tensors 
        convh = convh_im + pose_packed

        # pack bias
        fc_bias = self._weights.weights['{}_bias'.format(fc_name)]
        bias_packed = self._pack(tf.shape(convh_im)[1], tf.shape(convh_im)[2], fc_bias, vector=True)

        # add bias and apply activation
        convh = self._leaky_relu(convh + bias_packed, alpha=self._relu_coeff)

        return convh

    def _build_im_stream(self, input_node, input_pose_node, input_height, input_width, input_channels, drop_rate, layers, only_stream=False):
        self._logger.info('Building Image Stream...')

        if self._input_depth_mode == InputDepthMode.SUB:
            sub_mean = tf.constant(self._im_depth_sub_mean, dtype=tf.float32)
            sub_std = tf.constant(self._im_depth_sub_std, dtype=tf.float32)
            sub_im = tf.subtract(input_node, tf.tile(tf.reshape(input_pose_node, tf.constant((-1, 1, 1, 1))), tf.constant((1, input_height, input_width, 1))))
            norm_sub_im = tf.div(tf.subtract(sub_im, sub_mean), sub_std)
            input_node = norm_sub_im

        output_node = input_node
        prev_layer = "start" # dummy placeholder
        filter_dim = self._train_im_width
        last_index = len(layers.keys()) - 1
        for layer_index, (layer_name, layer_config) in enumerate(layers.iteritems()):
            layer_type = layer_config['type']
            if layer_type == 'conv':
                if prev_layer == 'fc':
                    raise ValueError('Cannot have conv layer after fc layer!')
                output_node, input_height, input_width, input_channels = self._build_conv_layer(output_node, input_height, input_width, input_channels, layer_config['filt_dim'], layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, norm=layer_config['norm'], pad=layer_config['pad'])
                prev_layer = layer_type
                if layer_config['pad'] == 'SAME':
                    filter_dim /= layer_config['pool_stride']
                else:
                    filter_dim = ((filter_dim - layer_config['filt_dim']) / layer_config['pool_stride']) + 1
            elif layer_type == 'fc':
                if layer_index == last_index and only_stream:
                    output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name, final_fc_layer=True)
                else:
                    output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name)
                prev_layer = layer_type
                filter_dim = 1 # because fully-convolutional layers at this point in the network have a filter_dim of 1
            elif layer_type == 'pc':
                raise ValueError('Cannot have pose connected layer in image stream!')
            elif layer_type == 'fc_merge':
                raise ValueError('Cannot have merge layer in image stream!')
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, -1

    def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, drop_rate, layers):
        self._logger.info('Building Merge Stream...')
        
        # first check if first layer is a merge layer
        if layers[layers.keys()[0]]['type'] != 'fc_merge':
            raise ValueError('First layer in merge stream must be of type fc_merge!')
            
        prev_layer = "start" # dummy placeholder
        last_index = len(layers.keys()) - 1
        filter_dim = 1 # because fully-convolutional layers at this point in the network have a filter_dim of 1
        fan_in = -1
        for layer_index, (layer_name, layer_config) in enumerate(layers.iteritems()):
            layer_type = layer_config['type']
            if layer_type == 'conv':
               raise ValueError('Cannot have conv layer in merge stream!')
            elif layer_type == 'fc':
                if layer_index == last_index:
                    output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name, final_fc_layer=True)
                else:
                    output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name)
                prev_layer = layer_type
            elif layer_type == 'pc':  
                raise ValueError('Cannot have pose connected layer in merge stream!')
            elif layer_type == 'fc_merge':
                output_node = self._build_fully_conv_merge_layer(input_stream_1, input_stream_2, filter_dim, layer_name)
                prev_layer = layer_type   
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, fan_in
