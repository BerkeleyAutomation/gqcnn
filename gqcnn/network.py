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
GQCNN network implemented in Tensorflow
Author: Vishal Satish
"""

import json
from collections import OrderedDict
import logging
import os
import math
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tcf

from .utils import read_pose_data, pose_dim, GripperMode, TrainingMode

def reduce_shape(shape):
    """ Get shape of a layer for flattening """
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape, 1)

class GQCNNWeights(object):
    """ Struct helper for storing weights """
    def __init__(self):
        self.weights = {}

class GQCNN(object):
    """ GQCNN network implemented in Tensorflow """

    def __init__(self, gqcnn_config):
        """
        Parameters
        ----------
        gqcnn_config :obj: dict
            python dictionary of configuration parameters such as architecture and basic data params such as batch_size for prediction,
            im_height, im_width, ...
        """
        self._sess = None
        self._weights = GQCNNWeights()
        self._graph = tf.Graph()
        self._parse_config(gqcnn_config)

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
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)

        gqcnn_config = train_config['gqcnn_config']
        
        # create GQCNN object and initialize weights and network
        gqcnn = GQCNN(gqcnn_config)
        gqcnn.init_weights_file(os.path.join(model_dir, 'model.ckpt'))
        gqcnn.init_mean_and_std(model_dir)
        training_mode = train_config['training_mode']
        if training_mode == TrainingMode.CLASSIFICATION:
            gqcnn.initialize_network(add_softmax=True)
        elif training_mode == TrainingMode.REGRESSION:
            gqcnn.initialize_network()
        else:
            raise ValueError('Invalid training mode: {}'.format(training_mode))
        return gqcnn

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
        return self._weights.weights

    def init_mean_and_std(self, model_dir):
        """ Initializes the mean and std to use for data normalization during prediction 

        Parameters
        ----------
        model_dir :obj: str
            path to model directory where means and standard deviations are stored
        """
        # load in means and stds 
        # pose format is: grasp center row, grasp center col, gripper depth, grasp theta, crop center row, crop center col, grip width
        self._im_mean = np.load(os.path.join(model_dir, 'im_mean.npy'))
        self._im_std = np.load(os.path.join(model_dir, 'im_std.npy'))
        self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))

        # read the certain parts of the pose mean/std that we desire
        self._pose_mean = read_pose_data(self._pose_mean, self._input_pose_mode)
        self._pose_std = read_pose_data(self._pose_std, self._input_pose_mode)
      
    def init_weights_file(self, ckpt_file):
        """ Initialize network weights from the specified model 

        Parameters
        ----------
        ckpt_file :obj: str
            Tensorflow checkpoint file from which to load model weights
        """
        with self._graph.as_default():
            # create new tf checkpoint reader
            reader = tf.train.NewCheckpointReader(ckpt_file)
        
            # create empty weight object
            self._weights = GQCNNWeights()

            # read/generate weight/bias variable names
            ckpt_vars = tcf.list_variables(ckpt_file)
            full_var_names = []
            short_names = []
            for variable, shape in ckpt_vars:
                full_var_names.append(variable)
                short_names.append(variable.split('/')[-1])
    
            # load variables
            for full_var_name, short_name in zip(full_var_names, short_names):
                self._weights.weights[short_name] = tf.Variable(reader.get_tensor(full_var_name))

    def _parse_config(self, gqcnn_config):
        """ Parses configuration file for this GQCNN 

        Parameters
        ----------
        gqcnn_config : dict
            python dictionary of configuration parameters such as architecure and basic data params such as batch_size for prediction,
            im_height, im_width, ... 
        """

        ##################### PARSING GQCNN CONFIG #####################
        # load tensor params
        self._batch_size = gqcnn_config['batch_size']
        self._train_im_height = gqcnn_config['im_height']
        self._train_im_width = gqcnn_config['im_width']
        self._im_height = self._train_im_height
        self._im_width = self._train_im_width
        self._num_channels = gqcnn_config['im_channels']
        try:
            self._gripper_mode = gqcnn_config['gripper_mode']
        except:
            logging.warning('Could not read gripper mode. Attempting legacy conversion')
            self._input_data_mode = gqcnn_config['input_data_mode']
            if self._input_data_mode == 'tf_image':
                self._gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
            elif self._input_data_mode == 'tf_image_suction':
                self._gripper_mode = GripperMode.LEGACY_SUCTION                
            
        # setup correct pose dimensions
        self._pose_dim = pose_dim(self._gripper_mode)

        # load architecture
        self._architecture = gqcnn_config['architecture']
        
        # load normalization constants
        self._normalization_radius = gqcnn_config['radius']
        self._normalization_alpha = gqcnn_config['alpha']
        self._normalization_beta = gqcnn_config['beta']
        self._normalization_bias = gqcnn_config['bias']

        # get ReLU coefficient
        self._relu_coeff = gqcnn_config['relu_coeff']

        # debugging
        self._debug = gqcnn_config['debug']
        self._rand_seed = gqcnn_config['seed']

        # initialize means and standard deviation to be 0 and 1, respectively
        self._im_mean = 0
        self._im_std = 1
        self._pose_mean = np.zeros(self._pose_dim)
        self._pose_std = np.ones(self._pose_dim)

        # create empty holder for feature handles
        self._feature_tensors = {}
    
    def initialize_network(self, train_im_node=None, train_pose_node=None, add_softmax=False, add_sigmoid=False):
        """ Set up input placeholders and build network.

        Parameters
        ----------
        train_im_node :obj:`tf.placeholder`
            images for training
        train_pose_node :obj:`tf.placeholder`
            poses for training
        add_softmax : bool
            whether or not to add a softmax layer to output of network
        """
        with self._graph.as_default():
            # set tf random seed if debugging
            if self._debug:
                tf.set_random_seed(self._rand_seed)

            # setup input placeholders
            if train_im_node is not None:
                # training
                self._input_im_node = tf.placeholder_with_default(train_im_node, (None, self._im_height, self._im_width, self._num_channels))
                self._input_pose_node = tf.placeholder_with_default(train_pose_node, (None, self._pose_dim))
            else:
                # inference using model instantiated from GQCNN.load()
                self._input_im_node = tf.placeholder(tf.float32, (self._batch_size, self._im_height, self._im_width, self._num_channels))
                self._input_pose_node = tf.placeholder(tf.float32, (self._batch_size, self._pose_dim))
            self._input_drop_rate_node = tf.placeholder_with_default(tf.constant(0.0), ())

            # build network
            self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, self._input_drop_rate_node)
            
            # add softmax function to output of network if specified(for regression)
            if add_softmax:
                self.add_softmax_to_output()
            # add sigmoid function to output of network if specified(for weighted cross-entropy loss)
            if add_sigmoid:
                self.add_sigmoid_to_output()

        # create feed tensors for prediction
        self._input_im_arr = np.zeros((self._batch_size, self._im_height, self._im_width, self._num_channels))
        self._input_pose_arr = np.zeros((self._batch_size, self._pose_dim))

    def open_session(self):
        """ Open tensorflow session """
        logging.info('Initializing TF Session...')
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            self.tf_config = tf.ConfigProto()
            # allow tf gpu_growth so tf does not lock-up all GPU memory
            self.tf_config.gpu_options.allow_growth = True
            self._sess = tf.Session(graph=self._graph, config=self.tf_config)
            self._sess.run(init)
            
        return self._sess

    def close_session(self):
        """ Close tensorflow session """
        logging.info('Closing TF Session...')
        with self._graph.as_default():
            self._sess.close()
            self._sess = None

    @property
    def batch_size(self):
        return self._batch_size

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def im_height(self):
        return self._im_height

    @property
    def im_width(self):
        return self._im_width

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def pose_dim(self):
        return self._pose_dim

    @property
    def input_pose_mode(self):
        return self._input_pose_mode

    @property
    def gripper_mode(self):
        return self._gripper_mode

    @property
    def input_im_node(self):
        return self._input_im_node

    @property
    def input_pose_node(self):
        return self._input_pose_node
    
    @property
    def input_drop_rate_node(self):
        return self._input_drop_rate_node

    @property
    def output(self):
        return self._output_tensor

    @property
    def weights(self):
        return self._weights

    @property
    def tf_graph(self):
        return self._graph
    
    @property
    def sess(self):
        return self._sess

    def set_im_mean(self, im_mean):
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

    def set_im_std(self, im_std):
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

    def set_pose_mean(self, pose_mean):
        """ Updates pose mean to be used for normalization when predicting 
        
        Parameters
        ----------
        pose_mean :obj:`numpy ndarray`
            pose mean to be used
        """
        self._pose_mean = pose_mean

    def get_pose_mean(self):
        """ Get the current pose mean to be used for normalization when predicting

        Returns
        -------
        :obj:`numpy ndarray`
            pose mean
        """
        return self._pose_mean

    def set_pose_std(self, pose_std):
        """ Updates pose standard deviation to be used for normalization when predicting 
        
        Parameters
        ----------
        pose_std :obj:`numpy ndarray`
            pose standard deviation to be used
        """
        self._pose_std = pose_std

    def get_pose_std(self):
        """ Get the current pose standard deviation to be used for normalization when predicting

        Returns
        -------
        :obj:`numpy ndarray`
            pose standard deviation
        """
        return self._pose_std

    def add_softmax_to_output(self):
        """ Adds softmax to output of network """
        with tf.name_scope('softmax'):
            logging.info('Building Softmax Layer...')
            self._output_tensor = tf.nn.softmax(self._output_tensor)

    def add_sigmoid_to_output(self):
        """ Adds sigmoid to output of network """
        with tf.name_scope('sigmoid'):
            logging.info('Building Sigmoid Layer...')
            self._output_tensor = tf.nn.sigmoid(self._output_tensor)

    def update_batch_size(self, batch_size):
        """ Updates the prediction batch size 

        Parameters
        ----------
        batch_size : float
            batch size to be used for prediction
        """
        self._batch_size = batch_size

    def _predict(self, image_arr, pose_arr, verbose=False):       
        # get prediction start time
        start_time = time.time()

        if verbose:
            logging.info('Predicting...')

        # setup for prediction
        num_batches = math.ceil(image_arr.shape[0] / self._batch_size)
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]

        output_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images and poses')

        # predict by filling in image array in batches
        with self._graph.as_default():
            if self._sess is None:
               raise RuntimeError('No TF session open. Please call open_session() first.')
            i = 0
            batch_idx = 0
            while i < num_images:
                if verbose:
                    logging.info('Predicting batch {} of {}'.format(batch_idx, num_batches))
                batch_idx += 1
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim
                
                self._input_im_arr[:dim, ...] = (
                        image_arr[cur_ind:end_ind, ...] - self._im_mean) / self._im_std 
                
                self._input_pose_arr[:dim, :] = (
                        pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

                gqcnn_output = self._sess.run(self._output_tensor,
                                                      feed_dict={self._input_im_node: self._input_im_arr,
                                                                 self._input_pose_node: self._input_pose_arr})

                # allocate output tensor if needed
                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))

                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]
                i = end_ind
        
        # get total prediction time
        pred_time = time.time() - start_time

        return output_arr

    def predict(self, image_arr, pose_arr, verbose=False):
        """ 
        Predict the probability of grasp success given a depth image and gripper pose

        Parameters
        ----------
        image_arr : :obj:`numpy ndarray`
            4D Tensor of depth images
        pose_arr : :obj:`numpy ndarray`
            Tensor of gripper poses
        """
        return self._predict(image_arr, pose_arr, verbose=verbose)
   
    def featurize(self, image_arr, pose_arr=None, feature_layer='conv1_1'):
        """ Featurize a set of images in batches """

        if feature_layer not in self._feature_tensors.keys():
            raise ValueError('Feature layer %s not recognized' %(feature_layer))
        
        # setup for prediction
        num_images = image_arr.shape[0]
        if pose_arr is not None:
            num_poses = pose_arr.shape[0]
            if num_images != num_poses:
                raise ValueError('Must provide same number of images and poses')
        output_arr = None

        # predict in batches
        with self._graph.as_default():
            if self._sess is None:
               raise RuntimeError('No TF session open. Please call open_session() first.')

            i = 0
            while i < num_images:
                logging.debug('Predicting file %d' % (i))
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim
                self._input_im_arr[:dim, :, :, :] = (
                    image_arr[cur_ind:end_ind, :, :, :] - self._im_mean) / self._im_std
                if pose_arr is not None:
                    self._input_pose_arr[:dim, :] = (
                        pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

                if pose_arr is not None:
                    gqcnn_output = self._sess.run(self._feature_tensors[feature_layer],
                                              feed_dict={self._input_im_node: self._input_im_arr,
                                                         self._input_pose_node: self._input_pose_arr})
                else:
                    gqcnn_output = self._sess.run(self._feature_tensors[feature_layer],
                                              feed_dict={self._input_im_node: self._input_im_arr})

                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))
                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]

                i = end_ind

        # truncate extraneous values off of end of output_arr
        output_arr = output_arr[:num_images]
        return output_arr
    
    def _leaky_relu(self, x, alpha=.1):
        return tf.maximum(alpha * x, x)
    
    def _build_conv_layer(self, input_node, input_height, input_width, input_channels, filter_h, filter_w, num_filt, pool_stride_h, pool_stride_w, pool_size, name, norm=False, pad='SAME'):
        logging.info('Building convolutional layer: {}...'.format(name))       
        with tf.name_scope(name):
            # initialize weights
            if '{}_weights'.format(name) in self._weights.weights.keys():
                convW = self._weights.weights['{}_weights'.format(name)]
                convb = self._weights.weights['{}_bias'.format(name)] 
            else:
                convW_shape = [filter_h, filter_w, input_channels, num_filt]

                fan_in = filter_h * filter_w * input_channels
                std = np.sqrt(2.0 / (fan_in))
                convW = tf.Variable(tf.truncated_normal(convW_shape, stddev=std), name='{}_weights'.format(name))
                convb = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_bias'.format(name))

                self._weights.weights['{}_weights'.format(name)] = convW
                self._weights.weights['{}_bias'.format(name)] = convb
            
            if pad == 'SAME':
                out_height = input_height / pool_stride_h
                out_width = input_width / pool_stride_w
            else:
                out_height = (((input_height - filter_h) / 1) + 1) / pool_stride_h
                out_width = (((input_width - filter_w) / 1) +1) / pool_stride_w
            out_channels = num_filt

            # build layer
            convh = tf.nn.conv2d(input_node, convW, strides=[
                                1, 1, 1, 1], padding=pad) + convb           
            convh = self._leaky_relu(convh, alpha=self._relu_coeff)
            
            if norm:
                convh = tf.nn.local_response_normalization(convh,
                                                            depth_radius=self._normalization_radius,
                                                            alpha=self._normalization_alpha,
                                                            beta=self._normalization_beta,
                                                            bias=self._normalization_bias)
            pool = tf.nn.max_pool(convh,
                                ksize=[1, pool_size, pool_size, 1],
                                strides=[1, pool_stride_h,
                                        pool_stride_w, 1],
                                padding='SAME')
            
            # add output to feature dict
            self._feature_tensors[name] = pool

            return pool, out_height, out_width, out_channels

    def _build_fc_layer(self, input_node, fan_in, out_size, name, input_is_multi, drop_rate, final_fc_layer=False):
        logging.info('Building fully connected layer: {}...'.format(name))
        
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            fcW = self._weights.weights['{}_weights'.format(name)]
            fcb = self._weights.weights['{}_bias'.format(name)] 
        else:
            std = np.sqrt(2.0 / (fan_in))
            fcW = tf.Variable(tf.truncated_normal([fan_in, out_size], stddev=std), name='{}_weights'.format(name))
            if final_fc_layer:
                fcb = tf.Variable(tf.constant(0.0, shape=[out_size]), name='{}_bias'.format(name))
            else:
                fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))

            self._weights.weights['{}_weights'.format(name)] = fcW
            self._weights.weights['{}_bias'.format(name)] = fcb

        # build layer
        if input_is_multi:
            reduced_dim1 = reduce_shape(input_node.get_shape())
            input_node = tf.reshape(input_node, [-1, reduced_dim1])
        if final_fc_layer:
            fc = tf.matmul(input_node, fcW) + fcb
        else:
            fc = self._leaky_relu(tf.matmul(input_node, fcW) + fcb, alpha=self._relu_coeff)

        fc = tf.nn.dropout(fc, 1 - drop_rate)

        # add output to feature dict
        self._feature_tensors[name] = fc

        return fc, out_size

    def _build_pc_layer(self, input_node, fan_in, out_size, name):
        logging.info('Building Fully-Connected Pose Layer: {}...'.format(name))
        
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            pcW = self._weights.weights['{}_weights'.format(name)]
            pcb = self._weights.weights['{}_bias'.format(name)] 
        else:
            std = np.sqrt(2.0 / (fan_in))
            pcW = tf.Variable(tf.truncated_normal([fan_in, out_size],
                                               stddev=std), name='{}_weights'.format(name))
            pcb = tf.Variable(tf.truncated_normal([out_size],
                                               stddev=std), name='{}_bias'.format(name))

            self._weights.weights['{}_weights'.format(name)] = pcW
            self._weights.weights['{}_bias'.format(name)] = pcb

        # build layer
        pc = self._leaky_relu(tf.matmul(input_node, pcW) +
                        pcb, alpha=self._relu_coeff)

        # add output to feature dict
        self._feature_tensors[name] = pc

        return pc, out_size

    def _build_fc_merge(self, input_fc_node_1, input_fc_node_2, fan_in_1, fan_in_2, out_size, drop_rate, name):
        logging.info('Building Merge Layer: {}...'.format(name))
        
        # initialize weights
        if '{}_input_1_weights'.format(name) in self._weights.weights.keys():
            input1W = self._weights.weights['{}_input_1_weights'.format(name)]
            input2W = self._weights.weights['{}_input_2_weights'.format(name)]
            fcb = self._weights.weights['{}_bias'.format(name)] 
        else:
            std = np.sqrt(2.0 / (fan_in_1 + fan_in_2))
            input1W = tf.Variable(tf.truncated_normal([fan_in_1, out_size], stddev=std), name='{}_input_1_weights'.format(name))
            input2W = tf.Variable(tf.truncated_normal([fan_in_2, out_size], stddev=std), name='{}_input_2_weights'.format(name))
            fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))

            self._weights.weights['{}_input_1_weights'.format(name)] = input1W
            self._weights.weights['{}_input_2_weights'.format(name)] = input2W
            self._weights.weights['{}_bias'.format(name)] = fcb

        # build layer
        fc = self._leaky_relu(tf.matmul(input_fc_node_1, input1W) +
                              tf.matmul(input_fc_node_2, input2W) +
                              fcb, alpha=self._relu_coeff)
        fc = tf.nn.dropout(fc, 1 - drop_rate)

        # add output to feature dict
        self._feature_tensors[name] = fc

        return fc, out_size


    def _build_im_stream(self, input_node, input_height, input_width, input_channels, drop_rate, layers):
        logging.info('Building Image Stream...')

        output_node = input_node
        prev_layer = "start"
        filter_dim = self._train_im_width
        for layer_name, layer_config in layers.iteritems():
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
                prev_layer_is_conv_or_res = False
                if prev_layer == 'conv':
                    prev_layer_is_conv = True
                    fan_in = input_height * input_width * input_channels
                output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv, drop_rate)
                prev_layer = layer_type
                filter_dim = 1
            elif layer_type == 'pc':
                raise ValueError('Cannot have pose-connected layer in image stream!')
            elif layer_type == 'fc_merge':
                raise ValueError('Cannot have merge layer in image stream!')
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, fan_in

    def _build_pose_stream(self, input_node, fan_in, layers):
        logging.info('Building Pose Stream...')
        output_node = input_node
        prev_layer = "start"
        for layer_name, layer_config in layers.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'conv':
               raise ValueError('Cannot have conv layer in pose stream')
            elif layer_type == 'fc':
                raise ValueError('Cannot have fc layer in pose stream')
            elif layer_type == 'pc':
                output_node, fan_in = self._build_pc_layer(output_node, fan_in, layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in pose stream")
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return output_node, fan_in

    def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, drop_rate, layers):
        logging.info('Building Merge Stream...')
        
        # first check if first layer is a merge layer
        if layers[layers.keys()[0]]['type'] != 'fc_merge':
            raise ValueError('First layer in merge stream must be a fc_merge layer!')
            
        prev_layer = "start"
        last_index = len(layers.keys()) - 1
        filter_dim = 1
        fan_in = -1
        for layer_index, (layer_name, layer_config) in enumerate(layers.iteritems()):
            layer_type = layer_config['type']
            if layer_type == 'conv':
               raise ValueError('Cannot have conv layer in merge stream!')
            elif layer_type == 'fc':
                if layer_index == last_index:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate, final_fc_layer=True)
                else:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate)
                prev_layer = layer_type
            elif layer_type == 'pc':  
                raise ValueError('Cannot have pose-connected layer in merge stream!')
            elif layer_type == 'fc_merge':
                output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], drop_rate, layer_name)
                prev_layer = layer_type   
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, fan_in

    def _build_network(self, input_im_node, input_pose_node, input_drop_rate_node):
        """ Builds network 

        Parameters
        ----------
        input_im_node : :obj:`tensorflow Placeholder`
            network input image placeholder
        input_pose_node : :obj:`tensorflow Placeholder`
            network input pose placeholder
        input_drop_rate_node: :obj:`tensorflow Placeholder`
            drop rate

        Returns
        -------
        :obj:`tensorflow Tensor`
            output of network
        """
        logging.info('Building Network...')
        with tf.name_scope('im_stream'):
            output_im_stream, fan_out_im = self._build_im_stream(input_im_node, self._im_height, self._im_width, self._num_channels, input_drop_rate_node, self._architecture['im_stream'])
        with tf.name_scope('pose_stream'):
            output_pose_stream, fan_out_pose = self._build_pose_stream(input_pose_node, self._pose_dim, self._architecture['pose_stream'])
        with tf.name_scope('merge_stream'):
            return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, input_drop_rate_node, self._architecture['merge_stream'])[0]
