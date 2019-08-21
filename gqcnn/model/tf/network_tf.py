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
GQ-CNN network implemented in Tensorflow.
Authors: Vishal Satish, Jeff Mahler
"""
import json
from collections import OrderedDict
import os
import math
import time
import operator
import sys
import logging

try:
    from functools import reduce
except Exception:
    pass

import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tcf

from gqcnn.utils import reduce_shape, read_pose_data, pose_dim, weight_name_to_layer_name, GeneralConstants, GripperMode, TrainingMode, InputDepthMode

class GQCNNWeights(object):
    """Helper struct for storing network weights."""
    def __init__(self):
        self.weights = {}

class GQCNNTF(object):
    """GQ-CNN network implemented in Tensorflow."""

    def __init__(self, gqcnn_config, verbose=True, log_file=None):
        """
        Parameters
        ----------
        gqcnn_config : dict
            python dictionary of network configuration parameters
        """
        self._sess = None
        self._graph = tf.Graph()

        # set up logger
        self._logger = logging.getLogger(self.__class__.__name__)
            
        self._weights = GQCNNWeights()
        self._parse_config(gqcnn_config)
        self._gqcnn_config = gqcnn_config

    @property
    def gqcnn_config(self):
        return self._gqcnn_config

    @property
    def train_config(self):
        return self._train_config

    @staticmethod
    def load(model_dir, verbose=True, log_file=None):
        """Instantiate a trained GQ-CNN for fine-tuning or inference. 

        Parameters
        ----------
        model_dir : str
            path to trained GQ-CNN

        Returns
        -------
        :obj:`GQCNNTF`
            initialized GQ-CNN 
        """
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file) as data_file:    
            train_config = json.load(data_file, object_pairs_hook=OrderedDict)

        # support for legacy configs
        try:
            gqcnn_config = train_config['gqcnn']
        except:
            gqcnn_config = train_config['gqcnn_config']            

            # convert old networks to new flexible arch format
            gqcnn_config['debug'] = 0
            gqcnn_config['seed'] = 0
            gqcnn_config['num_angular_bins'] = 0 # legacy networks had no angular support
            gqcnn_config['input_depth_mode'] = InputDepthMode.POSE_STREAM # legacy networks only supported depth integration through pose stream
            arch_config = gqcnn_config['architecture']
            if 'im_stream' not in arch_config.keys():
                new_arch_config = OrderedDict()
                new_arch_config['im_stream'] = OrderedDict()
                new_arch_config['pose_stream'] = OrderedDict()
                new_arch_config['merge_stream'] = OrderedDict()  

                layer_name = 'conv1_1'
                new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['im_stream'][layer_name]['type'] = 'conv'
                new_arch_config['im_stream'][layer_name]['pad'] = 'SAME'
                if 'padding' in arch_config[layer_name].keys():
                    new_arch_config['im_stream'][layer_name]['pad'] = arch_config[layer_name]['padding']

                layer_name = 'conv1_2'
                new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['im_stream'][layer_name]['type'] = 'conv' 
                new_arch_config['im_stream'][layer_name]['pad'] = 'SAME'
                if 'padding' in arch_config[layer_name].keys():
                    new_arch_config['im_stream'][layer_name]['pad'] = arch_config[layer_name]['padding']

                layer_name = 'conv2_1'
                new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['im_stream'][layer_name]['type'] = 'conv'
                new_arch_config['im_stream'][layer_name]['pad'] = 'SAME'
                if 'padding' in arch_config[layer_name].keys():
                    new_arch_config['im_stream'][layer_name]['pad'] = arch_config[layer_name]['padding']

                layer_name = 'conv2_2'
                new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['im_stream'][layer_name]['type'] = 'conv'
                new_arch_config['im_stream'][layer_name]['pad'] = 'SAME'
                if 'padding' in arch_config[layer_name].keys():
                    new_arch_config['im_stream'][layer_name]['pad'] = arch_config[layer_name]['padding']

                layer_name = 'conv3_1'
                if layer_name in arch_config.keys():
                    new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                    new_arch_config['im_stream'][layer_name]['type'] = 'conv'
                    new_arch_config['im_stream'][layer_name]['pad'] = 'SAME'
                    if 'padding' in arch_config[layer_name].keys():
                        new_arch_config['im_stream'][layer_name]['pad'] = arch_config[layer_name]['padding']

                layer_name = 'conv3_2'
                if layer_name in arch_config.keys():
                    new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                    new_arch_config['im_stream'][layer_name]['type'] = 'conv'
                    new_arch_config['im_stream'][layer_name]['pad'] = 'SAME'
                    if 'padding' in arch_config[layer_name].keys():
                        new_arch_config['im_stream'][layer_name]['pad'] = arch_config[layer_name]['padding']

                layer_name = 'fc3'
                new_arch_config['im_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['im_stream'][layer_name]['type'] = 'fc'            
                    
                layer_name = 'pc1'
                new_arch_config['pose_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['pose_stream'][layer_name]['type'] = 'pc'

                layer_name = 'pc2'
                if layer_name in arch_config.keys():
                    new_arch_config['pose_stream'][layer_name] = arch_config[layer_name]
                    new_arch_config['pose_stream'][layer_name]['type'] = 'pc'
                    
                layer_name = 'fc4'
                new_arch_config['merge_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['merge_stream'][layer_name]['type'] = 'fc_merge'            

                layer_name = 'fc5'
                new_arch_config['merge_stream'][layer_name] = arch_config[layer_name]
                new_arch_config['merge_stream'][layer_name]['type'] = 'fc'            

                gqcnn_config['architecture'] = new_arch_config
                
        # initialize weights and Tensorflow network
        gqcnn = GQCNNTF(gqcnn_config, verbose=verbose, log_file=log_file)
        gqcnn.init_weights_file(os.path.join(model_dir, 'model.ckpt'))
        gqcnn.init_mean_and_std(model_dir)
        gqcnn._train_config = train_config
        training_mode = train_config['training_mode']
        if training_mode == TrainingMode.CLASSIFICATION:
            gqcnn.initialize_network(add_softmax=True)
        elif training_mode == TrainingMode.REGRESSION:
            gqcnn.initialize_network()
        else:
            raise ValueError('Invalid training mode: {}'.format(training_mode))
        return gqcnn

    def init_mean_and_std(self, model_dir):
        """Loads the means and stds of a trained GQ-CNN to use for data normalization during inference. 

        Parameters
        ----------
        model_dir : str
            path to trained GQ-CNN directory where means and standard deviations are stored
        """
        # load in means and stds 
        if self._input_depth_mode == InputDepthMode.POSE_STREAM:
            try:
                self._im_mean = np.load(os.path.join(model_dir, 'im_mean.npy'))
                self._im_std = np.load(os.path.join(model_dir, 'im_std.npy'))
            except:
                # support for legacy file naming convention
                self._im_mean = np.load(os.path.join(model_dir, 'mean.npy'))
                self._im_std = np.load(os.path.join(model_dir, 'std.npy'))
            self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
            self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))

            # fix legacy #TODO: @Jeff, what needs to be fixed here? Or did I add this in?
            # read the certain parts of the pose mean/std that we desire
            if len(self._pose_mean.shape) > 0 and self._pose_mean.shape[0] != self._pose_dim:
                # handle multidim storage
                if len(self._pose_mean.shape) > 1 and self._pose_mean.shape[1] == self._pose_dim:
                    self._pose_mean = self._pose_mean[0,:]
                    self._pose_std = self._pose_std[0,:]
                else:
                    self._pose_mean = read_pose_data(self._pose_mean, self._gripper_mode)
                    self._pose_std = read_pose_data(self._pose_std, self._gripper_mode) 
        elif self._input_depth_mode == InputDepthMode.SUB:
            self._im_depth_sub_mean = np.load(os.path.join(model_dir, 'im_depth_sub_mean.npy')) 
            self._im_depth_sub_std = np.load(os.path.join(model_dir, 'im_depth_sub_std.npy'))
        elif self._input_depth_mode == InputDepthMode.IM_ONLY:
            self._im_mean = np.load(os.path.join(model_dir, 'im_mean.npy'))
            self._im_std = np.load(os.path.join(model_dir, 'im_std.npy'))
        else:
            raise ValueError('Unsupported input depth mode: {}'.format(self._input_depth_mode))
 
    def set_base_network(self, model_dir):
        """Initialize network weights for the base network. Used during fine-tuning.

        Parameters
        ----------
        model_dir : str
            path to GQ-CNN directory
        """
        # check architecture
        if 'base_model' not in self._architecture.keys():
            self._logger.warning('Architecuture has no base model. The network has not been modified')
            return False
        base_model_config = self._architecture['base_model']
        output_layer = base_model_config['output_layer']
        
        # read model
        ckpt_file = os.path.join(model_dir, 'model.ckpt')
        config_file = os.path.join(model_dir, 'architecture.json')
        base_arch = json.load(open(config_file, 'r'), object_pairs_hook=OrderedDict)

        # read base layer names
        self._base_layer_names = []
        found_base_layer = False
        use_legacy = not ('im_stream' in base_arch.keys())
        if use_legacy:
            layer_iter = iter(base_arch)
            while not found_base_layer:
                layer_name = layer_iter.next()
                self._base_layer_names.append(layer_name)
                if layer_name == output_layer:
                    found_base_layer = True
        else:
            stream_iter = iter(base_arch)
            while not found_base_layer:
                stream_name = stream_iter.next()
                stream_arch = base_arch[stream_name]
                layer_iter = iter(stream_arch)
                stop = False
                while not found_base_layer and not stop:
                    try:
                        layer_name = layer_iter.next()
                        self._base_layer_names.append(layer_name)
                        if layer_name == output_layer:
                            found_base_layer = True
                    except StopIteration:
                        stop = True
                            
        with self._graph.as_default():
            # create new tf checkpoint reader
            reader = tf.train.NewCheckpointReader(ckpt_file)
        
            # create empty weights object
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
                # check valid weights
                layer_name = weight_name_to_layer_name(short_name)

                # add weights
                if layer_name in self._base_layer_names:
                    self._weights.weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name)
                                                                    
                    
    def init_weights_file(self, ckpt_file):
        """Load trained GQ-CNN weights. 

        Parameters
        ----------
        ckpt_file : str
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
                self._weights.weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name)

    def _parse_config(self, gqcnn_config):
        """Parse configuration file.

        Parameters
        ----------
        gqcnn_config : dict
            python dictionary of configuration parameters
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
            # legacy support
            self._input_data_mode = gqcnn_config['input_data_mode']
            if self._input_data_mode == 'tf_image':
                self._gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
            elif self._input_data_mode == 'tf_image_suction':
                self._gripper_mode = GripperMode.LEGACY_SUCTION                
            elif self._input_data_mode == 'suction':
                self._gripper_mode = GripperMode.SUCTION
            elif self._input_data_mode == 'multi_suction':
                self._gripper_mode = GripperMode.MULTI_SUCTION                
            elif self._input_data_mode == 'parallel_jaw':
                self._gripper_mode = GripperMode.PARALLEL_JAW
            else:
                raise ValueError('Legacy input data mode: {} not supported!'.format(self._input_data_mode))
            self._logger.warning('Could not read gripper mode. Attempting legacy conversion to: {}'.format(self._gripper_mode))
            
        # setup gripper pose dimension depending on gripper mode
        self._pose_dim = pose_dim(self._gripper_mode)

        # load architecture
        self._architecture = gqcnn_config['architecture']

        # get input depth mode
        self._input_depth_mode = InputDepthMode.POSE_STREAM # legacy support
        if 'input_depth_mode' in gqcnn_config.keys():
            self._input_depth_mode = gqcnn_config['input_depth_mode']
        
        # load network local response normalization layer constants
        self._normalization_radius = gqcnn_config['radius']
        self._normalization_alpha = gqcnn_config['alpha']
        self._normalization_beta = gqcnn_config['beta']
        self._normalization_bias = gqcnn_config['bias']

        # get ReLU coefficient
        self._relu_coeff = 0.0 # legacy support
        if 'relu_coeff' in gqcnn_config.keys():
            self._relu_coeff = gqcnn_config['relu_coeff']

        # debugging
        self._debug = gqcnn_config['debug']
        self._rand_seed = gqcnn_config['seed']

        # initialize means and standard deviations to be 0 and 1, respectively
        if self._input_depth_mode == InputDepthMode.POSE_STREAM:
            self._im_mean = 0
            self._im_std = 1
            self._pose_mean = np.zeros(self._pose_dim)
            self._pose_std = np.ones(self._pose_dim)
        elif self._input_depth_mode == InputDepthMode.SUB:
            self._im_depth_sub_mean = 0
            self._im_depth_sub_std = 1
        elif self._input_depth_mode == InputDepthMode.IM_ONLY:
            self._im_mean = 0
            self._im_std = 1

        # get number of angular bins
        self._angular_bins = 0 # legacy support
        if 'angular_bins' in gqcnn_config.keys():
            self._angular_bins = gqcnn_config['angular_bins']

        # read multi-gripper indices if available
        self._gripper_types = None
        self._gripper_names = None
        self._tool_configs = None
        self._gripper_start_indices = None
        self._gripper_max_angles = None
        self._gripper_bin_widths = None
        self._gripper_num_angular_bins = None
        arch_config = gqcnn_config['architecture']
        if 'gripper_names' in arch_config.keys():
            self._gripper_names = arch_config['gripper_names']
        if 'tool_configs' in arch_config.keys():
            self._tool_configs = arch_config['tool_configs']
        if 'gripper_types' in arch_config.keys():
            self._gripper_types = arch_config['gripper_types']
            self._gripper_start_indices = arch_config['gripper_start_indices']
        if 'gripper_max_angles' in arch_config.keys():
            self._gripper_max_angles = arch_config['gripper_max_angles']
            self._gripper_bin_widths = arch_config['gripper_bin_widths']
        if 'num_angular_bins' in arch_config.keys():
            self._gripper_num_angular_bins = arch_config['num_angular_bins']
            self._angular_bins = np.sum([v for v in self._gripper_num_angular_bins.values()])
            
        # intermediate network feature handles
        self._feature_tensors = {}
  
        #  base layer names for fine-tuning
        self._base_layer_names = []
 
    def initialize_network(self, train_im_node=None, train_pose_node=None, add_softmax=False, add_sigmoid=False):
        """Set up input placeholders and build network.

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
                self._input_im_node = tf.placeholder_with_default(train_im_node, (None, self._im_height, self._im_width, self._num_channels), name='input_im')
                self._input_pose_node = None
                if self._input_depth_mode != InputDepthMode.SUB and self._input_depth_mode != InputDepthMode.IM_ONLY:
                    self._input_pose_node = tf.placeholder_with_default(train_pose_node, (None, self._pose_dim), name='input_pose')
            else:
                # inference only using GQ-CNN instantiated from GQCNNTF.load()
                self._input_im_node = tf.placeholder(GeneralConstants.TF_DTYPE, (self._batch_size, self._im_height, self._im_width, self._num_channels), name='input_im')
                self._input_pose_node = None
                if self._input_depth_mode != InputDepthMode.SUB and self._input_depth_mode != InputDepthMode.IM_ONLY:
                    self._input_pose_node = tf.placeholder(GeneralConstants.TF_DTYPE, (self._batch_size, self._pose_dim), name='input_pose')
            self._input_drop_rate_node = tf.placeholder_with_default(tf.constant(0.0, dtype=GeneralConstants.TF_DTYPE), ())

            # build network
            self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, self._input_drop_rate_node)
            
            # add softmax function to output of network(this is optional because 1) we might be doing regression or 2) we are training and Tensorflow has an optimized cross-entropy loss with the softmax already built-in)
            if add_softmax:
                self.add_softmax_to_output(num_outputs=self._angular_bins)
            # add sigmoid function to output of network(for weighted cross-entropy loss)
            if add_sigmoid:
                self.add_sigmoid_to_output()

        # create feed tensors for prediction
        self._input_im_arr = np.zeros((self._batch_size, self._im_height, self._im_width, self._num_channels))
        self._input_pose_arr = np.zeros((self._batch_size, self._pose_dim))

    def open_session(self):
        """Open Tensorflow session."""
        if self._sess is not None:
            self._logger.warning('Found already initialized TF Session...')
            return self._sess
        self._logger.debug('Initializing TF Session...')
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            self.tf_config = tf.ConfigProto()
            # allow Tensorflow gpu_growth so Tensorflow does not lock-up all GPU memory
            self.tf_config.gpu_options.allow_growth = True
            self._sess = tf.Session(graph=self._graph, config=self.tf_config)
            self._sess.run(init)
        return self._sess

    def close_session(self):
        """Close Tensorflow session."""
        if self._sess is None:
            self._logger.warning('No TF Session to close...')
            return
        self._logger.debug('Closing TF Session...')
        with self._graph.as_default():
            self._sess.close()
            self._sess = None

    def __del__(self):
        """Destructor that basically just makes sure the Tensorflow session has been closed."""
        if self._sess is not None:
            self.close_session()

    @property
    def input_depth_mode(self):
        return self._input_depth_mode

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
        return self._weights.weights

    @property
    def tf_graph(self):
        return self._graph
    
    @property
    def sess(self):
        return self._sess

    @property
    def angular_bins(self):
        return self._angular_bins

    @property
    def gripper_names(self):
        return self._gripper_names

    @property
    def tool_configs(self):
        return self._tool_configs
    
    @property
    def gripper_types(self):
        return self._gripper_types

    @property
    def gripper_start_indices(self):
        return self._gripper_start_indices

    @property
    def gripper_max_angles(self):
        return self._gripper_max_angles

    @property
    def gripper_bin_widths(self):
        return self._gripper_bin_widths

    @property
    def gripper_num_angular_bins(self):
        return self._gripper_num_angular_bins
    
    @property
    def stride(self):
        return reduce(operator.mul, [layer['pool_stride'] for layer in self._architecture['im_stream'].values() if layer['type']=='conv'])

    @property
    def filters(self):
        """Evaluate the filters of the first convolution layer.
        Returns
        -------
        :obj:`numpy.ndarray`
            filters(weights) from first convolution layer of the network
        """
        close_sess = False
        if self._sess is None:
            close_sess = True
            self.open_session()

        first_layer_name = list(self._architecture['im_stream'].keys())[0]
        try:
            filters = self._sess.run(self._weights.weights['{}_weights'.format(first_layer_name)])
        except:
            # legacy support
            raise Exception()
            # filters = self._sess.run(self._weights.weights['{}W'.format(first_layer_im_stream)])
 
        if close_sess:
            self.close_session()
        return filters

    def set_im_mean(self, im_mean):
        """Update image mean to be used for normalization during inference. 
        
        Parameters
        ----------
        im_mean : float
            image mean
        """
        self._im_mean = im_mean
    
    def get_im_mean(self):
        """Get the current image mean used for normalization during inference.

        Returns
        -------
        : float
            image mean
        """
        return self.im_mean

    def set_im_std(self, im_std):
        """Update image standard deviation to be used for normalization during inference. 
        
        Parameters
        ----------
        im_std : float
            image standard deviation
        """
        self._im_std = im_std

    def get_im_std(self):
        """Get the current image standard deviation to be used for normalization during inference.

        Returns
        -------
        : float
            image standard deviation
        """
        return self.im_std

    def set_pose_mean(self, pose_mean):
        """Update pose mean to be used for normalization during inference.
        
        Parameters
        ----------
        pose_mean :obj:`numpy.ndarray`
            pose mean
        """
        self._pose_mean = pose_mean

    def get_pose_mean(self):
        """Get the current pose mean to be used for normalization during inference.

        Returns
        -------
        :obj:`numpy.ndarray`
            pose mean
        """
        return self._pose_mean

    def set_pose_std(self, pose_std):
        """Update pose standard deviation to be used for normalization during inference.
        
        Parameters
        ----------
        pose_std :obj:`numpy.ndarray`
            pose standard deviation
        """
        self._pose_std = pose_std

    def get_pose_std(self):
        """Get the current pose standard deviation to be used for normalization during inference.

        Returns
        -------
        :obj:`numpy.ndarray`
            pose standard deviation
        """
        return self._pose_std

    def set_im_depth_sub_mean(self, im_depth_sub_mean):
        """Update mean of subtracted image and gripper depth to be used for normalization during inference.
        
        Parameters
        ----------
        im_depth_sub_mean : float
            mean of subtracted image and gripper depth
        """
        self._im_depth_sub_mean = im_depth_sub_mean

    def set_im_depth_sub_std(self, im_depth_sub_std):
        """Update standard deviation of subtracted image and gripper depth to be used for normalization during inference.
        
        Parameters
        ----------
        im_depth_sub_std : float
            standard deviation of subtracted image and gripper depth
        """
        self._im_depth_sub_std = im_depth_sub_std

    def add_softmax_to_output(self, num_outputs=0):
        """Adds softmax to output of network."""
        with tf.name_scope('softmax'):
            if num_outputs  > 0:
                self._logger.debug('Building Pair-wise Softmax Layer...')
                binwise_split_output = tf.split(self._output_tensor, num_outputs, axis=-1)
                binwise_split_output_soft = [tf.nn.softmax(s, name='output_%03d'%(i)) for i, s in enumerate(binwise_split_output)]
                self._output_tensor = tf.concat(binwise_split_output_soft, -1, name='output')
            else:
                self._logger.debug('Building Softmax Layer...')
                self._output_tensor = tf.nn.softmax(self._output_tensor, name='output')

    def add_sigmoid_to_output(self):
        """Adds sigmoid to output of network."""
        with tf.name_scope('sigmoid'):
            self._logger.debug('Building Sigmoid Layer...')
            self._output_tensor = tf.nn.sigmoid(self._output_tensor)

    def update_batch_size(self, batch_size):
        """Update the inference batch size. 

        Parameters
        ----------
        batch_size : float
            batch size to be used for inference
        """
        self._batch_size = batch_size

    def _predict(self, image_arr, pose_arr, verbose=False, save_raw_inputs_outputs=False):
        """Query predictions from network.

        Parameters
        ----------
        image_arr :obj:`numpy.ndarray`
            input images
        pose_arr :obj:`numpy.ndarray`
            input gripper poses
        verbose : bool
            whether or not to log progress, useful to turn off during training
        """       
        # get prediction start time
        start_time = time.time()

        if verbose:
            self._logger.debug('Predicting...')

        # setup for prediction
        num_batches = math.ceil(image_arr.shape[0] / self._batch_size)
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]

        output_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images as poses!')

        # predict in batches
        with self._graph.as_default():
            if self._sess is None:
               raise RuntimeError('No TF Session open. Please call open_session() first.')
            i = 0
            batch_idx = 0
            while i < num_images:
                if verbose:
                    self._logger.debug('Predicting batch {} of {}...'.format(batch_idx, num_batches))
                batch_idx += 1
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim

                # normalize the images and poses
                if self._input_depth_mode == InputDepthMode.POSE_STREAM:
                    self._input_im_arr[:dim, ...] = (
                        image_arr[cur_ind:end_ind, ...] - self._im_mean) / self._im_std 
                    self._input_pose_arr[:dim, :] = (
                        pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

                # subtract the depth and then normalize
                elif self._input_depth_mode == InputDepthMode.SUB:
                    # read batch
                    images = image_arr[cur_ind:end_ind, ...]
                    if len(pose_arr.shape) == 1:
                        poses = pose_arr[cur_ind:end_ind]
                    else:
                        poses = pose_arr[cur_ind:end_ind, :]

                    # subtract poses
                    images_sub = images - np.tile(np.reshape(poses, (-1, 1, 1, 1)), (1, images.shape[1], images.shape[2], 1))

                    # normalize
                    self._input_im_arr[:dim, ...] = (images_sub - self._im_depth_sub_mean) / self._im_depth_sub_std

                # normalize the images
                elif self._input_depth_mode == InputDepthMode.IM_ONLY:
                    self._input_im_arr[:dim, ...] = (
                        image_arr[cur_ind:end_ind, ...] - self._im_mean) / self._im_std

                # run forward inference
                if self._input_depth_mode == InputDepthMode.SUB or self._input_depth_mode == InputDepthMode.IM_ONLY:
                    # ignore pose input
                    gqcnn_output = self._sess.run(self._output_tensor,
                                                  feed_dict={self._input_im_node: self._input_im_arr})
                else:
                    # standard forward pass
                    gqcnn_output = self._sess.run(self._output_tensor,
                                                  feed_dict={self._input_im_node: self._input_im_arr,
                                                             self._input_pose_node: self._input_pose_arr})

                # save raw inputs and outputs
                if save_raw_inputs_outputs:
                    features = self._sess.run(self._feature_tensors['fc5'],
                                              feed_dict={self._input_im_node: self._input_im_arr})
                    for i in range(dim):
                        np.save('input_%d.npy' %(i), self._input_im_arr[i,...])
                    for i in range(dim):
                        np.save('output_%d.npy' %(i), features[i,...])
                        
                # allocate output tensor
                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))

                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]
                i = end_ind
        
        # get total prediction time
        pred_time = time.time() - start_time
        if verbose:
            self._logger.debug('Prediction took {} seconds.'.format(pred_time))

        return output_arr

    def predict(self, image_arr, pose_arr, verbose=False):
        """ 
        Predict the probability of grasp success given a depth image and gripper pose

        Parameters
        ----------
        image_arr :obj:`numpy ndarray`
            4D tensor of depth images
        pose_arr :obj:`numpy ndarray`
            tensor of gripper poses
        verbose : bool
            whether or not to log progress
        """
        return self._predict(image_arr, pose_arr, verbose=verbose)
   
    def featurize(self, image_arr, pose_arr=None, feature_layer='conv1_1', verbose=False):
        """Featurize a set of inputs.
        
        Parameters
        ----------
        image_arr :obj:`numpy ndarray` 
            4D tensor of depth images
        pose_arr :obj:`numpy ndarray`
            optional tensor of gripper poses
        feature_layer : str
            the network layer to featurize
        verbose : bool
            whether or not to log progress
        """
        # get featurization start time
        start_time = time.time()

        if verbose:
            self._logger.debug('Featurizing...')

        if feature_layer not in self._feature_tensors.keys():
            raise ValueError('Feature layer: {} not recognized.'.format(feature_layer))
        
        # setup for featurization
        num_images = image_arr.shape[0]
        if pose_arr is not None:
            num_poses = pose_arr.shape[0]
            if num_images != num_poses:
                raise ValueError('Must provide same number of images as poses!')
        output_arr = None

        # featurize in batches
        with self._graph.as_default():
            if self._sess is None:
               raise RuntimeError('No TF Session open. Please call open_session() first.')

            i = 0
            while i < num_images:
                if verbose:
                    self._logger.debug('Featurizing {} of {}...'.format(i, num_images))
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

        if verbose:
            self._logger.debug('Featurization took {} seconds'.format(time.time() - start_time))

        # truncate extraneous values off of end of output_arr
        output_arr = output_arr[:num_images] #TODO: @Jeff, this isn't needed, right?
        return output_arr
    
    def _leaky_relu(self, x, alpha=.1):
        return tf.maximum(alpha * x, x)
    
    def _build_conv_layer(self, input_node, input_height, input_width, input_channels, filter_h, filter_w, num_filt, pool_stride_h, pool_stride_w, pool_size, name, norm=False, pad='SAME'):
        self._logger.debug('Building convolutional layer: {}...'.format(name))       
        with tf.name_scope(name):
            # initialize weights
            if '{}_weights'.format(name) in self._weights.weights.keys():
                convW = self._weights.weights['{}_weights'.format(name)]
                convb = self._weights.weights['{}_bias'.format(name)] 
            elif '{}W'.format(name) in self._weights.weights.keys(): # legacy support
                self._logger.debug('Using old format for layer {}.'.format(name))
                convW = self._weights.weights['{}W'.format(name)]
                convb = self._weights.weights['{}b'.format(name)] 
            else:
                self._logger.debug('Reinitializing layer {}.'.format(name))
                convW_shape = [filter_h, filter_w, input_channels, num_filt]

                fan_in = filter_h * filter_w * input_channels
                std = np.sqrt(2.0 / (fan_in))
                convW = tf.Variable(tf.truncated_normal(convW_shape, stddev=std, dtype=GeneralConstants.TF_DTYPE),
                                    name='{}_weights'.format(name),
                                    dtype=GeneralConstants.TF_DTYPE)
                convb = tf.Variable(tf.truncated_normal([num_filt], stddev=std, dtype=GeneralConstants.TF_DTYPE),
                                    name='{}_bias'.format(name),
                                    dtype=GeneralConstants.TF_DTYPE)

                self._weights.weights['{}_weights'.format(name)] = convW
                self._weights.weights['{}_bias'.format(name)] = convb
            
            if pad == 'SAME':
                out_height = input_height // pool_stride_h
                out_width = input_width // pool_stride_w
            else:
                out_height = math.ceil(float(input_height - filter_h + 1) // pool_stride_h)
                out_width = math.ceil(float(input_width - filter_w + 1) // pool_stride_w)
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
        self._logger.debug('Building fully connected layer: {}...'.format(name))
        
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            fcW = self._weights.weights['{}_weights'.format(name)]
            fcb = self._weights.weights['{}_bias'.format(name)] 
        elif '{}W'.format(name) in self._weights.weights.keys(): # legacy support
            self._logger.debug('Using old format for layer {}.'.format(name))
            fcW = self._weights.weights['{}W'.format(name)]
            fcb = self._weights.weights['{}b'.format(name)] 
        else:
            self._logger.debug('Reinitializing layer {}.'.format(name))
            std = np.sqrt(2.0 / (fan_in))
            fcW = tf.Variable(tf.truncated_normal([fan_in, out_size], stddev=std, dtype=GeneralConstants.TF_DTYPE),
                              name='{}_weights'.format(name),
                              dtype=GeneralConstants.TF_DTYPE)

            if final_fc_layer:
                fcb = tf.Variable(tf.constant(0.0, shape=[out_size], dtype=GeneralConstants.TF_DTYPE),
                                  name='{}_bias'.format(name),
                                  dtype=GeneralConstants.TF_DTYPE)
            else:
                fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std, dtype=GeneralConstants.TF_DTYPE),
                                  name='{}_bias'.format(name),
                                  dtype=GeneralConstants.TF_DTYPE)                                  

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

    #TODO: This really doesn't need to it's own layer type...it does the same thing as _build_fc_layer()
    def _build_pc_layer(self, input_node, fan_in, out_size, name):
        self._logger.debug('Building Fully Connected Pose Layer: {}...'.format(name))
        
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            pcW = self._weights.weights['{}_weights'.format(name)]
            pcb = self._weights.weights['{}_bias'.format(name)] 
        elif '{}W'.format(name) in self._weights.weights.keys(): # legacy support
            self._logger.debug('Using old format for layer {}'.format(name))
            pcW = self._weights.weights['{}W'.format(name)]
            pcb = self._weights.weights['{}b'.format(name)] 
        else:
            self._logger.debug('Reinitializing layer {}'.format(name))
            std = np.sqrt(2.0 / (fan_in))
            pcW = tf.Variable(tf.truncated_normal([fan_in, out_size],
                                               stddev=std, dtype=GeneralConstants.TF_DTYPE),
                              name='{}_weights'.format(name),
                              dtype=GeneralConstants.TF_DTYPE)                                                                
            pcb = tf.Variable(tf.truncated_normal([out_size],
                                               stddev=std, dtype=GeneralConstants.TF_DTYPE),
                              name='{}_bias'.format(name),
                              dtype=GeneralConstants.TF_DTYPE)                              

            self._weights.weights['{}_weights'.format(name)] = pcW
            self._weights.weights['{}_bias'.format(name)] = pcb

        # build layer
        pc = self._leaky_relu(tf.matmul(input_node, pcW) +
                        pcb, alpha=self._relu_coeff)

        # add output to feature dict
        self._feature_tensors[name] = pc

        return pc, out_size

    def _build_fc_merge(self, input_fc_node_1, input_fc_node_2, fan_in_1, fan_in_2, out_size, drop_rate, name):
        self._logger.debug('Building Merge Layer: {}...'.format(name))
        
        # initialize weights
        if '{}_input_1_weights'.format(name) in self._weights.weights.keys():
            input1W = self._weights.weights['{}_input_1_weights'.format(name)]
            input2W = self._weights.weights['{}_input_2_weights'.format(name)]
            fcb = self._weights.weights['{}_bias'.format(name)] 
        elif '{}W_im'.format(name) in self._weights.weights.keys(): # legacy support
            self._logger.debug('Using old format for layer {}.'.format(name))
            input1W = self._weights.weights['{}W_im'.format(name)]
            input2W = self._weights.weights['{}W_pose'.format(name)]
            fcb = self._weights.weights['{}b'.format(name)] 
        else:
            self._logger.debug('Reinitializing layer {}.'.format(name))
            std = np.sqrt(2.0 / (fan_in_1 + fan_in_2))
            input1W = tf.Variable(tf.truncated_normal([fan_in_1, out_size], stddev=std, dtype=GeneralConstants.TF_DTYPE),
                                  name='{}_input_1_weights'.format(name),
                                  dtype=GeneralConstants.TF_DTYPE)
            input2W = tf.Variable(tf.truncated_normal([fan_in_2, out_size], stddev=std, dtype=GeneralConstants.TF_DTYPE),
                                  name='{}_input_2_weights'.format(name),
                                  dtype=GeneralConstants.TF_DTYPE)                                  
            fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std, dtype=GeneralConstants.TF_DTYPE),
                              name='{}_bias'.format(name),
                              dtype=GeneralConstants.TF_DTYPE)                              

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

    def _build_im_stream(self, input_node, input_pose_node, input_height, input_width, input_channels, drop_rate, layers, only_stream=False):
        self._logger.debug('Building Image Stream...')

        output_node = input_node
        prev_layer = "start" # dummy placeholder
        last_index = len(layers.keys()) - 1
        for layer_index, (layer_name, layer_config) in enumerate(layers.items()):
            layer_type = layer_config['type']
            if layer_type == 'conv':
                if prev_layer == 'fc':
                    raise ValueError('Cannot have conv layer after fc layer!')
                output_node, input_height, input_width, input_channels = self._build_conv_layer(output_node, input_height, input_width, input_channels, layer_config['filt_dim'], layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, norm=layer_config['norm'], pad=layer_config['pad'])
                prev_layer = layer_type 
            elif layer_type == 'fc':
                if layer_config['out_size'] == 0:
                    continue
                prev_layer_is_conv = False
                if prev_layer == 'conv':
                    prev_layer_is_conv = True
                    fan_in = input_height * input_width * input_channels
                if layer_index == last_index and only_stream:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv, drop_rate, final_fc_layer=True)
                else:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv, drop_rate)
                prev_layer = layer_type
            elif layer_type == 'pc':
                raise ValueError('Cannot have pose connected layer in image stream!')
            elif layer_type == 'fc_merge':
                raise ValueError('Cannot have merge layer in image stream!')
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, fan_in

    def _build_pose_stream(self, input_node, fan_in, layers):
        self._logger.debug('Building Pose Stream...')
        output_node = input_node
        prev_layer = "start" # dummy placeholder
        for layer_name, layer_config in layers.items():
            layer_type = layer_config['type']
            if layer_type == 'conv':
               raise ValueError('Cannot have conv layer in pose stream')
            elif layer_type == 'fc':
                raise ValueError('Cannot have fully connected layer in pose stream')
            elif layer_type == 'pc':
                if layer_config['out_size'] == 0:
                    continue
                output_node, fan_in = self._build_pc_layer(output_node, fan_in, layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in pose stream")
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return output_node, fan_in

    def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, drop_rate, layers):
        self._logger.debug('Building Merge Stream...')
        
        # first check if first layer is a merge layer
        if layers[list(layers.keys())[0]]['type'] != 'fc_merge':
            raise ValueError('First layer in merge stream must be a fc_merge layer!')
            
        prev_layer = "start"
        last_index = len(layers.keys()) - 1
        fan_in = -1
        for layer_index, (layer_name, layer_config) in enumerate(layers.items()):
            layer_type = layer_config['type']
            if layer_type == 'conv':
               raise ValueError('Cannot have conv layer in merge stream!')
            elif layer_type == 'fc':
                if layer_config['out_size'] == 0:
                    continue
                if layer_index == last_index:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate, final_fc_layer=True)
                else:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate)
                prev_layer = layer_type
            elif layer_type == 'pc':  
                raise ValueError('Cannot have pose connected layer in merge stream!')
            elif layer_type == 'fc_merge':
                if layer_config['out_size'] == 0:
                    continue
                output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], drop_rate, layer_name)
                prev_layer = layer_type   
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, fan_in

    def _build_network(self, input_im_node, input_pose_node, input_drop_rate_node):
        """Build GQ-CNN.

        Parameters
        ----------
        input_im_node :obj:`tf.placeholder`
            image placeholder
        input_pose_node :obj:`tf.placeholder`
            gripper pose placeholder
        input_drop_rate_node :obj:`tf.placeholder`
            drop rate placeholder

        Returns
        -------
        :obj:`tf.Tensor`
            tensor output of network
        """
        self._logger.debug('Building Network...')
        if self._input_depth_mode == InputDepthMode.POSE_STREAM:
            assert 'pose_stream' in self._architecture.keys() and 'merge_stream' in self._architecture.keys(), 'When using input depth mode "pose_stream", both pose stream and merge stream must be present!'
            with tf.name_scope('im_stream'):
                output_im_stream, fan_out_im = self._build_im_stream(input_im_node, input_pose_node, self._im_height, self._im_width, self._num_channels, input_drop_rate_node, self._architecture['im_stream'])
            with tf.name_scope('pose_stream'):
                output_pose_stream, fan_out_pose = self._build_pose_stream(input_pose_node, self._pose_dim, self._architecture['pose_stream'])
            with tf.name_scope('merge_stream'):
                return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, input_drop_rate_node, self._architecture['merge_stream'])[0]
        elif self._input_depth_mode == InputDepthMode.SUB or self._input_depth_mode == InputDepthMode.IM_ONLY:
            assert not ('pose_stream' in self._architecture.keys() or 'merge_stream' in self._architecture.keys()), 'When using input depth mode "{}", only im stream is allowed!'.format(self._input_depth_mode)
            with tf.name_scope('im_stream'):
                return self._build_im_stream(input_im_node, input_pose_node, self._im_height, self._im_width, self._num_channels, input_drop_rate_node, self._architecture['im_stream'], only_stream=True)[0]        
