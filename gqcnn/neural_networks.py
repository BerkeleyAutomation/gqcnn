"""
Quick wrapper for grasp quality neural network
Author: Jeff Mahler
"""

import copy
import json
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from autolab_core import YamlConfig
from gqcnn import InputDataMode

from spatial_transformer import transformer

import IPython

def reduce_shape(shape):
    """ Get shape of a layer for flattening """
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape, 1)


class GQCNNWeights(object):
    """ Struct helper for storing weights """
    weights = {}
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
        self._weights = GQCNNWeights()
        self._graph = tf.Graph()
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
        # load in means and stds for all 7 possible pose variables
        # grasp center row, grasp center col, gripper depth, grasp theta, crop center row, crop center col, grip width
        self._im_mean = np.load(os.path.join(model_dir, 'mean.npy'))
        self._im_std = np.load(os.path.join(model_dir, 'std.npy'))
        self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))

        # slice out the variables we want based on the input pose_dim, which
        # is dependent on the input data mode used to train the model
        if self._input_data_mode == InputDataMode.TF_IMAGE:
            # depth
            self._pose_mean = self._pose_mean[2]
            self._pose_std = self._pose_std[2]
        elif self._input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
            # depth, cx, cy
            self._pose_mean = np.concatenate([self._pose_mean[2:3], self._pose_mean[4:6]])
            self._pose_std = np.concatenate([self._pose_std[2:3], self._pose_std[4:6]])
        elif self._input_data_mode == InputDataMode.RAW_IMAGE:
            # u, v, depth, theta
            self._pose_mean = self._pose_mean[:4]
            self._pose_std = self._pose_std[:4]
        elif self._input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
            # u, v, depth, theta, cx, cy
            self._pose_mean = self._pose_mean[:6]
            self._pose_std = self._pose_std[:6]
        elif self._input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
            # depth, theta
            self._pose_mean = self._pose_mean[2:4]
            self._pose_std = self._pose_std[2:4]

    def init_weights_file(self, model_filename):
        """ Initialize network weights from the specified model 

        Parameters
        ----------
        model_filename :obj: str
            path to model to be loaded into weights
        """

        # read the input image
        with self._graph.as_default():

            # create new tf checkpoint reader
            reader = tf.train.NewCheckpointReader(model_filename)

            # create empty weight object
            self._weights = GQCnnWeights()

            # read in conv1 & conv2
            self._weights.conv1_1W = tf.Variable(reader.get_tensor("conv1_1W"))
            self._weights.conv1_1b = tf.Variable(reader.get_tensor("conv1_1b"))
            self._weights.conv1_2W = tf.Variable(reader.get_tensor("conv1_2W"))
            self._weights.conv1_2b = tf.Variable(reader.get_tensor("conv1_2b"))
            self._weights.conv2_1W = tf.Variable(reader.get_tensor("conv2_1W"))
            self._weights.conv2_1b = tf.Variable(reader.get_tensor("conv2_1b"))
            self._weights.conv2_2W = tf.Variable(reader.get_tensor("conv2_2W"))
            self._weights.conv2_2b = tf.Variable(reader.get_tensor("conv2_2b"))

            # if conv3 is to be used, read in conv3
            if self._use_conv3:
                self._weights.conv3_1W = tf.Variable(reader.get_tensor("conv3_1W"))
                self._weights.conv3_1b = tf.Variable(reader.get_tensor("conv3_1b"))
                self._weights.conv3_2W = tf.Variable(reader.get_tensor("conv3_2W"))
                self._weights.conv3_2b = tf.Variable(reader.get_tensor("conv3_2b"))

            # read in pc1
            self._weights.pc1W = tf.Variable(reader.get_tensor("pc1W"))
            self._weights.pc1b = tf.Variable(reader.get_tensor("pc1b"))

            # if pc2 is to be used, read in pc2
            if self._use_pc2:
                self._weights.pc2W = tf.Variable(reader.get_tensor("pc2W"))
                self._weights.pc2b = tf.Variable(reader.get_tensor("pc2b"))

            self._weights.fc3W = tf.Variable(reader.get_tensor("fc3W"))
            self._weights.fc3b = tf.Variable(reader.get_tensor("fc3b"))
            self._weights.fc4W_im = tf.Variable(reader.get_tensor("fc4W_im"))
            self._weights.fc4W_pose = tf.Variable(reader.get_tensor("fc4W_pose"))
            self._weights.fc4b = tf.Variable(reader.get_tensor("fc4b"))
            self._weights.fc5W = tf.Variable(reader.get_tensor("fc5W"))
            self._weights.fc5b = tf.Variable(reader.get_tensor("fc5b"))

    def reinitialize_layers(self, reinit_fc3, reinit_fc4, reinit_fc5):
        """ Re-initializes final fully-connected layers for fine-tuning 

        Parameters
        ----------
        reinit_fc3 : bool
            whether to re-initialize fc3
        reinit_fc4 : bool
            whether to re-initialize fc4
        reinit_fc5 : bool
            whether to re-initialize fc5
        """
        with self._graph.as_default():
            if reinit_fc3:
                fc3_std = np.sqrt(2.0 / (self.fc3_in_size))
                self._weights.fc3W = tf.Variable(tf.truncated_normal([self.fc3_in_size, self.fc3_out_size], stddev=fc3_std))
                self._weights.fc3b = tf.Variable(tf.truncated_normal([self.fc3_out_size], stddev=fc3_std))  
            if reinit_fc4:
                fc4_std = np.sqrt(2.0 / (self.fc4_in_size))
                self._weights.fc4W_im = tf.Variable(tf.truncated_normal([self.fc4_in_size, self.fc4_out_size], stddev=fc4_std))
                self._weights.fc4W_pose = tf.Variable(tf.truncated_normal([self.fc4_pose_in_size, self.fc4_out_size], stddev=fc4_std))
                self._weights.fc4b = tf.Variable(tf.truncated_normal([self.fc4_out_size], stddev=fc4_std))
            if reinit_fc5:
                fc5_std = np.sqrt(2.0 / (self.fc5_in_size))
                self._weights.fc5W = tf.Variable(tf.truncated_normal([self.fc5_in_size, self.fc5_out_size], stddev=fc5_std))
                self._weights.fc5b = tf.Variable(tf.constant(0.0, shape=[self.fc5_out_size]))
    
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
        if self._input_data_mode == InputDataMode.TF_IMAGE:
            # depth
            self._pose_dim = 1
        elif self._input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
            # depth, cx, cy
            self._pose_dim = 3
        elif self._input_data_mode == InputDataMode.RAW_IMAGE:
            # u, v, depth, theta
            self._pose_dim = 4
        elif self._input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
            # u, v, depth, theta, cx, cy
            self._pose_dim = 6
        elif self._input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
            # depth, theta
            self._pose_dim = 2

        # create feed tensors for prediction
        self._input_im_arr = np.zeros([self._batch_size, self._im_height,
                                       self._im_width, self._num_channels])
        self._input_pose_arr = np.zeros([self._batch_size, self._pose_dim])

        # load architecture
        self._architecture = config['architecture']
       	
        # load normalization constants
        self._normalization_radius = config['radius']
        self._normalization_alpha = config['alpha']
        self._normalization_beta = config['beta']
        self._normalization_bias = config['bias']

        # initialize means and standard deviation to be 0 and 1, respectively
        self._im_mean = 0
        self._im_std = 1
        self._pose_mean = np.zeros(self._pose_dim)
        self._pose_std = np.ones(self._pose_dim)

        # create empty holder for feature handles
        self._feature_tensors = {}

    def initialize_network(self, add_softmax=True):
        """ Set up input nodes and builds network.

        Parameters
        ----------
        add_softmax : float
            whether or not to add a softmax layer
        """
        with self._graph.as_default():
            # setup tf input placeholders and build network
            self._input_im_node = tf.placeholder(
                tf.float32, (self._batch_size, self._im_height, self._im_width, self._num_channels))
            self._input_pose_node = tf.placeholder(
                tf.float32, (self._batch_size, self._pose_dim))

            # build network
            self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, inference=True)
            if add_softmax:
                self.add_softmax_to_predict()

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
    
    @property
    def sess(self):
        return self._sess

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
        output_arr = np.zeros([num_images, 2])
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

                gqcnn_output = self._sess.run(self._output_tensor,
                                              feed_dict={self._input_im_node: self._input_im_arr,
                                                         self._input_pose_node: self._input_pose_arr})
                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]

                i = end_ind
            if close_sess:
                self.close_session()
        return output_arr
	
    def featurize(self, image_arr, pose_arr, feature_layer='conv2_2'):
        """ Featurize a set of images in batches """

        if feature_layer not in self._feature_tensors.keys():
            raise ValueError('Feature layer %s not recognized' %(feature_layer))
        
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

                gqcnn_output = self._sess.run(self._feature_tensors[feature_layer],
                                              feed_dict={self._input_im_node: self._input_im_arr,
                                                         self._input_pose_node: self._input_pose_arr})
                if output_arr is None:
                    output_arr = gqcnn_output
                else:
                    output_arr = np.r_[output_arr, gqcnn_output]

                i = end_ind
            if close_sess:
                self.close_session()
        # truncate extraneous values off of end of output_arr
        output_arr = output_arr[:num_images]
        return output_arr
    
    def _build_spatial_transformer(self, input_node, input_height, input_width, input_channels, num_transform_params, output_width, output_height, name):
        logging.info('Building spatial transformer layer: {}'.format(name))
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            transformW = self._weights.weights['{}_weights'.format(name)]
            transformb = self._weights.weights['{}_bias'.format(name)]
        else:
            transformW = tf.Variable(tf.zeros([input_height * input_width * input_channels, num_transform_params]), name='{}_weights'.format(name))

            initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
            transformb = tf.Variable(initial_value=initial, name='{}_bias'.format(name))
            
            self._weights.weights['{}_weights'.format(name)] = transformW
            self._weights.weights['{}_bias'.format(name)] = transformb

        # build localisation network
        loc_network = tf.matmul(tf.zeros([self._batch_size, input_height * input_width * input_channels]), transformW) + transformb
            
        # build transform layer
        transform_layer = transformer(input_node, loc_network, (output_width, output_height))

        # add output to feature dict
        self._feature_tensors[name] = transform_layer

        return transform_layer, output_height, output_width, input_channels

    def _build_conv_layer(self, input_node, input_height, input_width, input_channels, filter_h, filter_w, num_filt, pool_stride_h, pool_stride_w, pool_size, name, norm=False):
        logging.info('Building convolutional layer: {}'.format(name))
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

        out_height = input_height / pool_stride_h
        out_width = input_width / pool_stride_w
        out_channels = num_filt

        # build layer
        convh = tf.nn.relu(tf.nn.conv2d(input_node, convW, strides=[
                                1, 1, 1, 1], padding='SAME') + convb)
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

    def _build_fc_layer(self, input_node, fan_in, out_size, name, input_is_conv, drop_rate=0.0, final_fc_layer=False, inference=False):
        logging.info('Building fully connected layer: {}'.format(name))
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
        if input_is_conv:
            input_num_nodes = reduce_shape(input_node.get_shape())
            input_flat = tf.reshape(input_node, [-1, input_num_nodes])
            fc = tf.nn.relu(tf.matmul(input_flat, fcW) + fcb)
        else:
            fc = tf.nn.relu(tf.matmul(input_node, fcW) + fcb)

        if drop_rate > 0 and not inference:
            fc = tf.nn.dropout(fc, drop_rate)

        # add output to feature dict
        self._feature_tensors[name] = fc

        return fc, out_size

    def _build_pc_layer(self, input_node, fan_in, out_size, name):
        logging.info('Building Fully-Connected Pose Layer: {}'.format(name))
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
        pc = tf.nn.relu(tf.matmul(input_node, pcW) +
                        pcb)

        # add output to feature dict
        self._feature_tensors[name] = pc

        return pc, out_size

    def _build_fc_merge(self, input_fc_node_1, input_fc_node_2, fan_in_1, fan_in_2, out_size, name, drop_rate=0.0, inference=False):
        logging.info('Building Merge Layer: {}'.format(name))
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
        fc = tf.nn.relu(tf.matmul(input_fc_node_1, input1W) +
                                tf.matmul(input_fc_node_2, input2W) +
                                fcb)
        if drop_rate > 0 and not inference:
            fc = tf.nn.dropout(fc, drop_rate)

        # add output to feature dict
        self._feature_tensors[name] = fc

        return fc, out_size

    def _build_batch_norm(self, input_node, ep, inference=False):
        output_node = input_node
        batch_size = input_node.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(input_node, axes=[0, 1, 2])
        beta = tf.get_variable('batch_norm_beta', batch_size, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('batch_norm_gamma', batch_size, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        output_node = tf.nn.batch_normalization(output_node, mean, variance, beta, gamma, ep)

        return output_node

    def _build_residual_layer(self, input_node, input_channels, fan_in, num_filt, filt_h, filt_w, name, inference=False):
        if '{}_conv1_weights'.format(name) in self._weights.weights.keys():
            conv1W = self._weights.weights['{}_conv1_weights'.format(name)]
            conv1b = self._weights.weights['{}_conv1_bias'.format(name)]
            conv2W = self._weights.weights['{}_conv2_weights'.format(name)]
            conv2b = self._weights.weights['{}_conv2_bias'.format(name)] 
        else:
            std = np.sqrt(2.0 / fan_in)
            conv_shape = [filt_h, filt_w, input_channels, num_filt]
            conv1W = tf.Variable(tf.truncated_normal(conv_shape, stddev=std), name='{}_conv1_weights'.format(name))
            conv1b = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_conv1_bias'.format(name))
            conv2W = tf.Variable(tf.truncated_normal(conv_shape, stddev=std), name='{}_conv2_weights'.format(name))
            conv2b = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_conv2_bias'.format(name))


            self._weights.weights['{}_conv1_weights'.format(name)] = conv1W
            self._weights.weights['{}_conv1_bias'.format(name)] = conv1b
            self._weights.weights['{}_conv2_weights'.format(name)] = conv2W
            self._weights.weights['{}_conv2_bias'.format(name)] = conv2b

        #  implemented as x = BN + ReLU + Conv + BN + ReLU + Conv
        EP = .001
        output_node = input_node
        output_node = self._build_batch_norm(output_node, EP)
        output_node = tf.nn.relu(output_node)
        output_node = tf.nn.conv2d(output_node, conv1W, strides=[1, 1, 1, 1], padding='SAME') + conv1b
        output_node = self._build_batch_norm(output_node, EP)
        output_node = tf.nn.relu(output_node)
        output_node = tf.nn.conv2d(output_node, conv2W, strides=[1, 1, 1, 1], padding='SAME') + conv2b
        output_node = input_node + output_node

        return output_node 


    def _build_im_stream(self, input_node, input_height, input_width, input_channels, layers, inference=False):
        logging.info('Building Image Stream')
        output_node = input_node
        prev_layer = "start"
        for layer_name, layer_config in layers.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                output_node, input_height, input_width, input_channels = self._build_spatial_transformer(output_node, input_height, input_width, input_channels,
                    layer_config['num_transform_params'], layer_config['out_size'], layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'conv':
                if prev_layer == 'fc':
                    raise ValueError('Cannot have conv layer after fc layer')
                output_node, input_height, input_width, input_channels = self._build_conv_layer(output_node, input_height, input_width, input_channels, layer_config['filt_dim'],
                    layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, 
                    norm=layer_config['norm'])
                prev_layer = layer_type
            elif layer_type == 'fc':
                prev_layer_is_conv = False
                if prev_layer == 'conv':
                    prev_layer_is_conv = True
                    fan_in = input_height * input_width * input_channels
                if 'dropout_rate' in layer_config.keys():
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv, drop_rate=layer_config['drop_rate'], inference=inference)
                else:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv, inference=inference)
                prev_layer = layer_type
            elif layer_type == 'pc':
                raise ValueError('Cannot have pose-connected layer in image stream')
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in image stream")
            elif layer_type == 'residual':
                output_node = self._build_residual_layer(output_node, self._num_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
                layer_config['filt_dim'], layer_name)
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return output_node, fan_in

    def _build_pose_stream(self, input_node, fan_in, layers, inference=False):
        logging.info('Building Pose Stream')
        output_node = input_node
        prev_layer = "start"
        for layer_name, layer_config in layers.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                raise ValueError('Cannot have spatial transformer in pose stream')
            elif layer_type == 'conv':
               raise ValueError('Cannot have conv layer in pose stream')
            elif layer_type == 'fc':
                raise ValueError('Cannot have fc layer in pose stream')
            elif layer_type == 'pc':
                output_node, fan_in = self._build_pc_layer(output_node, fan_in, layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in pose stream")
            elif layer_type == 'residual':
                raise ValueError('Cannot have residual in pose stream')
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return output_node, fan_in

    def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, layers, inference=False):
        logging.info('Building Merge Stream')
        
        # first check if first layer is a merge layer
        if layers[layers.keys()[0]]['type'] != 'fc_merge':
            raise ValueError('First layer in merge stream must be a fc_merge layer')
            
        prev_layer = "start"
        last_index = len(layers.keys()) - 1
        for layer_index, (layer_name, layer_config) in enumerate(layers.iteritems()):
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                raise ValueError('Cannot have spatial transformer in merge stream')
            elif layer_type == 'conv':
               raise ValueError('Cannot have conv layer in merge stream')
            elif layer_type == 'fc':
                # TODO: Clean this giant if statement up
                if layer_index == last_index:
                    if 'drop_rate' in layer_config.keys():
                        output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, final_fc_layer=True, drop_rate=layer_config['drop_rate'], inference=inference)
                    else:
                        output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, final_fc_layer=True, inference=inference)
                else:
                    if 'drop_rate' in layer_config.keys():
                        output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate=layer_config['drop_rate'], inference=inference)
                    else:
                        output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, inference=inference)
                prev_layer = layer_type
            elif layer_type == 'pc':  
                raise ValueError('Cannot have pose-connected layer in merge stream')
            elif layer_type == 'fc_merge':
                if 'drop_rate' in layer_config.keys():
                    output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], layer_name, drop_rate=layer_config['drop_rate'], inference=inference)
                else:
                    output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], layer_name, inference=inference)
            elif layer_type == 'residual':
                raise ValueError('Cannot have residual in merge stream')
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node

    def _build_network(self, input_im_node, input_pose_node, inference=False):
        """ Builds neural network 

        Parameters
        ----------
        input_im_node : :obj:`tensorflow Placeholder`
            network input image placeholder
        input_pose_node : :obj:`tensorflow Placeholder`
            network input pose placeholder

        Returns
        -------
        :obj:`tensorflow Tensor`
            output of network
        """
        logging.info('Building Network')
        output_im_stream, fan_out_im = self._build_im_stream(input_im_node, self._im_height, self._im_width, self._num_channels, self._architecture['im_stream'], inference=inference)
        output_pose_stream, fan_out_pose = self._build_pose_stream(input_pose_node, self._pose_dim, self._architecture['pose_stream'], inference=inference)
        return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, self._architecture['merge_stream'], inference=inference)
