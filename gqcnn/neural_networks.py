"""
Wrapper for grasp quality neural networks. Implemented using Intel Nervana Neon backend.
Authors: Jeff Mahler, Vishal Satish
"""

import copy
import json
import logging
import numpy as np
import os
import sys
import IPython

from optimizer_constants import InputDataMode

from neon.models import Model
from neon.initializers import Kaiming
from neon.layers import Conv, Pooling, LRN, Sequential, Affine, Dropout, Linear, Bias, Activation, MergeMultistream
from neon.transforms import Rectlin, Softmax
from neon.backends import gen_backend

class GQCNN(object):
    """ Wrapper for GQ-CNN """

    def __init__(self, config, model=None):
        """
        Parameters
        ----------
        config :obj: dict
            python dictionary of configuration parameters such as architecure and basic data params such as batch_size for prediction,
            im_height, im_width, ...
        model :obj: str
        	Neon model to load if fine-tuning, analyzing, or predicting using a pre-existing model
        """
        self._parse_config(config)
        self._model = model

    @staticmethod
    def load(model_dir):
        """ Instantiates a GQCNN object using the Neon model found in model_dir 

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

        # create GQCNN object and initialize network
        gqcnn = GQCNN(gqcnn_config, os.path.join(model_dir, model.prm))
        gqcnn.initialize_network()
        gqcnn.init_mean_and_std(model_dir)

        return gqcnn

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
    
    def _parse_config(self, config):
        """ Parses configuration file for this GQCNN 

        Parameters
        ----------
        config : dict
            python dictionary of configuration parameters such as architecure and basic data params such as batch_size for prediction,
            im_height, im_width, ... 
        """

        # get backend type
        self._backend = config['backend']

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

        # load architecture
        self._architecture = config['architecture']
        self._use_pc2 = False
        if self._architecture['pc2']['out_size'] > 0:
            self._use_pc2 = True

        # get in and out sizes of conv, fully-connected layer and pose layers
        self.conv1_1_out_size = self._im_height / self._architecture['conv1_1']['pool_stride']
        self.conv1_2_out_size = self.conv1_1_out_size / self._architecture['conv1_2']['pool_stride']
        self.conv2_1_out_size = self.conv1_2_out_size / self._architecture['conv2_1']['pool_stride']
        self.conv2_2_out_size = self.conv2_1_out_size / self._architecture['conv2_2']['pool_stride']
        self.pc2_out_size = self._architecture['pc2']['out_size']
        self.pc1_in_size = self._pose_dim
        self.pc1_out_size = self._architecture['pc1']['out_size']
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

    def initialize_network(self, add_softmax=True):
        """ Sets up backend and builds network.

        Parameters
        ----------
        add_softmax : float
            whether or not to add a softmax layer
        """
        
        # first generate a neon backend
        self._be = gen_backend(backend=self._backend, batch_size=self._batch_size)

        # if there is currently no model, ex. during initial training from scratch, then build a new network 
        if self._model is None:
        	self._model, self._layers = self._build_network()
            
        # add softmax if specified
        if add_softmax:
            self.add_softmax_to_predict()

    @property
    def backend(self):
        return self._backend
    
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
    def model(self):
        return self._model

    @property
    def layers(self):
        return self._layers
    
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
        """ Adds softmax layer and re-build network"""
        softmax_layer = Activation(transform=Softmax(), name='softmax')
        self._layers.append(softmax_layer)
        self._model = Model(self._layers)

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

    def _build_network(self, drop_fc3=False, drop_fc4=False, fc3_drop_rate=0, fc4_drop_rate=0):
        """ Builds neural network 

        Parameters
        ----------
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
        :obj:`neon.Models Model`
            network model
        """

        # list to hold network layers
        layers = []

        # image path layers
        im_path_layers = []

        #################################################CONV1_1#########################################################
        # calculate the padding so that input and output dimensions are the same, equivalent to SAME in TensorFlow
        # NOTE: WE ASSUME THAT THE HEIGHT AND WIDTH DIMENSIONS ARE ALWAYS EQUAL SO WE ONLY EVER COMPUTE ONE OF THEM
        stride = 1
        out_dim = np.ceil(float(self._im_height) / float(stride))

        total_pad = max((out_dim - 1) * stride +
                    self._architecture['conv1_1']['filt_dim'] - self._im_height, 0)
        single_side_pad = int(total_pad // 2)

        # build conv layer
        conv1_1 = Conv((self._architecture['conv1_1']['filt_dim'], self._architecture['conv1_1']['filt_dim'], self._architecture['conv1_1']['num_filt']), 
        	init=Kaiming(), bias=Kaiming(),
            padding=single_side_pad, activation=Rectlin(), name="conv1_1")

        # build norm layer
        norm1_1 = None
        if self._architecture['conv1_1']['norm']:
                if self._architecture['conv1_1']['norm_type'] == "local_response":
                	norm1_1 = LRN(depth=self.normalization_radius, alpha=self.normalization_alpha, beta=self.normalization_beta, name="norm1_1")

        # build pool layer
        pool1_1_size = self._architecture['conv1_1']['pool_size']
        pool1_1_stride = self._architecture['conv1_1']['pool_stride']
        pool1_1 = Pooling((pool1_1_size, pool1_1_size), strides=pool1_1_stride, padding=single_side_pad, name='pool1_1')

        # add everything to the layers list
        im_path_layers.append(conv1_1)
        if norm1_1 is not None:
        	im_path_layers.append(norm1_1)
        im_path_layers.append(pool1_1)
        ####################################################################################################################



        #################################################CONV1_2#########################################################
        # calculate the padding so that input and output dimensions are the same, equivalent to SAME in TensorFlow
        # NOTE: WE ASSUME THAT THE HEIGHT AND WIDTH DIMENSIONS ARE ALWAYS EQUAL SO WE ONLY EVER COMPUTE ONE OF THEM
        stride = 1
        out_dim = np.ceil(float(self.conv1_1_out_size) / float(stride))

        total_pad = max((out_dim - 1) * stride +
                    self._architecture['conv1_2']['filt_dim'] - self.conv1_1_out_size, 0)
        single_side_pad = int(total_pad // 2)

        # build conv layer
        conv1_2 = Conv((self._architecture['conv1_2']['filt_dim'], self._architecture['conv1_2']['filt_dim'], self._architecture['conv1_2']['num_filt']), 
        	init=Kaiming(), bias=Kaiming(),
            padding=single_side_pad, activation=Rectlin(), name="conv1_2")

        # build norm layer
        norm1_2 = None
        if self._architecture['conv1_2']['norm']:
                if self._architecture['conv1_2']['norm_type'] == "local_response":
                	norm1_2 = LRN(depth=self.normalization_radius, alpha=self.normalization_alpha, beta=self.normalization_beta, name="norm1_2")

        # build pool layer
        pool1_2_size = self._architecture['conv1_2']['pool_size']
        pool1_2_stride = self._architecture['conv1_2']['pool_stride']
        pool1_2 = Pooling((pool1_2_size, pool1_2_size), strides=pool1_2_stride, padding=single_side_pad, name='pool1_2')

        # add everything to the layers list
        im_path_layers.append(conv1_2)
        if norm1_2 is not None:
        	im_path_layers.append(norm1_2)
        im_path_layers.append(pool1_2)
        ####################################################################################################################

        ################################################CONV2_1#########################################################
        # calculate the padding so that input and output dimensions are the same, equivalent to SAME in TensorFlow
        # NOTE: WE ASSUME THAT THE HEIGHT AND WIDTH DIMENSIONS ARE ALWAYS EQUAL SO WE ONLY EVER COMPUTE ONE OF THEM
        stride = 1
        out_dim = np.ceil(float(self.conv1_2_out_size) / float(stride))

        total_pad = max((out_dim - 1) * stride +
                    self._architecture['conv2_1']['filt_dim'] - self.conv1_2_out_size, 0)
        single_side_pad = int(total_pad // 2)

        # build conv layer
        conv2_1 = Conv((self._architecture['conv2_1']['filt_dim'], self._architecture['conv2_1']['filt_dim'], self._architecture['conv2_1']['num_filt']), 
        	init=Kaiming(), bias=Kaiming(),
            padding=single_side_pad, activation=Rectlin(), name="conv2_1")

        # build norm layer
        norm2_1 = None
        if self._architecture['conv2_1']['norm']:
                if self._architecture['conv2_1']['norm_type'] == "local_response":
                	norm2_1 = LRN(depth=self.normalization_radius, alpha=self.normalization_alpha, beta=self.normalization_beta, name="norm2_1")

        # build pool layer
        pool2_1_size = self._architecture['conv2_1']['pool_size']
        pool2_1_stride = self._architecture['conv2_1']['pool_stride']
        pool2_1 = Pooling((pool2_1_size, pool2_1_size), strides=pool2_1_stride, padding=single_side_pad, name='pool2_1')

        # add everything to the layers list
        im_path_layers.append(conv2_1)
        if norm2_1 is not None:
        	im_path_layers.append(norm2_1)
        im_path_layers.append(pool2_1)
        ####################################################################################################################

        ################################################CONV2_2#########################################################
        # calculate the padding so that input and output dimensions are the same, equivalent to SAME in TensorFlow
        # NOTE: WE ASSUME THAT THE HEIGHT AND WIDTH DIMENSIONS ARE ALWAYS EQUAL SO WE ONLY EVER COMPUTE ONE OF THEM
        stride = 1
        out_dim = np.ceil(float(self.conv2_1_out_size) / float(stride))

        total_pad = max((out_dim - 1) * stride +
                    self._architecture['conv2_2']['filt_dim'] - self.conv2_1_out_size, 0)
        single_side_pad = int(total_pad // 2)

        # build conv layer
        conv2_2 = Conv((self._architecture['conv2_2']['filt_dim'], self._architecture['conv2_2']['filt_dim'], self._architecture['conv2_2']['num_filt']), 
        	init=Kaiming(), bias=Kaiming(),
            padding=single_side_pad, activation=Rectlin(), name="conv2_2")

        # build norm layer
        norm2_2 = None
        if self._architecture['conv2_2']['norm']:
                if self._architecture['conv2_2']['norm_type'] == "local_response":
                	norm2_2 = LRN(depth=self.normalization_radius, alpha=self.normalization_alpha, beta=self.normalization_beta, name="norm2_2")

        # build pool layer
        pool2_2_size = self._architecture['conv2_2']['pool_size']
        pool2_2_stride = self._architecture['conv2_2']['pool_stride']
        pool2_2 = Pooling((pool2_2_size, pool2_2_size), strides=pool2_2_stride, padding=single_side_pad, name='pool2_2')

        # add everything to the layers list
        im_path_layers.append(conv2_2)
        if norm2_2 is not None:
        	im_path_layers.append(norm2_2)
        im_path_layers.append(pool2_2)
        ####################################################################################################################

        ################################################FC3#########################################################
        # build fully-connected layer
        fc3 = Affine(nout=self.fc3_out_size, init=Kaiming(), bias=Kaiming(), activation=Rectlin(), name='fc3')

        # drop fc3 if necessary
        fc3_drop = None
        if drop_fc3:
                fc3_drop = Dropout(keep=fc3_drop_rate, name="fc3_drop")
        
        # add everything to the layers list
        im_path_layers.append(fc3)
        if fc3_drop is not None:
        	im_path_layers.append(fc3_drop)
        ####################################################################################################################

        # form the image path 
        im_path = Sequential(im_path_layers)

        # pose path layers
        pose_path_layers = []

        ################################################PC1#########################################################
        # build fully-connected layer
        pc1 = Affine(nout=self.pc1_out_size, init=Kaiming(), bias=Kaiming(), activation=Rectlin(), name='pc1')
        pose_path_layers.append(pc1)
        ####################################################################################################################

        ################################################PC2#########################################################
        if self._use_pc2:
        	# build fully-connected layer
            pc2 = Affine(nout=self.pc2_out_size, init=Kaiming(), bias=Kaiming(), activation=Rectlin(), name='pc2')
            pose_path_layers.append(pc2)
        ####################################################################################################################

        # form the pose path
        pose_path = Sequential(pose_path_layers)

        # merge
        combined_layers = []
        combined_layers.append(MergeMultistream(layers=[im_path, pose_path], merge="stack"))

        ################################################FC4#########################################################
        # build fully-connected layer
        fc4 = Affine(nout=self.fc4_out_size, init=Kaiming(), bias=Kaiming(), activation=Rectlin(), name='fc4')

        # drop fc4 if necessary
        fc4_drop = None
        if drop_fc4:
                fc4_drop = Dropout(keep=fc4_drop_rate, name="fc4_drop")
        
        # add everything to the layers list
        combined_layers.append(fc4)
        if fc4_drop is not None:
        	combined_layers.append(fc4_drop)
        ####################################################################################################################

        ################################################FC5#########################################################
        # build fully-connected layer
        fc5 = Linear(nout=self.fc5_out_size, init=Kaiming(), name='fc5')
        fc5_bias = Bias(init=Kaiming(), name='fc5_bias')

        # add everything to the layers list
        combined_layers.append(fc5)
        combined_layers.append(fc5_bias)

        return Model(layers=combined_layers), combined_layers
