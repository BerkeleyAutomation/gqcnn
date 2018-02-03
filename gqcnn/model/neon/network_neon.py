"""
GQCNN network implemented in Intel Neon
Author: Vishal Satish
"""
import json
from collections import OrderedDict
import logging
import os

import numpy as np
import scipy.stats as stats

from neon.models import Model
from neon.initializers import Constant
from neon.initializers.initializer import Initializer
from neon.layers import Conv, Pooling, LRN, Sequential, Affine, Dropout, MergeMultistream, Activation
from neon.transforms import Rectlin, Softmax, Identity
from neon.backends import gen_backend

from gqcnn.utils.data_utils import parse_pose_data, parse_gripper_data
from gqcnn.utils.enums import InputPoseMode, InputGripperMode
from gqcnn.training.neon.gqcnn_predict_iterator import GQCNNPredictIterator

class CustomKaiming(Initializer):
    def __init__(self, local=True, name='CustomKaiming'):
        super(CustomKaiming, self).__init__(name=name)
        self.scale = None
        self.local = local

    def fill(self, param):
        if self.scale is None:
            fan_in = param.shape[0 if self.local else 1]
            self.scale = np.sqrt(2. / fan_in)

        upper_bound = 2
        lower_bound = -1 * upper_bound 
        truncated_norm = stats.truncnorm(lower_bound, upper_bound, scale=self.scale)
        param[:] = truncated_norm.rvs(np.prod(param.shape)).reshape(param.shape)

class GQCNNNeon(object):
    """ GQCNN network implemented in Intel Neon """

    def __init__(self, gqcnn_config, model_path=None):
        """
        Parameters
        ----------
        config :obj: dict
            python dictionary of configuration parameters such as architecure and basic data params such as batch_size for prediction,
            im_height, im_width, ...
        """
        self._be = None
        self._model_path = model_path
        self._parse_config(gqcnn_config)

        # fully-convolutional architecture not yet supported in Neon
        self._fully_conv = False

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
        gqcnn = GQCNNNeon(gqcnn_config, model_path=os.path.join(model_dir, 'model.prm'))
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
        # load in means and stds 
        # pose format is: grasp center row, grasp center col, gripper depth, grasp theta, crop center row, crop center col, grip width
        # gripper format is: min_width, force_limit, max_width, finger_radius
        self._im_mean = np.load(os.path.join(model_dir, 'im_mean.npy'))
        self._im_std = np.load(os.path.join(model_dir, 'im_std.npy'))
        self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
        self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))
        if self._gripper_dim > 0:
            self._gripper_mean = np.load(os.path.join(model_dir, 'gripper_mean.npy'))
            self._gripper_std = np.load(os.path.join(model_dir, 'gripper_std.npy'))

        # read the certain parts of the pose and gripper mean/std that we want
        self._pose_mean = parse_pose_data(self._pose_mean, self._input_pose_mode)
        self._pose_std = parse_pose_data(self._pose_std, self._input_pose_mode)
        if self._gripper_dim > 0:
            self._gripper_mean = parse_gripper_data(self._gripper_mean, self._input_gripper_mode)
            self._gripper_std = parse_gripper_data(self._gripper_std, self._input_gripper_mode)

    def reinitialize_layers(self, reinit_fc3, reinit_fc4, reinit_fc5):
        raise NotImplementedError("Fine-tuning functionality not yet supported in Neon. Please use Tensorflow backend for fine-tuning.")
    
    def _parse_config(self, gqcnn_config):
        """ Parses configuration file for this GQCNN 

        Parameters
        ----------
        config : dict
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
        self._input_pose_mode = gqcnn_config['input_pose_mode']
        self._input_gripper_mode = gqcnn_config['input_gripper_mode']

        # get backend type
        self._backend = gqcnn_config['backend']

        # setup correct pose dimensions 
        if self._input_pose_mode == InputPoseMode.TF_IMAGE:
            # depth
            self._pose_dim = 1
        elif self._input_pose_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
            # depth, cx, cy
            self._pose_dim = 3
        elif self._input_pose_mode == InputPoseMode.RAW_IMAGE:
            # u, v, depth, theta
            self._pose_dim = 4
        elif self._input_pose_mode == InputPoseMode.RAW_IMAGE_PERSPECTIVE:
            # u, v, depth, theta, cx, cy
            self._pose_dim = 6
        elif self._input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
            # depth, theta
            self._pose_dim = 2
        else:
            raise ValueError('Input pose mode: {} not understood'.format(self._input_pose_mode))

        if self._input_gripper_mode == InputGripperMode.WIDTH:
            self._gripper_dim = 1 # width
        elif self._input_gripper_mode == InputGripperMode.NONE:
            self._gripper_dim = 0 # no gripper channel
        elif self._input_gripper_mode == InputGripperMode.ALL:
            self._gripper_dim = 4 # width, palm depth, fx, fy
        elif self._input_gripper_mode == InputGripperMode.DEPTH_MASK:
            self._gripper_dim = 0 # no gripper channel
            self._num_channels += 2 # add channels for gripper depth masks
        else:
            raise ValueError('Input gripper mode: {} not understood'.format(self._input_gripper_mode))

        # load architecture
        self._architecture = gqcnn_config['architecture']
        
        # load normalization constants
        self._normalization_radius = gqcnn_config['radius']
        self._normalization_alpha = gqcnn_config['alpha']
        self._normalization_beta = gqcnn_config['beta']
        self._normalization_bias = gqcnn_config['bias']

        # initialize means and standard deviation to be 0 and 1, respectively
        self._im_mean = 0
        self._im_std = 1
        self._pose_mean = np.zeros(self._pose_dim)
        self._pose_std = np.ones(self._pose_dim)
        if self._gripper_dim > 0:
            self._gripper_mean = np.zeros(self._gripper_dim)
            self._gripper_std = np.ones(self._gripper_dim)

        # create empty holder for feature tensors
        self._feature_tensors = {}
    
        # initialize other misc parameters
        self._mask_and_inpaint = False
        self._drop_rate = 0.0

    def initialize_network(self, add_softmax=False):
        """ Sets up backend and builds network. """
        
        # first generate a neon backend
        if self._be is None:
            self.init_backend()

        if self._model_path is None:
            # if there is currently no model specified, ex. during initial training from scratch, then build a new network 
            self._layers = self._build_network()
            if add_softmax:
                self._add_softmax()
            self._model = Model(self._layers) 
        else:
            # else there is a model location specified and we have to instantiate it
            self._model = Model(self._model_path)

    def init_backend(self):
        self._be = gen_backend(backend=self._backend, batch_size=self._batch_size)

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
    def num_channels(self):
        return self._num_channels

    @property
    def pose_dim(self):
        return self._pose_dim

    @property
    def input_pose_mode(self):
        return self._input_pose_mode

    @property
    def model(self):
        return self._model

    @property
    def layers(self):
        return self._layers
    
    def update_drop_rate(self, drop_rate):
        self._drop_rate = drop_rate

    def set_mask_and_inpaint(self, mask_and_inpaint):
        self._mask_and_inpaint = mask_and_inpaint

    def _add_softmax(self):
        self._layers.append(Activation(transform=Softmax(), name='softmax'))

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

    def update_pose_std(self, pose_std):
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

    def update_gripper_mean(self, gripper_mean):
        """ Updates gripper parameter mean to be used for normalization when predicting 
        
        Parameters
        ----------
        gripper_mean :obj:`numpy ndarray`
            gripper parameter mean to be used
        """
        self._gripper_mean = gripper_mean

    def get_gripper_mean(self):
        """ Get the current gripper parameter mean to be used for normalization when predicting

        Returns
        -------
        :obj:`numpy ndarray`
            gripper parameter mean
        """
        return self._gripper_mean

    def update_gripper_std(self, gripper_std):
        """ Updates gripper parameter standard deviation to be used for normalization when predicting 
        
        Parameters
        ----------
        gripper_std :obj:`numpy ndarray`
            gripper parameter standard deviation to be used
        """
        self._gripper_std = gripper_std

    def get_gripper_std(self):
        """ Get the current gripper parameter standard deviation to be used for normalization when predicting

        Returns
        -------
        :obj:`numpy ndarray`
            gripper standard deviation
        """
        return self._gripper_std

    def update_gripper_depth_mask_mean(self, gripper_depth_mask_mean):
        """ Updates gripper depth mask mean to be used for normalization when predicting 
        
        Parameters
        ----------
        gripper_depth_mask_mean :obj:`numpy ndarray`
            gripper depth mask mean to be used
        """
        self._gripper_depth_mask_mean = gripper_depth_mask_mean

    def get_gripper_mean(self):
        """ Get the current gripper depth mask mean to be used for normalization when predicting

        Returns
        -------
        :obj:`numpy ndarray`
            gripper depth mask mean
        """
        return self._gripper_depth_mask_mean

    def update_gripper_depth_mask_std(self, gripper_depth_mask_std):
        """ Updates gripper depth mask standard deviation to be used for normalization when predicting 
        
        Parameters
        ----------
        gripper_depth_mask_std :obj:`numpy ndarray`
            gripper depth mask standard deviation to be used
        """
        self._gripper_depth_mask_std = gripper_depth_mask_std

    def get_gripper_std(self):
        """ Get the current gripper depth mask standard deviation to be used for normalization when predicting

        Returns
        -------
        :obj:`numpy ndarray`
            gripper depth mask standard deviation
        """
        return self._gripper_depth_mask_std

    def update_batch_size(self, batch_size):
        """ Updates the prediction batch size 

        Parameters
        ----------
        batch_size : float
            batch size to be used for prediction
        """
        self._batch_size = batch_size

    def update_backend(self, backend):
        self._backend = backend

    def predict(self, image_arr, pose_arr):
        """ Predict a set of images in batches """

        # setup for prediction
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]
        if num_images != num_poses:
            raise ValueError('Must provide same number of images and poses')
        
        # normalize image and pose data
        image_arr = (image_arr - self._im_mean) / self._im_std
        pose_arr = (pose_arr - self._pose_mean) / self._pose_std

        # create data iterator
        pred_iter = GQCNNPredictIterator(image_arr.reshape((image_arr.shape[0], self._im_width * self._im_width * self._num_channels)), 
            pose_arr, lshape=[(self._num_channels, self._im_height, self._im_width), (pose_arr.shape[1], )], name='prediction_iterator')

        # predict 
        output_arr = self._model.get_outputs(pred_iter)
        return output_arr[:image_arr.shape[0]]

    def featurize(self, image_arr, pose_arr, feature_layer='conv2_2'):
        """ Featurize a set of images in batches """
        raise NotImplementedError("Featurization of network not yet supported with Neon. Please use Tensorflow backend instead.")
    
    def _build_spatial_transformer(self, input_height, input_width, num_transform_params, output_width, output_height, name):
        raise NotImplementedError("Spatial transformer layer not yet supported with Neon. Please use Tensorflow backend instead.")

    def _build_conv_layer(self, input_height, input_width, filter_h, filter_w, num_filt, pool_stride_h, pool_stride_w, pool_size, name, use_norm=False):
        logging.info('Building convolutional layer: {}'.format(name))

        stride = 1
        out_dim = np.ceil(float(input_height) / float(stride))

        total_pad = max((out_dim - 1) * stride + filter_h - input_height, 0)
        single_side_pad = int(total_pad // 2)

        # build conv layer
        ck = CustomKaiming()
        conv = Conv((filter_h, filter_w, num_filt), init=ck, bias=ck, padding=single_side_pad, activation=Rectlin(), name=name)

        # build norm layer
        norm = None
        if use_norm:
            norm = LRN(depth=self._normalization_radius, ascale=self._normalization_alpha, bpower=self._normalization_beta)

        # build pool layer
        if pool_size == 1 and pool_stride_h == 1:
            pool = Pooling((pool_size, pool_size), strides={'str_h': pool_stride_h, 'str_w': pool_stride_w})
        else:
            pool = Pooling((pool_size, pool_size), strides={'str_h': pool_stride_h, 'str_w': pool_stride_w}, padding=0)

        # create a list of layers
        layers = []
        layers.append(conv)
        if norm is not None:
            layers.append(norm)
        layers.append(pool)

        out_height = input_height / pool_stride_h
        out_width = input_width / pool_stride_w
        out_channels = num_filt

        return layers, out_height, out_width, out_channels

    def _build_fully_conv_layer(self, input_node, filter_dim, fc_name):
        raise NotImplementedError('Fully-Convolutional layer not yet supported with Neon. Please use Tensorflow backend instead.')

    def _build_fully_conv_merge_layer(self, input_node_im, input_node_pose, filter_dim, fc_name):
        raise NotImplementedError('Fully-Convolutional Merge layer not yet supported with Neon. Please use Tensorflow backend instead.')
        
    def _build_fc_layer(self, out_size, name, drop_rate, final_fc_layer=False):
        logging.info('Building fully connected layer: {}'.format(name))
        
        ck = CustomKaiming(local=False)
        if final_fc_layer:
            binit = Constant()
            fc = Affine(nout=out_size, init=ck, bias=binit, activation=Identity(), name=name)
        else:
            fc = Affine(nout=out_size, init=ck, bias=ck, activation=Rectlin(), name=name)
        drop = Dropout(keep=1 - drop_rate)
        
        # create a list of layers
        layers = []
        layers.append(fc)
        layers.append(drop)

        return layers, out_size

    def _build_pc_layer(self, out_size, name):
        logging.info('Building Fully-Connected Pose Layer: {}'.format(name))
        
        ck = CustomKaiming(local=False)
        pc = Affine(nout=out_size, init=ck, bias=ck, activation=Rectlin(), name=name)

        return [pc], out_size

    def _build_gc_layer(self, input_node, fan_in, out_size, name):
        raise NotImplementedError('Gripper-Connected layer not yet supported with Neon. Please use Tensorflow backend instead.')

    def _build_fc_merge(self, input_stream_1, input_stream_2, out_size, drop_rate, name):
        logging.info('Building Merge Layer: {}'.format(name))
        
        layers = []
        layers.append(MergeMultistream(layers=[input_stream_1, input_stream_2], merge="stack"))

        l, out_size = self._build_fc_layer(out_size, name, drop_rate)
        layers.extend(l)

        return layers, out_size

    def _build_batch_norm(self, input_node, ep, drop_rate):
        raise NotImplementedError('Batch Norm layer not yet supported with Neon. Please use Tensorflow backend instead.')

    def _build_residual_layer(self, input_node, input_channels, fan_in, num_filt, filt_h, filt_w, drop_rate, name, first=False):
        raise NotImplementedError('Residual layer not yet supported with Neon. Please use Tensorflow backend instead.')  

    def _build_im_stream(self, input_height, input_width, input_channels, drop_rate, layer_dict):
        logging.info('Building Image Stream')
        layers = []
        prev_layer = "start"
        first_residual = True
        filter_dim = self._train_im_width
        for layer_name, layer_config in layer_dict.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                first_residual = True
                output_node, input_height, input_width, input_channels = self._build_spatial_transformer(input_height, input_width, input_channels,
                    layer_config['num_transform_params'], layer_config['out_size'], layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'conv':
                first_residual = True
                if prev_layer == 'fc':
                    raise ValueError('Cannot have conv layer after fc layer')
                l, input_height, input_width, input_channels = self._build_conv_layer(input_height, input_width, layer_config['filt_dim'],
                    layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, 
                    use_norm=layer_config['norm'])
                layers.extend(l)
                prev_layer = layer_type
                filter_dim /= layer_config['pool_stride']
            elif layer_type == 'fc':
                prev_layer_is_conv_or_res = False
                first_residual = True
                if prev_layer == 'conv' or prev_layer == 'residual':
                    prev_layer_is_conv_or_res = True
                    fan_in = input_height * input_width * input_channels
                if self._fully_conv:
                    output_node = self._build_fully_conv_layer(filter_dim, layer_name)
                else:
                    l, fan_in = self._build_fc_layer(layer_config['out_size'], layer_name, drop_rate)
                    layers.extend(l)
                    prev_layer = layer_type
                    filter_dim = 1
            elif layer_type == 'pc':
                raise ValueError('Cannot have pose-connected layer in image stream')
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in image stream")
            elif layer_type == 'residual':
                # TODO: currently we are assuming the layer before a res layer must be conv layer, fix this
                fan_in = input_height * input_width * input_channels
                if first_residual:
                    output_node, input_channels = self._build_residual_layer(input_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
                        layer_config['filt_dim'], drop_rate, layer_name, first=True)
                    first_residual = False
                else:
                    output_node, input_channels = self._build_residual_layer(input_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
                        layer_config['filt_dim'], drop_rate, layer_name)
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return Sequential(layers), fan_in

    def _build_pose_stream(self, fan_in, layer_dict):
        logging.info('Building Pose Stream')
        layers = []
        prev_layer = "start"
        for layer_name, layer_config in layer_dict.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                raise ValueError('Cannot have spatial transformer in pose stream')
            elif layer_type == 'conv':
               raise ValueError('Cannot have conv layer in pose stream')
            elif layer_type == 'fc':
                raise ValueError('Cannot have fc layer in pose stream')
            elif layer_type == 'pc':
                l, fan_in = self._build_pc_layer(layer_config['out_size'], layer_name)
                layers.extend(l)
                prev_layer = layer_type
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in pose stream")
            elif layer_type == 'residual':
                raise ValueError('Cannot have residual in pose stream')
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return Sequential(layers), fan_in

    def _build_gripper_stream(self, input_node, fan_in, layers):
        raise NotImplementedError('Gripper stream not yet supported with Neon. Please use Tensorflow backend instead.')

    def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, drop_rate, layer_dict, input_stream_3=None, fan_in_3=None):
        logging.info('Building Merge Stream')
        
        # first check if first layer is a merge layer
        if layer_dict[layer_dict.keys()[0]]['type'] != 'fc_merge':
            raise ValueError('First layer in merge stream must be a fc_merge layer')
            
        layers = []
        prev_layer = "start"
        last_index = len(layer_dict.keys()) - 1
        filter_dim = 1
        fan_in = -1
        for layer_index, (layer_name, layer_config) in enumerate(layer_dict.iteritems()):
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                raise ValueError('Cannot have spatial transformer in merge stream')
            elif layer_type == 'conv':
               raise ValueError('Cannot have conv layer in merge stream')
            elif layer_type == 'fc':
                if self._fully_conv:
                    output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name)
                else:
                    if layer_index == last_index:
                        l, fan_in = self._build_fc_layer(layer_config['out_size'], layer_name, drop_rate, final_fc_layer=True)
                    else:
                        l, fan_in = self._build_fc_layer(layer_config['out_size'], layer_name, drop_rate)
                    layers.extend(l)
                prev_layer = layer_type
            elif layer_type == 'pc':  
                raise ValueError('Cannot have pose-connected layer in merge stream')
            elif layer_type == 'fc_merge':
                if self._fully_conv:
                    output_node = self._build_fully_conv_merge_layer(input_stream_1, input_stream_2, filter_dim, layer_name)
                else:
                    if input_stream_3 is not None:
                        output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], drop_rate, layer_name, input_fc_node_3=input_stream_3, fan_in_3=fan_in_3)
                    else:
                        l, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, layer_config['out_size'], drop_rate, layer_name)
                layers.extend(l)
                prev_layer = layer_type   
            elif layer_type == 'residual':
                raise ValueError('Cannot have residual in merge stream')
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return layers, fan_in

    def _build_network(self):
        """ Builds neural network """
        logging.info('Building Network')
        output_im_stream, fan_out_im = self._build_im_stream(self._im_height, self._im_width, self._num_channels, self._drop_rate, self._architecture['im_stream'])
        output_pose_stream, fan_out_pose = self._build_pose_stream(self._pose_dim, self._architecture['pose_stream'])
        input_gripper_node = None
        if input_gripper_node is not None:
            output_gripper_stream, fan_out_gripper = self._build_gripper_stream(input_gripper_node, self._gripper_dim, self._architecture['gripper_stream'])
            if 'gripper_pose_merge_stream' in self._architecture.keys():
                output_gripper_pose_merge_stream, fan_out_gripper_pose_merge_stream = self._build_merge_stream(output_pose_stream, output_gripper_stream, fan_out_pose, fan_out_gripper, self._drop_rate, self._architecture['gripper_pose_merge_stream'])
                return self._build_merge_stream(output_im_stream, output_gripper_pose_merge_stream, fan_out_im, fan_out_gripper_pose_merge_stream, self.drop_rate, self._architecture['merge_stream'])[0]
            else:
                return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, self._drop_rate, self._architecture['merge_stream'], input_stream_3=output_gripper_stream, fan_in_3=fan_out_gripper)[0]
        else:
            return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, self._drop_rate, self._architecture['merge_stream'])[0]
