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

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tcf
from tensorflow.python.client import timeline

from gqcnn.utils.training_utils import TimelineLogger
from gqcnn.utils.data_utils import parse_pose_data, parse_gripper_data
from gqcnn.utils.enums import InputPoseMode, InputGripperMode, TrainingMode
from spatial_transformer import transformer

def reduce_shape(shape):
    """ Get shape of a layer for flattening """
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y
    return reduce(f, shape, 1)

class GQCNNWeights(object):
    """ Struct helper for storing weights """
    def __init__(self):
        self.weights = {}

class GQCNNTF(object):
    """ GQCNN network implemented in Tensorflow """

    def __init__(self, gqcnn_config, fully_conv_config=None):
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
        self._parse_config(gqcnn_config, fully_conv_config)
        self._rot = False
        self._rot_conv_filts = False

    @staticmethod
    def load(model_dir, fully_conv_config=None, conv_filt_rot=0.0):
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
        gqcnn = GQCNNTF(gqcnn_config, fully_conv_config=fully_conv_config)
        gqcnn._rot_conv_filts = False
        gqcnn.init_weights_file(os.path.join(model_dir, 'model.ckpt'), conv_filt_rot=conv_filt_rot)
        gqcnn.init_mean_and_std(model_dir)
        training_mode = train_config['training_mode']
        if training_mode == TrainingMode.CLASSIFICATION:
            gqcnn.initialize_network(add_softmax=True)
        elif training_mode == TrainingMode.REGRESSION:
            gqcnn.initialize_network()
        else:
            raise ValueError('Invalid training mode: {}'.format(training_mode))
        gqcnn.rotate_conv_filters(conv_filt_rot=conv_filt_rot)
#        gqcnn.init_mean_and_std(model_dir)
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
      
        if self._sub_im_depth:
            self.im_depth_sub_mean = np.load(os.path.join(model_dir, 'im_depth_sub_mean.npy'))
            self.im_depth_sub_std = np.load(os.path.join(model_dir, 'im_depth_sub_std.npy')) 

    def init_weights_file(self, ckpt_file, conv_filt_rot=0.0):
        """ Initialize network weights from the specified model 

        Parameters
        ----------
        model_filename :obj: str
            path to model to be loaded into weights
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

    def rotate_conv_filters(self, conv_filt_rot):
        with self._graph.as_default():
            # rotate conv filters
            self._rot_conv_filts = True if conv_filt_rot > 0.0 else False 
            short_names = self._weights.weights.keys()
            if self._rot_conv_filts:
                self._conv_filt_rot = conv_filt_rot
                self.open_session()
                for var_name in short_names:
                    if 'conv' in var_name and 'weights' in var_name and 'fully' not in var_name:
#                    if var_name == 'conv1_1_weights' or var_name == 'conv1_2_weights':
                        logging.info('Rotating weights for {}.'.format(var_name))
                        conv_W = self._weights.weights[var_name]
                        conv_W_np = self._sess.run(conv_W)
                        conv_W_np_orig = np.copy(conv_W_np)
                        rot_mat = cv.getRotationMatrix2D((float(conv_W_np.shape[1] - 1) / 2, float(conv_W_np.shape[0] - 1) / 2), conv_filt_rot, 1.0)
                        for i in range(conv_W_np.shape[2]):
                            for j in range(conv_W_np.shape[3]):
#                                conv_W_np[:, :, i, j] = np.copy(np.rot90(conv_W_np[:, :, i, j], k=1))
                                conv_W_np[:, :, i, j] = cv.warpAffine(conv_W_np[:, :, i, j], rot_mat, (conv_W_np.shape[1], conv_W_np.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
#                        import matplotlib.pyplot as plt
#                        plt.figure()
#                        for x in range(2):
#                             plt.clf()
#                             plt.subplot(121)
#                             plt.imshow(conv_W_np_orig[:, :, 0, x], cmap='gray')
#                             plt.subplot(122)
#                             plt.imshow(conv_W_np[:, :, 0, x], cmap='gray')
#                             plt.show()
                        self._weights.weights[var_name] = tf.Variable(conv_W_np)
                self.close_session()
#                self._rot = True
                self.initialize_network(add_softmax=True)

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
    
    def _parse_config(self, gqcnn_config, fully_conv_config):
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
            raise ValueError('Input pose mode %s not understood' %(self._input_pose_mode))

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
            raise ValueError('Input gripper mode %s not understood' %(self._input_gripper_mode))

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

        # create empty holder for feature handles
        self._feature_tensors = {}
    
        # initialize other misc parameters
        self._summary_writer = None
        self._mask_and_inpaint = False
        self._save_histograms = False
        self._angular_bins = gqcnn_config['angular_bins']

        self._sub_im_depth = False
        if 'sub_im_depth' in gqcnn_config.keys():
            self._sub_im_depth = gqcnn_config['sub_im_depth']
        self._norm_inputs = True
        if 'normalize_inputs' in gqcnn_config.keys():
            self._norm_inputs = gqcnn_config['normalize_inputs']
        self.sub_lambda = 1.0
        if 'sub_lambda' in gqcnn_config.keys():
            self.sub_lambda = gqcnn_config['sub_lambda']

        self._fully_conv = False
        if fully_conv_config:
            self._fully_conv = True
        ##################### PARSING FULLY CONVOLUTIONAL CONFIG #####################
        if self._fully_conv:
            # override the im_width and im_height with those from the fully_conv_config
            self._im_width = fully_conv_config['im_width']
            self._im_height = fully_conv_config['im_height']

    def initialize_network(self, train_im_node=None, train_pose_node=None, train_gripper_node=None, add_softmax=False):
        """ Set up input placeholders and build network.

        Parameters
        ----------
        add_softmax : float
            whether or not to add a softmax layer to output of network
        """
        with self._graph.as_default():
            # setup input placeholders
            if train_im_node is not None:
                self._input_im_node = tf.placeholder_with_default(train_im_node, (None, self._im_height, self._im_width, self._num_channels))
                self._input_pose_node = tf.placeholder_with_default(train_pose_node, (None, self._pose_dim))
                if self._gripper_dim > 0:
                    self._input_gripper_node = tf.placeholder(train_gripper_node, (None, self._gripper_dim))
            else:
                self._input_im_node = tf.placeholder(tf.float32, (self._batch_size, self._im_height, self._im_width, self._num_channels))
                self._input_pose_node = tf.placeholder(tf.float32, (self._batch_size, self._pose_dim))
                if self._gripper_dim > 0:
                    self._input_gripper_dim = tf.placeholder(train_gripper_node, (None, self._gripper_dim))
            self._input_drop_rate_node = tf.placeholder_with_default(tf.constant(0.0), ())
            self._input_distort_rot_ang_node = tf.placeholder_with_default(tf.constant(np.zeros((self._batch_size,)), dtype=tf.float32), (None,))

            # build network
            if self._gripper_dim > 0:
                self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, self._input_drop_rate_node, input_gripper_node=self._input_gripper_node)
            else:
                self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, self._input_drop_rate_node, self._input_distort_rot_ang_node)
            
            # add softmax function to output of network if specified
            if add_softmax:
                self.add_softmax_to_output()

        # create feed tensors for prediction
        self._input_im_arr = np.zeros((self._batch_size, self._im_height, self._im_width, self._num_channels))
        self._input_pose_arr = np.zeros((self._batch_size, self._pose_dim))
        if self._gripper_dim > 0:
            self._input_gripper_arr = np.zeros((self._batch_size, self._gripper_dim))

    def open_session(self):
        """ Open tensorflow session """
        logging.info('Initializig TF Session.')
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            self.tf_config = tf.ConfigProto()
            # allow tf gpu_growth so tf does not lock-up all GPU memory
            self.tf_config.gpu_options.allow_growth = True
            self._sess = tf.Session(graph=self._graph, config=self.tf_config)
            self._sess.run(init)
            
            # setup tf run options and metadata, used when profiling
            self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._run_metadata = tf.RunMetadata()
        return self._sess

    def close_session(self):
        """ Close tensorflow session """
        logging.info('Closing TF Session.')
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
    def input_pose_mode(self):
        return self._input_pose_mode

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
    def graph(self):
        return self._graph
    
    @property
    def sess(self):
        return self._sess
    
    def set_mask_and_inpaint(self, mask_and_inpaint):
        self._mask_and_inpaint = mask_and_inpaint
    
    def set_summary_writer(self, summary_writer):
        self._summary_writer = summary_writer

    def set_save_histograms(self, save_histograms):
        self._save_histograms = save_histograms

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
        
    def add_softmax_to_output(self):
        """ Adds softmax to output of network """
        with tf.name_scope('softmax'):
            if self._angular_bins > 0:
                logging.info('Building Pair-wise Softmax Layer')
                binwise_split_output = tf.split(self._output_tensor, self._angular_bins, axis=-1)
                binwise_split_output_soft = [tf.nn.softmax(s) for s in binwise_split_output]
                self._output_tensor = tf.concat(binwise_split_output_soft, -1)
            else:
                logging.info('Building Softmax Layer')
                self._output_tensor = tf.nn.softmax(self._output_tensor)

    def update_batch_size(self, batch_size):
        """ Updates the prediction batch size 

        Parameters
        ----------
        batch_size : float
            batch size to be used for prediction
        """
        self._batch_size = batch_size

    def _predict_optimized(self, image_arr, pose_arr, unique_im_map, timeline_save_file=None, max_timeline_updates=100, verbose=False):
        # setup TimelineLogger for run profiling if timeline_save_dir is not None
        log_timeline = False
        if timeline_save_file is not None:
            if verbose:
                logging.info('Profiling prediction using TimelineLogger')
            log_timeline = True
            timeline_save_dir = '/'.join(timeline_save_file.split('/')[:-1])
            timeline_save_fname = timeline_save_file.split('/')[-1]
            timeline_logger = TimelineLogger(timeline_save_dir)
        
        # get prediction start time
        timeline_update_time = 0
        start_time = time.time()

        if verbose:
            logging.info('Predicting...')

        # generate unique image map
        assert image_arr.shape[0] == unique_im_map.shape[0], 'Unique Image Map dim0 != image_arr dim0'
        if verbose:
            logging.info('Found unique image mapping, performing runtime optimizations')
            logging.info('Generating index map...')
        
#        map_start_time = time.time()
        arg_sorted_unique_im_map = np.argsort(unique_im_map)
        sorted_unique_im_map = unique_im_map[arg_sorted_unique_im_map]
        unique_vals, unique_start_ind = np.unique(sorted_unique_im_map, return_index=True)
        unique_im_idx_map = np.split(arg_sorted_unique_im_map, unique_start_ind[1:])
#        logging.info('Total map time: {}'.format(time.time() - map_start_time))        

        if verbose:
            logging.info('Found {} unique images'.format(unique_vals.shape[0]))

        # setup for prediction
        reset_metadata_and_options = False
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]
        output_arr = None
        input_feat_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images and poses')

        # first predict outputs of final conv layer for unique images
        unique_ims = image_arr[np.asarray([ind_group[0] for ind_group in unique_im_idx_map])]
#        logging.info('Unique ims shape {}'.format(unique_ims.shape))
#        feat_start_time = time.time()
        unique_conv_feat_arr = self.featurize(unique_ims, feature_layer=self._final_conv_name)
#        logging.info('Feat time: {}'.format(time.time() - feat_start_time))

        # broadcast unique image final conv features for all images
#        broad_start_time = time.time()
        broad_ind = np.zeros((num_images,), dtype=np.int32)
        for i, ind_group in enumerate(unique_im_idx_map):
            for idx in ind_group:
                broad_ind[idx] = i
        conv_feat_arr = unique_conv_feat_arr[broad_ind]
#        logging.info('Broad time: {}'.format(time.time() - broad_start_time))

        # next predict using final conv layer features and depths
        with self._graph.as_default():
            if self._sess is None:
               raise RuntimeError('No TF session open. Please call open_session() first.')
            i = 0
            batch_idx = 0
            while i < num_images:
                if verbose:
                    logging.info('Predicting batch {} of {}'.format(batch_idx, num_batches))
                batch_idx += 1
                if batch_idx > max_timeline_updates:
                    self._run_metadata = None
                    self._run_options = None
                    reset_metadata_and_options = True
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim

                if input_feat_arr is None:
#                    feat_arr_init_start_time = time.time()
                    input_feat_arr = np.zeros((self._batch_size,) + conv_feat_arr.shape[1:])
#                    logging.info('Feat arr init time: {}'.format(time.time() - feat_arr_init_start_time))
                
                input_feat_arr[:dim, ...] = conv_feat_arr[cur_ind:end_ind, ...]

                self._input_pose_arr[:dim, :] = (
                    pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

                gqcnn_output = self._sess.run(self._output_tensor,
                                                      feed_dict={self._final_conv_placeholder: input_feat_arr,
                                                                 self._input_pose_node: self._input_pose_arr},
                                                      options=self._run_options,
                                                      run_metadata=self._run_metadata)

                # update timeline for profiling if needed
                if log_timeline:
                    if verbose:
                        logging.info('Updating Timelinelogger')
                    if batch_idx < max_timeline_updates:
                        start_timeline_update_time = time.time()
                        timeline_logger.update_timeline(timeline.Timeline(self._run_metadata.step_stats).generate_chrome_trace_format())
                        timeline_update_time += time.time() - start_timeline_update_time
                    else:
                       if verbose:
                           logging.info('Skipping timeline update because max timeline update cap reached')

                # allocate output tensor if needed
                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))

                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]
                i = end_ind
        
        # get total prediction time
        pred_time = time.time() - start_time - timeline_update_time

        # save timeline for profiling if needed
        if log_timeline:
            if verbose:
                logging.info('Saving timeline')
            timeline_logger.save(timeline_save_fname)

        if reset_metadata_and_options:
            self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._run_metadata = tf.RunMetadata()  

        if log_timeline:
            return output_arr, pred_time
        else:
            return output_arr
 
    def _predict(self, image_arr, pose_arr, gripper_arr=None, gripper_depth_mask=False, timeline_save_file=None, max_timeline_updates=100, verbose=False):
         # setup TimelineLogger for run profiling if timeline_save_dir is not None
        log_timeline = False
        if timeline_save_file is not None:
            if verbose:
                logging.info('Profiling prediction using TimelineLogger')
            log_timeline = True
            timeline_save_dir = '/'.join(timeline_save_file.split('/')[:-1])
            timeline_save_fname = timeline_save_file.split('/')[-1]
            timeline_logger = TimelineLogger(timeline_save_dir)
        
        # get prediction start time
        timeline_update_time = 0
        start_time = time.time()

        if verbose:
            logging.info('Predicting...')

        # setup for prediction
        reset_metadata_and_options = False
        num_batches = math.ceil(image_arr.shape[0] / self._batch_size)
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]
        if gripper_arr is not None:
            num_gripper_parameters = gripper_arr.shape[0]

        output_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images and poses')
        if gripper_arr is not None:
            if num_images != num_gripper_parameters:
                raise ValueError('Must provide same number of images and gripper parameters')

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
                if batch_idx > max_timeline_updates:
                    self._run_metadata = None
                    self._run_options = None
                    reset_metadata_and_options = True
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim
                
                if self._norm_inputs:
                    self._input_im_arr[:dim, ...] = (
                        image_arr[cur_ind:end_ind, ...] - self._im_mean) / self._im_std 
                
                    self._input_pose_arr[:dim, :] = (
                        pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std
                else:
                    self._input_im_arr[:dim, ...] = image_arr[cur_ind:end_ind, ...]
                    self._input_pose_arr[:dim, :] = pose_arr[cur_ind:end_ind, :] 

                if gripper_arr is not None:
                    self._input_gripper_arr[:dim, :] = (
                        gripper_arr[cur_ind:end_ind, :] - self._gripper_mean) / self._gripper_std    

                if gripper_depth_mask:
                    self._input_im_arr[:dim, :, :, 1] = (
                        image_arr[cur_ind:end_ind, :, :, 1] - self._gripper_depth_mask_mean[0]) / self._gripper_depth_mask_std[0]
                    self._input_im_arr[:dim, :, :, 2] = (
                        image_arr[cur_ind:end_ind, :, :, 2] - self._gripper_depth_mask_mean[1]) / self._gripper_depth_mask_std[1]               

                if gripper_arr is not None:
                    gqcnn_output = self._sess.run(self._output_tensor,
                                                      feed_dict={self._input_im_node: self._input_im_arr,
                                                                 self._input_pose_node: self._input_pose_arr,
                                                                 self._input_gripper_node: self._input_gripper_arr},
                                                      options=self._run_options,
                                                      run_metadata=self._run_metadata)
                else:
                    gqcnn_output = self._sess.run(self._output_tensor,
                                                      feed_dict={self._input_im_node: self._input_im_arr,
                                                                 self._input_pose_node: self._input_pose_arr},
                                                      options=self._run_options,
                                                      run_metadata=self._run_metadata)

                # update timeline for profiling if needed
                if log_timeline:
                    if verbose:
                        logging.info('Updating Timelinelogger')
                    if batch_idx < max_timeline_updates:
                        start_timeline_update_time = time.time()
                        timeline_logger.update_timeline(timeline.Timeline(self._run_metadata.step_stats).generate_chrome_trace_format())
                        timeline_update_time += time.time() - start_timeline_update_time
                    else:
                       if verbose:
                           logging.info('Skipping timeline update because max timeline update cap reached')

                # allocate output tensor if needed
                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))

                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]
                i = end_ind
        
        # get total prediction time
        pred_time = time.time() - start_time - timeline_update_time

        # save timeline for profiling if needed
        if log_timeline:
            if verbose:
                logging.info('Saving timeline')
            timeline_logger.save(timeline_save_fname)

        if reset_metadata_and_options:
            self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self._run_metadata = tf.RunMetadata()  

        if log_timeline:
            return output_arr, pred_time
        else:
            return output_arr

    def predict(self, image_arr, pose_arr, gripper_arr=None, gripper_depth_mask=False, timeline_save_file=None, max_timeline_updates=100, unique_im_map=None, verbose=False):
        """ 
        Predict the probability of grasp success given a depth image, gripper pose, and
        optionally gripper parameters 

        Parameters
        ----------
        image_arr : :obj:`numpy ndarray`
            4D Tensor of depth images
        pose_arr : :obj:`numpy ndarray`
            Tensor of gripper poses
        gripper_arr : :obj:`numpy ndarray`
            optional Tensor of gripper parameters, if None will not be used for prediction
        """
        if unique_im_map is not None:
            return self._predict_optimized(image_arr, pose_arr, unique_im_map, timeline_save_file=timeline_save_file, max_timeline_updates=max_timeline_updates, verbose=verbose)       
        else:
            return self._predict(image_arr, pose_arr, gripper_arr=gripper_arr, gripper_depth_mask=gripper_depth_mask, timeline_save_file=timeline_save_file, max_timeline_updates=max_timeline_updates, verbose=verbose)
   
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
            if close_sess:
                self.close_session()

        # truncate extraneous values off of end of output_arr
        output_arr = output_arr[:num_images]
        return output_arr
    
    def _leaky_relu(self, x, alpha=.1):
        return tf.maximum(alpha * x, x)
    
    def _build_spatial_transformer(self, input_node, input_height, input_width, input_channels, num_transform_params, output_width, output_height, name):
        logging.info('Building spatial transformer layer: {}'.format(name))
        
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            transformW = self._weights.weights['{}_weights'.format(name)]
            transformb = self._weights.weights['{}_bias'.format(name)]
        else:
            transformW = tf.Variable(tf.zeros([input_height * input_width * input_channels, num_transform_params]), name='{}_weights'.format(name))

            initial = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
            transformb = tf.Variable(initial_value=initial, name='{}_bias'.format(name))
            
            self._weights.weights['{}_weights'.format(name)] = transformW
            self._weights.weights['{}_bias'.format(name)] = transformb

        # build localisation network
#        loc_network = tf.matmul(tf.zeros([64, input_height * input_width * input_channels]), transformW) + transformb
        orig_input_channels = input_channels
        loc_network, input_height, input_width, input_channels = self._build_conv_layer(input_node, self._input_distort_rot_ang_node, input_height, input_width, input_channels, 7, 7, 32, 1, 1, 1, 'loc_conv1')
        loc_network, input_height, input_width, input_channels = self._build_conv_layer(loc_network, self._input_distort_rot_ang_node, input_height, input_width, input_channels, 5, 5, 32, 2, 2, 2, 'loc_conv2')
        loc_fc_1_W = tf.Variable(tf.zeros([input_height * input_width * input_channels, 256]))
        loc_fc_1_b = tf.Variable(tf.zeros([256]))
        loc_network = tf.matmul(tf.reshape(loc_network, [-1, reduce_shape(loc_network.get_shape())]), loc_fc_1_W) + loc_fc_1_b
        
        loc_fc_2_W = tf.Variable(tf.zeros([256, 6]))
        loc_fc_2_b = tf.Variable(np.array([[1.0, 0, 0], [0, 1.0, 0]]).astype('float32').flatten())
        loc_network = tf.matmul(loc_network, loc_fc_2_W) + loc_fc_2_b

        # build transform layer
        transform_layer = transformer(input_node, loc_network, (output_width, output_height))

        # add output to feature dict
        self._feature_tensors[name] = transform_layer

        return transform_layer, output_height, output_width, orig_input_channels

    def _build_conv_layer(self, input_node, input_distort_rot_ang_node, input_height, input_width, input_channels, filter_h, filter_w, num_filt, pool_stride_h, pool_stride_w, pool_size, name, norm=False, pad='SAME', last_conv=False):
        logging.info('Building convolutional layer: {}'.format(name))       
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
            
#            if not inference:
#                if self._save_histograms:
#                    tf.summary.histogram('weights', convW, collections=["histogram"])
#                    tf.summary.histogram('bias', convb, collections=["histogram"])

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
            
#            if not inference:
#                if self._save_histograms:
#                    tf.summary.histogram('layer_raw', convh, collections=["histogram"])

            convh = self._leaky_relu(convh)
            
#            if not inference:
#                if self._save_histograms:
#                    tf.summary.histogram('layer_act', convh, collections=["histogram"])

            if norm:
                convh = tf.nn.local_response_normalization(convh,
                                                            depth_radius=self._normalization_radius,
                                                            alpha=self._normalization_alpha,
                                                            beta=self._normalization_beta,
                                                            bias=self._normalization_bias)
#            if not inference:
#                if self._save_histograms:
#                    tf.summary.histogram('layer_norm', convh, collections=["histogram"])

            pool = tf.nn.max_pool(convh,
                                ksize=[1, pool_size, pool_size, 1],
                                strides=[1, pool_stride_h,
                                        pool_stride_w, 1],
                                padding='SAME')
            
#            if not inference:
#                if self._save_histograms:
#                    tf.summary.histogram('layer_pool', pool, collections=["histogram"])     

            if self._rot_conv_filts and name == 'conv2_2':
                pool = tf.contrib.image.rotate(pool, -1 * self._conv_filt_rot * math.pi / 180.0)
            
#            if name == 'conv2_2':
#                rot_pool = tf.contrib.image.rotate(pool, input_distort_rot_ang_node * math.pi / 180)
#                pool = tf.contrib.image.rotate(rot_pool, -1 * input_distort_rot_ang_node * math.pi / 180)          
                
            # add output to feature dict
            self._feature_tensors[name] = pool

            if last_conv:
                self._final_conv_name = name
                logging.info('Building placeholder for final conv layer')
                self._final_conv_placeholder = tf.placeholder_with_default(pool, pool.get_shape())
                pool = self._final_conv_placeholder

            return pool, out_height, out_width, out_channels

    def _pack(self, dim_h, dim_w, data, vector=False):
        if vector:
            # first reshape vector into 3-dimensional tensor
            reshaped = tf.reshape(data, tf.concat([[1, 1], tf.shape(data)], 0))
         
            # then tile into tensor of shape dimxdimxdata.dim0
            packed = tf.tile(reshaped, [dim_h, dim_w, 1])
        else:
            # first reshape second dimension of tensor into 3-dimensional tensor
            reshaped = tf.reshape(data, tf.concat([tf.shape(data)[0:1], [1, 1], tf.shape(data)[1:]], 0))

            # then tile into tensor of shape bsizexdimxdimxdata.dim1
            packed = tf.tile(reshaped, [1, dim_h, dim_w, 1])

        return packed

    def _build_fully_conv_layer(self, input_node, filter_dim, fc_name, final_fc_layer=False):
        logging.info('Converting fc layer: {} to fully convolutional'.format(fc_name))
        
        if '{}_fully_conv_weights'.format(fc_name) in self._weights.weights.keys() and self._rot:
            convW = self._weights.weights['{}_fully_conv_weights'.format(fc_name)]
        else:
            # create new set of weights
            fcW = self._weights.weights['{}_weights'.format(fc_name)]
            convW = tf.Variable(tf.reshape(fcW, tf.concat([[filter_dim, filter_dim], [tf.shape(fcW)[0] / (filter_dim * filter_dim)], tf.shape(fcW)[1:]], 0)), name='{}_fully_conv_weights'.format(fc_name))
            self._weights.weights['{}_fully_conv_weights'.format(fc_name)] = convW
        convb = self._weights.weights['{}_bias'.format(fc_name)]

        # compute conv out(note that we use padding='VALID' here because we want and output size of 1x1xnum_filts for the original input size)
        convh = tf.nn.conv2d(input_node, convW, strides=[1, 1, 1, 1], padding='VALID')

        # pack bias into tensor of shape=tf.shape(convh)
        bias_packed = self._pack(tf.shape(convh)[1], tf.shape(convh)[2], convb, vector=True)

        # add bias term
        convh = convh + bias_packed

        # apply activation
        if not final_fc_layer:
            convh = self._leaky_relu(convh)

        # add output to feature_dict
        self._feature_tensors[fc_name] = convh

        return convh

    def _build_fully_conv_merge_layer(self, input_node_im, input_node_pose, filter_dim, fc_name):
        logging.info('Converting fc merge layer: {} to fully convolutional'.format(fc_name))

        # create fully convolutional layer for image stream
        if '{}_im_fully_conv_weights'.format(fc_name) in self._weights.weights.keys() and self._rot:
            convW = self._weights.weights['{}_im_fully_conv_weights'.format(fc_name)]
        else: 
            fcW_im = self._weights.weights['{}_input_1_weights'.format(fc_name)]
            convW = tf.Variable(tf.reshape(fcW_im, tf.concat([[filter_dim, filter_dim], [tf.shape(fcW_im)[0] / (filter_dim * filter_dim)], tf.shape(fcW_im)[1:]], 0)), name='{}_im_fully_conv_weights'.format(fc_name))
            self._weights.weights['{}_im_fully_conv_weights'.format(fc_name)] = convW
        convh_im = tf.nn.conv2d(input_node_im, convW, strides=[1, 1, 1, 1], padding='VALID')

        if not self._sub_im_depth:
            # compute matmul for pose stream
            fcW_pose = self._weights.weights['{}_input_2_weights'.format(fc_name)]
            pose_out = tf.matmul(input_node_pose, fcW_pose)

            # pack pose_out into a tensor of shape=tf.shape(convh_im)
            pose_packed = self._pack(tf.shape(convh_im)[1], tf.shape(convh_im)[2], pose_out)

            # add the im and pose tensors 
            convh = convh_im + pose_packed
        else:
            convh = convh_im

        # pack bias
        fc_bias = self._weights.weights['{}_bias'.format(fc_name)]
        bias_packed = self._pack(tf.shape(convh_im)[1], tf.shape(convh_im)[2], fc_bias, vector=True)

        # add bias and apply activation
        convh = self._leaky_relu(convh + bias_packed)

        return convh
        
    def _build_fc_layer(self, input_node, fan_in, out_size, name, input_is_multi, drop_rate, final_fc_layer=False):
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
        if input_is_multi:
            input_num_nodes = reduce_shape(input_node.get_shape())
            input_node = tf.reshape(input_node, [-1, input_num_nodes])
        if final_fc_layer:
            fc = tf.matmul(input_node, fcW) + fcb
        else:
            fc = self._leaky_relu(tf.matmul(input_node, fcW) + fcb)

        fc = tf.nn.dropout(fc, 1 - drop_rate)

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
        pc = self._leaky_relu(tf.matmul(input_node, pcW) +
                        pcb)

        # add output to feature dict
        self._feature_tensors[name] = pc

        return pc, out_size

    def _build_gc_layer(self, input_node, fan_in, out_size, name):
        logging.info('Building Fully-Connected Gripper Layer: {}'.format(name))
        
        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            gcW = self._weights.weights['{}_weights'.format(name)]
            gcb = self._weights.weights['{}_bias'.format(name)] 
        else:
            std = np.sqrt(2.0 / (fan_in))

            gcW = tf.Variable(tf.truncated_normal([fan_in, out_size],
                                               stddev=std), name='{}_weights'.format(name))
            gcb = tf.Variable(tf.truncated_normal([out_size],
                                               stddev=std), name='{}_bias'.format(name))

            self._weights.weights['{}_weights'.format(name)] = gcW
            self._weights.weights['{}_bias'.format(name)] = gcb

        # build layer
        gc = self._leaky_relu(tf.matmul(input_node, gcW) +
                        gcb)

        # add output to feature dict
        self._feature_tensors[name] = gc

        return gc, out_size

    def _build_fc_merge(self, input_fc_node_1, input_fc_node_2, fan_in_1, fan_in_2, out_size, drop_rate, name, input_fc_node_3=None, fan_in_3=None):
        logging.info('Building Merge Layer: {}'.format(name))
        
        if input_fc_node_3 is not None:
            # initialize weights
            if '{}_input_1_weights'.format(name) in self._weights.weights.keys():
                input1W = self._weights.weights['{}_input_1_weights'.format(name)]
                input2W = self._weights.weights['{}_input_2_weights'.format(name)]
                input3W = self._weights.weights['{}_input_3_weights'.format(name)]
                fcb = self._weights.weights['{}_bias'.format(name)] 
            else:
                std = np.sqrt(2.0 / (fan_in_1 + fan_in_2 + fan_in_3))
                input1W = tf.Variable(tf.truncated_normal([fan_in_1, out_size], stddev=std), name='{}_input_1_weights'.format(name))
                input2W = tf.Variable(tf.truncated_normal([fan_in_2, out_size], stddev=std), name='{}_input_2_weights'.format(name))
                input3W = tf.Variable(tf.truncated_normal([fan_in_3, out_size], stddev=std), name='{}_input_3_weights'.format(name))
                fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))

                self._weights.weights['{}_input_1_weights'.format(name)] = input1W
                self._weights.weights['{}_input_2_weights'.format(name)] = input2W
                self._weights.weights['{}_input_3_weights'.format(name)] = input3W
                self._weights.weights['{}_bias'.format(name)] = fcb

            # build layer
            fc = self._leaky_relu(tf.matmul(input_fc_node_1, input1W) +
                                    tf.matmul(input_fc_node_2, input2W) +
                                    tf.matmul(input_fc_node_3, input3W) +
                                    fcb)
        else:
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
            if not self._sub_im_depth:
                fc = self._leaky_relu(tf.matmul(input_fc_node_1, input1W) +
                                    tf.matmul(input_fc_node_2, input2W) +
                                    fcb)
            else:
                fc = self._leaky_relu(tf.matmul(input_fc_node_1, input1W) + fcb)
        fc = tf.nn.dropout(fc, 1 - drop_rate)

        # add output to feature dict
        self._feature_tensors[name] = fc

        return fc, out_size

    def _build_batch_norm(self, input_node, ep, drop_rate):
        output_node = input_node
        output_node = tf.layers.batch_normalization(output_node, training=tf.cast(drop_rate, dtype=tf.bool), epsilon=ep)
        return output_node

    def _build_residual_layer(self, input_node, input_channels, fan_in, num_filt, filt_h, filt_w, drop_rate, name, first=False):
        logging.info('Building Residual Layer: {}'.format(name))
        if '{}_conv1_weights'.format(name) in self._weights.weights.keys():
            conv1W = self._weights.weights['{}_conv1_weights'.format(name)]
            conv1b = self._weights.weights['{}_conv1_bias'.format(name)]
            conv2W = self._weights.weights['{}_conv2_weights'.format(name)]
            conv2b = self._weights.weights['{}_conv2_bias'.format(name)] 
        else:
            std = np.sqrt(2.0 / fan_in)
            conv1_shape = [filt_h, filt_w, input_channels, num_filt]
            conv2_shape = [filt_h, filt_w, num_filt, num_filt]
            conv1W = tf.Variable(tf.truncated_normal(conv1_shape, stddev=std), name='{}_conv1_weights'.format(name))
            conv1b = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_conv1_bias'.format(name))
            conv2W = tf.Variable(tf.truncated_normal(conv2_shape, stddev=std), name='{}_conv2_weights'.format(name))
            conv2b = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_conv2_bias'.format(name))


            self._weights.weights['{}_conv1_weights'.format(name)] = conv1W
            self._weights.weights['{}_conv1_bias'.format(name)] = conv1b
            self._weights.weights['{}_conv2_weights'.format(name)] = conv2W
            self._weights.weights['{}_conv2_bias'.format(name)] = conv2b

        #  implemented as x = BN + ReLU + Conv + BN + ReLU + Conv
        EP = .001
        output_node = input_node
        if not first:
            output_node = self._build_batch_norm(output_node, EP, drop_rate)
            output_node = self._leaky_relu(output_node)
        output_node = tf.nn.conv2d(output_node, conv1W, strides=[1, 1, 1, 1], padding='SAME') + conv1b
        output_node = self._build_batch_norm(output_node, EP, drop_rate)
        output_node = self._leaky_relu(output_node)
        output_node = tf.nn.conv2d(output_node, conv2W, strides=[1, 1, 1, 1], padding='SAME') + conv2b
        output_node = input_node + output_node

        # add output to feature dict
        self._feature_tensors[name] = output_node

        return output_node, num_filt  


    def _build_im_stream(self, input_node, input_pose_node, input_height, input_width, input_channels, drop_rate, input_distort_rot_ang_node, layers):
        logging.info('Building Image Stream')

        if self._sub_im_depth:
            sub_mean = tf.constant(self.im_depth_sub_mean, dtype=tf.float32)
            sub_std = tf.constant(self.im_depth_sub_std, dtype=tf.float32)
            sub_lambda = tf.constant(self.sub_lambda, dtype=tf.float32)
            orig_sub_im = tf.subtract(input_node, tf.tile(tf.reshape(input_pose_node, tf.constant((-1, 1, 1, 1))), tf.constant((1, input_height, input_width, 1))))
            self.orig_sub_im = orig_sub_im
            lambda_sub_im = tf.multiply(sub_lambda, orig_sub_im)
            self.lambda_sub_im = lambda_sub_im
            norm_sub_im = tf.div(tf.subtract(lambda_sub_im, sub_mean), sub_std)
            self.norm_sub_im = norm_sub_im
            input_node = norm_sub_im

#             input_node = tf.concat([input_node, tf.multiply(tf.tile(tf.reshape(input_pose_node, tf.constant((-1, 1, 1, 1))), tf.constant((1, 46, 46, 1))), tf.constant(1.75, dtype=tf.float32))], axis=3)            

#            orig_im = tf.add(tf.multiply(input_node, tf.constant(self._im_std, dtype=tf.float32)), tf.constant(self._im_mean, dtype=tf.float32))
#            orig_pose = tf.add(tf.multiply(input_pose_node, tf.constant(self._pose_std, dtype=tf.float32)), tf.constant(self._pose_mean, dtype=tf.float32))
#            self._orig_im = orig_im
#            self._orig_pose = orig_pose
#            mask = tf.subtract(orig_im, tf.tile(tf.reshape(orig_pose, tf.constant((-1, 1, 1, 1))), tf.constant((1, 46, 46, 1))))
#            input_node = tf.concat([input_node, mask], axis=3)
#            self._sub_im_depth_out = input_node

        output_node = input_node
        prev_layer = "start"
        first_residual = True
        filter_dim = self._train_im_width
        layer_idx = 0
        for layer_name, layer_config in layers.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                first_residual = True
                output_node, input_height, input_width, input_channels = self._build_spatial_transformer(output_node, input_height, input_width, input_channels,
                    layer_config['num_transform_params'], layer_config['out_size'], layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'conv':
                first_residual = True
                if prev_layer == 'fc':
                    raise ValueError('Cannot have conv layer after fc layer')
                if layers[layers.keys()[layer_idx + 1]]['type'] != 'conv':
                    output_node, input_height, input_width, input_channels = self._build_conv_layer(output_node, input_distort_rot_ang_node, input_height, input_width, input_channels, layer_config['filt_dim'], layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, norm=layer_config['norm'], pad=layer_config['pad'], last_conv=True)
                else:
                    output_node, input_height, input_width, input_channels = self._build_conv_layer(output_node, input_distort_rot_ang_node, input_height, input_width, input_channels, layer_config['filt_dim'],
                    layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, 
                    norm=layer_config['norm'], pad=layer_config['pad'])
                prev_layer = layer_type
                if layer_config['pad'] == 'SAME':
                    filter_dim /= layer_config['pool_stride']
                else:
                    filter_dim = ((filter_dim - layer_config['filt_dim']) / layer_config['pool_stride']) + 1
            elif layer_type == 'fc':
                prev_layer_is_conv_or_res = False
                first_residual = True
                if prev_layer == 'conv' or prev_layer == 'residual':
                    prev_layer_is_conv_or_res = True
                    fan_in = input_height * input_width * input_channels
                if self._fully_conv:
                    output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name)
                else:
                    output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv_or_res, drop_rate)
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
                    output_node, input_channels = self._build_residual_layer(output_node, input_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
                        layer_config['filt_dim'], drop_rate, layer_name, first=True)
                    first_residual = False
                else:
                    output_node, input_channels = self._build_residual_layer(output_node, input_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
                        layer_config['filt_dim'], drop_rate, layer_name)
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
            layer_idx += 1

        return output_node, fan_in

    def _build_pose_stream(self, input_node, fan_in, layers):
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

    def _build_gripper_stream(self, input_node, fan_in, layers):
        logging.info('Building Gripper Stream')
        output_node = input_node
        prev_layer = "start"
        for layer_name, layer_config in layers.iteritems():
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                raise ValueError('Cannot have spatial transformer in gripper stream')
            elif layer_type == 'conv':
               raise ValueError('Cannot have conv layer in gripper stream')
            elif layer_type == 'fc':
                raise ValueError('Cannot have fc layer in gripper stream')
            elif layer_type == 'pc':
                raise ValueError('Cannot have pc layer in gripper stream')
            elif layer_type == 'gc':
                output_node, fan_in = self._build_gc_layer(output_node, fan_in, layer_config['out_size'], layer_name)
                prev_layer = layer_type
            elif layer_type == 'fc_merge':
                raise ValueError("Cannot have merge layer in gripper stream")
            elif layer_type == 'residual':
                raise ValueError('Cannot have residual in gripper stream')
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))

        return output_node, fan_in

    def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, drop_rate, layers, input_stream_3=None, fan_in_3=None):
        logging.info('Building Merge Stream')
        
        # first check if first layer is a merge layer
        if layers[layers.keys()[0]]['type'] != 'fc_merge':
            raise ValueError('First layer in merge stream must be a fc_merge layer')
            
        prev_layer = "start"
        last_index = len(layers.keys()) - 1
        filter_dim = 1
        fan_in = -1
        for layer_index, (layer_name, layer_config) in enumerate(layers.iteritems()):
            layer_type = layer_config['type']
            if layer_type == 'spatial_transformer':
                raise ValueError('Cannot have spatial transformer in merge stream')
            elif layer_type == 'conv':
               raise ValueError('Cannot have conv layer in merge stream')
            elif layer_type == 'fc':
                if self._fully_conv:
                    if layer_index == last_index:
                        output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name, final_fc_layer=True)
                    else:
                        output_node = self._build_fully_conv_layer(output_node, filter_dim, layer_name)
                else:
                    if layer_index == last_index:
                        output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate, final_fc_layer=True)
                    else:
                        output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, False, drop_rate)
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
                        output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], drop_rate, layer_name)
                prev_layer = layer_type   
            elif layer_type == 'residual':
                raise ValueError('Cannot have residual in merge stream')
                prev_layer = layer_type
            else:
                raise ValueError("Unsupported layer type: {}".format(layer_type))
        return output_node, fan_in

    def _build_network(self, input_im_node, input_pose_node, input_drop_rate_node, input_distort_rot_ang_node, input_gripper_node=None):
        """ Builds neural network 

        Parameters
        ----------
        input_im_node : :obj:`tensorflow Placeholder`
            network input image placeholder
        input_pose_node : :obj:`tensorflow Placeholder`
            network input pose placeholder
        input_gripper_node: :obj:`tensorflow Placeholder`
            optional network input gripper parameter placeholder, if None then gripper_stream will not be
            used

        Returns
        -------
        :obj:`tensorflow Tensor`
            output of network
        """
        logging.info('Building Network')
        with tf.name_scope('im_stream'):
            output_im_stream, fan_out_im = self._build_im_stream(input_im_node, input_pose_node, self._im_height, self._im_width, self._num_channels, input_drop_rate_node, input_distort_rot_ang_node, self._architecture['im_stream'])
#             output_im_stream, fan_out_im = self._build_im_stream(input_im_node, input_pose_node, self._im_height, self._im_width, self._num_channels + 1, input_drop_rate_node, input_distort_rot_ang_node, self._architecture['im_stream'])
        with tf.name_scope('pose_stream'):
            output_pose_stream, fan_out_pose = self._build_pose_stream(input_pose_node, self._pose_dim, self._architecture['pose_stream'])
        if input_gripper_node is not None:
            with tf.name_scope('gripper_stream'):
                output_gripper_stream, fan_out_gripper = self._build_gripper_stream(input_gripper_node, self._gripper_dim, self._architecture['gripper_stream'])
            if 'gripper_pose_merge_stream' in self._architecture.keys():
                with tf.name_scope('gripper_pose_merge_stream'):
                    output_gripper_pose_merge_stream, fan_out_gripper_pose_merge_stream = self._build_merge_stream(output_pose_stream, output_gripper_stream, fan_out_pose, fan_out_gripper, input_drop_rate_node, self._architecture['gripper_pose_merge_stream'])
                with tf.name_scope('merge_stream'):
                    return self._build_merge_stream(output_im_stream, output_gripper_pose_merge_stream, fan_out_im, fan_out_gripper_pose_merge_stream, input_drop_rate_node, self._architecture['merge_stream'])[0]
            else:
                with tf.name_scope('merge_stream'):
                    return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, input_drop_rate_node, self._architecture['merge_stream'], input_stream_3=output_gripper_stream, fan_in_3=fan_out_gripper)[0]
        else:
            with tf.name_scope('merge_stream'):
                return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, input_drop_rate_node, self._architecture['merge_stream'])[0]
