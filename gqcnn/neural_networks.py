"""
Quick wrapper for grasp quality neural network
Author: Jeff Mahler
"""
import copy
import json
from collections import OrderedDict
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.framework as tcf 
import matplotlib.pyplot as plt

from autolab_core import YamlConfig
from gqcnn import InputPoseMode, InputGripperMode

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
			train_config = json.load(data_file, object_pairs_hook=OrderedDict)

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

	def _read_pose_data(self, pose_arr, input_pose_mode):
		""" Read the pose data and slice it according to the specified input_data_mode

		Parameters
		----------
		pose_arr: :obj:`ndArray`
			full pose data array read in from file
		input_pose_mode: :enum:`InputPoseMode`
			enum for input pose mode, see optimizer_constants.py for all
			possible input pose modes 

		Returns
		-------
		:obj:`ndArray`
			sliced pose_data corresponding to input pose mode
		"""
		if len(pose_arr.shape) == 1:
			pose_arr = np.asarray([pose_arr])
		if input_pose_mode == InputPoseMode.TF_IMAGE:
			# depth
			return pose_arr[:,2:3]
		elif input_pose_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
			# depth, cx, cy
			return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
		elif input_pose_mode == InputPoseMode.RAW_IMAGE:
			# u, v, depth, theta
			return pose_arr[:,:4]
		elif input_pose_mode == InputPoseMode.RAW_IMAGE_PERSPECTIVE:
			# u, v, depth, theta, cx, cy
			return pose_arr[:,:6]
		elif input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
			# depth, theta
			return pose_arr[:,2:4]
		else:
			raise ValueError('Input pose mode {} not supported'.format(input_pose_mode))

	def _read_gripper_data(self, gripper_param_arr, input_gripper_mode):
		if len(gripper_param_arr.shape) == 1:
			gripper_param_arr = np.asarray([gripper_param_arr])
		if input_gripper_mode == InputGripperMode.WIDTH:
			return gripper_param_arr[:, 2:3]
		else:
			raise ValueError('Input gripper mode {} not supportd'.format(input_gripper_mode))

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
		self._im_mean = np.load(os.path.join(model_dir, 'image_mean.npy'))
		self._im_std = np.load(os.path.join(model_dir, 'image_std.npy'))
		self._pose_mean = np.load(os.path.join(model_dir, 'pose_mean.npy'))
		self._pose_std = np.load(os.path.join(model_dir, 'pose_std.npy'))
		self._gripper_mean = np.load(os.path.join(model_dir, 'gripper_mean.npy'))
		self._gripper_std = np.load(os.path.join(model_dir, 'gripper_std.npy'))

		# read the certain parts of the pose and gripper mean/std that we want
		self._pose_mean = self._read_pose_data(self._pose_mean, self._input_pose_mode)
		self._pose_std = self._read_pose_data(self._pose_std, self._input_pose_mode)
		self._gripper_mean = self._read_gripper_data(self._gripper_mean, self._input_gripper_mode)
		self._gripper_std = self._read_gripper_data(self._gripper_std, self._input_gripper_mode)

	def init_weights_file(self, ckpt_file):
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
			ckpt_vars = tcf.list_variables(ckpt_file)
			full_var_names = []
			short_names = []
			for variable, shape in ckpt_vars:
				full_var_names.append(variable)
				short_names.append(variable.split('/')[-1])
	
			for full_var_name, short_name in zip(full_var_names, short_names):
				self._weights.weights[short_name] = tf.Variable(reader.get_tensor(full_var_name))

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
		self._input_pose_mode = config['input_pose_mode']
		self._input_gripper_mode = config['input_gripper_mode']

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
		else:
			raise ValueError('Input gripper mode %s not understood' %(self._input_gripper_mode))

		# create feed tensors for prediction
		self._input_im_arr = np.zeros((self._batch_size, self._im_height,
									   self._im_width, self._num_channels))
		self._input_pose_arr = np.zeros((self._batch_size, self._pose_dim))
		if self._gripper_dim > 0:
			self._input_gripper_arr = np.zeros((self._batch_size, self._gripper_dim))

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
		if self._gripper_dim > 0:
			self._gripper_mean = np.zeros(self._gripper_dim)
			self._gripper_std = np.ones(self._gripper_dim)

		# create empty holder for feature handles
		self._feature_tensors = {}
	
		# initialize other misc parameters
		self._summary_writer = None
		self._mask_and_inpaint = False
		self._save_histograms = False

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
			if self._gripper_dim > 0:
				self._input_gripper_node = tf.placeholder(
					tf.float32, (self._batch_size, self._gripper_dim))

			# build network
			if self._gripper_dim > 0:
				self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, input_gripper_node=self._input_gripper_node, inference=True)
			else:
				self._output_tensor = self._build_network(self._input_im_node, self._input_pose_node, inference=True)
			
			# add softmax function to output of inference network if specified
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

	def predict(self, image_arr, pose_arr, gripper_arr=None):
		""" Predict the probability of grasp success given a depth iamge, gripper pose, and
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

		# setup for prediction
		num_images = image_arr.shape[0]
		num_poses = pose_arr.shape[0]
		if gripper_arr is not None:
			num_gripper_parameters = gripper_arr.shape[0]

		output_arr = np.zeros([num_images, 2])
		if num_images != num_poses:
			raise ValueError('Must provide same number of images and poses')
		if gripper_arr is not None:
			if num_images != num_gripper_parameters:
				raise ValueError('Must provide same number of images and gripper parameters')

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

				# if using mask channel then do not normalize mask
				if self._mask_and_inpaint:
					self._input_im_arr[:dim, :, :, 0] = (
						image_arr[cur_ind:end_ind, :, :, 0] - self._im_mean) / self._im_std
				else:
					self._input_im_arr[:dim, :, :, :] = (
						image_arr[cur_ind:end_ind, :, :, :] - self._im_mean) / self._im_std
				
				self._input_pose_arr[:dim, :] = (
					pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

				if gripper_arr is not None:
					self._input_gripper_arr[:dim, :] = (
						gripper_arr[cur_ind:end_ind, :] - self._gripper_mean) / self._gripper_std					

				if gripper_arr is not None:
					gqcnn_output = self._sess.run(self._output_tensor,
													  feed_dict={self._input_im_node: self._input_im_arr,
																 self._input_pose_node: self._input_pose_arr,
																 self._input_gripper_node: self._input_gripper_arr})
				else:
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

			initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
			initial = initial.astype('float32')
			initial = initial.flatten()
			transformb = tf.Variable(initial_value=initial, name='{}_bias'.format(name))
			
			self._weights.weights['{}_weights'.format(name)] = transformW
			self._weights.weights['{}_bias'.format(name)] = transformb

		# build localisation network
		loc_network = tf.matmul(tf.zeros([input_node.get_shape().as_list()[0], input_height * input_width * input_channels]), transformW) + transformb
			
		# build transform layer
		transform_layer = transformer(input_node, loc_network, (output_width, output_height))

		# add output to feature dict
		self._feature_tensors[name] = transform_layer

		return transform_layer, output_height, output_width, input_channels

	def _build_conv_layer(self, input_node, input_height, input_width, input_channels, filter_h, filter_w, num_filt, pool_stride_h, pool_stride_w, pool_size, name, norm=False, inference=False):
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
			
			if not inference:
				if self._save_histograms:
					tf.summary.histogram('weights', convW, collections=["histogram"])
					tf.summary.histogram('bias', convb, collections=["histogram"])

			out_height = input_height / pool_stride_h
			out_width = input_width / pool_stride_w
			out_channels = num_filt

			# build layer
			convh = tf.nn.conv2d(input_node, convW, strides=[
								1, 1, 1, 1], padding='SAME') + convb
			
			if not inference:
				if self._save_histograms:
					tf.summary.histogram('layer_raw', convh, collections=["histogram"])

			convh = self._leaky_relu(convh)
			
			if not inference:
				if self._save_histograms:
					tf.summary.histogram('layer_act', convh, collections=["histogram"])

			if norm:
				convh = tf.nn.local_response_normalization(convh,
															depth_radius=self._normalization_radius,
															alpha=self._normalization_alpha,
															beta=self._normalization_beta,
															bias=self._normalization_bias)
			if not inference:
				if self._save_histograms:
					tf.summary.histogram('layer_norm', convh, collections=["histogram"])

			pool = tf.nn.max_pool(convh,
								ksize=[1, pool_size, pool_size, 1],
								strides=[1, pool_stride_h,
										pool_stride_w, 1],
								padding='SAME')
			
			if not inference:
				if self._save_histograms:
					tf.summary.histogram('layer_pool', pool, collections=["histogram"])	    

			# add output to feature dict
			self._feature_tensors[name] = pool

			return pool, out_height, out_width, out_channels

	def _build_fc_layer(self, input_node, fan_in, out_size, name, input_is_multi, drop_rate=0.0, final_fc_layer=False, inference=False):
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
			input_flat = tf.reshape(input_node, [-1, input_num_nodes])
			fc = self._leaky_relu(tf.matmul(input_flat, fcW) + fcb)
		else:
			fc = self._leaky_relu(tf.matmul(input_node, fcW) + fcb)

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

	def _build_fc_merge(self, input_fc_node_1, input_fc_node_2, fan_in_1, fan_in_2, out_size, name, input_fc_node_3=None, fan_in_3=None, drop_rate=0.0, inference=False):
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
			fc = self._leaky_relu(tf.matmul(input_fc_node_1, input1W) +
									tf.matmul(input_fc_node_2, input2W) +
									fcb)
		
		if drop_rate > 0 and not inference:
			fc = tf.nn.dropout(fc, drop_rate)

		# add output to feature dict
		self._feature_tensors[name] = fc

		return fc, out_size

	def _build_batch_norm(self, input_node, ep, inference=False):
		output_node = input_node
		output_node = tf.layers.batch_normalization(output_node, training=inference, epsilon = ep)
		return output_node

	def _build_residual_layer(self, input_node, input_channels, fan_in, num_filt, filt_h, filt_w, name, first=False, inference=False):
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
			output_node = self._build_batch_norm(output_node, EP)
			output_node = self._leaky_relu(output_node)
		output_node = tf.nn.conv2d(output_node, conv1W, strides=[1, 1, 1, 1], padding='SAME') + conv1b
		output_node = self._build_batch_norm(output_node, EP)
		output_node = self._leaky_relu(output_node)
		output_node = tf.nn.conv2d(output_node, conv2W, strides=[1, 1, 1, 1], padding='SAME') + conv2b
		output_node = input_node + output_node

		# add output to feature dict
		self._feature_tensors[name] = output_node

		return output_node, num_filt  


	def _build_im_stream(self, input_node, input_height, input_width, input_channels, layers, inference=False):
		logging.info('Building Image Stream')
		output_node = input_node
		prev_layer = "start"
		first_residual = True
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
				output_node, input_height, input_width, input_channels = self._build_conv_layer(output_node, input_height, input_width, input_channels, layer_config['filt_dim'],
					layer_config['filt_dim'], layer_config['num_filt'], layer_config['pool_stride'], layer_config['pool_stride'], layer_config['pool_size'], layer_name, 
					norm=layer_config['norm'], inference=inference)
				prev_layer = layer_type
			elif layer_type == 'fc':
				prev_layer_is_conv_or_res = False
				first_residual = True
				if prev_layer == 'conv' or prev_layer == 'residual':
					prev_layer_is_conv_or_res = True
					fan_in = input_height * input_width * input_channels
				if 'dropout_rate' in layer_config.keys():
					output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv_or_res, drop_rate=layer_config['drop_rate'], inference=inference)
				else:
					output_node, fan_in = self._build_fc_layer(output_node, fan_in, layer_config['out_size'], layer_name, prev_layer_is_conv_or_res, inference=inference)
				prev_layer = layer_type
			elif layer_type == 'pc':
				raise ValueError('Cannot have pose-connected layer in image stream')
			elif layer_type == 'fc_merge':
				raise ValueError("Cannot have merge layer in image stream")
			elif layer_type == 'residual':
				# TODO: currently we are assuming the layer before a res layer must be conv layer, fix this
				fan_in = input_height * input_width * input_channels
				if first_residual:
					output_node, input_channels = self._build_residual_layer(output_node, input_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
						layer_config['filt_dim'], layer_name, first=True)
					first_residual = False
				else:
					output_node, input_channels = self._build_residual_layer(output_node, input_channels, fan_in, layer_config['num_filt'], layer_config['filt_dim'],
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

	def _build_gripper_stream(self, input_node, fan_in, layers, inference=False):
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

	def _build_merge_stream(self, input_stream_1, input_stream_2, fan_in_1, fan_in_2, layers, input_stream_3=None, fan_in_3=None, inference=False):
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
				#TODO: Clean this giant if statement up
				if input_stream_3 is not None:
					if 'drop_rate' in layer_config.keys():
						output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], layer_name, input_fc_node_3=input_stream_3, fan_in_3=fan_in_3, drop_rate=layer_config['drop_rate'], inference=inference)
					else:
						output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], layer_name, input_fc_node_3=input_stream_3, fan_in_3=fan_in_3, inference=inference)
				else:
					if 'drop_rate' in layer_config.keys():
						output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], layer_name, drop_rate=layer_config['drop_rate'], inference=inference)
					else:
						output_node, fan_in = self._build_fc_merge(input_stream_1, input_stream_2, fan_in_1, fan_in_2, layer_config['out_size'], layer_name, inference=inference)
			elif layer_type == 'residual':
				raise ValueError('Cannot have residual in merge stream')
				prev_layer = layer_type
			else:
				raise ValueError("Unsupported layer type: {}".format(layer_type))
		return output_node, fan_in

	def _build_network(self, input_im_node, input_pose_node, input_gripper_node=None, inference=False):
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
			output_im_stream, fan_out_im = self._build_im_stream(input_im_node, self._im_height, self._im_width, self._num_channels, self._architecture['im_stream'], inference=inference)
		with tf.name_scope('pose_stream'):
			output_pose_stream, fan_out_pose = self._build_pose_stream(input_pose_node, self._pose_dim, self._architecture['pose_stream'], inference=inference)
		if input_gripper_node is not None:
			with tf.name_scope('gripper_stream'):
				output_gripper_stream, fan_out_gripper = self._build_gripper_stream(input_gripper_node, self._gripper_dim, self._architecture['gripper_stream'], inference=inference)
			if 'gripper_pose_merge_stream' in self._architecture.keys():
				with tf.name_scope('gripper_pose_merge_stream'):
					output_gripper_pose_merge_stream, fan_out_gripper_pose_merge_stream = self._build_merge_stream(output_pose_stream, output_gripper_stream, fan_out_pose, fan_out_gripper, self._architecture['gripper_pose_merge_stream'], inference=inference)
				with tf.name_scope('merge_stream'):
					return self._build_merge_stream(output_im_stream, output_gripper_pose_merge_stream, fan_out_im, fan_out_gripper_pose_merge_stream, self._architecture['merge_stream'], inference=inference)[0]
			else:
				with tf.name_scope('merge_stream'):
					return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, self._architecture['merge_stream'], input_stream_3=output_gripper_stream, fan_in_3=fan_out_gripper, inference=inference)[0]
		else:
			with tf.name_scope('merge_stream'):
				return self._build_merge_stream(output_im_stream, output_pose_stream, fan_out_im, fan_out_pose, self._architecture['merge_stream'], inference=inference)[0]
