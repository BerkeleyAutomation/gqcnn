"""
Nervana Dataset Wrapper for GQ-CNN datasets.

Author: Vishal Satish
"""
import os
import logging
import numpy as np
import cPickle as pkl
import random

from neon.data.datasets import Dataset

import autolab_core.utils as utils
from gqcnn import GQCNNTrainIterator
from gqcnn import ImageMode, TrainingMode, PreprocMode, InputDataMode, GeneralConstants, ImageFileTemplates

class GQCNNDataset(Dataset):
	def __init__(self, train_config):
		self.cfg = train_config
		self._setup()
    
    	@property
    	def _experiment_dir(self):
        	return self.experiment_dir    

	def gen_iterators(self):
		self._data_dict = {'train': GQCNNTrainIterator(self.im_filenames, self.pose_filenames, self.label_filenames, self.train_index_map, self.cfg, 
			self.data_mean, self.data_std, self.pose_mean, self.pose_std, distort=True, name='train_data')}
		self._data_dict['test'] = GQCNNTrainIterator(self.im_filenames, self.pose_filenames, self.label_filenames, self.val_index_map, self.cfg,
			self.data_mean, self.data_std, self.pose_mean, self.pose_std, name='val_data')
		return self._data_dict

	def _setup(self):
		""" Setup Dataset """

		# set up logger
		logging.getLogger().setLevel(logging.INFO)

		self.debug = self.cfg['debug']

		# set random seed for deterministic execution
		if self.debug:
			np.random.seed(GeneralConstants.SEED)
			random.seed(GeneralConstants.SEED)
			self.debug_num_files = self.cfg['debug_num_files']

		# setup output directories
		self._setup_output_dirs()

		# TODO: Move this
		self.data_dir = self.cfg['dataset_dir']
		self.image_mode = self.cfg['image_mode']
		self.data_split_mode = self.cfg['data_split_mode']
		self.train_pct = self.cfg['train_pct']
		self.total_pct = self.cfg['total_pct']
		self.target_metric_name = self.cfg['target_metric_name']
		self.metric_thresh = self.cfg['metric_thresh']


		# setup image and pose data files
		self._setup_data_filenames()

		# read data parameters from config file
		self._read_data_params()

		# compute train/test indices based on how the data is to be split
		if self.data_split_mode == 'image_wise':
			self._compute_indices_image_wise()
		elif self.data_split_mode == 'object_wise':
			self._compute_indices_object_wise()
		elif self.data_split_mode == 'stable_pose_wise':
			self._compute_indices_pose_wise()
		else:
			logging.error('Data Split Mode Not Supported')

		# compute means, std's, and normalization metrics
		self._compute_data_metrics()

	def _compute_data_metrics(self):
		""" Calculate image mean, image std, pose mean, pose std, normalization params """

		# compute data mean
		logging.info('Computing image mean')
		mean_filename = os.path.join(self.experiment_dir, 'mean.npy')
		std_filename = os.path.join(self.experiment_dir, 'std.npy')
		self.data_mean = 0
		self.data_std = 0
		random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
		num_summed = 0
		for k in random_file_indices.tolist():
			im_filename = self.im_filenames[k]
			im_data = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
			self.data_mean += np.sum(im_data[self.train_index_map[im_filename], :, :, :])
			num_summed += im_data[self.train_index_map[im_filename], :, :, :].shape[0]
		self.data_mean = self.data_mean / (num_summed * self.im_height * self.im_width)
		np.save(mean_filename, self.data_mean)

		for k in random_file_indices.tolist():
			im_filename = self.im_filenames[k]
			im_data = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
			self.data_std += np.sum((im_data[self.train_index_map[im_filename], :, :, :] - self.data_mean)**2)
		self.data_std = np.sqrt(self.data_std / (num_summed * self.im_height * self.im_width))
		np.save(std_filename, self.data_std)

		# compute pose mean
		logging.info('Computing pose mean')
		self.pose_mean_filename = os.path.join(self.experiment_dir, 'pose_mean.npy')
		self.pose_std_filename = os.path.join(self.experiment_dir, 'pose_std.npy')

		self.pose_mean = np.zeros(self.pose_shape)
		self.pose_std = np.zeros(self.pose_shape)
		num_summed = 0
		random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
		for k in random_file_indices.tolist():
			im_filename = self.im_filenames[k]
			pose_filename = self.pose_filenames[k]
			self.pose_data = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
			self.pose_mean += np.sum(self.pose_data[self.train_index_map[im_filename],:], axis=0)
			num_summed += self.pose_data[self.train_index_map[im_filename]].shape[0]
		self.pose_mean = self.pose_mean / num_summed

		for k in random_file_indices.tolist():
			im_filename = self.im_filenames[k]
			pose_filename = self.pose_filenames[k]
			self.pose_data = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
			self.pose_std += np.sum((self.pose_data[self.train_index_map[im_filename],:] - self.pose_mean)**2, axis=0)
		self.pose_std = np.sqrt(self.pose_std / num_summed)

		self.pose_std[self.pose_std==0] = 1.0

		np.save(self.pose_mean_filename, self.pose_mean)
		np.save(self.pose_std_filename, self.pose_std)

		# compute normalization parameters of the network
		logging.info('Computing metric stats')
		all_metrics = None
		all_val_metrics = None
		for im_filename, metric_filename in zip(self.im_filenames, self.label_filenames):
			self.metric_data = np.load(os.path.join(self.data_dir, metric_filename))['arr_0']
			indices = self.val_index_map[im_filename]
			val_metric_data = self.metric_data[indices]
			if all_metrics is None:
				all_metrics = self.metric_data
			else:
				all_metrics = np.r_[all_metrics, self.metric_data]
			if all_val_metrics is None:
				all_val_metrics = val_metric_data
			else:
				all_val_metrics = np.r_[all_val_metrics, val_metric_data]
		self.min_metric = np.min(all_metrics)
		self.max_metric = np.max(all_metrics)
		self.mean_metric = np.mean(all_metrics)
		self.median_metric = np.median(all_metrics)

		pct_pos_val = float(np.sum(all_val_metrics > self.metric_thresh)) / all_val_metrics.shape[0]
		logging.info('Percent positive in val set: ' + str(pct_pos_val))

	def _compute_indices_image_wise(self):
		""" Compute train and validation indices based on an image-wise split"""

		# get total number of training datapoints and set the decay_step
		num_datapoints = self.images_per_file * self.num_files
		self.num_train = int(self.train_pct * num_datapoints)

		# get training and validation indices
		all_indices = np.arange(num_datapoints)
		np.random.shuffle(all_indices)
		train_indices = np.sort(all_indices[:self.num_train])
		val_indices = np.sort(all_indices[self.num_train:])

		# make a map of the train and test indices for each file
		logging.info('Computing indices image-wise')
		train_index_map_filename = os.path.join(self.experiment_dir, 'train_indices_image_wise.pkl')
		self.val_index_map_filename = os.path.join(self.experiment_dir, 'val_indices_image_wise.pkl')
		if os.path.exists(train_index_map_filename):
			self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
			self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
                elif self.cfg['use_existing_indices']:
                        self.train_index_map = pkl.load(open(os.path.join(self.cfg['index_dir'], 'train_indices_image_wise.pkl'), 'r'))
                        self.val_index_map = pkl.load(open(os.path.join(self.cfg['index_dir'], 'val_indices_image_wise.pkl'), 'r'))
		else:
			self.train_index_map = {}
			self.val_index_map = {}
			for i, im_filename in enumerate(self.im_filenames):
				lower = i * self.images_per_file
				upper = (i+1) * self.images_per_file
				im_arr = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
				self.train_index_map[im_filename] = train_indices[(train_indices >= lower) & (train_indices < upper) &  (train_indices - lower < im_arr.shape[0])] - lower
				self.val_index_map[im_filename] = val_indices[(val_indices >= lower) & (val_indices < upper) & (val_indices - lower < im_arr.shape[0])] - lower
			pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
			pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))

	def _compute_indices_object_wise(self):
		""" Compute train and validation indices based on an object-wise split"""

		if self.obj_id_filenames is None:
			raise ValueError('Cannot use object-wise split. No object labels! Check the dataset_dir')

		# get total number of training datapoints and set the decay_step
		num_datapoints = self.images_per_file * self.num_files
		self.num_train = int(self.train_pct * num_datapoints)

		# get number of unique objects by taking last object id of last object id file
		self.obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
		last_file_object_ids = np.load(os.path.join(self.data_dir, self.obj_id_filenames[len(self.obj_id_filenames) - 1]))['arr_0']
		num_unique_objs = last_file_object_ids[len(last_file_object_ids) - 1]
		self.num_train_obj = int(self.train_pct * num_unique_objs)
		logging.debug('There are: ' + str(num_unique_objs) + 'unique objects in this dataset.')

		# get training and validation indices
		all_object_ids = np.arange(num_unique_objs + 1)
		np.random.shuffle(all_object_ids)
		train_object_ids = np.sort(all_object_ids[:self.num_train_obj])
		val_object_ids = np.sort(all_object_ids[self.num_train_obj:])

		# make a map of the train and test indices for each file
		logging.info('Computing indices object-wise')
		train_index_map_filename = os.path.join(self.experiment_dir, 'train_indices_object_wise.pkl')
		self.val_index_map_filename = os.path.join(self.experiment_dir, 'val_indices_object_wise.pkl')
		if os.path.exists(train_index_map_filename):
			self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
			self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
		else:
			self.train_index_map = {}
			self.val_index_map = {}
			for im_filename in self.im_filenames:
				# open up the corresponding obj_id file
				obj_ids = np.load(os.path.join(self.data_dir, 'object_labels_' + im_filename[-9:-4] + '.npz'))['arr_0']

				train_indices = []
				val_indices = []
				# for each obj_id if it is in train_object_ids then add it to train_indices else add it to val_indices
				for i, obj_id in enumerate(obj_ids):
					if obj_id in train_object_ids:
						train_indices.append(i)
					else:
						val_indices.append(i)

				self.train_index_map[im_filename] = np.asarray(train_indices, dtype=np.intc)
				self.val_index_map[im_filename] = np.asarray(val_indices, dtype=np.intc)
				train_indices = []
				val_indices = []

			pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
			pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))


	def _compute_indices_pose_wise(self):
		""" Compute train and validation indices based on an image-stable-pose-wise split"""

		# get total number of training datapoints and set the decay_step
		num_datapoints = self.images_per_file * self.num_files
		self.num_train = int(self.train_pct * num_datapoints)
		
		# get number of unique stable poses by taking last stable pose id of last stable pose id file
		self.stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))
		last_file_pose_ids = np.load(os.path.join(self.data_dir, self.stable_pose_filenames[len(self.stable_pose_filenames) - 1]))['arr_0']
		num_unique_stable_poses = last_file_pose_ids[len(last_file_pose_ids) - 1]
		self.num_train_poses = int(self.train_pct * num_unique_stable_poses)
		logging.debug('There are: ' + str(num_unique_stable_poses) + 'unique stable poses in this dataset.')

		# get training and validation indices
		all_pose_ids = np.arange(num_unique_stable_poses + 1)
		np.random.shuffle(all_pose_ids)
		train_pose_ids = np.sort(all_pose_ids[:self.num_train_poses])
		val_pose_ids = np.sort(all_pose_ids[self.num_train_poses:])

		# make a map of the train and test indices for each file
		logging.info('Computing indices stable-pose-wise')
		train_index_map_filename = os.path.join(self.experiment_dir, 'train_indices_stable_pose_wise.pkl')
		self.val_index_map_filename = os.path.join(self.experiment_dir, 'val_indices_stable_pose_wise.pkl')
		if os.path.exists(train_index_map_filename):
			self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
			self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
		else:
			self.train_index_map = {}
			self.val_index_map = {}
			for im_filename in self.im_filenames:
				# open up the corresponding obj_id file
				pose_ids = np.load(os.path.join(self.data_dir, 'pose_labels_' + im_filename[-9:-4] + '.npz'))['arr_0']

				train_indices = []
				val_indices = []
				# for each obj_id if it is in train_object_ids then add it to train_indices else add it to val_indices
				for i, pose_id in enumerate(pose_ids):
					if pose_id in train_pose_ids:
						train_indices.append(i)
					else:
						val_indices.append(i)

				self.train_index_map[im_filename] = np.asarray(train_indices, dtype=np.intc)
				self.val_index_map[im_filename] = np.asarray(val_indices, dtype=np.intc)
				train_indices = []
				val_indices = []

			pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
			pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))

	def _read_data_params(self):
		""" Read data parameters from configuration file """
		
		self.train_im_data = np.load(os.path.join(self.data_dir, self.im_filenames[0]))['arr_0']
		self.pose_data = np.load(os.path.join(self.data_dir, self.pose_filenames[0]))['arr_0']
		self.metric_data = np.load(os.path.join(self.data_dir, self.label_filenames[0]))['arr_0']
		self.images_per_file = self.train_im_data.shape[0]
		self.im_height = self.train_im_data.shape[1]
		self.im_width = self.train_im_data.shape[2]
		self.im_channels = self.train_im_data.shape[3]
		self.im_center = np.array([float(self.im_height-1)/2, float(self.im_width-1)/2])
		self.num_tensor_channels = self.cfg['num_tensor_channels']
		self.pose_shape = self.pose_data.shape[1]
		self.input_data_mode = self.cfg['input_data_mode']
		if self.input_data_mode == InputDataMode.TF_IMAGE:
			self.pose_dim = 1 # depth
		elif self.input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
			self.pose_dim = 3 # depth, cx, cy
		elif self.input_data_mode == InputDataMode.RAW_IMAGE:
			self.pose_dim = 4 # u, v, theta, depth
		elif self.input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
			self.pose_dim = 6 # u, v, theta, depth cx, cy
		else:
			raise ValueError('Input data mode %s not understood' %(self.input_data_mode))
		self.num_files = len(self.im_filenames)
		self.num_random_files = min(self.num_files, self.cfg['num_random_files'])
		self.num_categories = 2

	def _setup_data_filenames(self):
		""" Setup image and pose data filenames, subsample files, check validity of filenames/image mode """

		# read in filenames of training data(poses, images, labels)
		logging.info('Reading filenames')
		all_filenames = os.listdir(self.data_dir)
		if self.image_mode== ImageMode.BINARY:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tensor_template) > -1]
		elif self.image_mode== ImageMode.DEPTH:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tensor_template) > -1]
		elif self.image_mode== ImageMode.BINARY_TF:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.COLOR_TF:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.color_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.GRAY_TF:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.gray_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.DEPTH_TF:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.DEPTH_TF_TABLE:
			self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
		else:
			raise ValueError('Image mode %s not supported.' %(self.image_mode))

		self.pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]
		self.label_filenames = [f for f in all_filenames if f.find(self.target_metric_name) > -1]
		self.obj_id_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.object_labels_template) > -1]
		self.stable_pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.pose_labels_template) > -1]

		if self.debug:
			# sort
		        self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
            		self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
            		self.label_filenames.sort(key = lambda x: int(x[-9:-4]))
            		self.obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
            		self.stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))

            		# pack, shuffle and sample
            		zipped = zip(self.im_filenames, self.pose_filenames, self.label_filenames, self.obj_id_filenames, self.stable_pose_filenames)
            		random.shuffle(zipped)
            		zipped = zipped[:self.debug_num_files]

            		# unpack
            		self.im_filenames, self.pose_filenames, self.label_filenames, self.obj_id_filenames, self.stable_pose_filenames = zip(*zipped)
			self.im_filenames = list(self.im_filenames)
			self.pose_filenames = list(self.pose_filenames)
			self.label_filenames = list(self.label_filenames)
			self.obj_id_filenames = list(self.obj_id_filenames)
			self.stable_pose_filenames = list(self.stable_pose_filenames)

		self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.label_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))

		# check valid filenames
		if len(self.im_filenames) == 0 or len(self.pose_filenames) == 0 or len(self.label_filenames) == 0 or len(self.stable_pose_filenames) == 0:
			raise ValueError('One or more required training files in the dataset could not be found.')
		if len(self.obj_id_filenames) == 0:
			self.obj_id_filenames = None

		# subsample files
		self.num_files = len(self.im_filenames)
		num_files_used = int(self.total_pct * self.num_files)
		filename_indices = np.random.choice(self.num_files, size=num_files_used, replace=False)
		filename_indices.sort()
		self.im_filenames = [self.im_filenames[k] for k in filename_indices]
		self.pose_filenames = [self.pose_filenames[k] for k in filename_indices]
		self.label_filenames = [self.label_filenames[k] for k in filename_indices]
		if self.obj_id_filenames is not None:
			self.obj_id_filenames = [self.obj_id_filenames[k] for k in filename_indices]
		self.stable_pose_filenames = [self.stable_pose_filenames[k] for k in filename_indices]

		# create copy of image, pose, and label filenames because original cannot be accessed by load and enqueue op in the case that the error_rate_in_batches method is sorting the original
		self.im_filenames_copy = self.im_filenames[:]
		self.pose_filenames_copy = self.pose_filenames[:]
		self.label_filenames_copy = self.label_filenames[:]
		 
	def _setup_output_dirs(self):
		""" Setup output directories """

		# setup general output directory
		output_dir = self.cfg['output_dir']
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		experiment_id = utils.gen_experiment_id()
		self.experiment_dir = os.path.join(output_dir, 'model_%s' %(experiment_id))
		if not os.path.exists(self.experiment_dir):
			os.mkdir(self.experiment_dir)
		self.summary_dir = os.path.join(self.experiment_dir, 'tensorboard_summaries')
		if not os.path.exists(self.summary_dir):
			os.mkdir(self.summary_dir)
		else:
			# if the summary directory already exists, clean it out by deleting all files in it
			# we don't want tensorboard to get confused with old logs while debugging with the same directory
			old_files = os.listdir(self.summary_dir)
			for file in old_files:
				os.remove(os.path.join(self.summary_dir, file))

		logging.info('Saving model to %s' %(self.experiment_dir))

		# setup filter directory
		self.filter_dir = os.path.join(self.experiment_dir, 'filters')
		if not os.path.exists(self.filter_dir):
			os.mkdir(self.filter_dir)
