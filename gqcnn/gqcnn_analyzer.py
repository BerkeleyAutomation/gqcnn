"""
Class for analyzing a GQCNN model for grasp quality prediction
Author: Jeff Mahler
"""
import cPickle as pkl
import copy
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
from skimage.feature import hog
import scipy.misc as sm
import sys
import time

from gqcnn.model import get_gqcnn_model
from gqcnn.utils.learning_analysis import ClassificationResult
from gqcnn.utils.enums import InputPoseMode, ImageMode, InputGripperMode, DataFileTemplates

class GQCNNAnalyzer(object):
	""" Analyzes GQCNN models """

	def __init__(self, config):
		"""
		Parameters
		----------
		config : dict
			dictionary of configuration parameters
		"""
		self.cfg = config

	def analyze(self):
		""" Analyzes a GQCNN model """
		# setup for analysis
		self._setup()

		# run predictions
		self._run_predictions()

		# finally plot curves
		self._plot()

	def _setup(self):
		""" Setup for analysis """
		# setup logger
		logging.getLogger().setLevel(logging.INFO)
		logging.info('Setting up for analysis')

		# read config
		self.model_dir = self.cfg['model_dir']
		self.output_dir = self.cfg['output_dir']
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
        
        	self.backend = self.cfg['backend']

		self.font_size = self.cfg['font_size']
		self.dpi = self.cfg['dpi']

		self.models = self.cfg['models']

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
        	elif input_gripper_mode == InputGripperMode.ALL:
            		return gripper_param_arr
		else:
			raise ValueError('Input gripper mode {} not supportd'.format(input_gripper_mode))

	def _run_predictions(self):
		""" Run predictions to use for plotting """
		logging.info('Running Predictions')

		self.train_class_results = {}
		self.val_class_results = {}
		for model_name in self.models.keys():
			logging.info('Analyzing model %s' %(model_name))

			# read in model config
			model_subdir = os.path.join(self.model_dir, model_name)
			model_config_filename = os.path.join(model_subdir, 'config.json')
			with open(model_config_filename) as data_file:
					model_config = json.load(data_file)
			model_type = self.models[model_name]['type']
			model_tag = self.models[model_name]['tag']
			split_type = self.models[model_name]['split_type']

			# read in training params
			model_training_dataset_dir = model_config['dataset_dir']
			model_image_mode = model_config['image_mode']
			model_target_metric = model_config['target_metric_name']
			model_metric_thresh = model_config['metric_thresh']
			model_input_pose_mode = model_config['input_pose_mode']
			model_input_gripper_mode = model_config['input_gripper_mode']

			# create output dir
			model_output_dir = os.path.join(self.output_dir, model_name)
			if not os.path.exists(model_output_dir):
				os.mkdir(model_output_dir)

			# load
			logging.info('Loading model %s' %(model_name))
			if model_type == 'gqcnn':
				model = get_gqcnn_model(self.backend).load(model_subdir)
				# load indices based on dataset-split-type
				if split_type == 'image_wise':
					train_indices_filename = os.path.join(model_subdir, 'train_indices_image_wise.pkl')
					val_indices_filename = os.path.join(model_subdir, 'val_indices_image_wise.pkl')
				elif split_type == 'object_wise':
					train_indices_filename = os.path.join(model_subdir, 'train_indices_object_wise.pkl')
					val_indices_filename = os.path.join(model_subdir, 'val_indices_object_wise.pkl')
				elif split_type == 'stable_pose_wise':
					train_indices_filename = os.path.join(model_subdir, 'train_indices_stable_pose_wise.pkl')
					val_indices_filename = os.path.join(model_subdir, 'val_indices_stable_pose_wise.pkl')
 
				model.open_session()

				# visualize filters
				if self.models[model_name]['vis_conv']:
					conv1_filters = model.filters

					num_filt = conv1_filters.shape[3]
					d = int(np.ceil(np.sqrt(num_filt)))

					plt.clf()
					for k in range(num_filt):
						filt = conv1_filters[:,:,0,k]
						filt = sm.imresize(filt, 5.0, interp='bilinear', mode='F')
						plt.subplot(d,d,k+1)
						plt.imshow(filt, cmap=plt.cm.gray)
						plt.axis('off')
					figname = os.path.join(model_output_dir, 'conv1_filters.pdf')
					plt.savefig(figname, dpi=dpi)
					exit(0)
				
			else:
				model = pkl.load(open(os.path.join(model_subdir, 'model.pkl')))
				train_indices_filename = os.path.join(model_subdir, 'train_index_map.pkl')
				val_indices_filename = os.path.join(model_subdir, 'val_index_map.pkl')
				image_mean = np.load(os.path.join(model_subdir, 'image_mean.npy'))
				image_std = np.load(os.path.join(model_subdir, 'image_std.npy'))
				pose_mean = self._read_pose_data(np.load(os.path.join(model_subdir, 'pose_mean.npy')), model_input_pose_mode)
				pose_std = self._read_pose_data(np.load(os.path.join(model_subdir, 'pose_std.npy')), model_input_pose_mode)
				if model_input_gripper_mode != InputGripperMode.NONE:
					gripper_mean = self._read_gripper_data(np.load(os.path.join(model_subdir, 'gripper_mean.npy')), model_input_gripper_mode)
					gripper_std = self._read_gripper_data(np.load(os.path.join(model_subdir, 'gripper_std.npy')), model_input_gripper_mode)

			# read in training, val indices
			train_indices = pkl.load(open(train_indices_filename, 'r'))
			val_indices = pkl.load(open(val_indices_filename, 'r'))

			# get filenames
			filenames = [os.path.join(model_training_dataset_dir, f) for f in os.listdir(model_training_dataset_dir)]
			if model_image_mode == ImageMode.BINARY_TF:
				im_filenames = [f for f in filenames if f.find(FileTemplaltes.binary_im_tf_tensor_template) > -1]
			elif model_image_mode == ImageMode.DEPTH_TF:
				im_filenames = [f for f in filenames if f.find(FileTemplates.depth_im_tf_tensor_template) > -1]
			elif model_image_mode == ImageMode.DEPTH_TF_TABLE:
				im_filenames = [f for f in filenames if f.find(FileTemplates.depth_im_tf_table_tensor_template) > -1]
			else:
				raise ValueError('Model image mode %s not recognized' %(model_image_mode))
			pose_filenames = [f for f in filenames if f.find(FileTemplates.hand_poses_template) > -1]
			metric_filenames = [f for f in filenames if f.find(model_target_metric) > -1]
			if model_input_gripper_mode != InputGripperMode.NONE:
				gripper_filenames = [f for f in filenames if f.find(FileTemplates.gripper_params_template) > -1]

			# sort filenames for consistency
			im_filenames.sort(key = lambda x: int(x[-9:-4]))
			pose_filenames.sort(key = lambda x: int(x[-9:-4]))
			metric_filenames.sort(key = lambda x: int(x[-9:-4]))
			if model_input_gripper_mode != InputGripperMode.NONE:
				gripper_filenames.sort(key = lambda x: int(x[-9:-4]))

			num_files = len(im_filenames)
			cur_file_num = 0
			evaluation_time = 0

			# aggregate training and validation true labels and predicted probabilities
			train_preds = []
			train_labels = []
			val_preds = []
			val_labels = []
			if model_input_gripper_mode != InputGripperMode.NONE:
				zipped_filenames = zip(im_filenames, pose_filenames, metric_filenames, gripper_filenames)
			else:
				zipped_filenames = zip(im_filenames, pose_filenames, metric_filenames)

			for filename_batch in zipped_filenames:
				if cur_file_num % self.cfg['out_rate'] == 0:
					logging.info('Reading file %d of %d' %(cur_file_num+1, num_files))

				# get filenames
				if model_input_gripper_mode != InputGripperMode.NONE:
					im_filename, pose_filename, metric_filename, gripper_filename = filename_batch
				else:
					im_filename, pose_filename, metric_filename = filename_batch

				# read data
				image_arr = np.load(im_filename)['arr_0']            
				pose_arr = np.load(pose_filename)['arr_0']
				metric_arr = np.load(metric_filename)['arr_0']
				labels_arr = 1 * (metric_arr > model_metric_thresh)
				if model_input_gripper_mode != InputGripperMode.NONE:
					gripper_arr = np.load(gripper_filename)['arr_0']
				num_datapoints = image_arr.shape[0]

				if model_type == 'gqcnn':
					# slice correct part of pose_arr and/or gripper_arr corresponding to input_pose_mode and/or input_gripper_mode used for training model
					pose_arr = self._read_pose_data(pose_arr, model_input_pose_mode)
					if model_input_gripper_mode != InputGripperMode.NONE:
						gripper_arr = self._read_gripper_data(gripper_arr, model_input_gripper_mode)

				# predict
				pred_start = time.time()
				if model_type == 'gqcnn':
					if model_input_gripper_mode != InputGripperMode.NONE:
						pred_arr = model.predict(image_arr, pose_arr, gripper_arr=gripper_arr)
					else:
						pred_arr = model.predict(image_arr, pose_arr)
				else:
					pose_arr = (pose_arr - pose_mean) / pose_std

					if 'use_hog' in model_config.keys() and model_config['use_hog']:
						feature_arr = None
						for i in range(num_datapoints):
							image = image_arr[i,:,:,0]
							feature_descriptor = hog(image, orientations=model_config['hog_num_orientations'],
													 pixels_per_cell=(model_config['hog_pixels_per_cell'], model_config['hog_pixels_per_cell']),
													 cells_per_block=(model_config['hog_cells_per_block'], model_config['hog_cells_per_block']))
						feature_dim = feature_descriptor.shape[0]

						if feature_arr is None:
							feature_arr = np.zeros([num_datapoints, feature_dim+1])
						feature_arr[i,:] = np.r_[feature_descriptor, pose_arr[i]]
					else:
						feature_arr = np.c_[((image_arr - image_mean) / image_std).reshape(num_datapoints, -1),
											(pose_arr - pose_mean) / pose_std]

					if model_type == 'rf':
						pred_arr = model.predict_proba(feature_arr)
					elif model_type == 'svm':
						pred_arr = model.decision_function(feature_arr)
						pred_arr = pred_arr / (2*np.max(np.abs(pred_arr)))
						pred_arr = pred_arr - np.min(pred_arr)
						pred_arr = np.c_[1-pred_arr, pred_arr]
					else:
						raise ValueError('Model type %s not supported' %(model_type))

				pred_stop = time.time()                    
				evaluation_time += pred_stop - pred_start

				# break into training / val
				index_im_filename = im_filename
				new_train_indices = {}
				for key in train_indices.keys():
					new_train_indices[os.path.join(model_training_dataset_dir, key)] = train_indices[key]
				train_indices = new_train_indices
				new_val_indices = {}
				for key in val_indices.keys():
					new_val_indices[os.path.join(model_training_dataset_dir, key)] = val_indices[key]
				val_indices = new_val_indices

				train_preds.append(pred_arr[train_indices[index_im_filename]])
				train_labels.append(labels_arr[train_indices[index_im_filename]])
				val_preds.append(pred_arr[val_indices[index_im_filename]])
				val_labels.append(labels_arr[val_indices[index_im_filename]])            

				cur_file_num += 1

			# aggregate results
			train_class_result = ClassificationResult(copy.copy(train_preds), copy.copy(train_labels))
			val_class_result = ClassificationResult(copy.copy(val_preds), copy.copy(val_labels))

			self.train_class_results[model_name] = train_class_result
			self.val_class_results[model_name] = val_class_result

			train_class_result.save(os.path.join(model_output_dir, 'train_class_result.cres'))
			val_class_result.save(os.path.join(model_output_dir, 'val_class_result.cres'))

			if model_type == 'gqcnn':
				model.close_session()

			logging.info('Total evaluation time: %.3f sec' %(evaluation_time))

	def _plot(self):
		""" Plot analysis curves """
		logging.info('Beginning Plotting')

		colors = ['g', 'b', 'c', 'y', 'm', 'r', 'k', '#F7DC6F', '#A04000', '#BFC9CA', '#82E0AA', '#85C1E9', '#F1948A']
		styles = ['-', '--', '-.', ':', '-']

		# get stats, plot curves
		plt.clf()
		i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			train_class_result = self.train_class_results[model_name]
			logging.info('Model %s training error rate: %.3f' %(model_name, train_class_result.error_rate))
			train_class_result.precision_recall_curve(plot=True, color=colors[i],
													  style=styles[i if i < len(styles) else 0], label=model_name)
			i += 1
		plt.title('Training Precision Recall Curve', fontsize=self.font_size)
		handles, labels = plt.gca().get_legend_handles_labels()
		plt.legend(handles, labels, loc='best')
		figname = os.path.join(self.output_dir, 'train_precision_recall.pdf')
		plt.savefig(figname, dpi=self.dpi)
		plt.clf()
		i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			train_class_result = self.train_class_results[model_name]
			train_class_result.roc_curve(plot=True, color=colors[i],
										 style=styles[i if i < len(styles) else 0], label=model_name)
			i += 1
		plt.title('Training ROC Curve', fontsize=self.font_size)
		handles, labels = plt.gca().get_legend_handles_labels()
		plt.legend(handles, labels, loc='best')
		figname = os.path.join(self.output_dir, 'train_roc.pdf')
		plt.savefig(figname, dpi=self.dpi)

		plt.clf()
		i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			val_class_result = self.val_class_results[model_name]
			logging.info('Model %s validation error rate: %.3f' %(model_name, val_class_result.error_rate))
			val_class_result.precision_recall_curve(plot=True, color=colors[i],
													  style=styles[i if i < len(styles) else 0], label=model_name)
			i += 1
		plt.title('Validation Precision Recall Curve', fontsize=self.font_size)
		handles, labels = plt.gca().get_legend_handles_labels()
		plt.legend(handles, labels, loc='best')
		figname = os.path.join(self.output_dir, 'val_precision_recall.pdf')
		plt.savefig(figname, dpi=self.dpi)

		plt.clf()
		i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			val_class_result = self.val_class_results[model_name]
			val_class_result.roc_curve(plot=True, color=colors[i],
										 style=styles[i if i < len(styles) else 0], label=model_name)
			i += 1
		plt.title('Validation ROC Curve', fontsize=self.font_size)
		handles, labels = plt.gca().get_legend_handles_labels()
		plt.legend(handles, labels, loc='best')
		figname = os.path.join(self.output_dir, 'val_roc.pdf')
		plt.savefig(figname, dpi=self.dpi)

		# combined training and validation precision-recall curves plot
		plt.clf()
		i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			train_class_result = self.train_class_results[model_name]
			if model_tag is None:
				train_class_result.precision_recall_curve(plot=True, color=colors[i],
													  style=styles[i], label='Training')
			else:
				train_class_result.precision_recall_curve(plot=True, color=colors[i],
									  style=styles[i if i < len(styles) else 0], label='Training' + model_name)
			i += 1
        	i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			val_class_result = self.val_class_results[model_name]
			if model_tag is None:
				val_class_result.precision_recall_curve(plot=True, color=colors[i],
														  style=styles[i if i < len(styles) else 0], label='Validation')
			else:
				val_class_result.precision_recall_curve(plot=True, color=colors[i],
										  style=styles[i if i < len(styles) else 0], label='Validation' + model_name)
			i += 1
		
		plt.title('Precision Recall Curves', fontsize=self.font_size)
		handles, labels = plt.gca().get_legend_handles_labels()
		plt.legend(handles, labels, loc='best')
		figname = os.path.join(self.output_dir, 'precision_recall.pdf')
		plt.savefig(figname, dpi=self.dpi)


		# combined training and validation roc curves plot
		plt.clf()
		i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			train_class_result = self.train_class_results[model_name]
			if model_tag is None:
				train_class_result.roc_curve(plot=True, color=colors[i],
										 style=styles[i if i < len(styles) else 0], label='Training')
			else:
				train_class_result.roc_curve(plot=True, color=colors[i],
										 style=styles[i if i < len(styles) else 0], label='Training' + model_name)
			i += 1
        	i = 0
		for model_name in self.models.keys():
			model_tag = self.models[model_name]['tag']
			val_class_result = self.val_class_results[model_name]
			if model_tag is None:
				val_class_result.roc_curve(plot=True, color=colors[i],
										 style=styles[i if i < len(styles) else 0], label='Validation')
			else:
				val_class_result.roc_curve(plot=True, color=colors[i],
							 style=styles[i if i < len(styles) else 0], label='Validation' + model_name)
			i += 1

		plt.title('ROC Curves', fontsize=self.font_size)
		handles, labels = plt.gca().get_legend_handles_labels()
		plt.legend(handles, labels, loc='best')
		figname = os.path.join(self.output_dir, 'ROC.pdf')
		plt.savefig(figname, dpi=self.dpi)
