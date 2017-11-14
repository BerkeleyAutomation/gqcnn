"""
Optimizer class for training a gqcnn(Grasp Quality Neural Network).
Author: Vishal Satish
"""
import argparse
import copy
import cv2
import json
import logging
import numpy as np
import cPickle as pkl
import os
import random
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import skimage.restoration as sr
import signal
import sys
import shutil
import threading
import time
import urllib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import yaml
from autolab_core import YamlConfig
import autolab_core.utils as utils
import collections

import IPython as ip

from learning_analysis import ClassificationResult, RegressionResult
from optimizer_constants import ImageMode, TrainingMode, PreprocMode, InputPoseMode, InputGripperMode, GeneralConstants, FileTemplates
from train_stats_logger import TrainStatsLogger

class SGDOptimizer(object):
	""" Optimizer for gqcnn object """

	def __init__(self, gqcnn, config):
		"""
		Parameters
		----------
		gqcnn : :obj:`GQCNN`
			grasp quality neural network to optimize
		config : dict
			dictionary of configuration parameters
		"""
		self.gqcnn = gqcnn
		self.cfg = config
		self.tensorboard_has_launched = False

	def _create_loss(self):
		""" Creates a loss based on config file

		Returns
		-------
		:obj:`tensorflow Tensor`
			loss
		"""
		# TODO: Add Poisson Loss
		if self.cfg['loss'] == 'l2':
			return tf.nn.l2_loss(tf.sub(self.train_net_output, self.train_labels_node))
		elif self.cfg['loss'] == 'sparse':
			return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=self.train_labels_node, logits=self.train_net_output, name=None))

	def _create_optimizer(self, loss, batch, var_list, learning_rate):
		""" Create optimizer based on config file

		Parameters
		----------
		loss : :obj:`tensorflow Tensor`
			loss to use, generated with _create_loss()
		batch : :obj:`tf.Variable`
			variable to keep track of the current gradient step number
		var_list : :obj:`lst`
			list of tf.Variable objects to update to minimize loss(ex. network weights)
		learning_rate : float
			learning rate for training

		Returns
		-------
		:obj:`tf.train.Optimizer`
			optimizer
		"""    
		if self.cfg['optimizer'] == 'momentum':
			optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum_rate)
			return optimizer.minimize(loss, global_step=batch, var_list=var_list), optimizer
		elif self.cfg['optimizer'] == 'adam':
            		optimizer = tf.train.AdamOptimizer(learning_rate)
			return optimizer.minimize(loss, global_step=batch, var_list=var_list), optimizer
		elif self.cfg['optimizer'] == 'rmsprop':
            		optimizer = tf.train.RMSPropOptimizer(learning_rate)
			return optimizer.minimize(loss, global_step=batch, var_list=var_list), optimizer
		else:
			raise ValueError('Optimizer %s not supported' %(self.cfg['optimizer']))

	def _check_dead_queue(self):
		""" Checks to see if the queue is dead and if so closes the tensorflow session and cleans up the variables """
		if self.dead_event.is_set():
			# close self.session
			self.sess.close()
			
			# cleanup
			for layer_weights in self.weights.values():
				del layer_weights
			del self.saver
			del self.sess

	def _launch_tensorboard(self):
		""" Launches Tensorboard to visualize training """
		logging.info("Launching Tensorboard, Please navigate to localhost:6006 in your favorite web browser to view summaries")
		os.system('tensorboard --logdir=' + self.summary_dir + " &>/dev/null &")

	def _close_tensorboard(self):
		""" Closes Tensorboard """
		logging.info('Closing Tensorboard')
		tensorboard_pid = os.popen('pgrep tensorboard').read()
		os.system('kill ' + tensorboard_pid)

	def optimize(self):
		""" Perform optimization """
		start_time = time.time()

		# run setup 
		self._setup()
		
		# build training and validation networks
		with tf.name_scope('validation_network'):
			logging.info('Building Validation Network')
			self.gqcnn.initialize_network() # builds validation network inside gqcnn class
		with tf.name_scope('training_network'):
			logging.info('Building Training Network')
			if self.gripper_dim > 0:
				self.train_net_output = self.gqcnn._build_network(self.input_im_node, self.input_pose_node, input_gripper_node=self.input_gripper_node)
			else:
				self.train_net_output = self.gqcnn._build_network(self.input_im_node, self.input_pose_node)

		# once weights have been initialized create tf Saver for weights
		self.saver = tf.train.Saver()

		# form loss
		# part 1: error
		if self.training_mode == TrainingMode.CLASSIFICATION or self.preproc_mode == PreprocMode.NORMALIZATION:
			train_predictions = tf.nn.softmax(self.train_net_output)
			self.gqcnn.add_softmax_to_predict()
			with tf.name_scope('loss'):
				loss = self._create_loss()
		elif self.training_mode == TrainingMode.REGRESSION:
			train_predictions = self.train_net_output
			with tf.name_scope('loss'):
				loss = self._create_loss()

		# part 2: regularization
		layer_weights = self.weights.values()
		with tf.name_scope('regularization'):
			regularizers = tf.nn.l2_loss(layer_weights[0])
			for w in layer_weights[1:]:
				regularizers = regularizers + tf.nn.l2_loss(w)
			loss += self.train_l2_regularizer * regularizers

		# setup learning rate
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(
			self.base_lr,                # base learning rate.
			batch * self.train_batch_size,  # current index into the dataset.
			self.decay_step,          # decay step.
			self.decay_rate,                # decay rate.
			staircase=True)

		# setup variable list
		var_list = self.weights.values()
		if self.cfg['fine_tune'] and self.cfg['update_fc_only']:
			var_list = [v for k, v in self.weights.iteritems() if k.find('conv') == -1]
		elif self.cfg['fine_tune'] and self.cfg['update_conv0_only'] and self.use_conv0:
			var_list = [v for k, v in self.weights.iteritems() if k.find('conv0') > -1]

		# create optimizer
		with tf.name_scope('optimizer'):
			optimizer, true_optimizer = self._create_optimizer(loss, batch, var_list, learning_rate)
			# if flag to save histograms is on, then generate gradient histograms and finally setup the summaries
			if self.save_histograms:
				gradients = true_optimizer.compute_gradients(loss, var_list)
				for gradient, variable in gradients:
						if isinstance(gradient, ops.IndexedSlices):
							grad_values = gradient.values
						else:
							grad_values = gradient
						if grad_values is not None:
							var_name = variable.name.replace(":", "_")
							tf.summary.histogram("gradients/%s" % var_name, grad_values, collections=['histogram'])
		
				self.merged_histogram_summaries = tf.summary.merge_all('histogram')		
				self._setup_summaries()
			
		def handler(signum, frame):
			logging.info('caught CTRL+C, exiting...')
			self.term_event.set()

			### Forcefully Exit ####
			# TODO: remove this and figure out why queue thread does not properly exit
			logging.info('Forcefully Exiting Optimization')
			self.forceful_exit = True

			# forcefully kill the session to terminate any current graph ops that are stalling because the enqueue op has ended
			self.sess.close()

			# close tensorboard
			self._close_tensorboard()

			# pause and wait for queue thread to exit before continuing
			logging.info('Waiting for Queue Thread to Exit')
			while not self.queue_thread_exited:
				pass

			logging.info('Cleaning and Preparing to Exit Optimization')
				
			# cleanup
			for layer_weights in self.weights.values():
				del layer_weights
			del self.saver
			del self.sess

			# exit
			logging.info('Exiting Optimization')

			# forcefully exit the script
			exit(0)

		signal.signal(signal.SIGINT, handler)

		# now that everything in our graph is set up we write the graph to the summary event so 
		# it can be visualized in tensorboard
		self.summary_writer.add_graph(self.gqcnn.get_tf_graph())

		# begin optimization loop
		try:
			self.queue_thread = threading.Thread(target=self._load_and_enqueue)
			self.queue_thread.start()

			# init and run tf self.sessions
			init = tf.global_variables_initializer()
			self.sess.run(init)
			logging.info('Beginning Optimization')

			# create a TrainStatsLogger object to log training statistics at certain intervals
			self.train_stats_logger = TrainStatsLogger(self.experiment_dir)

			# loop through training steps
			training_range = xrange(int(self.num_epochs * self.num_train) // self.train_batch_size)
			for step in training_range:
				# check for dead queue
				self._check_dead_queue()

				# run optimization
				extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				#check_numeric_op = tf.add_check_numerics_ops()
                		if self.gripper_dim > 0:
				    _, l, lr, predictions, batch_labels, output, train_images, pose_node, gripper_node, _ = self.sess.run(
						[optimizer, loss, learning_rate, train_predictions, self.train_labels_node, self.train_net_output, self.input_im_node, self.input_pose_node, self.input_gripper_node, extra_update_ops], options=GeneralConstants.timeout_option)
                		else:
                    			_, l, lr, predictions, batch_labels, output, train_images, pose_node, _ = self.sess.run(
                        			[optimizer, loss, learning_rate, train_predictions, self.train_labels_node, self.train_net_output, self.input_im_node, self.input_pose_node, extra_update_ops], options=GeneralConstants.timeout_option)
				ex = np.exp(output - np.tile(np.max(output, axis=1)[:,np.newaxis], [1,2]))
				softmax = ex / np.tile(np.sum(ex, axis=1)[:,np.newaxis], [1,2])
				
				logging.debug('Max ' +  str(np.max(softmax[:,1])))
				logging.debug('Min ' + str(np.min(softmax[:,1])))
				logging.debug('Pred nonzero ' + str(np.sum(np.argmax(predictions, axis=1))))
				logging.debug('True nonzero ' + str(np.sum(batch_labels)))

				# log output
				if step % self.log_frequency == 0:
					elapsed_time = time.time() - start_time
					start_time = time.time()
					logging.info('Step %d (epoch %.2f), %.1f s' %
						  (step, float(step) * self.train_batch_size / self.num_train,
						   1000 * elapsed_time / self.eval_frequency))
					logging.info('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
					train_error = l
					if self.training_mode == TrainingMode.CLASSIFICATION:
						train_error = ClassificationResult([predictions], [batch_labels]).error_rate
						logging.info('Minibatch error: %.3f%%' %train_error)
					self.summary_writer.add_summary(self.sess.run(self.merged_log_summaries, feed_dict={self.minibatch_error_placeholder: train_error, self.minibatch_loss_placeholder: l, self.learning_rate_placeholder: lr}), step)
					sys.stdout.flush()

					# update the TrainStatsLogger
					self.train_stats_logger.update(train_eval_iter=step, train_loss=l, train_error=train_error, total_train_error=None, val_eval_iter=None, val_error=None, learning_rate=lr)

				# evaluate validation error
				if step % self.eval_frequency == 0:
					if self.cfg['eval_total_train_error']:
						train_error = self._error_rate_in_batches()
						logging.info('Training error: %.3f' %train_error)

						# update the TrainStatsLogger and save
						self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, total_train_error=train_error, val_eval_iter=None, val_error=None, learning_rate=None)
						self.train_stats_logger.log()

					val_error = self._error_rate_in_batches()
					self.summary_writer.add_summary(self.sess.run(self.merged_eval_summaries, feed_dict={self.val_error_placeholder: val_error}), step)
					logging.info('Validation error: %.3f' %val_error)
					sys.stdout.flush()

					# update the TrainStatsLogger
					self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, total_train_error=None, val_eval_iter=step, val_error=val_error, learning_rate=None)

					# save everything!
					self.train_stats_logger.log()
				
				# save the model
				if step % self.save_frequency == 0 and step > 0:
					self.saver.save(self.sess, os.path.join(self.experiment_dir, 'model_%05d.ckpt' %(step)))
					self.saver.save(self.sess, os.path.join(self.experiment_dir, 'model.ckpt'))


				# launch tensorboard only after the first iteration
				if not self.tensorboard_has_launched:
					self.tensorboard_has_launched = True
					self._launch_tensorboard()
				
				# write histogram summaries if specified
				if self.save_histograms:
					self.summary_writer.add_summary(self.sess.run([self.merged_histogram_summaries], feed_dict={self.gqcnn.input_im_node:train_images, self.gqcnn.input_pose_node:pose_node}))
					self.summary_writer.add_summary(merged_histogram_summaries)

			# get final logs
			val_error = self._error_rate_in_batches()
			logging.info('Final validation error: %.1f%%' %val_error)
			sys.stdout.flush()

			# update the TrainStatsLogger
			self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, total_train_error=None, val_eval_iter=step, val_error=val_error, learning_rate=None)

			# log & save everything!
			self.train_stats_logger.log()
			self.saver.save(self.sess, os.path.join(self.experiment_dir, 'model.ckpt'))

		except Exception as e:
			self.term_event.set()
			if not self.forceful_exit:
				self.sess.close() 
				for layer_weights in self.weights.values():
					del layer_weights
				del self.saver
				del self.sess
			raise

		# check for dead queue
		self._check_dead_queue()

		# close sessions
		self.term_event.set()

		# close tensorboard
		self._close_tensorboard()

		# TODO: remove this and figure out why queue thread does not properly exit
		self.sess.close()

		# pause and wait for queue thread to exit before continuing
		logging.info('Waiting for Queue Thread to Exit')
		while not self.queue_thread_exited:
			pass

		logging.info('Cleaning and Preparing to Exit Optimization')
		self.sess.close()
			
		# cleanup
		for layer_weights in self.weights.values():
			del layer_weights
		del self.saver
		del self.sess

		# exit
		logging.info('Exiting Optimization')

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
			return gripper_param_arr[:, 0:1]
		elif input_gripper_mode == InputGripperMode.ALL:
			return gripper_param_arr
		else:
			raise ValueError('Input gripper mode {} not supportd'.format(input_gripper_mode))

	def _setup_summaries(self):
		""" Sets up placeholders for summary values and creates summary writer """
		# we create placeholders for our python values because summary_scalar expects
		# a placeholder, not simply a python value 
		self.val_error_placeholder = tf.placeholder(tf.float32, [])
		self.minibatch_error_placeholder = tf.placeholder(tf.float32, [])
		self.minibatch_loss_placeholder = tf.placeholder(tf.float32, [])
		self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

		# we create summary scalars with tags that allow us to group them together so we can write different batches
		# of summaries at different intervals
		tf.summary.scalar('val_error', self.val_error_placeholder, collections=["eval_frequency"])
		tf.summary.scalar('minibatch_error', self.minibatch_error_placeholder, collections=["log_frequency"])
		tf.summary.scalar('minibatch_loss', self.minibatch_loss_placeholder, collections=["log_frequency"])
		tf.summary.scalar('learning_rate', self.learning_rate_placeholder, collections=["log_frequency"])
		self.merged_eval_summaries = tf.summary.merge_all("eval_frequency")
		self.merged_log_summaries = tf.summary.merge_all("log_frequency")

		# create a tf summary writer with the specified summary directory
		self.summary_writer = tf.summary.FileWriter(self.summary_dir)
		self.gqcnn.set_summary_writer(self.summary_writer)
		# initialize the variables again now that we have added some new ones
		with self.sess.as_default():
			tf.global_variables_initializer().run()

	def _setup_tensorflow(self):
		"""Setup Tensorflow placeholders, session, and queue """

		# setup nodes
		with tf.name_scope('train_data_node'):
			self.train_data_batch = tf.placeholder(tf.float32, (self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels))
		with tf.name_scope('train_pose_node'):
			self.train_poses_batch = tf.placeholder(tf.float32, (self.train_batch_size, self.pose_dim))
		if self.training_mode == TrainingMode.REGRESSION:
			train_label_dtype = tf.float32
			self.numpy_dtype = np.float32
		elif self.training_mode == TrainingMode.CLASSIFICATION:
			train_label_dtype = tf.int64
			self.numpy_dtype = np.int64
		else:
			raise ValueError('Training mode %s not supported' %(self.training_mode))
		with tf.name_scope('train_labels_node'):
			self.train_labels_batch = tf.placeholder(train_label_dtype, (self.train_batch_size,))
		if self.gripper_dim > 0:
			# feed gripper channel in network
			with tf.name_scope('train_gripper_node'):
				self.train_gripper_batch = tf.placeholder(tf.float32, (self.train_batch_size, self.gripper_dim))

		# create queue
		with tf.name_scope('data_queue'):
			if self.gripper_dim > 0:
				self.q = tf.FIFOQueue(self.queue_capacity, [tf.float32, tf.float32, tf.float32, train_label_dtype], shapes=[(self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels),
				 (self.train_batch_size, self.pose_dim), (self.train_batch_size, self.gripper_dim), (self.train_batch_size,)])
				self.enqueue_op = self.q.enqueue([self.train_data_batch, self.train_poses_batch, self.train_gripper_batch, self.train_labels_batch])
				self.train_labels_node = tf.placeholder(train_label_dtype, (self.train_batch_size,))
				self.input_im_node, self.input_pose_node, self.input_gripper_node, self.train_labels_node = self.q.dequeue()
			else:
				self.q = tf.FIFOQueue(self.queue_capacity, [tf.float32, tf.float32, train_label_dtype], shapes=[(self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels),
				 (self.train_batch_size, self.pose_dim), (self.train_batch_size,)])
				self.enqueue_op = self.q.enqueue([self.train_data_batch, self.train_poses_batch, self.train_labels_batch])
				self.train_labels_node = tf.placeholder(train_label_dtype, (self.train_batch_size,))
				self.input_im_node, self.input_pose_node, self.train_labels_node = self.q.dequeue()

		# setup weights using gqcnn
		if self.cfg['fine_tune']:
			# this assumes that a gqcnn was passed in that was initialized with weights from a model using GQCNN.load(), so all that has to
			# be done is to possibly reinitialize fc3/fc4/fc5
			self.gqcnn.reinitialize_layers(self.cfg['reinit_fc3'], self.cfg['reinit_fc4'], self.cfg['reinit_fc5'])

		# get weights
		self.weights = self.gqcnn.get_weights()

		# open a tf session for the gqcnn object and store it also as the optimizer session
		self.sess = self.gqcnn.open_session()

		# setup term event/dead event
		self.term_event = threading.Event()
		self.term_event.clear()
		self.dead_event = threading.Event()
		self.dead_event.clear()

	def _compute_data_metrics(self):
		""" Calculate image mean, image std, pose mean, pose std, gripper mean, gripper std, metric params """

		# compute data mean
		logging.info('Computing image mean')
		mean_filename = os.path.join(self.experiment_dir, 'image_mean.npy')
		std_filename = os.path.join(self.experiment_dir, 'image_std.npy')
		if self.cfg['fine_tune']:
			self.data_mean = self.gqcnn.get_im_mean()
			self.data_std = self.gqcnn.get_im_std()
		else:
			self.data_mean = 0
			self.data_std = 0
			random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
			num_summed = 0
			for k in random_file_indices.tolist():
				im_filename = self.im_filenames[k]
				im_data = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
				if self.cfg['border_distortion']:
					self._distort(im_data, only_dropout=True)
				self.data_mean += np.sum(im_data[self.train_index_map[im_filename], :, :, :])
				num_summed += im_data[self.train_index_map[im_filename], :, :, :].shape[0]
			self.data_mean = self.data_mean / (num_summed * self.im_height * self.im_width)

			for k in random_file_indices.tolist():
				im_filename = self.im_filenames[k]
				im_data = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
				self.data_std += np.sum((im_data[self.train_index_map[im_filename], :, :, :] - self.data_mean)**2)
			self.data_std = np.sqrt(self.data_std / (num_summed * self.im_height * self.im_width))
			
			np.save(mean_filename, self.data_mean)
			np.save(std_filename, self.data_std)

		# compute pose mean
		logging.info('Computing pose mean')
		self.pose_mean_filename = os.path.join(self.experiment_dir, 'pose_mean.npy')
		self.pose_std_filename = os.path.join(self.experiment_dir, 'pose_std.npy')
		if self.cfg['fine_tune']:
			self.pose_mean = self.gqcnn.get_pose_mean()
			self.pose_std = self.gqcnn.get_pose_std()
		else:
			self.pose_mean = np.zeros(self.pose_shape)
			self.pose_std = np.zeros(self.pose_shape)
			num_summed = 0
			random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
			for k in random_file_indices.tolist():
				im_filename = self.im_filenames[k]
				pose_filename = self.pose_filenames[k]
				self.pose_data = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
				if self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
					rand_indices = np.random.choice(self.pose_data.shape[0], size=self.pose_data.shape[0]/2, replace=False)
					self.pose_data[rand_indices, 3] = -self.pose_data[rand_indices, 3]
				pose_data = self.pose_data[self.train_index_map[im_filename],:]
				pose_data = pose_data[np.isfinite(pose_data[:,3]),:]
				self.pose_mean += np.sum(pose_data, axis=0)
				num_summed += pose_data.shape[0]
			self.pose_mean = self.pose_mean / num_summed

			for k in random_file_indices.tolist():
				im_filename = self.im_filenames[k]
				pose_filename = self.pose_filenames[k]
				self.pose_data = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
				if self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
					self.pose_data[rand_indices, 3] = -self.pose_data[rand_indices, 3]
				pose_data = self.pose_data[self.train_index_map[im_filename],:]
				pose_data = pose_data[np.isfinite(pose_data[:,3]), :]
				self.pose_std += np.sum((pose_data - self.pose_mean)**2, axis=0)
			self.pose_std = np.sqrt(self.pose_std / num_summed)

			self.pose_std[self.pose_std==0] = 1.0

			np.save(self.pose_mean_filename, self.pose_mean)
			np.save(self.pose_std_filename, self.pose_std)

		# compute gripper param mean
		if self.gripper_dim > 0:
			logging.info('Computing gripper mean')
			self.gripper_mean_filename = os.path.join(self.experiment_dir, 'gripper_mean.npy')
			self.gripper_std_filename = os.path.join(self.experiment_dir, 'gripper_std.npy')
			if self.cfg['fine_tune']:
				self.gripper_mean = self.gqcnn.get_gripper_mean()
				self.gripper_std = self.gqcnn.get_gripper_std()
			else:
				self.gripper_mean = np.zeros(self.gripper_shape)
				self.gripper_std = np.zeros(self.gripper_shape)
				num_summed = 0
				random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
				for k in random_file_indices.tolist():
					im_filename = self.im_filenames[k]
					gripper_filename = self.gripper_param_filenames[k]
					self.gripper_data = np.load(os.path.join(self.data_dir, gripper_filename))['arr_0']
					gripper_data = self.gripper_data[self.train_index_map[im_filename],:]
					self.gripper_mean += np.sum(gripper_data, axis=0)
					num_summed += gripper_data.shape[0]
				self.gripper_mean = self.gripper_mean / num_summed

				for k in random_file_indices.tolist():
					im_filename = self.im_filenames[k]
					gripper_filename = self.gripper_param_filenames[k]
					self.gripper_data = np.load(os.path.join(self.data_dir, gripper_filename))['arr_0']
					gripper_data = self.gripper_data[self.train_index_map[im_filename],:]
					self.gripper_std += np.sum((gripper_data - self.gripper_mean)**2, axis=0)
				self.gripper_std = np.sqrt(self.gripper_std / num_summed)

				self.gripper_std[self.gripper_std==0] = 1.0

				np.save(self.gripper_mean_filename, self.gripper_mean)
				np.save(self.gripper_std_filename, self.gripper_std)

		if self.cfg['fine_tune']:
			out_mean_filename = os.path.join(self.experiment_dir, 'image_mean.npy')
			out_std_filename = os.path.join(self.experiment_dir, 'image_std.npy')
			out_pose_mean_filename = os.path.join(self.experiment_dir, 'pose_mean.npy')
			out_pose_std_filename = os.path.join(self.experiment_dir, 'pose_std.npy')
			out_gripper_mean_filename = os.path.join(self.experiment_dir, 'gripper_mean.npy')
			out_gripper_std_filename = os.path.join(self.experiment_dir, 'gripper_std.npy')
			np.save(out_mean_filename, self.data_mean)
			np.save(out_std_filename, self.data_std)
			np.save(out_pose_mean_filename, self.pose_mean)
			np.save(out_pose_std_filename, self.pose_std)
			np.save(out_gripper_mean_filename, self.gripper_mean)
			np.save(out_gripper_std_filename, self.gripper_std)

		# update gqcnn im mean & std
		self.gqcnn.update_im_mean(self.data_mean)
		self.gqcnn.update_im_std(self.data_std)

		# update gqcnn pose mean & std
		self.gqcnn.update_pose_mean(self._read_pose_data(self.pose_mean, self.input_pose_mode))
		self.gqcnn.update_pose_std(self._read_pose_data(self.pose_std, self.input_pose_mode))
        
	        if self.gripper_dim > 0:
		    # update gqcnn gripper mean & std
		    self.gqcnn.update_gripper_mean(self._read_gripper_data(self.gripper_mean, self.input_gripper_mode))
		    self.gqcnn.update_gripper_std(self._read_gripper_data(self.gripper_std, self.input_gripper_mode))

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
		self.decay_step = self.decay_step_multiplier * self.num_train

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

		# throw an excpetion if the object ids are not in the dataset
		if not self._obj_files_exist:
			raise RuntimeError('Object Id Files were not found in dataset')

		# get total number of training datapoints and set the decay_step
		num_datapoints = self.images_per_file * self.num_files
		self.num_train = int(self.train_pct * num_datapoints)
		self.decay_step = self.decay_step_multiplier * self.num_train

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

		# throw an excpetion if the stable_pose_labels are not in the dataset
		if not self._stable_pose_files_exist:
			raise RuntimeError('Stable Pose Files were not found in dataset')

		# get total number of training datapoints and set the decay_step
		num_datapoints = self.images_per_file * self.num_files
		self.num_train = int(self.train_pct * num_datapoints)
		self.decay_step = self.decay_step_multiplier * self.num_train
		
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

	def _read_training_params(self):
		""" Read training parameters from configuration file """

		self.data_dir = self.cfg['dataset_dir']
		self.image_mode = self.cfg['image_mode']
		self.data_split_mode = self.cfg['data_split_mode']
		self.train_pct = self.cfg['train_pct']
		self.total_pct = self.cfg['total_pct']

		self.train_batch_size = self.cfg['train_batch_size']
		self.val_batch_size = self.cfg['val_batch_size']

		# update the GQCNN's batch_size param to this one
		self.gqcnn.update_batch_size(self.val_batch_size)

		self.num_epochs = self.cfg['num_epochs']
		self.eval_frequency = self.cfg['eval_frequency']
		self.save_frequency = self.cfg['save_frequency']
		self.log_frequency = self.cfg['log_frequency']
		self.vis_frequency = self.cfg['vis_frequency']

		self.queue_capacity = self.cfg['queue_capacity']
		self.queue_sleep = self.cfg['queue_sleep']

		self.train_l2_regularizer = self.cfg['train_l2_regularizer']
		self.base_lr = self.cfg['base_lr']
		self.decay_step_multiplier = self.cfg['decay_step_multiplier']
		self.decay_rate = self.cfg['decay_rate']
		self.momentum_rate = self.cfg['momentum_rate']
		self.max_training_examples_per_load = self.cfg['max_training_examples_per_load']

		self.target_metric_name = self.cfg['target_metric_name']
		self.metric_thresh = self.cfg['metric_thresh']
		self.training_mode = self.cfg['training_mode']
		self.preproc_mode = self.cfg['preproc_mode']

		self.save_histograms = self.cfg['save_histograms']

		if self.train_pct < 0 or self.train_pct > 1:
			raise ValueError('Train percentage must be in range [0,1]')

		if self.total_pct < 0 or self.total_pct > 1:
			raise ValueError('Train percentage must be in range [0,1]')

		self.gqcnn.set_mask_and_inpaint(self.cfg['mask_and_inpaint'])
		
	def _read_data_params(self):
		""" Read data parameters from configuration file """

		self.train_im_data = np.load(os.path.join(self.data_dir, self.im_filenames[0]))['arr_0']
		self.pose_data = np.load(os.path.join(self.data_dir, self.pose_filenames[0]))['arr_0']
		self.gripper_data = np.load(os.path.join(self.data_dir, self.gripper_param_filenames[0]))['arr_0']
		self.metric_data = np.load(os.path.join(self.data_dir, self.label_filenames[0]))['arr_0']
		self.images_per_file = self.train_im_data.shape[0]
		self.im_height = self.train_im_data.shape[1]
		self.im_width = self.train_im_data.shape[2]
		self.im_channels = self.train_im_data.shape[3]
		self.im_center = np.array([float(self.im_height-1)/2, float(self.im_width-1)/2])
		self.num_tensor_channels = self.cfg['num_tensor_channels']
		self.pose_shape = self.pose_data.shape[1]
		self.gripper_shape = self.gripper_data.shape[1]
		self.input_pose_mode = self.cfg['input_pose_mode']
		self.input_gripper_mode = self.cfg['input_gripper_mode']
		
		# update pose dimension according to input_pose_mode for creation of tensorflow placeholders
		if self.input_pose_mode == InputPoseMode.TF_IMAGE:
			self.pose_dim = 1 # depth
		elif self.input_pose_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
			self.pose_dim = 3 # depth, cx, cy
		elif self.input_pose_mode == InputPoseMode.RAW_IMAGE:
			self.pose_dim = 4 # u, v, theta, depth
		elif self.input_pose_mode == InputPoseMode.RAW_IMAGE_PERSPECTIVE:
			self.pose_dim = 6 # u, v, theta, depth cx, cy
		elif self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
			self.pose_dim = 2 # depth, theta
		else:
			raise ValueError('Input pose mode %s not understood' %(self.input_pose_mode))

		# update gripper dimension according to input_gripper_mode for creation of tensorflow placeholders
		if self.input_gripper_mode == InputGripperMode.WIDTH:
			self.gripper_dim = 1 # width
		elif self.input_gripper_mode == InputGripperMode.NONE:
			self.gripper_dim = 0 # no gripper channel
		elif self.input_gripper_mode == InputGripperMode.ALL:
			self.gripper_dim = 4 # width, palm depth, fx, fy
        elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK or self.input_gripper_mode == SEG_MASK:
            self.gripper_dim = 0 # no gripper channel
            self.num_tensor_channels += 1 # mask will be added as channel to depth image
		else:
			raise ValueError('Input gripper mode %s not understood' %(self.input_gripper_mode))

		self.num_files = len(self.im_filenames)
		self.num_random_files = min(self.num_files, self.cfg['num_random_files'])
		self.num_categories = 2

	def _setup_denoising_and_synthetic(self):
		""" Setup denoising and synthetic data parameters """

		if self.cfg['multiplicative_denoising']:
			self.gamma_shape = self.cfg['gamma_shape']
			self.gamma_scale = 1.0 / self.gamma_shape

	def _setup_data_filenames(self):
		""" Setup data filenames, subsample files, check validity of filenames"""

		# read in filenames of training data(poses, images, labels, obj_id's, stable_poses, gripper_params)
		logging.info('Reading filenames')
		all_filenames = os.listdir(self.data_dir)
		if self.image_mode== ImageMode.BINARY:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.binary_im_tensor_template) > -1]
		elif self.image_mode== ImageMode.DEPTH:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.depth_im_tensor_template) > -1]
		elif self.image_mode== ImageMode.BINARY_TF:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.binary_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.COLOR_TF:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.color_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.GRAY_TF:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.gray_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.DEPTH_TF:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.depth_im_tf_tensor_template) > -1]
		elif self.image_mode== ImageMode.DEPTH_TF_TABLE:
			self.im_filenames = [f for f in all_filenames if f.find(FileTemplates.depth_im_tf_table_tensor_template) > -1]
		else:
			raise ValueError('Image mode %s not supported.' %(self.image_mode))

		self.pose_filenames = [f for f in all_filenames if f.find(FileTemplates.hand_poses_template) > -1]
		self.label_filenames = [f for f in all_filenames if f.find(self.target_metric_name) > -1]
		# since these are not required in the dataset, we fill them with FileTemplates.FILENAME_PLACEHOLDER just to prevent sorting exceptions down the line 
		# however, if they do not exist then exceptions will be thrown if the user tries to use object_wise/pose_wise splits 
		# or tries to input the gripper paramters to the network during training
		self.obj_id_filenames = [f if (f.find(FileTemplates.object_labels_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]
		self._obj_files_exist = True
		if self.obj_id_filenames[0] == FileTemplates.FILENAME_PLACEHOLDER:
			self._obj_files_exist = False
		self.stable_pose_filenames = [f if (f.find(FileTemplates.pose_labels_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]
		self._stable_pose_files_exist = True
		if self.stable_pose_filenames[0] == FileTemplates.FILENAME_PLACEHOLDER:
			self._stable_pose_files_exist = False
		self.gripper_param_filenames = [f if (f.find(FileTemplates.gripper_params_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]
        self.gripper_depth_mask_fingertip_filenames = [f if (f.find(FileTemplates.fingertip_depth_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]
        self.gripper_depth_mask_palm_filenames = [f if (f.find(FileTemplates.palm_depth_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]
        self.gripper_seg_mask_fingertip_filenames = [f if (f.find(FileTemplates.fingertip_segmask_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]
        self.gripper_seg_mask_palm_filenames = [f if (f.find(FileTemplates.palm_segmask_template) > -1) else FileTemplates.FILENAME_PLACEHOLDER for f in all_filenames]

		if self.debug:
			# sort
			self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
			self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
			self.label_filenames.sort(key = lambda x: int(x[-9:-4]))
			self.obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
			self.stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))
			self.gripper_param_filenames.sort(key = lambda x: int(x[-9:-4]))
            self.gripper_depth_mask_fingertip_filenames.sort(key = lambda x: int(x[-9:-4]))
            self.gripper_depth_mask_palm_filenames.sort(key = lambda x: int(x[-9:-4]))
            self.gripper_seg_mask_fingertip_filenames.sort(key = lambda x: int(x[-9:-4]))
            self.gripper_seg_mask_palm_filenames.sort(key = lambda x: int(x[-9:-4]))

			# pack, shuffle and sample
			zipped = zip(self.im_filenames, self.pose_filenames, self.label_filenames, self.obj_id_filenames, 
				self.stable_pose_filenames, self.gripper_param_filenames, self.gripper_depth_mask_fingertip_filenames, self.gripper_depth_mask_palm_filenames, self.gripper_seg_mask_fingertip_filenames, self.gripper_seg_mask_palm_filenames)

			random.shuffle(zipped)
			zipped = zipped[:self.debug_num_files]

			# unpack
			self.im_filenames, self.pose_filenames, self.label_filenames, self.obj_id_filenames, self.stable_pose_filenames, self.gripper_param_filenames, self.gripper_depth_mask_fingertip_filenames, self.gripper_depth_mask_palm_filenames, self.gripper_seg_mask_fingertip_filenames, self.gripper_seg_mask_palm_filenames = zip(*zipped)

		self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.label_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.gripper_param_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.gripper_depth_mask_fingertip_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.gripper_depth_mask_palm_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.gripper_seg_mask_fingertip_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.gripper_seg_mask_palm_filenames.sort(key = lambda x: int(x[-9:-4]))

		# check valid filenames
		if len(self.im_filenames) == 0 or len(self.pose_filenames) == 0 or len(self.label_filenames) == 0:
			raise RuntimeError('One or more required training files(Images, Poses, Labels) in the dataset could not be found.')

		# subsample files
		self.num_files = len(self.im_filenames)
		num_files_used = int(self.total_pct * self.num_files)
		filename_indices = np.random.choice(self.num_files, size=num_files_used, replace=False)
		filename_indices.sort()
		self.im_filenames = [self.im_filenames[k] for k in filename_indices]
		self.pose_filenames = [self.pose_filenames[k] for k in filename_indices]
		self.label_filenames = [self.label_filenames[k] for k in filename_indices]
		self.obj_id_filenames = [self.obj_id_filenames[k] for k in filename_indices]   
		self.stable_pose_filenames = [self.stable_pose_filenames[k] for k in filename_indices]
		self.gripper_param_filenames = [self.gripper_param_filenames[k] for k in filename_indices]
		self.gripper_depth_mask_fingertip_filenames = [self.gripper_depth_mask_fingertip_filenames[k] for k in filename_indices]
        self.gripper_depth_mask_palm_filenames = [self.gripper_depth_mask_palm_filenames[k] for k in filename_indices]
        self.gripper_seg_mask_fingertip_filenames = [self.gripper_seg_mask_fingertip_filenames[k] for k in filename_indices]
        self.gripper_seg_mask_palm_filenames = [self.gripper_seg_mask_palm_filenames[k] for k in filename_indices]

		# create copy of image, pose, gripper_param, and label filenames because original cannot be accessed by load and enqueue op in the case that 		     the error_rate_in_batches method is sorting the original
		self.im_filenames_copy = self.im_filenames[:]
		self.pose_filenames_copy = self.pose_filenames[:]
		self.label_filenames_copy = self.label_filenames[:]
		self.gripper_depth_mask_fingertip_filenames_copy = self.gripper_depth_mask_fingertip_filenames[:]
        self.gripper_depth_mask_palm_filenames_copy = self.gripper_depth_mask_palm_filenames[:]
        self.gripper_seg_mask_fingertip_filenames_copy = self.gripper_seg_mask_fingertip_filenames[:]
        self.gripper_seg_mask_palm_filenames_copy = self.gripper_seg_mask_palm_filenames[:]
        self.gripper_param_filenames_copy = self.gripper_param_filenames[:]

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

	def _copy_config(self):
		""" Keep a copy of original config files """

		out_config_filename = os.path.join(self.experiment_dir, 'config.json')
		tempOrderedDict = collections.OrderedDict()
		for key in self.cfg.keys():
			tempOrderedDict[key] = self.cfg[key]
		with open(out_config_filename, 'w') as outfile:
			json.dump(tempOrderedDict, outfile)
		this_filename = sys.argv[0]
		out_train_filename = os.path.join(self.experiment_dir, 'training_script.py')
		shutil.copyfile(this_filename, out_train_filename)
		out_architecture_filename = os.path.join(self.experiment_dir, 'architecture.json')
		json.dump(self.cfg['gqcnn_config']['architecture'], open(out_architecture_filename, 'w'))

	def _setup(self):
		""" Setup for optimization """

		# set up logger
		logging.getLogger().setLevel(logging.INFO)

		self.debug = self.cfg['debug']
		
		# initialize thread exit booleans
		self.queue_thread_exited = False
		self.forceful_exit = False

		# set random seed for deterministic execution
		if self.debug:
			np.random.seed(GeneralConstants.SEED)
			random.seed(GeneralConstants.SEED)
			self.debug_num_files = self.cfg['debug_num_files']

		# setup output directories
		self._setup_output_dirs()

		# copy config file
		self._copy_config()

		# read training parameters from config file
		self._read_training_params()

		# setup denoising and synthetic data parameters
		self._setup_denoising_and_synthetic()

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
			raise ValueError('Data Split Mode Not Supported')

		# compute means, std's, and normalization metrics
		self._compute_data_metrics()

		# setup tensorflow session/placeholders/queue
		self._setup_tensorflow()

		# setup summaries for visualizing metrics in tensorboard
		# do this here if we are not saving histograms, else it will be done later after gradient/weight/etc. histograms have been setup
		if not self.save_histograms:
			self._setup_summaries()
		
		self._num_original_train_images_saved = 0
		self._num_distorted_train_images_saved = 0
		self._num_original_val_images_saved = 0
		self._num_distorted_val_images_saved = 0
  
	def _load_and_enqueue(self):
		""" Loads and Enqueues a batch of images, poses, labels, and possibly gripper paramters for training """

		# read parameters of gaussian process
		self.gp_rescale_factor = self.cfg['gaussian_process_scaling_factor']
		self.gp_sample_height = int(self.im_height / self.gp_rescale_factor)
		self.gp_sample_width = int(self.im_width / self.gp_rescale_factor)
		self.gp_num_pix = self.gp_sample_height * self.gp_sample_width
		self.gp_sigma = self.cfg['gaussian_process_sigma']

		while not self.term_event.is_set():
			time.sleep(self.cfg['queue_sleep'])

			# loop through data
			num_queued = 0
			start_i = 0
			end_i = 0
			file_num = 0

			# init buffers
			train_data = np.zeros(
						[self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels]).astype(np.float32)
			train_poses = np.zeros([self.train_batch_size, self.pose_dim]).astype(np.float32)
			label_data = np.zeros(self.train_batch_size).astype(self.numpy_dtype)
			if self.gripper_dim > 0:
				train_gripper = np.zeros((self.train_batch_size, self.gripper_dim)).astype(np.float32)

			while start_i < self.train_batch_size:
				# compute num remaining
				num_remaining = self.train_batch_size - num_queued

				# gen file index uniformly at random
				file_num = np.random.choice(len(self.im_filenames_copy), size=1)[0]
				train_data_filename = self.im_filenames_copy[file_num]

				self.train_data_arr = np.load(os.path.join(self.data_dir, train_data_filename))[
										 'arr_0'].astype(np.float32)
				self.train_poses_arr = np.load(os.path.join(self.data_dir, self.pose_filenames_copy[file_num]))[
										  'arr_0'].astype(np.float32)
				self.train_label_arr = np.load(os.path.join(self.data_dir, self.label_filenames_copy[file_num]))[
										  'arr_0'].astype(np.float32)
				if self.gripper_dim > 0:
					self.train_gripper_arr = np.load(os.path.join(self.data_dir, self.gripper_param_filenames_copy[file_num]))[
										  'arr_0'].astype(np.float32)
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK or self.input_gripper_mode == InputGripperMode.SEG_MASK:
                    self.train_gripper_depth_mask_fingertip_arr = np.load(os.path.join(self.data_dir, self.gripper_depth_mask_fingertip_filenames_copy[file_num]))[ 'arr_0'].astype(np.float32)
                    self.train_gripper_depth_mask_palm_arr = np.load(os.path.join(self.data_dir, self.gripper_depth_mask_palm_filenames_copy[file_num]))[ 'arr_0'].astype(np.float32)
                    self.train_gripper_seg_mask_fingertip_arr = np.load(os.path.join(self.data_dir, self.gripper_seg_mask_fingertip_filenames_copy[file_num]))[ 'arr_0'].astype(np.float32)
                    self.train_gripper_seg_mask_palm_arr = np.load(os.path.join(self.data_dir, self.gripper_seg_mask_palm_filenames_copy[file_num]))[ 'arr_0'].astype(np.float32)

				# TODO: Remove this?
				if self.pose_dim == 1 and self.train_poses_arr.shape[1] == 6:
					self.train_poses_arr = self.train_poses_arr[:, :4]

				# get batch indices uniformly at random
				train_ind = self.train_index_map[train_data_filename]
				np.random.shuffle(train_ind)
				if self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
					tp_tmp = self._read_pose_data(self.train_poses_arr.copy(), self.input_pose_mode)
					train_ind = train_ind[np.isfinite(tp_tmp[train_ind,1])]
				upper = min(num_remaining, train_ind.shape[
							0], self.max_training_examples_per_load)
				ind = train_ind[:upper]
				num_loaded = ind.shape[0]
				end_i = start_i + num_loaded

				# subsample data
				self.train_data_arr = self.train_data_arr[ind, ...]
				self.train_poses_arr = self.train_poses_arr[ind, :]
				self.train_label_arr = self.train_label_arr[ind]
				if self.gripper_dim > 0:
					self.train_gripper_arr = self.train_gripper_arr[ind]
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK or self.input_gripper_mode == InputGripperMode.SEG_MASK:
                    self.train_gripper_depth_mask_fingertip_arr = self.train_gripper_depth_mask_fingertip_arr[ind]
                    self.train_gripper_depth_mask_palm_arr = self.train_gripper_depth_mask_palm_arr[ind]
                    self.train_gripper_seg_mask_fingertip_arr = self.train_gripper_seg_mask_fingertip_arr[ind]
                    self.train_gripper_seg_mask_palm_arr = self.train_gripper_seg_mask_palm_arr[ind]
				self.num_images = self.train_data_arr.shape[0]
				
				# save undistorted train images for debugging
				if self.cfg['save_original_train_images']:
					if self._num_original_train_images_saved < self.cfg['num_original_train_images']:
						output_dir = os.path.join(self.experiment_dir, 'original_train_images')
						if not os.path.exists(output_dir):
							os.mkdir(output_dir)
						np.savez_compressed(os.path.join(output_dir, 'original_image_{}'.format(self._num_original_train_images_saved)), self.train_data_arr[0, :, :, 0])
						self._num_original_train_images_saved += 1	

				# add noise to images
				if self.cfg['mask_and_inpaint']:
					# allocate mask tensor if required
					mask_arr = np.zeros((self.train_data_arr.shape[0], self.train_data_arr.shape[1], self.train_data_arr.shape[2], self.num_tensor_channels))
					mask_arr[:, :, :, 0] = self.train_data_arr[:, :, :, 0]
					self.train_data_arr = mask_arr
					self.train_data_arr, self.train_poses_arr = self._distort(self.train_data_arr, self.train_poses_arr, mask_and_inpaint=True)
				else:
					self.train_data_arr, self.train_poses_arr = self._distort(self.train_data_arr, self.train_poses_arr)

				# save distorted train images for debugging 
				if self.cfg['save_distorted_train_images']:
					if self._num_distorted_train_images_saved < self.cfg['num_distorted_train_images']:
						output_dir = os.path.join(self.experiment_dir, 'distorted_train_images')
						if not os.path.exists(output_dir):
							os.mkdir(output_dir)
						np.savez_compressed(os.path.join(output_dir, 'distorted_image_{}'.format(self._num_distorted_train_images_saved)), self.train_data_arr[0, :, :, 0])
						self._num_distorted_train_images_saved += 1

			    self.train_data_arr[:, :, :, 0] = (self.train_data_arr[:, :, :, 0] - self.data_mean) / self.data_std
				self.train_poses_arr = (self.train_poses_arr - self.pose_mean) / self.pose_std
				if self.gripper_dim > 0:
					self.train_gripper_arr = (self.train_gripper_arr - self.gripper_mean) / self.gripper_std
		
				# normalize labels
				if self.training_mode == TrainingMode.REGRESSION:
					if self.preproc_mode == PreprocMode.NORMALIZATION:
						self.train_label_arr = (self.train_label_arr - self.min_metric) / (self.max_metric - self.min_metric)
				elif self.training_mode == TrainingMode.CLASSIFICATION:
					self.train_label_arr = 1 * (self.train_label_arr > self.metric_thresh)
					self.train_label_arr = self.train_label_arr.astype(self.numpy_dtype)

				# enqueue training data batch
				train_data[start_i:end_i, ...] = self.train_data_arr
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                    train_data[:, :, :, 1] = self.train_gripper_depth_mask_palm_arr
                    train_data[:, :, :, 2] = self.train_gripper_depth_mask_fingertip_arr
                if self.input_gripper_mode == InputGripperMode.SEG_MASK:
                    train_data[:, :, :, 1] = self.train_gripper_seg_mask_palm_arr
                    train_data[:, :, :, 2] = self.train_gripper_seg_mask_fingertip_arr
				train_poses[start_i:end_i,:] = self._read_pose_data(self.train_poses_arr, self.input_pose_mode)
				label_data[start_i:end_i] = self.train_label_arr
				if self.gripper_dim > 0:
					train_gripper[start_i:end_i] = self._read_gripper_data(self.train_gripper_arr, self.input_gripper_mode)

				del self.train_data_arr
				del self.train_poses_arr
				del self.train_label_arr
				if self.gripper_dim > 0:
					del self.train_gripper_arr
		
				# update start index
				start_i = end_i
				num_queued += num_loaded
		  
			# send data to queue
			if not self.term_event.is_set():
				try:
					if self.gripper_dim > 0:
						self.sess.run(self.enqueue_op, feed_dict={self.train_data_batch: train_data,
												self.train_poses_batch: train_poses,
												self.train_gripper_batch: train_gripper,
												self.train_labels_batch: label_data})
					else:
						self.sess.run(self.enqueue_op, feed_dict={self.train_data_batch: train_data,
												self.train_poses_batch: train_poses,
												self.train_labels_batch: label_data})
				except:
					pass

		del train_data
		del train_poses
		del label_data
		if self.gripper_dim > 0:
			del train_gripper

		self.dead_event.set()
		logging.info('Queue Thread Exiting')
		self.queue_thread_exited = True

	def _distort(self, im_arr, pose_arr=None, only_dropout=False, mask_and_inpaint=False):
		""" Adds noise to a batch of images and poses"""
		# denoising and synthetic data generation
		if self.cfg['multiplicative_denoising'] and not only_dropout:
			mult_samples = ss.gamma.rvs(self.gamma_shape, scale=self.gamma_scale, size=len(im_arr))
			mult_samples = mult_samples[:,np.newaxis,np.newaxis,np.newaxis]
			im_arr = im_arr * np.tile(mult_samples, [1, self.im_height, self.im_width, self.im_channels])

		# randomly dropout regions of the image for robustness
		if self.cfg['image_dropout'] and not only_dropout:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['image_dropout_rate']:
					image = self.im_arr[i,:,:,0]
					nonzero_px = np.where(image > 0)
					nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
					num_nonzero = nonzero_px.shape[0]
					num_dropout_regions = ss.poisson.rvs(self.cfg['dropout_poisson_mean']) 
					
					# sample ellipses
					dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
					x_radii = ss.gamma.rvs(self.cfg['dropout_radius_shape'], scale=self.cfg['dropout_radius_scale'], size=num_dropout_regions)
					y_radii = ss.gamma.rvs(self.cfg['dropout_radius_shape'], scale=self.cfg['dropout_radius_scale'], size=num_dropout_regions)

					# set interior pixels to zero
					for j in range(num_dropout_regions):
						ind = dropout_centers[j]
						dropout_center = nonzero_px[ind, :]
						x_radius = x_radii[j]
						y_radius = y_radii[j]
						dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=image.shape)
						image[dropout_px_y, dropout_px_x] = 0.0
					im_arr[i,:,:,0] = image

		# dropout a region around the areas of the image with high gradient
		if self.cfg['gradient_dropout'] and not only_dropout:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['gradient_dropout_rate']:
					image = im_arr[i,:,:,0]
					grad_mag = sf.gaussian_gradient_magnitude(image, sigma=self.cfg['gradient_dropout_sigma'])
					thresh = ss.gamma.rvs(self.cfg['gradient_dropout_shape'], self.cfg['gradient_dropout_scale'], size=1)
					high_gradient_px = np.where(grad_mag > thresh)
					image[high_gradient_px[0], high_gradient_px[1]] = 0.0
				im_arr[i,:,:,0] = image

		# add correlated Gaussian noise
		if self.cfg['gaussian_process_denoising'] and not only_dropout:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['gaussian_process_rate']:
					image = im_arr[i,:,:,0]
					gp_noise = ss.norm.rvs(scale=self.gp_sigma, size=self.gp_num_pix).reshape(self.gp_sample_height, self.gp_sample_width)
					gp_noise = sm.imresize(gp_noise, self.gp_rescale_factor, interp='bicubic', mode='F')
					image[image > 0] += gp_noise[image > 0]
					im_arr[i,:,:,0] = image

		# run open and close filters to 
		if self.cfg['morphological'] and not only_dropout:
			for i in range(len(im_arr)):
				image = im_arr[i,:,:,0]
				sample = np.random.rand()
				morph_filter_dim = ss.poisson.rvs(self.cfg['morph_poisson_mean'])                         
				if sample < self.cfg['morph_open_rate']:
					image = snm.grey_opening(image, size=morph_filter_dim)
				else:
					closed_train_image = snm.grey_closing(image, size=morph_filter_dim)
					
					# set new closed pixels to the minimum depth, mimicing the table
					new_nonzero_px = np.where((image == 0) & (closed_train_image > 0))
					closed_train_image[new_nonzero_px[0], new_nonzero_px[1]] = np.min(image[image>0])
					image = closed_train_image.copy()

				im_arr[i,:,:,0] = image                        

		# randomly dropout borders of the image for robustness
		if self.cfg['border_distortion']:
			for i in range(len(im_arr)):
				if np.random.rand() < self.cfg['border_distortion_rate']:
					image = im_arr[i,:,:,0]
					original = image.copy()
					mask = np.zeros(image.shape)
					grad_mag = sf.gaussian_gradient_magnitude(image, sigma=self.cfg['border_grad_sigma'])
					if self.cfg['visualize_border_distortion']:
						plt.clf()
						plt.subplot(1, 2, 1)
						plt.imshow(grad_mag, cmap=plt.cm.gray_r)
					high_gradient_px = np.where(grad_mag > self.cfg['border_grad_thresh'])
					high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
					num_nonzero = high_gradient_px.shape[0]
					if num_nonzero == 0:
						continue
					num_dropout_regions = ss.poisson.rvs(self.cfg['border_poisson_mean']) 
					# logging.info('Num_dropouts:{}'.format(num_dropout_regions))

					# sample ellipses
					dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
					x_radii = ss.gamma.rvs(self.cfg['border_radius_shape'], scale=self.cfg['border_radius_scale'], size=num_dropout_regions)
					y_radii = ss.gamma.rvs(self.cfg['border_radius_shape'], scale=self.cfg['border_radius_scale'], size=num_dropout_regions)

					# set interior pixels to zero or one
					for j in range(num_dropout_regions):
						ind = dropout_centers[j]
						dropout_center = high_gradient_px[ind, :]
						x_radius = x_radii[j]
						y_radius = y_radii[j]
						# logging.info('Center {}: {}, x_radius: {}, y_radius: {}'.format(j, dropout_center, x_radius, y_radius))
						dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=image.shape)
					
						if self.cfg['border_fill_type'] == 'zero':
							image[dropout_px_y, dropout_px_x] = 0.0
							mask[dropout_px_y, dropout_px_x] = 1
						elif self.cfg['border_fill_type'] == 'inf':
							image[dropout_px_y, dropout_px_x] = np.inf
							mask[dropout_px_y, dropout_px_x] = np.inf
						elif self.cfg['border_fill_type'] == 'machine_max':
							image[dropout_px_y, dropout_px_x] = np.finfo(np.float64).max
							mask[dropout_px_y, dropout_px_x] = np.finfo(np.float64).max
				
					if self.cfg['visualize_border_distortion']:
						vmin = 0
						vmax = .7
						plt.subplot(1, 2, 2)
						plt.imshow(image, cmap=plt.cm.gray_r, vmin=vmin, vmax=vmax)
						plt.show()
					if mask_and_inpaint:
						image = sr.inpaint.inpaint_biharmonic(image, mask)
						image = image.reshape((32, 32, 1))
						mask = mask.reshape((32, 32, 1))
						image = np.c_[image, mask]
						inpainted_image = image[:, :, 0]
						mask = image[:, :, 1]
						if self.cfg['visualize_border_distort_mask_inpaint']:
							plt.clf()
							plt.subplot(1, 3, 1)
							plt.imshow(original, cmap=plt.cm.gray_r)
							plt.subplot(1, 3, 2)
							plt.imshow(inpainted_image, cmap=plt.cm.gray_r)
							plt.subplot(1, 3, 3)
							plt.imshow(mask, cmap=plt.cm.gray)
							plt.show()
						im_arr[i] = image
					else: 
						im_arr[i,:,:,0] = image


		# randomly replace background pixels with constant depth
		if self.cfg['background_denoising'] and not only_dropout:
			for i in range(len(im_arr)):
				image = self.im_arr[i,:,:,0]                
				if np.random.rand() < self.cfg['background_rate']:
					image[image > 0] = self.cfg['background_min_depth'] + (self.cfg['background_max_depth'] - self.cfg['background_min_depth']) * np.random.rand()
				im_arr[i,:,:,0] = image

		# symmetrize images and poses
		if self.cfg['symmetrize'] and not only_dropout:
			for i in range(len(im_arr)):
				image = im_arr[i,:,:,0]
				# rotate with 50% probability
				if np.random.rand() < 0.5:
					theta = 180.0
					rot_map = cv2.getRotationMatrix2D(tuple(self.im_center), theta, 1)
					image = cv2.warpAffine(image, rot_map, (self.im_height, self.im_width), flags=cv2.INTER_NEAREST)
					if self.pose_dim > 4:
						pose_arr[i,4] = -pose_arr[i,4]
						pose_arr[i,5] = -pose_arr[i,5]
				# reflect left right with 50% probability
				if np.random.rand() < 0.5:
					image = np.fliplr(image)
					if self.pose_dim > 4:
						pose_arr[i,5] = -pose_arr[i,5]
				# reflect up down with 50% probability
				if np.random.rand() < 0.5:
					image = np.flipud(image)
					if self.pose_dim > 4:
						pose_arr[i,4] = -pose_arr[i,4]
				im_arr[i,:,:,0] = image

		return im_arr, pose_arr

	def _error_rate_in_batches(self):
		""" Get all predictions for a dataset by running it in small batches

		Returns
		-------
		: float
			validation error
		"""
		error_rates = []
		self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.label_filenames.sort(key = lambda x: int(x[-9:-4]))
		self.gripper_param_filenames.sort(key = lambda x: int(x[-9:-4]))

		for data_filename, pose_filename, gripper_filename, label_filename in zip(self.im_filenames, self.pose_filenames, self.gripper_param_filenames, self.label_filenames):
			# load next file
			data = np.load(os.path.join(self.data_dir, data_filename))['arr_0']
			poses = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
			labels = np.load(os.path.join(self.data_dir, label_filename))['arr_0']
			if self.gripper_dim > 0:
				gripper_params = np.load(os.path.join(self.data_dir, gripper_filename))['arr_0']
			
			val_indices = self.val_index_map[data_filename]
		
			# if no datapoints from this file are in validation then just continue
			if len(val_indices) == 0:
				continue
		
			if self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
				tp_tmp = self._read_pose_data(poses.copy(), self.input_pose_mode)
				val_indices = val_indices[np.isfinite(tp_tmp[val_indices,1])]

			data = data[val_indices,...]
			poses = self._read_pose_data(poses[val_indices, :], self.input_pose_mode)
			if self.gripper_dim > 0:
				gripper_params = self._read_gripper_data(gripper_params[val_indices, :], self.input_gripper_mode)
			labels = labels[val_indices,...]

			if self.training_mode == TrainingMode.REGRESSION:
				if self.preproc_mode == PreprocMode.NORMALIZATION:
					labels = (labels - self.min_metric) / (self.max_metric - self.min_metric)
			elif self.training_mode == TrainingMode.CLASSIFICATION:
				labels = 1 * (labels > self.metric_thresh)
				labels = labels.astype(np.uint8)
			
			# save undistorted validation images for debugging 
			if self.cfg['save_original_val_images']:
				if self._num_original_val_images_saved < self.cfg['num_original_val_images']:
					output_dir = os.path.join(self.experiment_dir, 'original_val_images')
					if not os.path.exists(output_dir):
						os.mkdir(output_dir)
					np.savez_compressed(os.path.join(output_dir, 'original_image_{}'.format(self._num_original_val_images_saved)), data[0, :, :, 0])
					self._num_original_val_images_saved += 1

			# allocate mask channel if needed
			if self.cfg['mask_and_inpaint']:
				mask_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], self.num_tensor_channels))
				mask_data[:, :, :, 0] = data[:, :, :, 0]
				data = mask_data

			# get predictions
			if self.cfg['distort_val_data']:
				self._distort(data, poses)

			if self.gripper_dim > 0:
				predictions = self.gqcnn.predict(data, poses, gripper_arr=gripper_params)
			else:
				predictions = self.gqcnn.predict(data, poses)
			
			# save distorted validation images for debugging
			if self.cfg['save_distorted_val_images']:
				if self._num_distorted_val_images_saved < self.cfg['num_distorted_val_images']:
					output_dir = os.path.join(self.experiment_dir, 'distorted_val_images')
					if not os.path.exists(output_dir):
						os.mkdir(output_dir)
					np.savez_compressed(os.path.join(output_dir, 'distorted_image_{}'.format(self._num_distorted_val_images_saved)), data[0, :, :, 0])
					self._num_distorted_val_images_saved += 1

			# get error rate
			if self.training_mode == TrainingMode.CLASSIFICATION:
				error_rates.append(ClassificationResult([predictions], [labels]).error_rate)
			else:
				error_rates.append(RegressionResult([predictions], [labels]).error_rate)
			
		# clean up
		del data
		del poses
		del labels
		if self.gripper_dim > 0:
			del gripper_params

		# return average error rate over all files (assuming same size)
		return np.mean(error_rates)
