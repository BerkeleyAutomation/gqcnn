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
Optimizer class for training a gqcnn(Grasp Quality Neural Network) object.
Author: Vishal Satish and Jeff Mahler
"""
import argparse
import copy
import cv2
import gc
import json
import logging
import numbers
import numpy as np
import cPickle as pkl
import os
import random
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import signal
import sys
import shutil
import threading
import time
import urllib
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from autolab_core import YamlConfig
import autolab_core.utils as utils
import collections

import IPython

from autolab_core import BinaryClassificationResult, RegressionResult, TensorDataset
import autolab_core.utils as utils

from .optimizer_constants import ImageMode, TrainingMode, GripperMode, GeneralConstants
from .train_stats_logger import TrainStatsLogger
from .utils import pose_dim, read_pose_data

class GQCNNOptimizer(object):
    """ Optimizer for gqcnn object """

    def __init__(self, gqcnn,
                 dataset_dir,
                 output_dir,
                 config):
        """
        Parameters
        ----------
        gqcnn : :obj:`GQCNN`
            grasp quality neural network to optimize
        dataset_dir : str
            path to the training / validation dataset
        output_dir : str
            path to save the model output
        config : dict
            dictionary of configuration parameters
        """
        self.gqcnn = gqcnn
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.cfg = config
        self.tensorboard_has_launched = False

    def _create_loss(self):
        """ Creates a loss based on config file

        Returns
        -------
        :obj:`tensorflow Tensor`
            loss
        """
        if self.cfg['loss'] == 'l2':
            return (1.0 / self.train_batch_size) * tf.nn.l2_loss(tf.subtract(tf.nn.sigmoid(self.train_net_output), self.train_labels_node))
        elif self.cfg['loss'] == 'sparse':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=self.train_labels_node, logits=self.train_net_output, name=None))
        elif self.cfg['loss'] == 'weighted_cross_entropy':
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.reshape(self.train_labels_node, [-1,1]),
                                                                           logits=self.train_net_output,
                                                                           pos_weight=self.pos_weight,
                                                                           name=None))

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
            return tf.train.MomentumOptimizer(learning_rate,
                                              self.momentum_rate).minimize(loss,
                                                                           global_step=batch,
                                                                           var_list=var_list)
        elif self.cfg['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                                  global_step=batch,
                                                                  var_list=var_list)
        elif self.cfg['optimizer'] == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate).minimize(loss,
                                                                     global_step=batch,
                                                                     var_list=var_list)
        else:
            raise ValueError('Optimizer %s not supported' %(self.cfg['optimizer']))

    def _check_dead_queue(self):
        """ Checks to see if the queue is dead and if so closes the tensorflow session and cleans up the variables """
        if self.dead_event.is_set():
            # close self.session
            self.sess.close()
            
            # cleanup
            for layer_weights in self.weights.__dict__.values():
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
        
        # read and setup dropouts from config
        drop_fc3 = False
        if 'drop_fc3' in self.cfg.keys() and self.cfg['drop_fc3']:
            drop_fc3 = True
        drop_fc4 = False
        if 'drop_fc4' in self.cfg.keys() and self.cfg['drop_fc4']:
            drop_fc4 = True
        
        fc3_drop_rate = self.cfg['fc3_drop_rate']
        fc4_drop_rate = self.cfg['fc4_drop_rate']
        
        # build training and validation networks
        with tf.name_scope('validation_network'):
            if self.training_mode == TrainingMode.REGRESSION:
                self.gqcnn.initialize_network(add_softmax=False, add_sigmoid=True) # builds validation network inside gqcnn class
            elif self.cfg['loss'] != 'weighted_cross_entropy':
                self.gqcnn.initialize_network(add_softmax=True, add_sigmoid=False) # builds validation network inside gqcnn class
            else:
                self.gqcnn.initialize_network(add_softmax=False, add_sigmoid=True) # builds validation network inside gqcnn class                
        with tf.name_scope('training_network'):
            self.train_net_output = self.gqcnn._build_network(self.input_im_node, self.input_pose_node, drop_fc3, drop_fc4, fc3_drop_rate , fc4_drop_rate)

            # form loss
            # part 1: error
            if self.training_mode == TrainingMode.CLASSIFICATION:
                if self.cfg['loss'] == 'weighted_cross_entropy':
                    train_predictions = tf.nn.sigmoid(self.train_net_output)
                else:
                    train_predictions = tf.nn.softmax(self.train_net_output)
                with tf.name_scope('loss'):
                    loss = self._create_loss()
            elif self.training_mode == TrainingMode.REGRESSION:
                train_predictions = tf.nn.sigmoid(self.train_net_output)
                with tf.name_scope('loss'):
                    loss = self._create_loss()

            # part 2: regularization
            layer_weights = self.weights.__dict__.values()
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
        var_list = self.weights.__dict__.values()
        if self.cfg['fine_tune'] and self.cfg['update_fc_only']:
            var_list = [v for k, v in self.weights.__dict__.iteritems() if k.find('conv') == -1]
        elif self.cfg['fine_tune'] and self.cfg['update_conv0_only'] and self.use_conv0:
            var_list = [v for k, v in self.weights.__dict__.iteritems() if k.find('conv0') > -1]

        # create optimizer
        with tf.name_scope('optimizer'):
            optimizer = self._create_optimizer(loss, batch, var_list, learning_rate)

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
            for layer_weights in self.weights.__dict__.values():
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
        self.summary_writer.add_graph(self.gqcnn.tf_graph)

        # begin optimization loop
        try:
            self.queue_thread = threading.Thread(target=self._load_and_enqueue)
            self.queue_thread.start()

            # init and run tf self.sessions
            init = tf.global_variables_initializer()
            self.sess.run(init)
            logging.info('Beginning Optimization')

            # create a TrainStatsLogger object to log training statistics at certain intervals
            self.train_stats_logger = TrainStatsLogger(self.model_dir)

            # loop through training steps
            training_range = xrange(int(self.num_epochs * self.num_train) // self.train_batch_size)
            for step in training_range:
                # check for dead queue
                self._check_dead_queue()

                # run optimization
                step_start = time.time()
                _, l, lr, predictions, batch_labels, output, train_images, conv1_1W, conv1_1b, train_poses = self.sess.run(
                        [optimizer, loss, learning_rate, train_predictions, self.train_labels_node, self.train_net_output, self.input_im_node, self.weights.conv1_1W, self.weights.conv1_1b, self.input_pose_node], options=GeneralConstants.timeout_option)
                step_stop = time.time()
                logging.info('Step took %.3f sec' %(step_stop-step_start))
                
                if self.training_mode == TrainingMode.REGRESSION:
                    logging.info('Max ' +  str(np.max(predictions)))
                    logging.info('Min ' + str(np.min(predictions)))
                elif self.cfg['loss'] != 'weighted_cross_entropy':
                    ex = np.exp(output - np.tile(np.max(output, axis=1)[:,np.newaxis], [1,2]))
                    softmax = ex / np.tile(np.sum(ex, axis=1)[:,np.newaxis], [1,2])
		        
                    logging.info('Max ' +  str(np.max(softmax[:,1])))
                    logging.info('Min ' + str(np.min(softmax[:,1])))
                    logging.info('Pred nonzero ' + str(np.sum(softmax[:,1] > 0.5)))
                    logging.info('True nonzero ' + str(np.sum(batch_labels)))
                else:
                    sigmoid = 1.0 / (1.0 + np.exp(-output))
                    logging.info('Max ' +  str(np.max(sigmoid)))
                    logging.info('Min ' + str(np.min(sigmoid)))
                    logging.info('Pred nonzero ' + str(np.sum(sigmoid > 0.5)))
                    logging.info('True nonzero ' + str(np.sum(batch_labels > 0.5)))

                if np.isnan(l) or np.any(np.isnan(train_poses)):
                    IPython.embed()
                    logging.info('Exiting...')
                    break
                    
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
                        classification_result = BinaryClassificationResult(predictions[:,1], batch_labels)
                        train_error = classification_result.error_rate
                        
                    logging.info('Minibatch error: %.3f' %(train_error))
                        
                    self.summary_writer.add_summary(self.sess.run(self.merged_log_summaries, feed_dict={self.minibatch_error_placeholder: train_error, self.minibatch_loss_placeholder: l, self.learning_rate_placeholder: lr}), step)
                    sys.stdout.flush()

                    # update the TrainStatsLogger
                    self.train_stats_logger.update(train_eval_iter=step, train_loss=l, train_error=train_error, total_train_error=None, val_eval_iter=None, val_error=None, learning_rate=lr)

                # evaluate validation error
                if step % self.eval_frequency == 0 and step > 0:
                    if self.cfg['eval_total_train_error']:
                        train_error = self._error_rate_in_batches(validation_set=False)
                        logging.info('Training error: %.3f' %(train_error))

                        # update the TrainStatsLogger and save
                        self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, total_train_error=train_error, val_eval_iter=None, val_error=None, learning_rate=None)
                        self.train_stats_logger.log()

                    val_error = self._error_rate_in_batches()
                    self.summary_writer.add_summary(self.sess.run(self.merged_eval_summaries, feed_dict={self.val_error_placeholder: val_error}), step)
                    logging.info('Validation error: %.3f' %(val_error))
                    sys.stdout.flush()

                    # update the TrainStatsLogger
                    self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, total_train_error=None, val_eval_iter=step, val_error=val_error, learning_rate=None)

                    # save everything!
                    self.train_stats_logger.log()

                # save filters
                if step % self.vis_frequency == 0:
                    # conv1_1
                    num_filt = conv1_1W.shape[3]
                    d = int(np.ceil(np.sqrt(num_filt)))

                    plt.clf()
                    for i in range(num_filt):
                        plt.subplot(d,d,i+1)
                        plt.imshow(conv1_1W[:,:,0,i], cmap=plt.cm.gray, interpolation='nearest')
                        plt.axis('off')
                        plt.title('b=%.3f' %(conv1_1b[i]), fontsize=10)
                    plt.savefig(os.path.join(self.filter_dir, 'conv1_1W_%05d.jpg' %(step)))
                    
                # save the model
                if step % self.save_frequency == 0 and step > 0:
                    self.saver.save(self.sess, os.path.join(self.model_dir, 'model_%05d.ckpt' %(step)))
                    self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'))

                # launch tensorboard only after the first iteration
                if not self.tensorboard_has_launched:
                    self.tensorboard_has_launched = True
                    self._launch_tensorboard()

            # get final logs
            val_error = self._error_rate_in_batches()
            logging.info('Final validation error: %.3f%%' %val_error)
            sys.stdout.flush()

            # update the TrainStatsLogger
            self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, total_train_error=None, val_eval_iter=step, val_error=val_error, learning_rate=None)

            # log & save everything!
            self.train_stats_logger.log()
            self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'))

        except Exception as e:
            self.term_event.set()
            if not self.forceful_exit:
                self.sess.close() 
                for layer_weights in self.weights.__dict__.values():
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
        for layer_weights in self.weights.__dict__.values():
            del layer_weights
        del self.saver
        del self.sess

        # exit
        logging.info('Exiting Optimization')

    def _compute_data_metrics(self):
        """ Calculate image mean, image std, pose mean, pose std, normalization params """
        # subsample tensors (for faster runtime)
        random_file_indices = np.random.choice(self.num_tensors,
                                               size=self.num_random_files,
                                               replace=False)
        
        # compute image stats
        im_mean_filename = os.path.join(self.model_dir, 'mean.npy')
        im_std_filename = os.path.join(self.model_dir, 'std.npy')
        if self.cfg['fine_tune']:
            self.im_mean = self.gqcnn.im_mean
            self.im_std = self.gqcnn.im_std
        elif os.path.exists(im_mean_filename) and os.path.exists(im_std_filename):
            self.im_mean = np.load(im_mean_filename)
            self.im_std = np.load(im_std_filename)
        else:
            self.im_mean = 0
            self.im_std = 0

            # compute mean
            logging.info('Computing image mean')
            num_summed = 0
            for k, i in enumerate(random_file_indices):
                if k % self.preproc_log_frequency == 0:
                    logging.info('Adding file %d of %d to image mean estimate' %(k+1, random_file_indices.shape[0]))
                im_data = self.dataset.tensor(self.im_field_name, i).data
                train_indices = self.train_index_map[i]
                self.im_mean += np.sum(im_data[train_indices, ...])
                num_summed += self.train_index_map[i].shape[0] * im_data.shape[1] * im_data.shape[2]
            self.im_mean = self.im_mean / num_summed

            # compute std
            logging.info('Computing image std')
            for k, i in enumerate(random_file_indices):
                if k % self.preproc_log_frequency == 0:
                    logging.info('Adding file %d of %d to image std estimate' %(k+1, random_file_indices.shape[0]))
                im_data = self.dataset.tensor(self.im_field_name, i).data
                train_indices = self.train_index_map[i]
                self.im_std += np.sum((im_data[train_indices, ...] - self.im_mean)**2)
            self.im_std = np.sqrt(self.im_std / num_summed)

            # save
            np.save(im_mean_filename, self.im_mean)
            np.save(im_std_filename, self.im_std)

        # update gqcnn
        self.gqcnn.set_im_mean(self.im_mean)
        self.gqcnn.set_im_std(self.im_std)

        # compute pose stats
        pose_mean_filename = os.path.join(self.model_dir, 'pose_mean.npy')
        pose_std_filename = os.path.join(self.model_dir, 'pose_std.npy')
        if self.cfg['fine_tune']:
            self.pose_mean = self.gqcnn.pose_mean
            self.pose_std = self.gqcnn.pose_std
        elif os.path.exists(pose_mean_filename) and os.path.exists(pose_std_filename):
            self.pose_mean = np.load(pose_mean_filename)
            self.pose_std = np.load(pose_std_filename)
        else:
            self.pose_mean = np.zeros(self.raw_pose_shape)
            self.pose_std = np.zeros(self.raw_pose_shape)

            # compute mean
            num_summed = 0
            logging.info('Computing pose mean')
            for k, i in enumerate(random_file_indices):
                if k % self.preproc_log_frequency == 0:
                    logging.info('Adding file %d of %d to pose mean estimate' %(k+1, random_file_indices.shape[0]))
                pose_data = self.dataset.tensor(self.pose_field_name, i).data
                train_indices = self.train_index_map[i]
                if self.gripper_mode == GripperMode.SUCTION:
                    rand_indices = np.random.choice(pose_data.shape[0],
                                                    size=pose_data.shape[0]/2,
                                                    replace=False)
                    pose_data[rand_indices, 4] = -pose_data[rand_indices, 4]
                elif self.gripper_mode == GripperMode.LEGACY_SUCTION:
                    rand_indices = np.random.choice(pose_data.shape[0],
                                                    size=pose_data.shape[0]/2,
                                                    replace=False)
                    pose_data[rand_indices, 3] = -pose_data[rand_indices, 3]
                pose_data = pose_data[train_indices,:]
                pose_data = pose_data[np.isfinite(pose_data[:,3]),:]
                self.pose_mean += np.sum(pose_data, axis=0)
                num_summed += pose_data.shape[0]
            self.pose_mean = self.pose_mean / num_summed

            # compute std
            logging.info('Computing pose std')
            for k, i in enumerate(random_file_indices):
                if k % self.preproc_log_frequency == 0:
                    logging.info('Adding file %d of %d to pose std estimate' %(k+1, random_file_indices.shape[0]))
                pose_data = self.dataset.tensor(self.pose_field_name, i).data
                train_indices = self.train_index_map[i]
                if self.gripper_mode == GripperMode.SUCTION:
                    rand_indices = np.random.choice(pose_data.shape[0],
                                                    size=pose_data.shape[0]/2,
                                                    replace=False)
                    pose_data[rand_indices, 4] = -pose_data[rand_indices, 4]
                elif self.gripper_mode == GripperMode.LEGACY_SUCTION:
                    rand_indices = np.random.choice(pose_data.shape[0],
                                                    size=pose_data.shape[0]/2,
                                                    replace=False)
                    pose_data[rand_indices, 3] = -pose_data[rand_indices, 3]
                pose_data = pose_data[train_indices,:]
                pose_data = pose_data[np.isfinite(pose_data[:,3]),:]
                self.pose_std += np.sum((pose_data - self.pose_mean)**2, axis=0)
            self.pose_std = np.sqrt(self.pose_std / num_summed)
            self.pose_std[self.pose_std==0] = 1.0

            # save
            self.pose_mean = read_pose_data(self.pose_mean, self.gripper_mode)
            self.pose_std = read_pose_data(self.pose_std, self.gripper_mode)
            np.save(pose_mean_filename, self.pose_mean)
            np.save(pose_std_filename, self.pose_std)

        # update gqcnn
        self.gqcnn.set_pose_mean(self.pose_mean)
        self.gqcnn.set_pose_std(self.pose_std)

        # check for invalid values
        if np.any(np.isnan(self.pose_mean)) or np.any(np.isnan(self.pose_std)):
            logging.error('Pose mean or pose std is NaN! Check the input dataset')
            IPython.embed()
            exit(0)

        # save mean and std to file for finetuning
        if self.cfg['fine_tune']:
            out_mean_filename = os.path.join(self.model_dir, 'mean.npy')
            out_std_filename = os.path.join(self.model_dir, 'std.npy')
            out_pose_mean_filename = os.path.join(self.model_dir, 'pose_mean.npy')
            out_pose_std_filename = os.path.join(self.model_dir, 'pose_std.npy')
            np.save(out_mean_filename, self.im_mean)
            np.save(out_std_filename, self.im_std)
            np.save(out_pose_mean_filename, self.pose_mean)
            np.save(out_pose_std_filename, self.pose_std)
            
        # compute normalization parameters of the network
        pct_pos_train_filename = os.path.join(self.model_dir, 'pct_pos_train.npy')
        pct_pos_val_filename = os.path.join(self.model_dir, 'pct_pos_val.npy')
        if os.path.exists(pct_pos_train_filename) and os.path.exists(pct_pos_val_filename):
            pct_pos_train = np.load(pct_pos_train_filename)
            pct_pos_val = np.load(pct_pos_val_filename)
        else:
            logging.info('Computing metric stats')
            all_train_metrics = None
            all_val_metrics = None
    
            # read metrics
            for k, i in enumerate(random_file_indices):
                if k % self.preproc_log_frequency == 0:
                    logging.info('Adding file %d of %d to metric stat estimates' %(k+1, random_file_indices.shape[0]))
                metric_data = self.dataset.tensor(self.label_field_name, i).data
                train_indices = self.train_index_map[i]
                val_indices = self.val_index_map[i]
                train_metric_data = metric_data[train_indices]
                val_metric_data = metric_data[val_indices]
                
                if all_train_metrics is None:
                    all_train_metrics = train_metric_data
                else:
                    all_train_metrics = np.r_[all_train_metrics, train_metric_data]

                if all_val_metrics is None:
                    all_val_metrics = val_metric_data
                else:
                    all_val_metrics = np.r_[all_val_metrics, val_metric_data]

            # compute train stats
            self.min_metric = np.min(all_train_metrics)
            self.max_metric = np.max(all_train_metrics)
            self.mean_metric = np.mean(all_train_metrics)
            self.median_metric = np.median(all_train_metrics)

            # save metrics
            pct_pos_train = float(np.sum(all_train_metrics > self.metric_thresh)) / all_train_metrics.shape[0]
            np.save(pct_pos_train_filename, np.array(pct_pos_train))

            pct_pos_val = float(np.sum(all_val_metrics > self.metric_thresh)) / all_val_metrics.shape[0]
            np.save(pct_pos_val_filename, np.array(pct_pos_val))

        logging.info('Percent positive in train: ' + str(pct_pos_train))
        logging.info('Percent positive in val: ' + str(pct_pos_val))
        
    def _compute_indices_image_wise(self):
        """ Compute train and validation indices based on an image-wise train-val split"""

        # get total number of training datapoints and set the decay_step
        self.num_train = int(self.train_pct * self.dataset.num_datapoints)

        # make a map of the train and test indices for each file
        logging.info('Computing indices image-wise')
        train_index_map_filename = os.path.join(self.model_dir, 'train_indices_image_wise.pkl')
        val_index_map_filename = os.path.join(self.model_dir, 'val_indices_image_wise.pkl')
        if os.path.exists(train_index_map_filename) and os.path.exists(val_index_map_filename):
            self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
            self.val_index_map = pkl.load(open(val_index_map_filename, 'r'))
        else:
            # get training and validation indices
            all_indices = np.arange(self.dataset.num_datapoints)
            np.random.shuffle(all_indices)
            train_indices = np.sort(all_indices[:self.num_train])
            val_indices = np.sort(all_indices[self.num_train:])

            # create a mapping from tensors to indices
            self.train_index_map = {}
            self.val_index_map = {}
            i = 0
            for i in range(self.num_tensors):
                logging.info('Computing indices for file %d' %(i))
                datapoint_indices = self.dataset.datapoint_indices_for_tensor(i)
                lower = np.min(datapoint_indices)
                upper = np.max(datapoint_indices)                
                self.train_index_map[i] = train_indices[(train_indices >= lower) & (train_indices < upper)] - lower
                self.val_index_map[i] = val_indices[(val_indices >= lower) & (val_indices < upper)] - lower
                if i % 10 == 0:
                    gc.collect()
                i += 1
            pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
            pkl.dump(self.val_index_map, open(val_index_map_filename, 'w'))

    def _compute_indices_object_wise(self):
        """ Compute train and validation indices based on an object-wise train-val split"""
        # check that object-wise splits are possible
        if 'split' not in self.dataset.field_names:
            raise ValueError('Object splits not available. Must be pre-computed for the dataset.')

        # make a map of the train and test indices for each file
        logging.info('Computing indices object-wise')
        train_index_map_filename = os.path.join(self.model_dir, 'train_indices_object_wise.pkl')
        val_index_map_filename = os.path.join(self.model_dir, 'val_indices_object_wise.pkl')
        if os.path.exists(train_index_map_filename) and os.path.exists(val_index_map_filename):
            self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
            self.val_index_map = pkl.load(open(val_index_map_filename, 'r'))
        else:
            self.train_index_map = {}
            self.val_index_map = {}
            for i in range(self.dataset.num_tensors):
                logging.info('Computing indices for file %d' %(i))
                self.train_index_map[i] = np.zeros(0)
                self.val_index_map[i] = np.zeros(0)
                split_arr = self.dataset.tensor('split', i)
                self.train_index_map[i] = np.where(split_arr == TRAIN_ID)[0]
                self.val_index_map[i] = np.where(split_arr == TEST_ID)[0]
                del split_arr
                if i % 10 == 0:
                    gc.collect()
            pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
            pkl.dump(self.val_index_map, open(val_index_map_filename, 'w'))

    def _setup_output_dirs(self):
        """ Setup output directories """
        # create a directory for the model
        model_id = utils.gen_experiment_id()
        self.model_dir = os.path.join(self.output_dir, 'model_%s' %(model_id))
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # create the summary dir
        self.summary_dir = os.path.join(self.model_dir, 'tensorboard_summaries')
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)
        else:
            # if the summary directory already exists, clean it out by deleting all files in it
            # we don't want tensorboard to get confused with old logs while debugging with the same directory
            old_files = os.listdir(self.summary_dir)
            for file in old_files:
                os.remove(os.path.join(self.summary_dir, file))

        # setup filter directory
        self.filter_dir = os.path.join(self.model_dir, 'filters')
        if not os.path.exists(self.filter_dir):
            os.mkdir(self.filter_dir)

        logging.info('Saving model to %s' %(self.model_dir))
            
    def _setup_logging(self):
        """ Copy the original config files """
        # save config
        out_config_filename = os.path.join(self.model_dir, 'config.json')
        tempOrderedDict = collections.OrderedDict()
        for key in self.cfg.keys():
            tempOrderedDict[key] = self.cfg[key]
        with open(out_config_filename, 'w') as outfile:
            json.dump(tempOrderedDict,
                      outfile,
                      indent=GeneralConstants.JSON_INDENT)

        # setup logging
        self.log_filename = os.path.join(self.model_dir, 'training.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        hdlr = logging.FileHandler(self.log_filename)
        hdlr.setFormatter(formatter)
        logging.getLogger().addHandler(hdlr)
            
        # save training script    
        this_filename = sys.argv[0]
        out_train_filename = os.path.join(self.model_dir, 'training_script.py')
        shutil.copyfile(this_filename, out_train_filename)

        # save architecture
        out_architecture_filename = os.path.join(self.model_dir, 'architecture.json')
        json.dump(self.cfg['gqcnn']['architecture'],
                  open(out_architecture_filename, 'w'),
                  indent=GeneralConstants.JSON_INDENT)
        
    def _read_training_params(self):
        """ Read training parameters from configuration file """
        # splits
        self.data_split_mode = self.cfg['data_split_mode']
        self.train_pct = self.cfg['train_pct']
        self.total_pct = self.cfg['total_pct']

        # training sizes
        self.train_batch_size = self.cfg['train_batch_size']
        self.val_batch_size = self.cfg['val_batch_size']
        self.max_files_eval = None
        if 'max_files_eval' in self.cfg.keys():
            self.max_files_eval = self.cfg['max_files_eval']
        
        # update the GQCNN's batch_size param to this one
        logging.info("updating val_batch_size to %d" %(self.val_batch_size))
        self.gqcnn.set_batch_size(self.val_batch_size)

        # logging
        self.num_epochs = self.cfg['num_epochs']
        self.eval_frequency = self.cfg['eval_frequency']
        self.save_frequency = self.cfg['save_frequency']
        self.vis_frequency = self.cfg['vis_frequency']
        self.log_frequency = self.cfg['log_frequency']

        # optimization
        self.train_l2_regularizer = self.cfg['train_l2_regularizer']
        self.base_lr = self.cfg['base_lr']
        self.decay_step_multiplier = self.cfg['decay_step_multiplier']
        self.decay_rate = self.cfg['decay_rate']
        self.momentum_rate = self.cfg['momentum_rate']
        self.max_training_examples_per_load = self.cfg['max_training_examples_per_load']

        # metrics
        self.target_metric_name = self.cfg['target_metric_name']
        self.metric_thresh = self.cfg['metric_thresh']
        self.training_mode = self.cfg['training_mode']

        # preproc
        self.preproc_log_frequency = self.cfg['preproc_log_frequency']
        self.num_random_files = self.cfg['num_random_files']

        # re-weighting positives / negatives
        self.pos_weight = 0.0
        if 'pos_weight' in self.cfg.keys():
            self.pos_weight = self.cfg['pos_weight']
            self.pos_accept_prob = 1.0
            self.neg_accept_prob = 1.0
            if self.pos_weight > 1:
                self.neg_accept_prob = 1.0 / self.pos_weight
            else:
                self.pos_accept_prob = self.pos_weight
                
        if self.train_pct < 0 or self.train_pct > 1:
            raise ValueError('Train percentage must be in range [0,1]')

        if self.total_pct < 0 or self.total_pct > 1:
            raise ValueError('Train percentage must be in range [0,1]')

        
    def _setup_denoising_and_synthetic(self):
        """ Setup denoising and synthetic data parameters """
        # multiplicative denoising
        if self.cfg['multiplicative_denoising']:
            self.gamma_shape = self.cfg['gamma_shape']
            self.gamma_scale = 1.0 / self.gamma_shape
        # gaussian process noise    
        if self.cfg['gaussian_process_denoising']:
            self.gp_rescale_factor = self.cfg['gaussian_process_scaling_factor']
            self.gp_sample_height = int(self.im_height / self.gp_rescale_factor)
            self.gp_sample_width = int(self.im_width / self.gp_rescale_factor)
            self.gp_num_pix = self.gp_sample_height * self.gp_sample_width
            self.gp_sigma = self.cfg['gaussian_process_sigma']

    def _open_dataset(self):
        """ Open the dataset """
        # read in filenames of training data(poses, images, labels)
        self.dataset = TensorDataset.open(self.dataset_dir)
        self.num_datapoints = self.dataset.num_datapoints
        self.num_tensors = self.dataset.num_tensors
        self.datapoints_per_file = self.dataset.datapoints_per_file
        self.num_random_files = min(self.num_tensors, self.num_random_files)
        
    def _compute_data_params(self):
        """ Compute parameters of the dataset """
        # image params
        self.im_field_name = self.cfg['image_field_name']
        self.im_height = self.dataset.config['fields'][self.im_field_name]['height']
        self.im_width = self.dataset.config['fields'][self.im_field_name]['width']
        self.im_channels = self.dataset.config['fields'][self.im_field_name]['channels']
        self.im_center = np.array([float(self.im_height-1)/2, float(self.im_width-1)/2])

        # poses
        self.pose_field_name = self.cfg['pose_field_name']
        self.gripper_mode = self.gqcnn.gripper_mode
        self.pose_dim = pose_dim(self.gripper_mode)
        self.raw_pose_shape = self.dataset.config['fields'][self.pose_field_name]['height']
        
        # outputs
        self.label_field_name = self.target_metric_name
        self.num_categories = 2

        # compute the number of train and val examples
        self.num_train = 0
        self.num_val = 0
        for train_indices in self.train_index_map.values():
            self.num_train += train_indices.shape[0]
        for val_indices in self.train_index_map.values():
            self.num_val += val_indices.shape[0]

        # set params based on the number of training examples (convert epochs to steps)
        self.eval_frequency = int(self.eval_frequency * (self.num_train / self.train_batch_size))
        self.save_frequency = int(self.save_frequency * (self.num_train / self.train_batch_size))
        self.vis_frequency = int(self.vis_frequency * (self.num_train / self.train_batch_size))
        self.decay_step = self.decay_step_multiplier * self.num_train

    def _setup_tensorflow(self):
        """Setup Tensorflow placeholders, session, and queue """

        # setup nodes
        with tf.name_scope('train_data_node'):
            self.train_data_batch = tf.placeholder(tf.float32, (self.train_batch_size, self.im_height, self.im_width, self.im_channels))
        with tf.name_scope('train_pose_node'):
            self.train_poses_batch = tf.placeholder(tf.float32, (self.train_batch_size, self.pose_dim))
        if self.training_mode == TrainingMode.REGRESSION:
            train_label_dtype = tf.float32
            self.numpy_dtype = np.float32
        elif self.training_mode == TrainingMode.CLASSIFICATION:
            train_label_dtype = tf.int64
            self.numpy_dtype = np.int64
            if self.cfg['loss'] == 'weighted_cross_entropy':
                train_label_dtype = tf.float32
                self.numpy_dtype = np.float32            
        else:
            raise ValueError('Training mode %s not supported' %(self.training_mode))
        with tf.name_scope('train_labels_node'):
            self.train_labels_batch = tf.placeholder(train_label_dtype, (self.train_batch_size,))

        # create queue
        with tf.name_scope('data_queue'):
            self.q = tf.FIFOQueue(GeneralConstants.QUEUE_CAPACITY, [tf.float32, tf.float32, train_label_dtype], shapes=[(self.train_batch_size, self.im_height, self.im_width, self.im_channels), (self.train_batch_size, self.pose_dim), (self.train_batch_size,)])
            self.enqueue_op = self.q.enqueue([self.train_data_batch, self.train_poses_batch, self.train_labels_batch])
            self.train_labels_node = tf.placeholder(train_label_dtype, (self.train_batch_size,))
            self.input_im_node, self.input_pose_node, self.train_labels_node = self.q.dequeue()

        # setup weights using gqcnn
        if self.cfg['fine_tune']:
            # check that the gqcnn has weights, re-initialize if not
            try:
                self.weights = self.gqcnn.weights
            except:
                self.gqcnn.init_weights_gaussian()                

            # this assumes that a gqcnn was passed in that was initialized with weights from a model using GQCNN.load(), so all that has to
            # be done is to possibly reinitialize fc3/fc4/fc5
            reinit_pc1 = False
            if 'reinit_pc1' in self.cfg.keys():
                reinit_pc1 = self.cfg['reinit_pc1']
            self.gqcnn.reinitialize_layers(self.cfg['reinit_fc3'], self.cfg['reinit_fc4'], self.cfg['reinit_fc5'], reinit_pc1=reinit_pc1)
        else:
            self.gqcnn.init_weights_gaussian()

        # get weights
        self.weights = self.gqcnn.weights

        # open a tf session for the gqcnn object and store it also as the optimizer session
        self.saver = tf.train.Saver()
        self.sess = self.gqcnn.open_session()

        # setup term event/dead event
        self.term_event = threading.Event()
        self.term_event.clear()
        self.dead_event = threading.Event()
        self.dead_event.clear()

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

        # initialize the variables again now that we have added some new ones
        with self.sess.as_default():
            tf.global_variables_initializer().run()
        
    def _setup(self):
        """ Setup for optimization """

        # set up logger
        logging.getLogger().setLevel(logging.INFO)

        # initialize thread exit booleans
        self.queue_thread_exited = False
        self.forceful_exit = False

        # set random seed for deterministic execution
        self.debug = self.cfg['debug']
        if self.debug:
            np.random.seed(self.cfg['seed'])
            random.seed(self.cfg['seed'])

        # setup output directories
        self._setup_output_dirs()

        # setup logging
        self._setup_logging()

        # read training parameters from config file
        self._read_training_params()

        # setup denoising and synthetic data parameters
        self._setup_denoising_and_synthetic()

        # setup image and pose data files
        self._open_dataset()

        # compute train/test indices based on how the data is to be split
        if self.data_split_mode == 'image_wise':
            self._compute_indices_image_wise()
        elif self.data_split_mode == 'object_wise':
            self._compute_indices_object_wise()
        else:
            logging.error('Data split mode %s not supported!' %(self.data_split_mode))

        # compute data parameters
        self._compute_data_params()
            
        # compute means, std's, and normalization metrics
        self._compute_data_metrics()

        # setup tensorflow session/placeholders/queue
        self._setup_tensorflow()

        # setup summaries for visualizing metrics in tensorboard
        self._setup_summaries()

    def _load_and_enqueue(self):
        """ Loads and Enqueues a batch of images for training """
        # open dataset
        dataset = TensorDataset.open(self.dataset_dir)

        while not self.term_event.is_set():
            # sleep between reads
            time.sleep(GeneralConstants.QUEUE_SLEEP)

            # loop through data
            num_queued = 0
            start_i = 0
            end_i = 0
            file_num = 0
            queue_start = time.time()

            # init buffers
            train_images = np.zeros(
                [self.train_batch_size, self.im_height, self.im_width, self.im_channels]).astype(np.float32)
            train_poses = np.zeros([self.train_batch_size, self.pose_dim]).astype(np.float32)
            train_labels = np.zeros(self.train_batch_size).astype(self.numpy_dtype)
            
            while start_i < self.train_batch_size:
                # compute num remaining
                num_remaining = self.train_batch_size - num_queued
                
                # gen file index uniformly at random
                file_num = np.random.choice(self.num_tensors, size=1)[0]

                read_start = time.time()
                train_images_tensor = dataset.tensor(self.im_field_name, file_num)
                train_poses_tensor = dataset.tensor(self.pose_field_name, file_num)
                train_labels_tensor = dataset.tensor(self.label_field_name, file_num)
                read_stop = time.time()
                logging.debug('Reading data took %.3f sec' %(read_stop - read_start))
                logging.debug('File num: %d' %(file_num))
                
                # get batch indices uniformly at random
                train_ind = self.train_index_map[file_num]
                np.random.shuffle(train_ind)
                if self.gripper_mode == GripperMode.LEGACY_SUCTION:
                    tp_tmp = read_pose_data(train_poses_tensor.data, self.gripper_mode)
                    train_ind = train_ind[np.isfinite(tp_tmp[train_ind,1])]
                    
                # filter positives and negatives
                if self.training_mode == TrainingMode.CLASSIFICATION and self.pos_weight != 0.0:
                    labels = 1 * (train_labels_tensor.data > self.metric_thresh)
                    np.random.shuffle(train_ind)
                    filtered_ind = []
                    for index in train_ind:
                        if labels[index] == 0 and np.random.rand() < self.neg_accept_prob:
                            filtered_ind.append(index)
                        elif labels[index] == 1 and np.random.rand() < self.pos_accept_prob:
                            filtered_ind.append(index)
                    train_ind = np.array(filtered_ind)

                # samples train indices
                upper = min(num_remaining, train_ind.shape[0], self.max_training_examples_per_load)
                ind = train_ind[:upper]
                num_loaded = ind.shape[0]
                if num_loaded == 0:
                    logging.debug('Loaded zero examples!!!!')
                    continue
                
                # subsample data
                train_images_arr = train_images_tensor.data[ind, ...]
                train_poses_arr = train_poses_tensor.data[ind, ...]
                train_label_arr = train_labels_tensor[ind]
                num_images = train_images_arr.shape[0]

                # resize images
                rescale_factor = float(self.im_height) / train_images_arr.shape[1]
                if rescale_factor != 1.0:
                    resized_train_images_arr = np.zeros([num_images,
                                                         self.im_height,
                                                         self.im_width,
                                                         self.im_channels]).astype(np.float32)
                    for i in range(num_images):
                        for c in range(train_images_arr.shape[3]):
                            resized_train_images_arr[i,:,:,c] = sm.imresize(train_images_arr[i,:,:,c],
                                                                            rescale_factor,
                                                                            interp='bicubic', mode='F')
                    train_images_arr = resized_train_images_arr
                
                # add noises to images
                train_images_arr, train_poses_arr = self._distort(train_images_arr, train_poses_arr)

                # slice poses
                train_poses_arr = read_pose_data(train_poses_arr,
                                                 self.gripper_mode)

                # standardize inputs and outpus
                train_images_arr = (train_images_arr - self.im_mean) / self.im_std
                train_poses_arr = (train_poses_arr - self.pose_mean) / self.pose_std
                train_label_arr = 1 * (train_label_arr > self.metric_thresh)
                train_label_arr = train_label_arr.astype(self.numpy_dtype)

                # compute the number of examples loaded
                num_loaded = train_images_arr.shape[0]
                end_i = start_i + num_loaded
                    
                # enqueue training data batch
                train_images[start_i:end_i, ...] = train_images_arr.copy()
                train_poses[start_i:end_i,:] = train_poses_arr.copy()
                train_labels[start_i:end_i] = train_label_arr.copy()

                del train_images_arr
                del train_poses_arr
                del train_label_arr
		
                # update start index
                start_i = end_i
                num_queued += num_loaded

            # send data to queue
            if not self.term_event.is_set():
                try:
                    self.sess.run(self.enqueue_op, feed_dict={self.train_data_batch: train_images,
                                                              self.train_poses_batch: train_poses,
                                                              self.train_labels_batch: train_labels})
                    queue_stop = time.time()
                    logging.debug('Queue batch took %.3f sec' %(queue_stop - queue_start))
                except:
                    pass
        del train_images
        del train_poses
        del train_labels
        self.dead_event.set()
        logging.info('Queue Thread Exiting')
        self.queue_thread_exited = True

    def _distort(self, image_arr, pose_arr):
        """ Adds noise to a batch of images """
        # read params
        num_images = image_arr.shape[0]
        
        # denoising and synthetic data generation
        if self.cfg['multiplicative_denoising']:
            mult_samples = ss.gamma.rvs(self.gamma_shape, scale=self.gamma_scale, size=num_loaded)
            mult_samples = mult_samples[:,np.newaxis,np.newaxis,np.newaxis]
            image_arr = image_arr * np.tile(mult_samples, [1, self.im_height, self.im_width, self.im_channels])

        # add correlated Gaussian noise
        if self.cfg['gaussian_process_denoising']:
            for i in range(num_images):
                if np.random.rand() < self.cfg['gaussian_process_rate']:
                    train_image = image_arr[i,:,:,0]
                    gp_noise = ss.norm.rvs(scale=self.gp_sigma, size=self.gp_num_pix).reshape(self.gp_sample_height, self.gp_sample_width)
                    gp_noise = sm.imresize(gp_noise, self.gp_rescale_factor, interp='bicubic', mode='F')
                    train_image[train_image > 0] += gp_noise[train_image > 0]
                    image_arr[i,:,:,0] = train_image

        # symmetrize images
        if self.cfg['symmetrize']:
            for i in range(num_images):
                train_image = image_arr[i,:,:,0]
                # rotate with 50% probability
                if np.random.rand() < 0.5:
                    theta = 180.0
                    rot_map = cv2.getRotationMatrix2D(tuple(self.im_center), theta, 1)
                    train_image = cv2.warpAffine(train_image, rot_map, (self.im_height, self.im_width), flags=cv2.INTER_NEAREST)

                    if self.gripper_mode == GripperMode.LEGACY_SUCTION:
                        pose_arr[:,3] = -pose_arr[:,3]
                    elif self.gripper_mode == GripperMode.SUCTION:
                        pose_arr[:,4] = -pose_arr[:,4]
                # reflect left right with 50% probability
                if np.random.rand() < 0.5:
                    train_image = np.fliplr(train_image)
                # reflect up down with 50% probability
                if np.random.rand() < 0.5:
                    train_image = np.flipud(train_image)

                    if self.gripper_mode == GripperMode.LEGACY_SUCTION:
                        pose_arr[:,3] = -pose_arr[:,3]
                    elif self.gripper_mode == GripperMode.SUCTION:
                        pose_arr[:,4] = -pose_arr[:,4]
                image_arr[i,:,:,0] = train_image
        return image_arr, pose_arr

    def _error_rate_in_batches(self, num_files_eval=None, validation_set=True):
        """ Get all predictions for a dataset by running it in small batches

        Returns
        -------
        : float
            validation error
        """
        error_rates = []

        # subsample files
        file_indices = np.arange(self.num_tensors)
        if num_files_eval is None:
            num_files_eval = self.max_files_eval
        np.random.shuffle(file_indices)
        if self.max_files_eval is not None and num_files_eval > 0:
            file_indices = file_indices[:num_files_eval]

        for i in file_indices:
            # load next file
            images = self.dataset.tensor(self.im_field_name, i).data
            poses = self.dataset.tensor(self.pose_field_name, i).data
            labels = self.dataset.tensor(self.label_field_name, i).data

            # if no datapoints from this file are in validation then just continue
            if validation_set:
                indices =  self.val_index_map[i]
            else:
                indices =  self.train_index_map[i]                    
            if len(indices) == 0:
                continue

            images = images[indices,...]
            poses = read_pose_data(poses[indices,:],
                                   self.gripper_mode)
            labels = labels[indices]

            if self.training_mode == TrainingMode.CLASSIFICATION:
                labels = 1 * (labels > self.metric_thresh)
                labels = labels.astype(np.uint8)
                    
            # get predictions
            predictions = self.gqcnn.predict(images, poses)
            
            # get error rate
            if self.training_mode == TrainingMode.CLASSIFICATION:
                error_rates.append(BinaryClassificationResult(predictions[:,1], labels).error_rate)
            else:
                error_rates.append(RegressionResult(predictions, labels).error_rate)
            
            # clean up
            del images
            del poses
            del labels

        # return average error rate over all files (assuming same size)
        return np.mean(error_rates)
