# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Trains a GQ-CNN network using Tensorflow backend.

Author
------
Vishal Satish & Jeff Mahler
"""
import collections
import json
import multiprocessing as mp
import os
import random
import shutil
import signal
import subprocess
import sys
import time

from past.builtins import xrange
import cv2
import numpy as np
import scipy.stats as ss
import tensorflow as tf

from autolab_core import (BinaryClassificationResult, RegressionResult,
                          TensorDataset, Logger)
from autolab_core.constants import JSON_INDENT
import autolab_core.utils as utils

from ...utils import (TrainingMode, GripperMode, InputDepthMode,
                      GeneralConstants, TrainStatsLogger, GQCNNTrainingStatus,
                      GQCNNFilenames, pose_dim, read_pose_data,
                      weight_name_to_layer_name, is_py2, imresize)

if is_py2():
    import Queue
else:
    import queue as Queue


class GQCNNTrainerTF(object):
    """Trains a GQ-CNN with Tensorflow backend."""

    def __init__(self,
                 gqcnn,
                 dataset_dir,
                 split_name,
                 output_dir,
                 config,
                 name=None,
                 progress_dict=None,
                 verbose=True):
        """
        Parameters
        ----------
        gqcnn : :obj:`GQCNN`
            Grasp quality neural network to optimize.
        dataset_dir : str
            Path to the training/validation dataset.
        split_name : str
            Name of the split to train on.
        output_dir : str
            Path to save the model output.
        config : dict
            Dictionary of configuration parameters.
        name : str
            Name of the the model.
        """
        self.gqcnn = gqcnn
        self.dataset_dir = dataset_dir
        self.split_name = split_name
        self.output_dir = output_dir
        self.cfg = config
        self.tensorboard_has_launched = False
        self.model_name = name
        self.progress_dict = progress_dict
        self.finetuning = False

        # Create a directory for the model.
        if self.model_name is None:
            model_id = utils.gen_experiment_id()
            self.model_name = "model_{}".format(model_id)
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # Set up logger.
        self.logger = Logger.get_logger(self.__class__.__name__,
                                        log_file=os.path.join(
                                            self.model_dir, "training.log"),
                                        silence=(not verbose),
                                        global_log_file=verbose)

        # Check default split.
        if split_name is None:
            self.logger.warning("Using default image-wise split.")
            self.split_name = "image_wise"

        # Update cfg for saving.
        self.cfg["dataset_dir"] = self.dataset_dir
        self.cfg["split_name"] = self.split_name

    def _create_loss(self):
        """Creates a loss based on config file.

        Returns
        -------
        :obj:`tensorflow Tensor`
            Loss.
        """
        if self.cfg["loss"] == "l2":
            return (1.0 / self.train_batch_size) * tf.nn.l2_loss(
                tf.subtract(tf.nn.sigmoid(self.train_net_output),
                            self.train_labels_node))
        elif self.cfg["loss"] == "sparse":
            if self._angular_bins > 0:
                log = tf.reshape(
                    tf.dynamic_partition(self.train_net_output,
                                         self.train_pred_mask_node, 2)[1],
                    (-1, 2))
                return tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        _sentinel=None,
                        labels=self.train_labels_node,
                        logits=log))
            else:
                return tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        _sentinel=None,
                        labels=self.train_labels_node,
                        logits=self.train_net_output,
                        name=None))
        elif self.cfg["loss"] == "weighted_cross_entropy":
            return tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=tf.reshape(self.train_labels_node, [-1, 1]),
                    logits=self.train_net_output,
                    pos_weight=self.pos_weight,
                    name=None))

    def _create_optimizer(self, loss, batch, var_list, learning_rate):
        """Create optimizer based on config file.

        Parameters
        ----------
        loss : :obj:`tensorflow Tensor`
            Loss to use, generated with `_create_loss`.
        batch : :obj:`tf.Variable`
            Variable to keep track of the current gradient step number.
        var_list : :obj:`lst`
            List of tf.Variable objects to update to minimize loss (ex. network
            weights).
        learning_rate : float
            Learning rate for training.

        Returns
        -------
        :obj:`tf.train.Optimizer`
            Optimizer.
        """
        # Instantiate optimizer.
        if self.cfg["optimizer"] == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                   self.momentum_rate)
        elif self.cfg["optimizer"] == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.cfg["optimizer"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise ValueError("Optimizer '{}' not supported".format(
                self.cfg["optimizer"]))

        # Compute gradients.
        gradients, variables = zip(
            *optimizer.compute_gradients(loss, var_list=var_list))

        # Clip gradients to prevent exploding gradient problem.
        gradients, global_grad_norm = tf.clip_by_global_norm(
            gradients, self.max_global_grad_norm)

        # Generate op to apply gradients.
        apply_grads = optimizer.apply_gradients(zip(gradients, variables),
                                                global_step=batch)

        return apply_grads, global_grad_norm

    def _launch_tensorboard(self):
        """Launches Tensorboard to visualize training."""
        FNULL = open(os.devnull, "w")
        tensorboard_launch_msg = ("Launching Tensorboard, please navigate to"
                                  " localhost:{} in your favorite web browser"
                                  " to view summaries.")
        self.logger.info(tensorboard_launch_msg.format(self._tensorboard_port))
        self._tensorboard_proc = subprocess.Popen([
            "tensorboard", "--port",
            str(self._tensorboard_port), "--logdir", self.summary_dir
        ],
                                                  stdout=FNULL)

    def _close_tensorboard(self):
        """Closes Tensorboard process."""
        self.logger.info("Closing Tensorboard...")
        self._tensorboard_proc.terminate()

    def train(self):
        """Perform optimization."""
        with self.gqcnn.tf_graph.as_default():
            self._train()

    def _train(self):
        """Perform optimization."""
        # Run setup.
        if self.progress_dict is not None:
            self.progress_dict[
                "training_status"] = GQCNNTrainingStatus.SETTING_UP
        self._setup()

        # Build network.
        self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node)

        # Optimize weights.
        if self.progress_dict is not None:
            self.progress_dict[
                "training_status"] = GQCNNTrainingStatus.TRAINING
        self._optimize_weights()

    def finetune(self, base_model_dir):
        """Perform fine-tuning.

        Parameters
        ----------
        base_model_dir : str
            Path to the pre-trained base model to use.
        """
        with self.gqcnn.tf_graph.as_default():
            self._finetune(base_model_dir)

    def _finetune(self, base_model_dir):
        """Perform fine-tuning.

        Parameters
        ----------
        base_model_dir : str
            Path to the pre-trained base model to use.
        """
        # Set flag and base model for fine-tuning.
        self.finetuning = True
        self.base_model_dir = base_model_dir

        # Run setup.
        self._setup()

        # Build network.
        self.gqcnn.set_base_network(base_model_dir)
        self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node)

        # Optimize weights.
        if self.progress_dict is not None:
            self.progress_dict[
                "training_status"] = GQCNNTrainingStatus.TRAINING
        self._optimize_weights(finetune=True)

    def _optimize_weights(self, finetune=False):
        """Optimize the network weights."""
        start_time = time.time()

        # Setup output.
        self.train_net_output = self.gqcnn.output
        if self.training_mode == TrainingMode.CLASSIFICATION:
            if self.cfg["loss"] == "weighted_cross_entropy":
                self.gqcnn.add_sigmoid_to_output()
            else:
                self.gqcnn.add_softmax_to_output()
        elif self.training_mode == TrainingMode.REGRESSION:
            self.gqcnn.add_sigmoid_to_output()
        else:
            raise ValueError("Training mode: {} not supported !".format(
                self.training_mode))
        train_predictions = self.gqcnn.output
        drop_rate_in = self.gqcnn.input_drop_rate_node
        self.weights = self.gqcnn.weights

        # Once weights have been initialized, create TF saver for weights.
        self.saver = tf.train.Saver()

        # Form loss.
        with tf.name_scope("loss"):
            # Part 1: error.
            loss = self._create_loss()
            unregularized_loss = loss

            # Part 2: regularization.
            layer_weights = list(self.weights.values())
            with tf.name_scope("regularization"):
                regularizers = tf.nn.l2_loss(layer_weights[0])
                for w in layer_weights[1:]:
                    regularizers = regularizers + tf.nn.l2_loss(w)
            loss += self.train_l2_regularizer * regularizers

        # Setup learning rate.
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.base_lr,  # Base learning rate.
            batch * self.train_batch_size,  # Current index into the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # decay rate.
            staircase=True)

        # Setup variable list.
        var_list = list(self.weights.values())
        if finetune:
            var_list = []
            for weights_name, weights_val in self.weights.items():
                layer_name = weight_name_to_layer_name(weights_name)
                if self.optimize_base_layers or \
                        layer_name not in self.gqcnn._base_layer_names:
                    var_list.append(weights_val)

        # Create optimizer.
        with tf.name_scope("optimizer"):
            apply_grad_op, global_grad_norm = self._create_optimizer(
                loss, batch, var_list, learning_rate)

        # Add a handler for SIGINT for graceful exit.
        def handler(signum, frame):
            self.logger.info("caught CTRL+C, exiting...")
            self._cleanup()
            exit(0)

        signal.signal(signal.SIGINT, handler)

        # Now that everything in our graph is set up, we write the graph to the
        # summary event so it can be visualized in Tensorboard.
        self.summary_writer.add_graph(self.gqcnn.tf_graph)

        # Begin optimization loop.
        try:
            # Start prefetch queue workers.
            self.prefetch_q_workers = []
            seed = self._seed
            for i in range(self.num_prefetch_q_workers):
                if self.num_prefetch_q_workers > 1 or not self._debug:
                    seed = np.random.randint(GeneralConstants.SEED_SAMPLE_MAX)
                p = mp.Process(target=self._load_and_enqueue, args=(seed, ))
                p.start()
                self.prefetch_q_workers.append(p)

            # Init TF variables.
            init = tf.global_variables_initializer()
            self.sess.run(init)

            self.logger.info("Beginning Optimization...")

            # Create a `TrainStatsLogger` object to log training statistics at
            # certain intervals.
            self.train_stats_logger = TrainStatsLogger(self.model_dir)

            # Loop through training steps.
            training_range = xrange(
                int(self.num_epochs * self.num_train) // self.train_batch_size)
            for step in training_range:
                # Run optimization.
                step_start = time.time()
                if self._angular_bins > 0:
                    images, poses, labels, masks = self.prefetch_q.get()
                    _, l, ur_l, lr, predictions, raw_net_output = \
                        self.sess.run(
                            [
                                apply_grad_op, loss, unregularized_loss,
                                learning_rate, train_predictions,
                                self.train_net_output
                            ],
                            feed_dict={
                                drop_rate_in: self.drop_rate,
                                self.input_im_node: images,
                                self.input_pose_node: poses,
                                self.train_labels_node: labels,
                                self.train_pred_mask_node: masks
                            },
                            options=GeneralConstants.timeout_option)
                else:
                    images, poses, labels = self.prefetch_q.get()
                    _, l, ur_l, lr, predictions, raw_net_output = \
                        self.sess.run(
                            [
                                apply_grad_op, loss, unregularized_loss,
                                learning_rate, train_predictions,
                                self.train_net_output
                            ],
                            feed_dict={
                                drop_rate_in: self.drop_rate,
                                self.input_im_node: images,
                                self.input_pose_node: poses,
                                self.train_labels_node: labels
                            },
                            options=GeneralConstants.timeout_option)
                step_stop = time.time()
                self.logger.info("Step took {} sec.".format(
                    str(round(step_stop - step_start, 3))))

                if self.training_mode == TrainingMode.REGRESSION:
                    self.logger.info("Max " + str(np.max(predictions)))
                    self.logger.info("Min " + str(np.min(predictions)))
                elif self.cfg["loss"] != "weighted_cross_entropy":
                    if self._angular_bins == 0:
                        ex = np.exp(raw_net_output - np.tile(
                            np.max(raw_net_output, axis=1)[:, np.newaxis],
                            [1, 2]))
                        softmax = ex / np.tile(
                            np.sum(ex, axis=1)[:, np.newaxis], [1, 2])

                        self.logger.info("Max " + str(np.max(softmax[:, 1])))
                        self.logger.info("Min " + str(np.min(softmax[:, 1])))
                        self.logger.info("Pred nonzero " +
                                         str(np.sum(softmax[:, 1] > 0.5)))
                        self.logger.info("True nonzero " + str(np.sum(labels)))

                else:
                    sigmoid = 1.0 / (1.0 + np.exp(-raw_net_output))
                    self.logger.info("Max " + str(np.max(sigmoid)))
                    self.logger.info("Min " + str(np.min(sigmoid)))
                    self.logger.info("Pred nonzero " +
                                     str(np.sum(sigmoid > 0.5)))
                    self.logger.info("True nonzero " +
                                     str(np.sum(labels > 0.5)))

                if np.isnan(l) or np.any(np.isnan(poses)):
                    self.logger.error(
                        "Encountered NaN in loss or training poses!")
                    raise Exception

                # Log output.
                if step % self.log_frequency == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    self.logger.info("Step {} (epoch {}), {} s".format(
                        step,
                        str(
                            round(
                                step * self.train_batch_size / self.num_train,
                                3)),
                        str(round(1000 * elapsed_time / self.eval_frequency,
                                  2))))
                    self.logger.info(
                        "Minibatch loss: {}, learning rate: {}".format(
                            str(round(l, 3)), str(round(lr, 6))))
                    if self.progress_dict is not None:
                        self.progress_dict["epoch"] = str(
                            round(
                                step * self.train_batch_size / self.num_train,
                                2))

                    train_error = l
                    if self.training_mode == TrainingMode.CLASSIFICATION:
                        if self._angular_bins > 0:
                            predictions = predictions[masks.astype(
                                bool)].reshape((-1, 2))
                        classification_result = BinaryClassificationResult(
                            predictions[:, 1], labels)
                        train_error = classification_result.error_rate

                    self.logger.info("Minibatch error: {}".format(
                        str(round(train_error, 3))))

                    self.summary_writer.add_summary(
                        self.sess.run(
                            self.merged_log_summaries,
                            feed_dict={
                                self.minibatch_error_placeholder: train_error,
                                self.minibatch_loss_placeholder: l,
                                self.learning_rate_placeholder: lr
                            }), step)
                    sys.stdout.flush()

                    # Update the `TrainStatsLogger`.
                    self.train_stats_logger.update(train_eval_iter=step,
                                                   train_loss=l,
                                                   train_error=train_error,
                                                   total_train_error=None,
                                                   val_eval_iter=None,
                                                   val_error=None,
                                                   learning_rate=lr)

                # Evaluate model.
                if step % self.eval_frequency == 0 and step > 0:
                    if self.cfg["eval_total_train_error"]:
                        train_result = self._error_rate_in_batches(
                            validation_set=False)
                        self.logger.info("Training error: {}".format(
                            str(round(train_result.error_rate, 3))))

                        # Update the `TrainStatsLogger` and save.
                        self.train_stats_logger.update(
                            train_eval_iter=None,
                            train_loss=None,
                            train_error=None,
                            total_train_error=train_result.error_rate,
                            total_train_loss=train_result.cross_entropy_loss,
                            val_eval_iter=None,
                            val_error=None,
                            learning_rate=None)
                        self.train_stats_logger.log()

                    if self.train_pct < 1.0:
                        val_result = self._error_rate_in_batches()
                        self.summary_writer.add_summary(
                            self.sess.run(
                                self.merged_eval_summaries,
                                feed_dict={
                                    self.val_error_placeholder: val_result.
                                    error_rate
                                }), step)
                        self.logger.info("Validation error: {}".format(
                            str(round(val_result.error_rate, 3))))
                        self.logger.info("Validation loss: {}".format(
                            str(round(val_result.cross_entropy_loss, 3))))
                    sys.stdout.flush()

                    # Update the `TrainStatsLogger`.
                    if self.train_pct < 1.0:
                        self.train_stats_logger.update(
                            train_eval_iter=None,
                            train_loss=None,
                            train_error=None,
                            total_train_error=None,
                            val_eval_iter=step,
                            val_loss=val_result.cross_entropy_loss,
                            val_error=val_result.error_rate,
                            learning_rate=None)
                    else:
                        self.train_stats_logger.update(train_eval_iter=None,
                                                       train_loss=None,
                                                       train_error=None,
                                                       total_train_error=None,
                                                       val_eval_iter=step,
                                                       learning_rate=None)

                    # Save everything!
                    self.train_stats_logger.log()

                # Save the model.
                if step % self.save_frequency == 0 and step > 0:
                    self.saver.save(
                        self.sess,
                        os.path.join(self.model_dir,
                                     GQCNNFilenames.INTER_MODEL.format(step)))
                    self.saver.save(
                        self.sess,
                        os.path.join(self.model_dir,
                                     GQCNNFilenames.FINAL_MODEL))

                # Launch tensorboard only after the first iteration.
                if not self.tensorboard_has_launched:
                    self.tensorboard_has_launched = True
                    self._launch_tensorboard()

            # Get final errors and flush the stdout pipeline.
            final_val_result = self._error_rate_in_batches()
            self.logger.info("Final validation error: {}".format(
                str(round(final_val_result.error_rate, 3))))
            self.logger.info("Final validation loss: {}".format(
                str(round(final_val_result.cross_entropy_loss, 3))))
            if self.cfg["eval_total_train_error"]:
                final_train_result = self._error_rate_in_batches(
                    validation_set=False)
                self.logger.info("Final training error: {}".format(
                    final_train_result.error_rate))
                self.logger.info("Final training loss: {}".format(
                    final_train_result.cross_entropy_loss))
            sys.stdout.flush()

            # Update the `TrainStatsLogger`.
            self.train_stats_logger.update(
                train_eval_iter=None,
                train_loss=None,
                train_error=None,
                total_train_error=None,
                val_eval_iter=step,
                val_loss=final_val_result.cross_entropy_loss,
                val_error=final_val_result.error_rate,
                learning_rate=None)

            # Log & save everything!
            self.train_stats_logger.log()
            self.saver.save(
                self.sess,
                os.path.join(self.model_dir, GQCNNFilenames.FINAL_MODEL))

        except Exception as e:
            self._cleanup()
            raise e

        self._cleanup()

    def _compute_data_metrics(self):
        """Calculate image mean, image std, pose mean, pose std, normalization
        params."""
        # Subsample tensors for faster runtime.
        random_file_indices = np.random.choice(self.num_tensors,
                                               size=self.num_random_files,
                                               replace=False)

        if self.gqcnn.input_depth_mode == InputDepthMode.POSE_STREAM:
            # Compute image stats.
            im_mean_filename = os.path.join(self.model_dir,
                                            GQCNNFilenames.IM_MEAN)
            im_std_filename = os.path.join(self.model_dir,
                                           GQCNNFilenames.IM_STD)
            if os.path.exists(im_mean_filename) and os.path.exists(
                    im_std_filename):
                self.im_mean = np.load(im_mean_filename)
                self.im_std = np.load(im_std_filename)
            else:
                self.im_mean = 0
                self.im_std = 0

                # Compute mean.
                self.logger.info("Computing image mean")
                num_summed = 0
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info(
                            "Adding file {} of {} to image mean estimate".
                            format(k + 1, random_file_indices.shape[0]))
                    im_data = self.dataset.tensor(self.im_field_name, i).arr
                    train_indices = self.train_index_map[i]
                    if train_indices.shape[0] > 0:
                        self.im_mean += np.sum(im_data[train_indices, ...])
                        num_summed += self.train_index_map[i].shape[
                            0] * im_data.shape[1] * im_data.shape[2]
                self.im_mean = self.im_mean / num_summed

                # Compute std.
                self.logger.info("Computing image std")
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info(
                            "Adding file {} of {} to image std estimate".
                            format(k + 1, random_file_indices.shape[0]))
                    im_data = self.dataset.tensor(self.im_field_name, i).arr
                    train_indices = self.train_index_map[i]
                    if train_indices.shape[0] > 0:
                        self.im_std += np.sum(
                            (im_data[train_indices, ...] - self.im_mean)**2)
                self.im_std = np.sqrt(self.im_std / num_summed)

                # Save.
                np.save(im_mean_filename, self.im_mean)
                np.save(im_std_filename, self.im_std)

            # Update GQ-CNN instance.
            self.gqcnn.set_im_mean(self.im_mean)
            self.gqcnn.set_im_std(self.im_std)

            # Compute pose stats.
            pose_mean_filename = os.path.join(self.model_dir,
                                              GQCNNFilenames.POSE_MEAN)
            pose_std_filename = os.path.join(self.model_dir,
                                             GQCNNFilenames.POSE_STD)
            if os.path.exists(pose_mean_filename) and os.path.exists(
                    pose_std_filename):
                self.pose_mean = np.load(pose_mean_filename)
                self.pose_std = np.load(pose_std_filename)
            else:
                self.pose_mean = np.zeros(self.raw_pose_shape)
                self.pose_std = np.zeros(self.raw_pose_shape)

                # Compute mean.
                num_summed = 0
                self.logger.info("Computing pose mean")
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info(
                            "Adding file {} of {} to pose mean estimate".
                            format(k + 1, random_file_indices.shape[0]))
                    pose_data = self.dataset.tensor(self.pose_field_name,
                                                    i).arr
                    train_indices = self.train_index_map[i]
                    if self.gripper_mode == GripperMode.SUCTION:
                        rand_indices = np.random.choice(
                            pose_data.shape[0],
                            size=pose_data.shape[0] // 2,
                            replace=False)
                        pose_data[rand_indices,
                                  4] = -pose_data[rand_indices, 4]
                    elif self.gripper_mode == GripperMode.LEGACY_SUCTION:
                        rand_indices = np.random.choice(
                            pose_data.shape[0],
                            size=pose_data.shape[0] // 2,
                            replace=False)
                        pose_data[rand_indices,
                                  3] = -pose_data[rand_indices, 3]
                    if train_indices.shape[0] > 0:
                        pose_data = pose_data[train_indices, :]
                        pose_data = pose_data[np.isfinite(pose_data[:, 3]), :]
                        self.pose_mean += np.sum(pose_data, axis=0)
                        num_summed += pose_data.shape[0]
                self.pose_mean = self.pose_mean / num_summed

                # Compute std.
                self.logger.info("Computing pose std")
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info(
                            "Adding file {} of {} to pose std estimate".format(
                                k + 1, random_file_indices.shape[0]))
                    pose_data = self.dataset.tensor(self.pose_field_name,
                                                    i).arr
                    train_indices = self.train_index_map[i]
                    if self.gripper_mode == GripperMode.SUCTION:
                        rand_indices = np.random.choice(
                            pose_data.shape[0],
                            size=pose_data.shape[0] // 2,
                            replace=False)
                        pose_data[rand_indices,
                                  4] = -pose_data[rand_indices, 4]
                    elif self.gripper_mode == GripperMode.LEGACY_SUCTION:
                        rand_indices = np.random.choice(
                            pose_data.shape[0],
                            size=pose_data.shape[0] // 2,
                            replace=False)
                        pose_data[rand_indices,
                                  3] = -pose_data[rand_indices, 3]
                    if train_indices.shape[0] > 0:
                        pose_data = pose_data[train_indices, :]
                        pose_data = pose_data[np.isfinite(pose_data[:, 3]), :]
                        self.pose_std += np.sum(
                            (pose_data - self.pose_mean)**2, axis=0)
                self.pose_std = np.sqrt(self.pose_std / num_summed)
                self.pose_std[self.pose_std == 0] = 1.0

                # Save.
                self.pose_mean = read_pose_data(self.pose_mean,
                                                self.gripper_mode)
                self.pose_std = read_pose_data(self.pose_std,
                                               self.gripper_mode)
                np.save(pose_mean_filename, self.pose_mean)
                np.save(pose_std_filename, self.pose_std)

            # Update GQ-CNN instance.
            self.gqcnn.set_pose_mean(self.pose_mean)
            self.gqcnn.set_pose_std(self.pose_std)

            # Check for invalid values.
            if np.any(np.isnan(self.pose_mean)) or np.any(
                    np.isnan(self.pose_std)):
                self.logger.error(
                    "Pose mean or pose std is NaN! Check the input dataset")
                exit(0)

        elif self.gqcnn.input_depth_mode == InputDepthMode.SUB:
            # Compute (image - depth) stats.
            im_depth_sub_mean_filename = os.path.join(
                self.model_dir, GQCNNFilenames.IM_DEPTH_SUB_MEAN)
            im_depth_sub_std_filename = os.path.join(
                self.model_dir, GQCNNFilenames.IM_DEPTH_SUB_STD)
            if os.path.exists(im_depth_sub_mean_filename) and os.path.exists(
                    im_depth_sub_std_filename):
                self.im_depth_sub_mean = np.load(im_depth_sub_mean_filename)
                self.im_depth_sub_std = np.load(im_depth_sub_std_filename)
            else:
                self.im_depth_sub_mean = 0
                self.im_depth_sub_std = 0

                # Compute mean.
                self.logger.info("Computing (image - depth) mean.")
                num_summed = 0
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info("Adding file {} of {}...".format(
                            k + 1, random_file_indices.shape[0]))
                    im_data = self.dataset.tensor(self.im_field_name, i).arr
                    depth_data = read_pose_data(
                        self.dataset.tensor(self.pose_field_name, i).arr,
                        self.gripper_mode)
                    sub_data = im_data - np.tile(
                        np.reshape(depth_data, (-1, 1, 1, 1)),
                        (1, im_data.shape[1], im_data.shape[2], 1))
                    train_indices = self.train_index_map[i]
                    if train_indices.shape[0] > 0:
                        self.im_depth_sub_mean += np.sum(
                            sub_data[train_indices, ...])
                        num_summed += self.train_index_map[i].shape[
                            0] * im_data.shape[1] * im_data.shape[2]
                self.im_depth_sub_mean = self.im_depth_sub_mean / num_summed

                # Compute std.
                self.logger.info("Computing (image - depth) std.")
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info("Adding file {} of {}...".format(
                            k + 1, random_file_indices.shape[0]))
                    im_data = self.dataset.tensor(self.im_field_name, i).arr
                    depth_data = read_pose_data(
                        self.dataset.tensor(self.pose_field_name, i).arr,
                        self.gripper_mode)
                    sub_data = im_data - np.tile(
                        np.reshape(depth_data, (-1, 1, 1, 1)),
                        (1, im_data.shape[1], im_data.shape[2], 1))
                    train_indices = self.train_index_map[i]
                    if train_indices.shape[0] > 0:
                        self.im_depth_sub_std += np.sum(
                            (sub_data[train_indices, ...] -
                             self.im_depth_sub_mean)**2)
                self.im_depth_sub_std = np.sqrt(self.im_depth_sub_std /
                                                num_summed)

                # Save.
                np.save(im_depth_sub_mean_filename, self.im_depth_sub_mean)
                np.save(im_depth_sub_std_filename, self.im_depth_sub_std)

            # Update GQ-CNN instance.
            self.gqcnn.set_im_depth_sub_mean(self.im_depth_sub_mean)
            self.gqcnn.set_im_depth_sub_std(self.im_depth_sub_std)

        elif self.gqcnn.input_depth_mode == InputDepthMode.IM_ONLY:
            # Compute image stats.
            im_mean_filename = os.path.join(self.model_dir,
                                            GQCNNFilenames.IM_MEAN)
            im_std_filename = os.path.join(self.model_dir,
                                           GQCNNFilenames.IM_STD)
            if os.path.exists(im_mean_filename) and os.path.exists(
                    im_std_filename):
                self.im_mean = np.load(im_mean_filename)
                self.im_std = np.load(im_std_filename)
            else:
                self.im_mean = 0
                self.im_std = 0

                # Compute mean.
                self.logger.info("Computing image mean.")
                num_summed = 0
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info("Adding file {} of {}...".format(
                            k + 1, random_file_indices.shape[0]))
                    im_data = self.dataset.tensor(self.im_field_name, i).arr
                    train_indices = self.train_index_map[i]
                    if train_indices.shape[0] > 0:
                        self.im_mean += np.sum(im_data[train_indices, ...])
                        num_summed += self.train_index_map[i].shape[
                            0] * im_data.shape[1] * im_data.shape[2]
                self.im_mean = self.im_mean / num_summed

                # Compute std.
                self.logger.info("Computing image std.")
                for k, i in enumerate(random_file_indices):
                    if k % self.preproc_log_frequency == 0:
                        self.logger.info("Adding file {} of {}...".format(
                            k + 1, random_file_indices.shape[0]))
                    im_data = self.dataset.tensor(self.im_field_name, i).arr
                    train_indices = self.train_index_map[i]
                    if train_indices.shape[0] > 0:
                        self.im_std += np.sum(
                            (im_data[train_indices, ...] - self.im_mean)**2)
                self.im_std = np.sqrt(self.im_std / num_summed)

                # Save.
                np.save(im_mean_filename, self.im_mean)
                np.save(im_std_filename, self.im_std)

            # Update GQ-CNN instance.
            self.gqcnn.set_im_mean(self.im_mean)
            self.gqcnn.set_im_std(self.im_std)

        # Compute normalization parameters of the network.
        pct_pos_train_filename = os.path.join(self.model_dir,
                                              GQCNNFilenames.PCT_POS_TRAIN)
        pct_pos_val_filename = os.path.join(self.model_dir,
                                            GQCNNFilenames.PCT_POS_VAL)
        if os.path.exists(pct_pos_train_filename) and os.path.exists(
                pct_pos_val_filename):
            pct_pos_train = np.load(pct_pos_train_filename)
            pct_pos_val = np.load(pct_pos_val_filename)
        else:
            self.logger.info("Computing grasp quality metric stats.")
            all_train_metrics = None
            all_val_metrics = None

            # Read metrics.
            for k, i in enumerate(random_file_indices):
                if k % self.preproc_log_frequency == 0:
                    self.logger.info("Adding file {} of {}...".format(
                        k + 1, random_file_indices.shape[0]))
                metric_data = self.dataset.tensor(self.label_field_name, i).arr
                train_indices = self.train_index_map[i]
                val_indices = self.val_index_map[i]

                if train_indices.shape[0] > 0:
                    train_metric_data = metric_data[train_indices]
                    if all_train_metrics is None:
                        all_train_metrics = train_metric_data
                    else:
                        all_train_metrics = np.r_[all_train_metrics,
                                                  train_metric_data]

                if val_indices.shape[0] > 0:
                    val_metric_data = metric_data[val_indices]
                    if all_val_metrics is None:
                        all_val_metrics = val_metric_data
                    else:
                        all_val_metrics = np.r_[all_val_metrics,
                                                val_metric_data]

            # Compute train stats.
            self.min_metric = np.min(all_train_metrics)
            self.max_metric = np.max(all_train_metrics)
            self.mean_metric = np.mean(all_train_metrics)
            self.median_metric = np.median(all_train_metrics)

            # Save metrics.
            pct_pos_train = np.sum(all_train_metrics > self.metric_thresh
                                   ) / all_train_metrics.shape[0]
            np.save(pct_pos_train_filename, np.array(pct_pos_train))

            if self.train_pct < 1.0:
                pct_pos_val = np.sum(all_val_metrics > self.metric_thresh
                                     ) / all_val_metrics.shape[0]
                np.save(pct_pos_val_filename, np.array(pct_pos_val))

        self.logger.info("Percent positive in train: " + str(pct_pos_train))
        if self.train_pct < 1.0:
            self.logger.info("Percent positive in val: " + str(pct_pos_val))

        if self._angular_bins > 0:
            self.logger.info("Calculating angular bin statistics...")
            bin_counts = np.zeros((self._angular_bins, ))
            for m in range(self.num_tensors):
                pose_arr = self.dataset.tensor(self.pose_field_name, m).arr
                angles = pose_arr[:, 3]
                neg_ind = np.where(angles < 0)
                angles = np.abs(angles) % self._max_angle
                angles[neg_ind] *= -1
                g_90 = np.where(angles > (self._max_angle / 2))
                l_neg_90 = np.where(angles < (-1 * (self._max_angle / 2)))
                angles[g_90] -= self._max_angle
                angles[l_neg_90] += self._max_angle
                # TODO(vsatish): Actually fix this.
                angles *= -1  # Hack to fix reverse angle convention.
                angles += (self._max_angle / 2)
                for i in range(angles.shape[0]):
                    bin_counts[int(angles[i] // self._bin_width)] += 1
            self.logger.info("Bin counts: {}.".format(bin_counts))

    def _compute_split_indices(self):
        """Compute train and validation indices for each tensor to speed data
        accesses."""
        # Read indices.
        train_indices, val_indices, _ = self.dataset.split(self.split_name)

        # Loop through tensors, assigning indices to each file.
        self.train_index_map = {}
        for i in range(self.dataset.num_tensors):
            self.train_index_map[i] = []

        for i in train_indices:
            tensor_index = self.dataset.tensor_index(i)
            datapoint_indices = self.dataset.datapoint_indices_for_tensor(
                tensor_index)
            lowest = np.min(datapoint_indices)
            self.train_index_map[tensor_index].append(i - lowest)

        for i, indices in self.train_index_map.items():
            self.train_index_map[i] = np.array(indices)

        self.val_index_map = {}
        for i in range(self.dataset.num_tensors):
            self.val_index_map[i] = []

        for i in val_indices:
            tensor_index = self.dataset.tensor_index(i)
            if tensor_index not in self.val_index_map:
                self.val_index_map[tensor_index] = []
            datapoint_indices = self.dataset.datapoint_indices_for_tensor(
                tensor_index)
            lowest = np.min(datapoint_indices)
            self.val_index_map[tensor_index].append(i - lowest)

        for i, indices in self.val_index_map.items():
            self.val_index_map[i] = np.array(indices)

    def _setup_output_dirs(self):
        """Setup output directories."""
        self.logger.info("Saving model to: {}".format(self.model_dir))

        # Create the summary dir.
        self.summary_dir = os.path.join(self.model_dir,
                                        "tensorboard_summaries")
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)
        else:
            # If the summary directory already exists, clean it out by deleting
            # all files in it. We don't want Tensorboard to get confused with
            # old logs while reusing the same directory.
            old_files = os.listdir(self.summary_dir)
            for f in old_files:
                os.remove(os.path.join(self.summary_dir, f))

        # Setup filter directory.
        self.filter_dir = os.path.join(self.model_dir, "filters")
        if not os.path.exists(self.filter_dir):
            os.mkdir(self.filter_dir)

    def _save_configs(self):
        """Save training configuration."""
        # Update config for fine-tuning.
        if self.finetuning:
            self.cfg["base_model_dir"] = self.base_model_dir

        # Save config.
        out_config_filename = os.path.join(self.model_dir,
                                           GQCNNFilenames.SAVED_CFG)
        tempOrderedDict = collections.OrderedDict()
        for key in self.cfg:
            tempOrderedDict[key] = self.cfg[key]
        with open(out_config_filename, "w") as outfile:
            json.dump(tempOrderedDict, outfile, indent=JSON_INDENT)

        # Save training script.
        this_filename = sys.argv[0]
        out_train_filename = os.path.join(self.model_dir, "training_script.py")
        shutil.copyfile(this_filename, out_train_filename)

        # Save architecture.
        out_architecture_filename = os.path.join(self.model_dir,
                                                 GQCNNFilenames.SAVED_ARCH)
        json.dump(self.cfg["gqcnn"]["architecture"],
                  open(out_architecture_filename, "w"),
                  indent=JSON_INDENT)

    def _read_training_params(self):
        """Read training parameters from configuration file."""
        # Splits.
        self.train_pct = self.cfg["train_pct"]
        self.total_pct = self.cfg["total_pct"]

        # Training sizes.
        self.train_batch_size = self.cfg["train_batch_size"]
        self.val_batch_size = self.cfg["val_batch_size"]
        self.max_files_eval = None
        if "max_files_eval" in self.cfg:
            self.max_files_eval = self.cfg["max_files_eval"]

        # Logging.
        self.num_epochs = self.cfg["num_epochs"]
        self.eval_frequency = self.cfg["eval_frequency"]
        self.save_frequency = self.cfg["save_frequency"]
        self.log_frequency = self.cfg["log_frequency"]

        # Optimization.
        self.train_l2_regularizer = self.cfg["train_l2_regularizer"]
        self.base_lr = self.cfg["base_lr"]
        self.decay_step_multiplier = self.cfg["decay_step_multiplier"]
        self.decay_rate = self.cfg["decay_rate"]
        self.momentum_rate = self.cfg["momentum_rate"]
        self.max_training_examples_per_load = self.cfg[
            "max_training_examples_per_load"]
        self.drop_rate = self.cfg["drop_rate"]
        self.max_global_grad_norm = self.cfg["max_global_grad_norm"]
        self.optimize_base_layers = False
        if "optimize_base_layers" in self.cfg:
            self.optimize_base_layers = self.cfg["optimize_base_layers"]

        # Metrics.
        self.target_metric_name = self.cfg["target_metric_name"]
        self.metric_thresh = self.cfg["metric_thresh"]
        self.training_mode = self.cfg["training_mode"]
        if self.training_mode != TrainingMode.CLASSIFICATION:
            raise ValueError(
                "Training mode '{}' not currently supported!".format(
                    self.training_mode))

        # Tensorboad.
        self._tensorboard_port = self.cfg["tensorboard_port"]

        # Preprocessing.
        self.preproc_log_frequency = self.cfg["preproc_log_frequency"]
        self.num_random_files = self.cfg["num_random_files"]
        self.max_prefetch_q_size = GeneralConstants.MAX_PREFETCH_Q_SIZE
        if "max_prefetch_q_size" in self.cfg:
            self.max_prefetch_q_size = self.cfg["max_prefetch_q_size"]
        self.num_prefetch_q_workers = GeneralConstants.NUM_PREFETCH_Q_WORKERS
        if "num_prefetch_q_workers" in self.cfg:
            self.num_prefetch_q_workers = self.cfg["num_prefetch_q_workers"]

        # Re-weighting positives/negatives.
        self.pos_weight = 0.0
        if "pos_weight" in self.cfg:
            self.pos_weight = self.cfg["pos_weight"]
            self.pos_accept_prob = 1.0
            self.neg_accept_prob = 1.0
            if self.pos_weight > 1:
                self.neg_accept_prob = 1 / self.pos_weight
            else:
                self.pos_accept_prob = self.pos_weight

        if self.train_pct < 0 or self.train_pct > 1:
            raise ValueError("Train percentage must be in range [0,1]")

        if self.total_pct < 0 or self.total_pct > 1:
            raise ValueError("Total percentage must be in range [0,1]")

        # Input normalization.
        self._norm_inputs = True
        if self.gqcnn.input_depth_mode == InputDepthMode.SUB:
            self._norm_inputs = False

        # Angular training.
        self._angular_bins = self.gqcnn.angular_bins
        self._max_angle = self.gqcnn.max_angle

        # During angular training, make sure symmetrization in denoising is
        # turned off and also set the angular bin width.
        if self._angular_bins > 0:
            symmetrization_msg = ("Symmetrization denoising must be turned off"
                                  " during angular training.")
            assert not self.cfg["symmetrize"], symmetrization_msg
            self._bin_width = self._max_angle / self._angular_bins

        # Debugging.
        self._debug = self.cfg["debug"]
        self._seed = self.cfg["seed"]
        if self._debug:
            if self.num_prefetch_q_workers > 1:
                self.logger.warning(
                    "Deterministic execution is not possible with "
                    "more than one prefetch queue worker even in debug mode.")
            # This reduces initialization time for fast debugging.
            self.num_random_files = self.cfg["debug_num_files"]

            np.random.seed(self._seed)
            random.seed(self._seed)

    def _setup_denoising_and_synthetic(self):
        """Setup denoising and synthetic data parameters."""
        # Multiplicative denoising.
        if self.cfg["multiplicative_denoising"]:
            self.gamma_shape = self.cfg["gamma_shape"]
            self.gamma_scale = 1 / self.gamma_shape

        # Gaussian process noise.
        if self.cfg["gaussian_process_denoising"]:
            self.gp_rescale_factor = self.cfg[
                "gaussian_process_scaling_factor"]
            self.gp_sample_height = int(self.im_height /
                                        self.gp_rescale_factor)
            self.gp_sample_width = int(self.im_width / self.gp_rescale_factor)
            self.gp_num_pix = self.gp_sample_height * self.gp_sample_width
            self.gp_sigma = self.cfg["gaussian_process_sigma"]

    def _open_dataset(self):
        """Open the dataset."""
        # Read in filenames of training data (poses, images, labels).
        self.dataset = TensorDataset.open(self.dataset_dir)
        self.num_datapoints = self.dataset.num_datapoints
        self.num_tensors = self.dataset.num_tensors
        self.datapoints_per_file = self.dataset.datapoints_per_file
        self.num_random_files = min(self.num_tensors, self.num_random_files)

        # Read split.
        if not self.dataset.has_split(self.split_name):
            self.logger.info(
                "Dataset split: {} not found. Creating new split...".format(
                    self.split_name))
            self.dataset.make_split(self.split_name, train_pct=self.train_pct)
        else:
            self.logger.info("Training split: {} found in dataset.".format(
                self.split_name))
        self._compute_split_indices()

    def _compute_data_params(self):
        """Compute parameters of the dataset."""
        # Image params.
        self.im_field_name = self.cfg["image_field_name"]
        self.im_height = self.dataset.config["fields"][
            self.im_field_name]["height"]
        self.im_width = self.dataset.config["fields"][
            self.im_field_name]["width"]
        self.im_channels = self.dataset.config["fields"][
            self.im_field_name]["channels"]
        # NOTE: There was originally some weird math going on here...
        self.im_center = np.array([self.im_height // 2, self.im_width // 2])

        # Poses.
        self.pose_field_name = self.cfg["pose_field_name"]
        self.gripper_mode = self.gqcnn.gripper_mode
        self.pose_dim = pose_dim(self.gripper_mode)
        self.raw_pose_shape = self.dataset.config["fields"][
            self.pose_field_name]["height"]

        # Outputs.
        self.label_field_name = self.target_metric_name
        self.num_categories = 2

        # Compute the number of train and val examples.
        self.num_train = 0
        self.num_val = 0
        for train_indices in self.train_index_map.values():
            self.num_train += train_indices.shape[0]
        for val_indices in self.train_index_map.values():
            self.num_val += val_indices.shape[0]

        # Set params based on the number of training examples (convert epochs
        # to steps).
        self.eval_frequency = int(
            np.ceil(self.eval_frequency *
                    (self.num_train / self.train_batch_size)))
        self.save_frequency = int(
            np.ceil(self.save_frequency *
                    (self.num_train / self.train_batch_size)))
        self.decay_step = self.decay_step_multiplier * self.num_train

    def _setup_tensorflow(self):
        """Setup Tensorflow placeholders, session, and queue."""

        # Set up training label and NumPy data types.
        if self.training_mode == TrainingMode.REGRESSION:
            train_label_dtype = tf.float32
            self.numpy_dtype = np.float32
        elif self.training_mode == TrainingMode.CLASSIFICATION:
            train_label_dtype = tf.int64
            self.numpy_dtype = np.int64
            if self.cfg["loss"] == "weighted_cross_entropy":
                train_label_dtype = tf.float32
                self.numpy_dtype = np.float32
        else:
            raise ValueError("Training mode '{}' not supported".format(
                self.training_mode))

        # Set up placeholders.
        self.train_labels_node = tf.placeholder(train_label_dtype,
                                                (self.train_batch_size, ))
        self.input_im_node = tf.placeholder(
            tf.float32, (self.train_batch_size, self.im_height, self.im_width,
                         self.im_channels))
        self.input_pose_node = tf.placeholder(
            tf.float32, (self.train_batch_size, self.pose_dim))
        if self._angular_bins > 0:
            self.train_pred_mask_node = tf.placeholder(
                tf.int32, (self.train_batch_size, self._angular_bins * 2))

        # Create data prefetch queue.
        self.prefetch_q = mp.Queue(self.max_prefetch_q_size)

        # Get weights.
        self.weights = self.gqcnn.weights

        # Open a TF session for the GQ-CNN instance and store it also as the
        # optimizer session.
        self.sess = self.gqcnn.open_session()

        # Setup data prefetch queue worker termination event.
        self.term_event = mp.Event()
        self.term_event.clear()

    def _setup_summaries(self):
        """Sets up placeholders for summary values and creates summary
        writer."""
        # Create placeholders for Python values because `tf.summary.scalar`
        # expects a placeholder.
        self.val_error_placeholder = tf.placeholder(tf.float32, [])
        self.minibatch_error_placeholder = tf.placeholder(tf.float32, [])
        self.minibatch_loss_placeholder = tf.placeholder(tf.float32, [])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Tag the `tf.summary.scalar`s so that we can group them together and
        # write different batches of summaries at different intervals.
        tf.summary.scalar("val_error",
                          self.val_error_placeholder,
                          collections=["eval_frequency"])
        tf.summary.scalar("minibatch_error",
                          self.minibatch_error_placeholder,
                          collections=["log_frequency"])
        tf.summary.scalar("minibatch_loss",
                          self.minibatch_loss_placeholder,
                          collections=["log_frequency"])
        tf.summary.scalar("learning_rate",
                          self.learning_rate_placeholder,
                          collections=["log_frequency"])
        self.merged_eval_summaries = tf.summary.merge_all("eval_frequency")
        self.merged_log_summaries = tf.summary.merge_all("log_frequency")

        # Create a TF summary writer with the specified summary directory.
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)

        # Initialize the variables again now that we have added some new ones.
        with self.sess.as_default():
            tf.global_variables_initializer().run()

    def _cleanup(self):
        self.logger.info("Cleaning and preparing to exit optimization...")

        # Set termination event for prefetch queue workers.
        self.logger.info("Terminating prefetch queue workers...")
        self.term_event.set()

        # Flush prefetch queue.
        # NOTE: This prevents a deadlock with the worker process queue buffers.
        self._flush_prefetch_queue()

        # Join prefetch queue worker processes.
        for p in self.prefetch_q_workers:
            p.join()

        # Close tensorboard if started.
        if self.tensorboard_has_launched:
            self._close_tensorboard()

        # Close Tensorflow session.
        self.gqcnn.close_session()

    def _flush_prefetch_queue(self):
        """Flush prefetch queue."""
        self.logger.info("Flushing prefetch queue...")
        for i in range(self.prefetch_q.qsize()):
            self.prefetch_q.get()

    def _setup(self):
        """Setup for training."""
        # Initialize data prefetch queue thread exit booleans.
        self.queue_thread_exited = False
        self.forceful_exit = False

        # Setup output directories.
        self._setup_output_dirs()

        # Save training configuration.
        self._save_configs()

        # Read training parameters from config file.
        self._read_training_params()

        # Setup image and pose data files.
        self._open_dataset()

        # Compute data parameters.
        self._compute_data_params()

        # Setup denoising and synthetic data parameters.
        self._setup_denoising_and_synthetic()

        # Compute means, std's, and normalization metrics.
        self._compute_data_metrics()

        # Setup Tensorflow session/placeholders/queue.
        self._setup_tensorflow()

        # Setup summaries for visualizing metrics in Tensorboard.
        self._setup_summaries()

    def _load_and_enqueue(self, seed):
        """Loads and enqueues a batch of images for training."""

        # When the parent process receives a SIGINT, it will itself handle
        # cleaning up child processes.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Set the random seed explicitly to prevent all workers from possible
        # inheriting the same seed on process initialization.
        np.random.seed(seed)
        random.seed(seed)

        # Open dataset.
        dataset = TensorDataset.open(self.dataset_dir)

        while not self.term_event.is_set():
            # Loop through data.
            num_queued = 0
            start_i = 0
            end_i = 0
            file_num = 0
            queue_start = time.time()

            # Init buffers.
            train_images = np.zeros([
                self.train_batch_size, self.im_height, self.im_width,
                self.im_channels
            ]).astype(np.float32)
            train_poses = np.zeros([self.train_batch_size,
                                    self.pose_dim]).astype(np.float32)
            train_labels = np.zeros(self.train_batch_size).astype(
                self.numpy_dtype)
            if self._angular_bins > 0:
                train_pred_mask = np.zeros(
                    (self.train_batch_size, self._angular_bins * 2),
                    dtype=bool)

            while start_i < self.train_batch_size:
                # Compute num remaining.
                num_remaining = self.train_batch_size - num_queued

                # Generate tensor index uniformly at random.
                file_num = np.random.choice(self.num_tensors, size=1)[0]

                read_start = time.time()
                train_images_tensor = dataset.tensor(self.im_field_name,
                                                     file_num)
                train_poses_tensor = dataset.tensor(self.pose_field_name,
                                                    file_num)
                train_labels_tensor = dataset.tensor(self.label_field_name,
                                                     file_num)
                read_stop = time.time()
                self.logger.debug("Reading data took {} sec".format(
                    str(round(read_stop - read_start, 3))))
                self.logger.debug("File num: {}".format(file_num))

                # Get batch indices uniformly at random.
                train_ind = self.train_index_map[file_num]
                np.random.shuffle(train_ind)
                if self.gripper_mode == GripperMode.LEGACY_SUCTION:
                    tp_tmp = read_pose_data(train_poses_tensor.data,
                                            self.gripper_mode)
                    train_ind = train_ind[np.isfinite(tp_tmp[train_ind, 1])]

                # Filter positives and negatives.
                if self.training_mode == TrainingMode.CLASSIFICATION and \
                        self.pos_weight != 0.0:
                    labels = 1 * (train_labels_tensor.arr > self.metric_thresh)
                    np.random.shuffle(train_ind)
                    filtered_ind = []
                    for index in train_ind:
                        if labels[index] == 0 and np.random.rand(
                        ) < self.neg_accept_prob:
                            filtered_ind.append(index)
                        elif labels[index] == 1 and np.random.rand(
                        ) < self.pos_accept_prob:
                            filtered_ind.append(index)
                    train_ind = np.array(filtered_ind)

                # Samples train indices.
                upper = min(num_remaining, train_ind.shape[0],
                            self.max_training_examples_per_load)
                ind = train_ind[:upper]
                num_loaded = ind.shape[0]
                if num_loaded == 0:
                    self.logger.warning("Queueing zero examples!!!!")
                    continue

                # Subsample data.
                train_images_arr = train_images_tensor.arr[ind, ...]
                train_poses_arr = train_poses_tensor.arr[ind, ...]
                angles = train_poses_arr[:, 3]
                train_label_arr = train_labels_tensor.arr[ind]
                num_images = train_images_arr.shape[0]

                # Resize images.
                rescale_factor = self.im_height / train_images_arr.shape[1]
                if rescale_factor != 1.0:
                    resized_train_images_arr = np.zeros([
                        num_images, self.im_height, self.im_width,
                        self.im_channels
                    ]).astype(np.float32)
                    for i in range(num_images):
                        for c in range(train_images_arr.shape[3]):
                            resized_train_images_arr[i, :, :, c] = imresize(
                                train_images_arr[i, :, :, c],
                                rescale_factor,
                                interp="bicubic")
                    train_images_arr = resized_train_images_arr

                # Add noises to images.
                train_images_arr, train_poses_arr = self._distort(
                    train_images_arr, train_poses_arr)

                # Slice poses.
                train_poses_arr = read_pose_data(train_poses_arr,
                                                 self.gripper_mode)

                # Standardize inputs and outputs.
                if self._norm_inputs:
                    train_images_arr = (train_images_arr -
                                        self.im_mean) / self.im_std
                    if self.gqcnn.input_depth_mode == \
                            InputDepthMode.POSE_STREAM:
                        train_poses_arr = (train_poses_arr -
                                           self.pose_mean) / self.pose_std
                train_label_arr = 1 * (train_label_arr > self.metric_thresh)
                train_label_arr = train_label_arr.astype(self.numpy_dtype)

                if self._angular_bins > 0:
                    bins = np.zeros_like(train_label_arr)
                    # Form prediction mask to use when calculating loss.
                    neg_ind = np.where(angles < 0)
                    angles = np.abs(angles) % self._max_angle
                    angles[neg_ind] *= -1
                    g_90 = np.where(angles > (self._max_angle / 2))
                    l_neg_90 = np.where(angles < (-1 * (self._max_angle / 2)))
                    angles[g_90] -= self._max_angle
                    angles[l_neg_90] += self._max_angle
                    # TODO(vsatish): Actually fix this.
                    angles *= -1  # Hack to fix reverse angle convention.
                    angles += (self._max_angle / 2)
                    train_pred_mask_arr = np.zeros(
                        (train_label_arr.shape[0], self._angular_bins * 2))
                    for i in range(angles.shape[0]):
                        bins[i] = angles[i] // self._bin_width
                        train_pred_mask_arr[i,
                                            int((angles[i] //
                                                 self._bin_width) * 2)] = 1
                        train_pred_mask_arr[
                            i, int((angles[i] // self._bin_width) * 2 + 1)] = 1

                # Compute the number of examples loaded.
                num_loaded = train_images_arr.shape[0]
                end_i = start_i + num_loaded

                # Enqueue training data batch.
                train_images[start_i:end_i, ...] = train_images_arr.copy()
                train_poses[start_i:end_i, :] = train_poses_arr.copy()
                train_labels[start_i:end_i] = train_label_arr.copy()
                if self._angular_bins > 0:
                    train_pred_mask[start_i:end_i] = train_pred_mask_arr.copy()

                # Update start index.
                start_i = end_i
                num_queued += num_loaded

            # Send data to queue.
            if not self.term_event.is_set():
                try:
                    if self._angular_bins > 0:
                        self.prefetch_q.put_nowait(
                            (train_images, train_poses, train_labels,
                             train_pred_mask))
                    else:
                        self.prefetch_q.put_nowait(
                            (train_images, train_poses, train_labels))
                except Queue.Full:
                    time.sleep(GeneralConstants.QUEUE_SLEEP)
                queue_stop = time.time()
                self.logger.debug("Queue batch took {} sec".format(
                    str(round(queue_stop - queue_start, 3))))

    def _distort(self, image_arr, pose_arr):
        """Adds noise to a batch of images."""
        # Read params.
        num_images = image_arr.shape[0]

        # Denoising and synthetic data generation.
        if self.cfg["multiplicative_denoising"]:
            mult_samples = ss.gamma.rvs(self.gamma_shape,
                                        scale=self.gamma_scale,
                                        size=num_images)
            mult_samples = mult_samples[:, np.newaxis, np.newaxis, np.newaxis]
            image_arr = image_arr * np.tile(
                mult_samples,
                [1, self.im_height, self.im_width, self.im_channels])

        # Add correlated Gaussian noise.
        if self.cfg["gaussian_process_denoising"]:
            for i in range(num_images):
                if np.random.rand() < self.cfg["gaussian_process_rate"]:
                    train_image = image_arr[i, :, :, 0]
                    gp_noise = ss.norm.rvs(scale=self.gp_sigma,
                                           size=self.gp_num_pix).reshape(
                                               self.gp_sample_height,
                                               self.gp_sample_width)
                    gp_noise = imresize(gp_noise,
                                        self.gp_rescale_factor,
                                        interp="bicubic")
                    train_image[train_image > 0] += gp_noise[train_image > 0]
                    image_arr[i, :, :, 0] = train_image

        # Symmetrize images.
        if self.cfg["symmetrize"]:
            for i in range(num_images):
                train_image = image_arr[i, :, :, 0]
                # Rotate with 50% probability.
                if np.random.rand() < 0.5:
                    theta = 180.0
                    rot_map = cv2.getRotationMatrix2D(tuple(self.im_center),
                                                      theta, 1)
                    train_image = cv2.warpAffine(
                        train_image,
                        rot_map, (self.im_height, self.im_width),
                        flags=cv2.INTER_NEAREST)

                    if self.gripper_mode == GripperMode.LEGACY_SUCTION:
                        pose_arr[:, 3] = -pose_arr[:, 3]
                    elif self.gripper_mode == GripperMode.SUCTION:
                        pose_arr[:, 4] = -pose_arr[:, 4]
                # Reflect left right with 50% probability.
                if np.random.rand() < 0.5:
                    train_image = np.fliplr(train_image)
                # Reflect up down with 50% probability.
                if np.random.rand() < 0.5:
                    train_image = np.flipud(train_image)

                    if self.gripper_mode == GripperMode.LEGACY_SUCTION:
                        pose_arr[:, 3] = -pose_arr[:, 3]
                    elif self.gripper_mode == GripperMode.SUCTION:
                        pose_arr[:, 4] = -pose_arr[:, 4]
                image_arr[i, :, :, 0] = train_image
        return image_arr, pose_arr

    def _error_rate_in_batches(self, num_files_eval=None, validation_set=True):
        """Compute error and loss over either training or validation set.

        Returns
        -------
        :obj:"autolab_core.BinaryClassificationResult`
            validation error
        """
        all_predictions = []
        all_labels = []

        # Subsample files.
        file_indices = np.arange(self.num_tensors)
        if num_files_eval is None:
            num_files_eval = self.max_files_eval
        np.random.shuffle(file_indices)
        if self.max_files_eval is not None and num_files_eval > 0:
            file_indices = file_indices[:num_files_eval]

        for i in file_indices:
            # Load next file.
            images = self.dataset.tensor(self.im_field_name, i).arr
            poses = self.dataset.tensor(self.pose_field_name, i).arr
            raw_poses = np.array(poses, copy=True)
            labels = self.dataset.tensor(self.label_field_name, i).arr

            # If no datapoints from this file are in validation then just
            # continue.
            if validation_set:
                indices = self.val_index_map[i]
            else:
                indices = self.train_index_map[i]
            if len(indices) == 0:
                continue

            images = images[indices, ...]
            poses = read_pose_data(poses[indices, :], self.gripper_mode)
            raw_poses = raw_poses[indices, :]
            labels = labels[indices]

            if self.training_mode == TrainingMode.CLASSIFICATION:
                labels = 1 * (labels > self.metric_thresh)
                labels = labels.astype(np.uint8)

            if self._angular_bins > 0:
                # Form mask to extract predictions from ground-truth angular
                # bins.
                angles = raw_poses[:, 3]
                neg_ind = np.where(angles < 0)
                angles = np.abs(angles) % self._max_angle
                angles[neg_ind] *= -1
                g_90 = np.where(angles > (self._max_angle / 2))
                l_neg_90 = np.where(angles < (-1 * (self._max_angle / 2)))
                angles[g_90] -= self._max_angle
                angles[l_neg_90] += self._max_angle
                # TODO(vsatish): Actually fix this.
                angles *= -1  # Hack to fix reverse angle convention.
                angles += (self._max_angle / 2)
                pred_mask = np.zeros((labels.shape[0], self._angular_bins * 2),
                                     dtype=bool)
                for i in range(angles.shape[0]):
                    pred_mask[i, int(
                        (angles[i] // self._bin_width) * 2)] = True
                    pred_mask[i,
                              int((angles[i] // self._bin_width) * 2 +
                                  1)] = True

            # Get predictions.
            predictions = self.gqcnn.predict(images, poses)

            if self._angular_bins > 0:
                predictions = predictions[pred_mask].reshape((-1, 2))

            # Update.
            all_predictions.extend(predictions[:, 1].tolist())
            all_labels.extend(labels.tolist())

        # Get learning result.
        result = None
        if self.training_mode == TrainingMode.CLASSIFICATION:
            result = BinaryClassificationResult(all_predictions, all_labels)
        else:
            result = RegressionResult(all_predictions, all_labels)
        return result
