"""
Class for training a GQCNN using Tensorflow backend.
Author: Vishal Satish
"""
import logging
import cPickle as pkl
import os
import random
import signal
import time
import threading

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from gqcnn.utils.learning_analysis import ClassificationResult, RegressionResult
from gqcnn.utils.train_stats_logger import TrainStatsLogger
from gqcnn.utils.data_utils import parse_pose_data, parse_gripper_data, \
    compute_data_metrics, compute_grasp_label_metrics, denoise
from gqcnn.utils.training_utils import copy_config, compute_indices_image_wise, \
    compute_indices_pose_wise, compute_indices_object_wise, get_decay_step, \
    setup_data_filenames, setup_output_dirs
from gqcnn.utils.enums import GeneralConstants, DataSplitMode, InputPoseMode, InputGripperMode, TrainingMode

class GQCNNTrainerTF(object):
    """ Trains GQCNN with Tensorflow backend """

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
        """ Creates loss

        Returns
        -------
        :obj:`tensorflow Tensor`
            loss
        """
        # TODO: Add Poisson Loss
        if self.cfg['loss'] == 'l2':
            return tf.nn.l2_loss(tf.sub(self.train_net_output, self.train_labels_node))
        elif self.cfg['loss'] == 'cross_entropy':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, 
                labels=self.train_labels_node, logits=self.train_net_output, name=None))

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
            raise ValueError('Optimizer %s not supported' % (self.cfg['optimizer']))

    def _check_dead_queue(self):
        """ Checks to see if the queue is dead and if so closes the 
        tensorflow session and cleans up the variables """
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
        logging.info(
            "Launching Tensorboard, Please navigate to localhost:6006 in your favorite web browser to view summaries")
        os.system('tensorboard --logdir=' + self.summary_dir + " &>/dev/null &")

    def _close_tensorboard(self):
        """ Closes Tensorboard """
        logging.info('Closing Tensorboard')
        tensorboard_pid = os.popen('pgrep tensorboard').read()
        os.system('kill ' + tensorboard_pid)

    def train(self):
        """ Perform Optimization  """
        with self.gqcnn.get_tf_graph().as_default():
            self._train()

    def _train(self):
        start_time = time.time()

        # run setup
        self._setup()

        # build network
        if self.training_mode == TrainingMode.CLASSIFICATION:
            if self.gripper_dim > 0:
                self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node, self.input_gripper_node)
            else:
                self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node)
            self.train_net_output = self.gqcnn.output
            self.gqcnn.add_softmax_to_output()
        elif self.training_mode == TrainingMode.REGRESSION:
            if self.gripper_dim > 0:
                self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node, self.input_gripper_node)
            else:
                self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node, self.input_gripper_node)
            self.train_net_output = self.gqcnn.output
        train_predictions = self.gqcnn.output
        drop_rate = self.gqcnn.input_drop_rate_node

        # once weights have been initialized create tf Saver for weights
        self.saver = tf.train.Saver()

        ############# FORM LOSS #############
        # part 1: error
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
            optimizer, true_optimizer = self._create_optimizer(
                loss, batch, var_list, learning_rate)

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
                            tf.summary.histogram("gradients/%s" %
                                                 var_name, grad_values, collections=['histogram'])

                self.merged_histogram_summaries = tf.summary.merge_all('histogram')
                self._setup_summaries()

        def handler(signum, frame):
            logging.info('caught CTRL+C, exiting...')
            self.term_event.set()

            ### Forcefully Exit ####
            # TODO: remove this and figure out why queue thread does not properly exit
            logging.info('Forcefully Exiting Optimization')
            self.forceful_exit = True

            # forcefully kill the session to terminate any current graph 
            # ops that are stalling because the enqueue op has ended
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

            # initialize all tf global variables
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

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # check_numeric_op = tf.add_check_numerics_ops()

                # fprop + bprop
                _, l, lr, predictions, batch_labels, net_output = self.sess.run([optimizer, loss, learning_rate,
                    train_predictions, self.train_labels_node, self.train_net_output], 
                    feed_dict={drop_rate: self.drop_rate}, options=GeneralConstants.timeout_option)
                
                ex = np.exp(net_output - np.tile(np.max(net_output, axis=1)[:, np.newaxis], [1, 2]))
                softmax = ex / np.tile(np.sum(ex, axis=1)[:, np.newaxis], [1, 2])

                logging.debug('Max: ' + str(np.max(softmax[:, 1])))
                logging.debug('Min: ' + str(np.min(softmax[:, 1])))
                logging.debug('Pred nonzero: ' + str(np.sum(np.argmax(predictions, axis=1))))
                logging.debug('True nonzero: ' + str(np.sum(batch_labels)))

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
                        logging.info('Minibatch error: %.3f%%' % train_error)
                    self.summary_writer.add_summary(self.sess.run(self.merged_log_summaries, feed_dict={
                        self.minibatch_error_placeholder: train_error, self.minibatch_loss_placeholder: l, self.learning_rate_placeholder: lr}), step)

                    # update the TrainStatsLogger
                    self.train_stats_logger.update(train_eval_iter=step, train_loss=l, train_error=train_error, 
                        total_train_error=None, val_eval_iter=None, val_error=None, learning_rate=lr)

                # evaluate validation error
                if step % self.eval_frequency == 0:
                    if self.cfg['eval_total_train_error']:
                        train_error = self._error_rate_in_batches()
                        logging.info('Training error: %.3f' % train_error)

                        # update the TrainStatsLogger and save
                        self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, 
                            total_train_error=train_error, val_eval_iter=None, val_error=None, learning_rate=None)
                        self.train_stats_logger.log()

                    val_error = self._error_rate_in_batches()
                    self.summary_writer.add_summary(self.sess.run(self.merged_eval_summaries, feed_dict={
                                                    self.val_error_placeholder: val_error}), step)
                    logging.info('Validation error: %.3f' % val_error)

                    # update the TrainStatsLogger
                    self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, 
                        total_train_error=None, val_eval_iter=step, val_error=val_error, learning_rate=None)

                    # save everything
                    self.train_stats_logger.log()

                # save the model
                if step % self.save_frequency == 0 and step > 0:
                    self.saver.save(self.sess, os.path.join(
                        self.experiment_dir, 'model_%05d.ckpt' % (step)))
                    self.saver.save(self.sess, os.path.join(
                        self.experiment_dir, 'model.ckpt'))

                # launch tensorboard only after the first iteration
                if not self.tensorboard_has_launched:
                    self.tensorboard_has_launched = True
                    self._launch_tensorboard()

                # write histogram summaries if specified
                if self.save_histograms:
                    self.summary_writer.add_summary(self.sess.run([self.merged_histogram_summaries], 
                        feed_dict={self.gqcnn.input_im_node: train_images, self.gqcnn.input_pose_node: pose_node}))
                    self.summary_writer.add_summary(merged_histogram_summaries)

            # get final logs
            val_error = self._error_rate_in_batches()
            logging.info('Final validation error: %.1f%%' % val_error)

            # update the TrainStatsLogger
            self.train_stats_logger.update(train_eval_iter=None, train_loss=None, train_error=None, 
                total_train_error=None, val_eval_iter=step, val_error=val_error, learning_rate=None)

            # log & save everything
            self.train_stats_logger.log()
            self.saver.save(self.sess, self.exp_path_gen('model.ckpt'))

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

    def _save_index_maps(self, train_idx_map, val_idx_map, train_fname, val_fname):
        with open(self.exp_path_gen(train_fname), 'w') as fhandle:
            pkl.dump(train_idx_map, fhandle)
        with open(self.exp_path_gen(val_fname), 'w') as fhandle:
            pkl.dump(val_idx_map, fhandle)

    def _compute_indices(self, data_split_mode, *computation_args):
        train_idx_map_fname = 'train_indices_{}.pkl'.format(data_split_mode)
        val_idx_map_fname = 'val_indices_{}.pkl'.format(data_split_mode)
        train_idx_map_fpath = self.exp_path_gen(train_idx_map_fname)
        val_idx_map_fpath = self.exp_path_gen(val_idx_map_fname)
        if os.path.exists(train_idx_map_fpath):
            with open(train_idx_map_fpath, 'r') as fhandle:
                train_idx_map = pkl.load(fhandle)
            with open(val_idx_map_fname, 'r') as fhandle:
                val_idx_map = pkl.load(fhandle)
        elif self.cfg['use_existing_indices']:
            with open(os.path.join(self.cfg['index_dir'], train_idx_map_fname)) as fhandle:
                train_idx_map = pkl.load(fhandle)
            with open(os.path.join(self.cfg['index_dir'], val_idx_map_fname)) as fhandle:
                val_idx_map = pkl.load(fhandle)
        else:
            if data_split_mode == DataSplitMode.IMAGE_WISE:
                train_idx_map, val_idx_map = compute_indices_image_wise(*computation_args)
            elif data_split_mode == DataSplitMode.OBJECT_WISE:
                train_idx_map, val_idx_map = compute_indices_object_wise(*computation_args)
            else:
                train_idx_map, val_idx_map = compute_indices_pose_wise(*computation_args)

        # save indices
        self._save_index_maps(train_idx_map, val_idx_map, train_idx_map_fname, val_idx_map_fname)

        return train_idx_map, val_idx_map

    def _compute_data_metrics(self):
        if self.cfg['fine_tune']:
            self.im_mean = self.gqcnn.get_im_mean()
            self.im_std = self.gqcnn.get_im_std()
            self.pose_mean = self.gqcnn.get_pose_mean()
            self.pose_std = self.gqcnn.get_pose_std()
            if self.gripper_dim > 0:
                self.gripper_mean = self.gqcnn.get_gripper_mean()
                self.gripper_std = self.gqcnn.get_gripper_std()
            elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                self.gripper_depth_mask_mean = self.gqcnn.get_gripper_depth_mask_mean()
                self.gripper_depth_mask_std = self.gqcnn.get_gripper_depth_mask_std()
        else:
            im_mean_fname = self.exp_path_gen('im_mean.npy')
            im_std_fname = self.exp_path_gen('im_std.npy')
            pose_mean_fname = self.exp_path_gen('pose_mean.npy')
            pose_std_fname = self.exp_path_gen('pose_std.npy')
            if self.gripper_dim > 0:
                gripper_mean_fname = self.exp_path_gen('gripper_mean.npy')
                gripper_std_fname = self.exp_path_gen('gripper_std.npy')
                self.image_mean, self.image_std, self.pose_mean, self.pose_std, self.gripper_mean, self.gripper_std, _, _ = compute_data_metrics(
                    self.experiment_dir, self.data_dir, self.im_height, self.im_width, self.pose_shape, self.input_pose_mode, self.train_index_map, 
                    self.im_filenames, self.pose_filenames, gripper_param_filenames=self.gripper_param_filenames, total_gripper_param_elems=self.gripper_shape, 
                    num_random_files=self.num_random_files)

                np.save(gripper_mean_fname, self.gripper_mean)
                np.save(gripper_std_fname, self.gripper_std)

                self.gqcnn.update_gripper_mean(parse_gripper_data(self.gripper_mean, self.input_gripper_mode))
                self.gqcnn.update_gripper_std(parse_gripper_data(self.gripper_std, self.input_gripper_mode))

            elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                gripper_depth_mask_mean_fname = self.exp_path_gen('gripper_depth_mask_mean.npy')
                gripper_depth_mask_std_fname = self.exp_path_gen('gripper_depth_mask_std.npy')
                self.image_mean, self.image_std, self.pose_mean, self.pose_std, _, _, self.gripper_depth_mask_mean, self.gripper_depth_mask_std = compute_data_metrics(
                    self.experiment_dir, self.data_dir, self.im_height, self.im_width, self.pose_shape, self.input_pose_mode, self.train_index_map, 
                    self.im_filenames, self.pose_filenames, gripper_depth_mask_filenames=self.gripper_depth_mask_filenames, num_random_files=self.num_random_files)

                np.save(gripper_depth_mask_mean_fname, self.gripper_depth_mask_mean)
                np.save(gripper_depth_mask_std_fname, self.gripper_depth_mask_std)

                self.gqcnn.update_gripper_depth_mask_mean(self.gripper_depth_mask_mean)
                self.gqcnn.update_gripper_depth_mask_std(self.gripper_depth_mask_std)

            else:
                self.im_mean, self.im_std, self.pose_mean, self.pose_std, _, _, _, _ = compute_data_metrics(
                    self.experiment_dir, self.data_dir, self.im_height, self.im_width, self.pose_shape, self.input_pose_mode, self.train_index_map, 
                    self.im_filenames, self.pose_filenames, num_random_files=self.num_random_files)

            np.save(im_mean_fname, self.im_mean)
            np.save(im_std_fname, self.im_std)
            np.save(pose_mean_fname, self.pose_mean)
            np.save(pose_std_fname, self.pose_std)

            self.gqcnn.update_im_mean(self.im_mean)
            self.gqcnn.update_im_std(self.im_std)
            self.gqcnn.update_pose_mean(parse_pose_data(self.pose_mean, self.input_pose_mode))
            self.gqcnn.update_pose_std(parse_pose_data(self.pose_std, self.input_pose_mode))

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
        tf.summary.scalar('val_error', self.val_error_placeholder,
                          collections=["eval_frequency"])
        tf.summary.scalar(
            'minibatch_error', self.minibatch_error_placeholder, collections=["log_frequency"])
        tf.summary.scalar(
            'minibatch_loss', self.minibatch_loss_placeholder, collections=["log_frequency"])
        tf.summary.scalar(
            'learning_rate', self.learning_rate_placeholder, collections=["log_frequency"])
        self.merged_eval_summaries = tf.summary.merge_all("eval_frequency")
        self.merged_log_summaries = tf.summary.merge_all("log_frequency")

        # create a tf summary writer with the specified summary directory
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        self.gqcnn.set_summary_writer(self.summary_writer)

        # initialize the variables again now that we have added some new ones
        with self.sess.as_default():
            tf.global_variables_initializer().run()

    def _setup_data_pipeline(self):
        """Setup Tensorflow data pipeline for reading in data from dataset and forwarding it to model for training"""

        # setup nodes
        with tf.name_scope('train_data_node'):
            self.train_data_batch = tf.placeholder(
                tf.float32, (self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels))
        with tf.name_scope('train_pose_node'):
            self.train_poses_batch = tf.placeholder(
                tf.float32, (self.train_batch_size, self.pose_dim))
        if self.training_mode == TrainingMode.REGRESSION:
            train_label_dtype = tf.float32
            self.numpy_dtype = np.float32
        elif self.training_mode == TrainingMode.CLASSIFICATION:
            train_label_dtype = tf.int64
            self.numpy_dtype = np.int64
        else:
            raise ValueError('Training mode %s not supported' % (self.training_mode))
        with tf.name_scope('train_labels_node'):
            self.train_labels_batch = tf.placeholder(
                train_label_dtype, (self.train_batch_size,))
        if self.gripper_dim > 0:
            with tf.name_scope('train_gripper_node'):
                self.train_gripper_batch = tf.placeholder(
                    tf.float32, (self.train_batch_size, self.gripper_dim))

        # create data queue to fetch data from dataset in batches
        with tf.name_scope('data_queue'):
            if self.gripper_dim > 0:
                self.q = tf.FIFOQueue(self.queue_capacity, [tf.float32, tf.float32, tf.float32, train_label_dtype], shapes=[(self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels),
                 (self.train_batch_size, self.pose_dim), (self.train_batch_size, self.gripper_dim), (self.train_batch_size,)])
                self.enqueue_op = self.q.enqueue(
                    [self.train_data_batch, self.train_poses_batch, self.train_gripper_batch, self.train_labels_batch])
                self.train_labels_node = tf.placeholder(
                    train_label_dtype, (self.train_batch_size,))
                self.input_im_node, self.input_pose_node, self.input_gripper_node, self.train_labels_node = self.q.dequeue()
            else:
                self.q = tf.FIFOQueue(self.queue_capacity, [tf.float32, tf.float32, train_label_dtype], shapes=[(self.train_batch_size, self.im_height, self.im_width, self.num_tensor_channels),
                 (self.train_batch_size, self.pose_dim), (self.train_batch_size,)])
                self.enqueue_op = self.q.enqueue(
                    [self.train_data_batch, self.train_poses_batch, self.train_labels_batch])
                self.train_labels_node = tf.placeholder(
                    train_label_dtype, (self.train_batch_size,))
                self.input_im_node, self.input_pose_node, self.train_labels_node = self.q.dequeue()

        # setup weights using gqcnn
        if self.cfg['fine_tune']:
            # this assumes that a gqcnn was passed in that was initialized with weights from a model using GQCNN.load(), so all that has to
            # be done is to possibly reinitialize fc3/fc4/fc5
            self.gqcnn.reinitialize_layers(
                self.cfg['reinit_fc3'], self.cfg['reinit_fc4'], self.cfg['reinit_fc5'])

        # get weights
        self.weights = self.gqcnn.get_weights()

        # open a tf session for the gqcnn object and store it also as the optimizer session
        self.sess = self.gqcnn.open_session()

        # setup term event/dead event
        self.term_event = threading.Event()
        self.term_event.clear()
        self.dead_event = threading.Event()
        self.dead_event.clear()
    
    def _read_training_params(self):
        """ Read training parameters from configuration file """

        self.data_dir = self.cfg['dataset_dir']
        self.image_mode = self.cfg['image_mode']
        self.data_split_mode = self.cfg['data_split_mode']
        self.train_pct = self.cfg['train_pct']
        self.total_pct = self.cfg['total_pct']

        self.train_batch_size = self.cfg['train_batch_size']
        self.val_batch_size = self.cfg['val_batch_size']
        # update the GQCNN's batch_size param to val_batch_size
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
        self.drop_rate = self.cfg['drop_rate']

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
        self.metric_data = np.load(os.path.join(self.data_dir, self.label_filenames[0]))['arr_0']
        self.images_per_file = self.train_im_data.shape[0]
        self.im_height = self.train_im_data.shape[1]
        self.im_width = self.train_im_data.shape[2]
        self.im_channels = self.train_im_data.shape[3]

        self.num_tensor_channels = self.cfg['num_tensor_channels']
        self.pose_shape = self.pose_data.shape[1]
        self.input_pose_mode = self.cfg['input_pose_mode']
        self.input_gripper_mode = self.cfg['input_gripper_mode']

        # update pose dimension according to input_pose_mode for creation of tensorflow placeholders
        if self.input_pose_mode == InputPoseMode.TF_IMAGE:
            self.pose_dim = 1  # depth
        elif self.input_pose_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
            self.pose_dim = 3  # depth, cx, cy
        elif self.input_pose_mode == InputPoseMode.RAW_IMAGE:
            self.pose_dim = 4  # u, v, theta, depth
        elif self.input_pose_mode == InputPoseMode.RAW_IMAGE_PERSPECTIVE:
            self.pose_dim = 6  # u, v, theta, depth cx, cy
        elif self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
            self.pose_dim = 2  # depth, theta
        else:
            raise ValueError('Input pose mode %s not understood' %
                             (self.input_pose_mode))

        # update gripper dimension according to input_gripper_mode for creation of tensorflow placeholders
        if self.input_gripper_mode == InputGripperMode.WIDTH:
            self.gripper_dim = 1  # width
        elif self.input_gripper_mode == InputGripperMode.NONE:
            self.gripper_dim = 0  # no gripper channel
        elif self.input_gripper_mode == InputGripperMode.ALL:
            self.gripper_dim = 4  # width, palm depth, fx, fy
        elif self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
            self.gripper_dim = 0  # no gripper channel
            self.num_tensor_channels += 2  # masks will be added as channels to depth image
        else:
            raise ValueError('Input gripper mode %s not understood' %
                             (self.input_gripper_mode))
        
        if self.gripper_dim > 0:
            self.gripper_data = np.load(os.path.join(self.data_dir, self.gripper_param_filenames[0]))['arr_0']
            self.gripper_shape = self.gripper_data.shape[1]        

        self.num_files = len(self.im_filenames)
        self.num_random_files = min(self.num_files, self.cfg['num_random_files'])
        self.num_categories = 2

        self.denoising_params = self.cfg['denoise']

    def _setup(self):
        """ Setup for training """

        # get debug flag and number of files to use when debugging
        self.debug = self.cfg['debug']
        self.debug_num_files = self.cfg['debug_num_files']
        
        # initialize thread exit booleans
        self.queue_thread_exited = False
        self.forceful_exit = False

        # set random seed for deterministic execution if in debug mode
        if self.debug:
            np.random.seed(GeneralConstants.SEED)
            random.seed(GeneralConstants.SEED)

        # setup output directories
        output_dir = self.cfg['output_dir']
        self.experiment_dir, self.summary_dir, self.filter_dir = setup_output_dirs(output_dir)

        # create python lambda function to help create file paths to experiment_dir
        self.exp_path_gen = lambda fname: os.path.join(self.experiment_dir, fname)

        # copy config file
        copy_config(self.experiment_dir, self.cfg)

        # read training parameters from config file
        self._read_training_params()

        # read dataset filenames
        self.im_filenames, self.pose_filenames, self.label_filenames, self.gripper_param_filenames, \
        self.gripper_depth_mask_filenames, self.gripper_seg_mask_filenames,self.im_filenames_copy, \
        self.pose_filenames_copy, self.label_filenames_copy, self.gripper_param_filenames_copy, \
        self.gripper_depth_mask_filenames_copy, self.gripper_seg_mask_filenames_copy, self.obj_id_filenames, \
        self.stable_pose_filenames, self.num_files = setup_data_filenames(self.data_dir, self.image_mode, self.target_metric_name, self.total_pct, self.debug, self.debug_num_files)

        # read data parameters from config file
        self._read_data_params()

        # compute total number of datapoints in dataset(rounded up to num_datapoints_per_file)
        self.num_datapoints = self.images_per_file * self.num_files

        # compute the number of training datapoints to use in training loop counter
        self.num_train = self.train_pct * self.num_datapoints

        steps_per_epoch = self.num_datapoints * self.train_pct / self.train_batch_size
        # if self.eval_frequency == -1, change it to reflect a single epoch
        if self.eval_frequency == -1:
            self.eval_frequency = steps_per_epoch

        # if self.save_frequency == -1, change it to reflect a single epoch
        if self.save_frequency == -1:
            self.save_frequency == steps_per_epoch

        # compute train/test indices based on how the data is to be split
        if self.data_split_mode == DataSplitMode.IMAGE_WISE:
            self.train_index_map, self.val_index_map = self._compute_indices(DataSplitMode.IMAGE_WISE, self.data_dir, self.images_per_file, self.num_datapoints, self.train_pct, self.im_filenames)
        elif self.data_split_mode == DataSplitMode.OBJECT_WISE:
            self.train_index_map, self.val_index_map = self._compute_indices(DataSplitMode.OBJECT_WISE, self.data_dir, self.train_pct, self.im_filenames, self.obj_id_filenames)
        elif self.data_split_mode == DataSplitMode.STABLE_POSE_WISE:
            self.train_index_map, self.val_index_map = self._compute_indices(DataSplitMode.STABLE_POSE_WISE, self.data_dir, self.train_pct, self.im_filenames, self.stable_pose_filenames)
        else:
            raise ValueError('Data split mode: {} not supported'.format(self.data_split_mode))

        # calculate learning rate decay step
        self.decay_step = get_decay_step(self.train_pct, self.images_per_file, self.decay_step_multiplier)

        # compute data metrics
        self._compute_data_metrics()

        # compute grasp label metrics
        self.min_grasp_metric, self.max_grasp_metric, self.mean_grasp_metric, self.median_grasp_metric, self.pct_pos_val = compute_grasp_label_metrics(
            self.data_dir, self.im_filenames, self.label_filenames, self.val_index_map, self.metric_thresh)
        logging.info('Percent positive in val set: ' + str(self.pct_pos_val))

        # setup tensorflow data pipeline
        self._setup_data_pipeline()

        # setup summaries for visualizing metrics in tensorboard
        # do this here if we are not saving histograms, else it will be done later after gradient/weight/etc. histograms have been setup
        if not self.save_histograms:
            self._setup_summaries()
        
        self._num_original_train_images_saved = 0
        self._num_distorted_train_images_saved = 0
        self._num_original_val_images_saved = 0
        self._num_distorted_val_images_saved = 0
  
    def _load_and_enqueue(self):
        """ Loads and Enqueues a batch of images, poses, labels, and possibly gripper parameters for training """

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

                train_data_arr = np.load(os.path.join(self.data_dir, train_data_filename))[
                                         'arr_0'].astype(np.float32)
                train_poses_arr = np.load(os.path.join(self.data_dir, self.pose_filenames_copy[file_num]))[
                                          'arr_0'].astype(np.float32)
                train_label_arr = np.load(os.path.join(self.data_dir, self.label_filenames_copy[file_num]))[
                                          'arr_0'].astype(np.float32)
                if self.gripper_dim > 0:
                    train_gripper_arr = np.load(os.path.join(self.data_dir, self.gripper_param_filenames_copy[file_num]))[
                                          'arr_0'].astype(np.float32)
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                    train_gripper_depth_mask_arr = np.load(os.path.join(self.data_dir, self.gripper_depth_mask_filenames_copy[file_num]))[ 'arr_0'].astype(np.float32)
                    train_gripper_seg_mask_arr = np.load(os.path.join(self.data_dir, self.gripper_seg_mask_filenames_copy[file_num]))[ 'arr_0'].astype(np.float32)
                   
                # get batch indices uniformly at random
                train_ind = self.train_index_map[train_data_filename]
                np.random.shuffle(train_ind)
                if self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
                    tp_tmp = parse_pose_data(self.train_poses_arr.copy(), self.input_pose_mode)
                    train_ind = train_ind[np.isfinite(tp_tmp[train_ind,1])]
                upper = min(num_remaining, train_ind.shape[
                            0], self.max_training_examples_per_load)
                ind = train_ind[:upper]
                num_loaded = ind.shape[0]
                end_i = start_i + num_loaded

                # subsample data
                train_data_arr = train_data_arr[ind, ...]
                train_poses_arr = train_poses_arr[ind, :]
                train_label_arr = train_label_arr[ind]
                if self.gripper_dim > 0:
                    train_gripper_arr = train_gripper_arr[ind]
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                    train_gripper_depth_mask_arr = train_gripper_depth_mask_arr[ind]
                    train_gripper_seg_mask_arr = train_gripper_seg_mask_arr[ind]
                self.num_images = train_data_arr.shape[0]
                
                # save undistorted train images for debugging
                if self.cfg['save_original_train_images']:
                    if self._num_original_train_images_saved < self.cfg['num_original_train_images']:
                        output_dir = self.exp_path_gen('original_train_images')
                        if not os.path.exists(output_dir):
                            os.mkdir(output_dir)
                        np.savez_compressed(os.path.join(output_dir, 'original_image_{}'.format(self._num_original_train_images_saved)), train_data_arr[0, :, :, 0])
                        self._num_original_train_images_saved += 1  

                # add noise to images
                if self.cfg['mask_and_inpaint']:
                    # allocate mask tensor if required
                    mask_arr = np.zeros((train_data_arr.shape[0], train_data_arr.shape[1], train_data_arr.shape[2], self.num_tensor_channels))
                    mask_arr[:, :, :, 0] = train_data_arr[:, :, :, 0]
                    train_data_arr = mask_arr
                    train_data_arr, train_poses_arr = denoise(train_data_arr, self.im_height, self.im_width, self.im_channels, self.denoising_params, pose_arr=train_poses_arr, pose_dim=self.pose_dim, mask_and_inpaint=True)
                else:
                     i = 0
#                    train_data_arr, train_poses_arr = denoise(train_data_arr, self.im_height, self.im_width, self.im_channels, self.denoising_params, pose_arr=train_poses_arr, pose_dim=self.pose_dim)

                # save distorted train images for debugging 
                if self.cfg['save_distorted_train_images']:
                    if self._num_distorted_train_images_saved < self.cfg['num_distorted_train_images']:
                        output_dir = self.exp_path_gen('distorted_train_images')
                        if not os.path.exists(output_dir):
                            os.mkdir(output_dir)
                        np.savez_compressed(os.path.join(output_dir, 'distorted_image_{}'.format(self._num_distorted_train_images_saved)), train_data_arr[0, :, :, 0])
                        self._num_distorted_train_images_saved += 1

                train_data_arr[:, :, :, 0] = (train_data_arr[:, :, :, 0] - self.im_mean) / self.im_std
                train_poses_arr = (train_poses_arr - self.pose_mean) / self.pose_std
                if self.gripper_dim > 0:
                    train_gripper_arr = (train_gripper_arr - self.gripper_mean) / self.gripper_std
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                    train_gripper_depth_mask_arr[:, :, :, 0] = (train_gripper_depth_mask_arr[:, :, :, 0] - self.gripper_depth_mask_mean[0]) / self.gripper_depth_mask_std[0]
                    train_gripper_depth_mask_arr[:, :, :, 1] = (train_gripper_depth_mask_arr[:, :, :, 1] - self.gripper_depth_mask_mean[1]) / self.gripper_depth_mask_std[1]

                # normalize labels
                if self.training_mode == TrainingMode.REGRESSION:
                    if self.preproc_mode == PreprocMode.NORMALIZATION:
                        train_label_arr = (train_label_arr - self.min_grasp_metric) / (self.max_grasp_metric - self.min_grasp_metric)
                elif self.training_mode == TrainingMode.CLASSIFICATION:
                    train_label_arr = 1 * (train_label_arr > self.metric_thresh)
                    train_label_arr = train_label_arr.astype(self.numpy_dtype)

                # enqueue training data batch
                train_data[start_i:end_i, :, :, 0] = train_data_arr[:, :, :, 0]
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                    train_data[start_i:end_i, :, :, 1] = train_gripper_depth_mask_arr[:, :, :,  0]
                    train_data[start_i:end_i, :, :, 2] = train_gripper_depth_mask_arr[:, :, :,  1]
                train_poses[start_i:end_i,:] = parse_pose_data(train_poses_arr, self.input_pose_mode)
                label_data[start_i:end_i] = train_label_arr
                if self.gripper_dim > 0:
                    train_gripper[start_i:end_i] = parse_gripper_data(train_gripper_arr, self.input_gripper_mode)

                del train_data_arr
                del train_poses_arr
                del train_label_arr
                if self.gripper_dim > 0:
                    del train_gripper_arr
                if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                    del train_gripper_depth_mask_arr
                    del train_gripper_seg_mask_arr
        
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
        self.gripper_depth_mask_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.gripper_seg_mask_filenames.sort(key = lambda x: int(x[-9:-4]))

        for data_filename, pose_filename, gripper_filename, gripper_depth_mask_filename, gripper_seg_mask_filename, label_filename in zip(self.im_filenames, self.pose_filenames, self.gripper_param_filenames, self.gripper_depth_mask_filenames, self.gripper_seg_mask_filenames, self.label_filenames):
            # load next file
            data = np.load(os.path.join(self.data_dir, data_filename))['arr_0']
            poses = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
            labels = np.load(os.path.join(self.data_dir, label_filename))['arr_0']
            if self.gripper_dim > 0:
                gripper_params = np.load(os.path.join(self.data_dir, gripper_filename))['arr_0']
            if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                gripper_depth_mask = np.load(os.path.join(self.data_dir, gripper_depth_mask_filename))[ 'arr_0'].astype(np.float32)
                gripper_seg_mask = np.load(os.path.join(self.data_dir, gripper_seg_mask_filename))[ 'arr_0'].astype(np.float32)

            val_indices = self.val_index_map[data_filename]
        
            # if no datapoints from this file are in validation then just continue
            if len(val_indices) == 0:
                continue
        
            if self.input_pose_mode == InputPoseMode.TF_IMAGE_SUCTION:
                tp_tmp = parse_pose_data(poses.copy(), self.input_pose_mode)
                val_indices = val_indices[np.isfinite(tp_tmp[val_indices,1])]

            data = data[val_indices,...]
            poses = parse_pose_data(poses[val_indices, :], self.input_pose_mode)
            if self.gripper_dim > 0:
                gripper_params = parse_gripper_data(gripper_params[val_indices, :], self.input_gripper_mode)
            if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                gripper_depth_mask = gripper_depth_mask[val_indices]
                gripper_seg_mask = gripper_seg_mask[val_indices]
            labels = labels[val_indices,...]

            if self.training_mode == TrainingMode.REGRESSION:
                if self.preproc_mode == PreprocMode.NORMALIZATION:
                    labels = (labels - self.min_grasp_metric) / (self.max_grasp_metric - self.min_grasp_metric)
            elif self.training_mode == TrainingMode.CLASSIFICATION:
                labels = 1 * (labels > self.metric_thresh)
                labels = labels.astype(np.uint8)
            
            # save undistorted validation images for debugging 
            if self.cfg['save_original_val_images']:
                if self._num_original_val_images_saved < self.cfg['num_original_val_images']:
                    output_dir = self.exp_path_gen('original_val_images')
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    np.savez_compressed(os.path.join(output_dir, 'original_image_{}'.format(self._num_original_val_images_saved)), data[0, :, :, 0])
                    self._num_original_val_images_saved += 1

            # allocate mask channel if needed
            if self.cfg['mask_and_inpaint']:
                mask_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], self.num_tensor_channels))
                mask_data[:, :, :, 0] = data[:, :, :, 0]
                data = mask_data

            # distort
            if self.cfg['distort_val_data']:
                data, poses = denoise(data, self.im_height, self.im_width, self.im_channels, self.denoising_params, pose_arr=poses, pose_dim=self.pose_dim)

            # add gripper mask channels if needed
            if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
                # allocate tensor for image data
                data_new = np.zeros(list(data.shape[:-1]) + [self.num_tensor_channels])
                data_new[:, :, :, 0] = data[:, :, :, 0]
                data_new[:, :, :, 1] = gripper_depth_mask[:, :, :, 0]
                data_new[:, :, :, 2] = gripper_depth_mask[:, :, :, 1]
                data = data_new

            # get predictions
            if self.gripper_dim > 0:
                predictions = self.gqcnn.predict(data, poses, gripper_depth_mask=(self.input_gripper_mode == InputGripperMode.DEPTH_MASK), gripper_arr=gripper_params)
            else:
                predictions = self.gqcnn.predict(data, poses, gripper_depth_mask=(self.input_gripper_mode == InputGripperMode.DEPTH_MASK))
            
            # save distorted validation images for debugging
            if self.cfg['save_distorted_val_images']:
                if self._num_distorted_val_images_saved < self.cfg['num_distorted_val_images']:
                    output_dir = self.exp_path_gen('distorted_val_images')
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
        if self.input_gripper_mode == InputGripperMode.DEPTH_MASK:
            del gripper_depth_mask
            del gripper_seg_mask

        # return average error rate over all files (assuming same size)
        return np.mean(error_rates)
