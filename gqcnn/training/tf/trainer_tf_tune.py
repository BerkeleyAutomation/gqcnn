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
Trains a GQCNN network using Tensorflow backend.
Author: Vishal Satish and Jeff Mahler
"""
import argparse
import collections
import copy
import json
import cPickle as pkl
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import multiprocessing as mp
import Queue

import cv2
import numpy as np
import scipy.misc as sm
import scipy.stats as ss
import tensorflow as tf

from autolab_core import BinaryClassificationResult, RegressionResult, TensorDataset, YamlConfig, Logger
from autolab_core.constants import *
import autolab_core.utils as utils

from gqcnn.utils import ImageMode, TrainingMode, GripperMode, InputDepthMode, GeneralConstants, TrainStatsLogger, pose_dim, read_pose_data, weight_name_to_layer_name, GQCNNTrainingStatus
from ray.tune import Trainable
from gqcnn import get_gqcnn_model
from .trainer_tf import GQCNNTrainerTF


class GQCNNTrainerTFTune(GQCNNTrainerTF, Trainable):
    """ Trains a GQ-CNN with Tensorflow backend. """

    def __init__(self, config, logger_creator=None):
        Trainable.__init__(self, config, logger_creator)


    def _setup(self, config):
        self.gqcnn = get_gqcnn_model()(config["gqcnn"])
        self.dataset_dir = config["dataset_dir"]
        self.split_name = config["split_name"]
        self.output_dir = config["output_dir"]
        self.cfg = config
        self.tensorboard_has_launched = False
        self.model_name = config["model_name"]
        self.finetuning = False

        # create a directory for the model
        if self.model_name is None:
            model_id = utils.gen_experiment_id()
            self.model_name = 'model_%s' %(model_id)
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # set up logger
        self.logger = Logger.get_logger(self.__class__.__name__, log_file=os.path.join(self.model_dir, 'training.log'))

        # check default split
        if self.split_name is None:
            self.logger.warning('Using default image-wise split.')
            self.split_name = 'image_wise'

        GQCNNTrainerTF._setup(self)

        self.gqcnn.initialize_network(self.input_im_node, self.input_pose_node)

        self._setup_training()

        # update cfg for saving
        self.cfg['dataset_dir'] = self.dataset_dir
        self.cfg['split_name'] = self.split_name

    def _setup_training(self):
        # setup output
        self.train_net_output = self.gqcnn.output
        if self.training_mode == TrainingMode.CLASSIFICATION:
            if self.cfg['loss'] == 'weighted_cross_entropy':
                self.gqcnn.add_sigmoid_to_output()
            else:
                self.gqcnn.add_softmax_to_output()
        elif self.training_mode == TrainingMode.REGRESSION:
            self.gqcnn.add_sigmoid_to_output()
        else:
            raise ValueError('Training mode: {} not supported !'.format(self.training_mode))
        self.train_predictions = self.gqcnn.output
        self.drop_rate_in = self.gqcnn.input_drop_rate_node
        self.weights = self.gqcnn.weights

        # once weights have been initialized create tf Saver for weights
        self.saver = tf.train.Saver()

        # form loss
        with tf.name_scope('loss'):
            # part 1: error
            loss = self._create_loss()
            self.unregularized_loss = loss

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
        # create optimizer
        with tf.name_scope('optimizer'):
            apply_grad_op, global_grad_norm = self._create_optimizer(loss, batch, var_list, learning_rate)

        # add a handler for SIGINT for graceful exit
        def handler(signum, frame):
            self.logger.info('caught CTRL+C, exiting...')
            self._cleanup()
            exit(0)
        signal.signal(signal.SIGINT, handler)

        # now that everything in our graph is set up, we write the graph to the summary event so it can be visualized in tensorboard
        self.summary_writer.add_graph(self.gqcnn.tf_graph)

        # start prefetch queue workers
        self.prefetch_q_workers = []
        seed = self._seed
        for i in range(self.num_prefetch_q_workers):
            if self.num_prefetch_q_workers > 1 or not self._debug:
                seed = np.random.randint(GeneralConstants.SEED_SAMPLE_MAX)
            p = mp.Process(target=self._load_and_enqueue, args=(seed,))
            p.start()
            self.prefetch_q_workers.append(p)

        # init TF variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.logger.info('Beginning Optimization...')

        # create a TrainStatsLogger object to log training statistics at certain intervals
        self.train_stats_logger = TrainStatsLogger(self.model_dir)

        # loop through training steps
        self.training_range = list(range(int(self.num_epochs * self.num_train) // self.train_batch_size))

    def train(self):
        Trainable.train(self)

    def _train(self):
        step = self.training_range.pop()
        result = self._step(step)
        if not self.training_range:
            result["done"] = True
        return result

    def _stop(self):
        self._cleanup()
