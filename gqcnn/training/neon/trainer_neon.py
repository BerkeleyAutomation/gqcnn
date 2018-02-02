"""
Class for training a GQCNN using Intel Neon backend.
Author: Vishal Satish
"""
import logging
import os
import random

import numpy as np

from gqcnn.utils.training_utils import setup_output_dirs, copy_config
from gqcnn.utils.enums import TrainingMode
from gqcnn.training.neon.gqcnn_dataset import GQCNNDataset

from neon.transforms.cost import Metric
from neon.transforms import CrossEntropyMulti, Accuracy
from neon.optimizers import GradientDescentMomentum, ExpSchedule
from neon.callbacks import Callbacks
from neon.callbacks.callbacks import MetricCallback, SerializeModelCallback
from neon.layers import GeneralizedCost

class ErrorRate(Metric):
    """ Custom metric for calculating error rate during training """
    def __init__(self):
        self._acc_metric = Accuracy()
        self.metric_names = ['error_rate']

    def __call__(self, y, t, calcrange=slice(0, None)):
        return 1 - self._acc_metric(y, t, calcrange=calcrange)

class GQCNNTrainerNeon(object):
    """ Trains GQCNN with Neon backend """

    def __init__(self, gqcnn, config):
        """
        Parameters
        ----------
        gqcnn : :obj:`GQCNNNeon`
            grasp quality neural network to optimize
        config : dict
            dictionary of configuration parameters
        """
        self._gqcnn = gqcnn
        self.cfg = config

    def _create_loss(self):
        """ Creat loss based on config """
        if self.cfg['loss'] == 'cross_entropy':
            return GeneralizedCost(costfunc=CrossEntropyMulti())
        else:
            raise ValueError('Loss: {} not supported'.format(self._cfg['loss']))

    def _create_optimizer(self, learning_rate, momentum_rate, weight_decay, schedule):
        """ Create optimizer based on config """
        if self.cfg['optimizer'] == 'momentum':
            return GradientDescentMomentum(learning_rate, momentum_rate, wdecay=weight_decay, schedule=schedule)
        else:
            raise ValueError('Optimizer %s not supported' % (self.cfg['optimizer']))

    def _learning_schedule(self, decay_rate):
        """ Create exponential learning schedule with specified decay rate"""
        return ExpSchedule(decay_rate)

    def train(self):
        """ Perform optimization """

        # setup for training
        self._setup()

        # build network
        if self.training_mode == TrainingMode.CLASSIFICATION:
            self._gqcnn.initialize_network(add_softmax=True)
        elif self.training_mode == TrainingMode.REGRESSION:
            self._gqcnn.initialize_network()
        else:
            raise ValueError('Training mode: {} not supported.'.format(self.training_mode))
        self._model = self._gqcnn.model

        # create loss
        self._loss = self._create_loss()

        # setup learning rate
        self._learn_schedule = self._learning_schedule(self.decay_rate)

        # create optimizer
        self._optimizer = self._create_optimizer(self.base_lr, self.momentum_rate, self.train_l2_regularizer, 
            self._learn_schedule)

        # create callbacks
        self._callbacks = Callbacks(self._model, eval_set=self._val_iter, 
            eval_freq=self.eval_frequency, output_file=self.exp_path_gen('data.h5'))
        self._callbacks.add_callback(MetricCallback(eval_set=self._val_iter, metric=ErrorRate()))
        self._callbacks.add_callback(SerializeModelCallback(self.exp_path_gen('model_ckpt.prm'), 
            epoch_freq=self.save_frequency))

        # begin optimization
        logging.info('Beginning Optimization')
        self._model.fit(dataset=self._train_iter, cost=self._loss, optimizer=self._optimizer, 
            num_epochs=self.num_epochs, callbacks=self._callbacks)
        
        # save final model
        self._model.save_params(self.exp_path_gen('model.prm'))        
        
        # set term event for data fetch queue
        self._train_iter.set_term_event()

        # exit
        logging.info('Exiting Optimization')

    def _read_training_params(self):
        """ Read training parameters from config"""

        self.data_dir = self.cfg['dataset_dir']
        self.image_mode = self.cfg['image_mode']
        self.data_split_mode = self.cfg['data_split_mode']
        self.train_pct = self.cfg['train_pct']
        self.total_pct = self.cfg['total_pct']

        self.train_batch_size = self.cfg['train_batch_size']
        # update the GQCNN's batch size to self.train_batch_size,
        # we will use the same batch size for training and validation since Neon backends have a fixed batch size
        self._gqcnn.update_batch_size(self.train_batch_size)

        self.num_epochs = self.cfg['num_epochs']
        self.eval_frequency = self.cfg['eval_frequency']
        self.save_frequency = self.cfg['save_frequency']

        self.queue_capacity = self.cfg['queue_capacity']
        self.queue_sleep = self.cfg['queue_sleep']

        self.train_l2_regularizer = self.cfg['train_l2_regularizer']
        self.base_lr = self.cfg['base_lr']
        self.decay_rate = self.cfg['decay_rate']
        self.momentum_rate = self.cfg['momentum_rate']
        self.drop_rate = self.cfg['drop_rate']
        # update the GQCNN's drop rate
        self._gqcnn.update_drop_rate(self.drop_rate)

        self.target_metric_name = self.cfg['target_metric_name']
        self.metric_thresh = self.cfg['metric_thresh']
        self.training_mode = self.cfg['training_mode']
        self.preproc_mode = self.cfg['preproc_mode']

        self._backend = self.cfg['backend']
        # override GQCNN backend 
        self._gqcnn.update_backend(self._backend)

        if self.train_pct < 0 or self.train_pct > 1:
            raise ValueError('Train percentage must be in range [0,1]')

        if self.total_pct < 0 or self.total_pct > 1:
            raise ValueError('Train percentage must be in range [0,1]')

    def _setup(self):
        """ Setup for training """

        # get debug flag and number of files to use when debugging
        self.debug = self.cfg['debug']
        self.debug_num_files = self.cfg['debug_num_files']

        # set random seed for deterministic execution if in debug mode
        if self.debug:
            np.random.seed(GeneralConstants.SEED)
            random.seed(GeneralConstants.SEED)

        # setup output directories
        output_dir = self.cfg['output_dir']
        self.experiment_dir, _, _ = setup_output_dirs(output_dir)

        # create python lambda function to help create file paths to experiment_dir
        self.exp_path_gen = lambda fname: os.path.join(self.experiment_dir, fname)

        # copy config file
        copy_config(self.experiment_dir, self.cfg)

        # read training parameters from config file
        self._read_training_params()

        # create dataset
        self._dataset = GQCNNDataset(self._gqcnn, self.experiment_dir, self.total_pct, self.train_pct, self.data_split_mode, self.target_metric_name, self.metric_thresh, self.data_dir, self.queue_sleep, 
            self.queue_capacity, self.image_mode, self.training_mode, self.preproc_mode, self.cfg, 
            debug=self.debug, debug_num_files=self.debug_num_files)

        steps_per_epoch = self._dataset.num_datapoints * self.train_pct / self.train_batch_size
        # if self.eval_frequency == -1, change it to reflect a single epoch(this is so it plays well with Tensorflow implementation)
        if self.eval_frequency == -1:
            self.eval_frequency = 1
        else:
            self.eval_frequency = int(math.ceil(float(self.eval_frequency) / steps_per_epoch))
        # if self.save_frequency == -1, change it to reflect a single epoch(this is so it plays well with Tensorflow implementation)
        if self.save_frequency == -1:
            self.save_frequency = 1
        else:
            self.save_frequency = int(math.ceil(float(self.eval_frequency) / steps_per_epoch))

        # generate backend prematurely so that iterators can access batch size
        self._gqcnn.init_backend()

        # get data iterators
        self._data_iters = self._dataset.gen_iterators()
        self._train_iter = self._data_iters['train']
        self._val_iter = self._data_iters['val']
