"""
Optimizer class for training a gqcnn(Grasp Quality Neural Network) object.
Author: Vishal Satish
"""
import copy
import json
import logging
import numbers
import numpy as np
import cPickle as pkl
import os
import random
import sys
import shutil
import matplotlib.pyplot as plt
import yaml
import collections
import time

import IPython

import autolab_core.utils as utils
from gqcnn import ImageMode, TrainingMode, PreprocMode, InputDataMode, GeneralConstants, ImageFileTemplates
from gqcnn import GQCNNDataset

from neon.transforms import CrossEntropyMulti, Misclassification, PrecisionRecall
from neon.optimizers import GradientDescentMomentum, ExpSchedule, StepSchedule, Schedule
from neon.callbacks import Callbacks
from neon.callbacks.callbacks import MetricCallback
from neon.layers import GeneralizedCost
from neon.backends import gen_backend


# class l2_regularized_cross_entropy_cost(Cost):
#     def __init__(self, l2_regularizer, cross_entropy_loss):
#         self.cross_entropy_loss = cross_entropy_loss
#         self.l2_regularizer = l2_regularizer

#     def __call__(self, y, t, params):
#         l = self.cross_entropy_loss(y, t)
#         reg_l = 0
#         for p in params:
#             reg_l += self.be.sum(self.be.square(p)) / 2.0
#         return l + self.l2_regularizer*reg_l

#     def bprop(self, y, t):
#         self.cross_entropy.brop(y, t)
        

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

    def _create_loss(self):
        """ Creates a loss based on config file

        Returns
        -------
        :obj:`tensorflow Tensor`
            loss
        """
        if self.cfg['loss'] == 'cross_entropy':
            return GeneralizedCost(costfunc=CrossEntropyMulti())
        else:
            raise ValueError('Loss %s not supported' % (self.cfg['loss']))

    def _create_optimizer(self, learning_rate, momentum_rate, weight_decay, schedule):
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
            return GradientDescentMomentum(learning_rate, momentum_rate, schedule=Schedule())
        else:
            raise ValueError('Optimizer %s not supported' % (self.cfg['optimizer']))

    def _learning_schedule(self, decay_rate):
        return ExpSchedule(decay_rate)
        epochs = [x for x in range(1, 25)]
        lrs = [0.0094999988,
               0.0085737491,
               0.0081450613,
               0.0073509179,
               0.0069833719,
               0.0063024932,
               0.0059873681,
               0.0054036002,
               0.0051334193,
               0.0046329112,
               0.0044012656,
               0.0039721425,
               0.0037735351,
               0.0034056152,
               0.0032353343,
               0.0029198891,
               0.0027738949,
               0.00250344,
               0.0023782679,
               0.0021463865,
               0.0020390672,
               0.0018402583,
               0.0017482453,
               0.0015777914]

    # lrs = [ .01,
    #  0.01,
    #  0.0081450613,
    #  0.0073509179,
    #  0.0069833719,
    #  0.0063024932,
    #  0.0059873681,
    #  0.0054036002,
    #  0.0051334193,
    #  0.0046329112,
    #  0.0044012656,
    #  0.0039721425,
    #  0.0037735351,
    #  0.0034056152,
    #  0.0032353343,
    #  0.0029198891,
    #  0.0027738949,
    #  0.00250344,
    #  0.0023782679,
    #  0.0021463865,
    #  0.0020390672,
    #  0.0018402583,
    #  0.0017482453,
    #  0.0015777914]
    # return StepSchedule(step_config=epochs, change=lrs)

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
        # self.gqcnn.initialize_network() # builds validation network inside gqcnn class
        self._be = self.gqcnn._be

        self._train_model, _ = self.gqcnn._build_network(drop_fc3, drop_fc4, fc3_drop_rate,
                                                         fc4_drop_rate)  # builds training network with dropouts

        # create loss
        self._loss = self._create_loss()

        # setup learning rate
        self._learn_schedule = self._learning_schedule(self.decay_rate)

        # create optimizer
        self._optimizer = self._create_optimizer(self.base_lr, self.momentum_rate, self.train_l2_regularizer,
                                                 self._learn_schedule)

        # create callbacks
        eval_freq = self.eval_frequency * (
                float(self.train_batch_size) / (self._train_iter.ndata + self._val_iter.ndata))
        self._callbacks = Callbacks(self._train_model, train_set=self._train_iter, eval_set=self._val_iter, eval_freq=1,
                                    output_file="./data.h5")
        self._callbacks.add_callback(MetricCallback(eval_set=self._val_iter, metric=Misclassification()))
        self._callbacks.add_callback(MetricCallback(eval_set=self._val_iter, metric=PrecisionRecall(2)))
        self._callbacks.add_hist_callback(plot_per_mini=True, filter_key=['W', 'dW'])

        # begin optimization
        logging.info('Beginning Optimization')
        self._train_model.fit(dataset=self._train_iter, cost=self._loss, optimizer=self._optimizer,
                              num_epochs=self.num_epochs, callbacks=self._callbacks)

        # exit
        logging.info('Exiting Optimization')

    def _read_training_params(self):
        """ Read training parameters from configuration file """

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

        # set random seed for deterministic execution
        if self.debug:
            np.random.seed(GeneralConstants.SEED)
            random.seed(GeneralConstants.SEED)
            self.debug_num_files = self.cfg['debug_num_files']

        # copy config file
        # self._copy_config()

        # read training parameters from config file
        self._read_training_params()

        # create dataset
        self._dataset = GQCNNDataset(self.cfg)

        # get data iterators
        self._data_iters = self._dataset.gen_iterators()
        self._train_iter = self._data_iters['train']
        self._val_iter = self._data_iters['test']

    # gen_backend(backend='gpu', batch_size=64)
    # iterator = self._val_iter.__iter__()
    # from perception import DepthImage
    # from visualization import Visualizer2D as vis
    # unpack_func = lambda gpu_batch: (gpu_batch[0][0].get().reshape((32, 32, 64)), gpu_batch[0][1].get(), gpu_batch[1].get())
    # # IPython.embed()
    # while self._val_iter.nbatches:
    # 	images, poses, labels = unpack_func(iterator.next())
    # 	depth_image = DepthImage(images[:, :, 0])
    # 	pose = poses[:, 0]
    # 	label = labels[:, 0]
    # 	vis.imshow(depth_image)
    # 	print(pose, label)
    # 	vis.show()
