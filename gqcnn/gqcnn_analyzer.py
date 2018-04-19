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

import autolab_core.utils as utils
from autolab_core import BinaryClassificationResult, TensorDataset
from autolab_core.constants import *

from . import GQCNN
from .optimizer_constants import GripperMode, ImageMode
from .utils import *

class GQCNNAnalyzer(object):
    """ Analyzes GQCNN models """

    def __init__(self, config):
        """
        Parameters
        ----------
        model_dir : str
            path to the model to analyze
        output_dir : str
            path to save the model output
        config : dict
            dictionary of configuration parameters
        """
        self.cfg = config
        self._parse_config()

    def _parse_config(self):
        """ Read params from the config file """
        # plotting params
        self.log_rate = self.cfg['log_rate']
        self.font_size = self.cfg['font_size']
        self.dpi = self.cfg['dpi']
        self.num_bins = self.cfg['num_bins']
        
    def analyze(self, model_dir,
                output_dir,
                dataset_config=None):
        """ Analyzes a GQCNN model.

        Parameters
        ----------
        model_dir : str
            path to the model
        output_dir : str
            path to save the analysis
        dataset_config : dict
            dictionary of parameters for the dataset to test on

        Returns
        -------
        :obj:`autolab_core.BinaryClassificationResult`
            result on training data
        :obj:`autolab_core.BinaryClassificationResult`
            result on validation data
        """
        # determine model output dir
        model_name = ''
        model_root = model_dir
        while model_name == '' and model_root != '':
            model_root, model_name = os.path.split(model_root)

        model_output_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        logging.info('Analyzing model %s' %(model_name))
            
        # run predictions
        train_result, val_result = self._run_prediction_single_model(model_dir,
                                                                     model_output_dir,
                                                                     dataset_config)

        # finally plot curves
        self._plot(model_output_dir, train_result, val_result)

        return train_result, val_result
        
    def _run_prediction_single_model(self, model_dir,
                                     model_output_dir,
                                     dataset_config):
        """ Analyze the performance of a single model. """
        # read in model config
        model_config_filename = os.path.join(model_dir, 'config.json')
        with open(model_config_filename) as data_file:
            model_config = json.load(data_file)

        # load model
        logging.info('Loading model %s' %(model_dir))
        gqcnn = GQCNN.load(model_dir)
        gqcnn.open_session()
        gripper_mode = gqcnn.gripper_mode
        
        # read params from the config
        if dataset_config is None:
            dataset_dir = model_config['dataset_dir']
            split_name = model_config['split_name']
            image_field_name = model_config['image_field_name']
            pose_field_name = model_config['pose_field_name']
            metric_name = model_config['target_metric_name']
            metric_thresh = model_config['metric_thresh']
        else:
            dataset_dir = dataset_config['dataset_dir']
            split_name = dataset_config['split_name']
            image_field_name = dataset_config['image_field_name']
            pose_field_name = dataset_config['pose_field_name']
            metric_name = dataset_config['target_metric_name']
            metric_thresh = dataset_config['metric_thresh']
            gripper_mode = dataset_config['gripper_mode']
            
        logging.info('Loading dataset %s' %(dataset_dir))
        dataset = TensorDataset.open(dataset_dir)
        train_indices, val_indices, _ = dataset.split(split_name)
        
        # visualize conv filters
        conv1_filters = gqcnn.filters
        num_filt = conv1_filters.shape[3]
        d = utils.sqrt_ceil(num_filt)
        plt.clf()
        for k in range(num_filt):
            filt = conv1_filters[:,:,0,k]
            plt.subplot(d,d,k+1)
            plt.imshow(filt, cmap=plt.cm.gray, interpolation='none')
            plt.axis('off')
            figname = os.path.join(model_output_dir, 'conv1_filters.pdf')
        plt.savefig(figname, dpi=self.dpi)
        
        # aggregate training and validation true labels and predicted probabilities
        all_predictions = []
        all_labels = []
        for i in range(dataset.num_tensors):
            # log progress
            if i % self.log_rate == 0:
                logging.info('Predicting tensor %d of %d' %(i+1, dataset.num_tensors))

            # read in data
            image_arr = dataset.tensor(image_field_name, i).arr
            pose_arr = read_pose_data(dataset.tensor(pose_field_name, i).arr,
                                      gripper_mode)
            metric_arr = dataset.tensor(metric_name, i).arr
            label_arr = 1 * (metric_arr > metric_thresh)
            label_arr = label_arr.astype(np.uint8)

            # predict with GQ-CNN
            predictions = gqcnn.predict(image_arr, pose_arr)
            
            # aggregate
            all_predictions.extend(predictions[:,1].tolist())
            all_labels.extend(label_arr.tolist())
            
        # close session
        gqcnn.close_session()            

        # create arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        train_predictions = all_predictions[train_indices]
        val_predictions = all_predictions[val_indices]
        train_labels = all_labels[train_indices]
        val_labels = all_labels[val_indices]
        
        # aggregate results
        train_result = BinaryClassificationResult(train_predictions, train_labels)
        val_result = BinaryClassificationResult(val_predictions, val_labels)
        train_result.save(os.path.join(model_output_dir, 'train_result.cres'))
        val_result.save(os.path.join(model_output_dir, 'val_result.cres'))

        # save summary stats
        train_summary_stats = {
            'error_rate': train_result.error_rate,
            'ap_score': train_result.ap_score,
            'auc_score': train_result.auc_score
        }
        train_stats_filename = os.path.join(model_output_dir, 'train_stats.json')
        json.dump(train_summary_stats, open(train_stats_filename, 'w'),
                  indent=JSON_INDENT,
                  sort_keys=True)

        val_summary_stats = {
            'error_rate': val_result.error_rate,
            'ap_score': val_result.ap_score,
            'auc_score': val_result.auc_score
        }
        val_stats_filename = os.path.join(model_output_dir, 'val_stats.json')
        json.dump(val_summary_stats, open(val_stats_filename, 'w'),
                  indent=JSON_INDENT,
                  sort_keys=True)        
        
        return train_result, val_result

    def _plot(self, model_output_dir, train_result, val_result):
        """ Plot analysis curves """
        logging.info('Plotting')

        _, model_name = os.path.split(model_output_dir)
        
        # set params
        colors = ['g', 'b', 'c', 'y', 'm', 'r']
        styles = ['-', '--', '-.', ':', '-'] 
        num_colors = len(colors)
        num_styles = len(styles)
        
        # get stats, plot curves
        logging.info('Model %s training error rate: %.3f' %(model_name, train_result.error_rate))
        logging.info('Model %s validation error rate: %.3f' %(model_name, val_result.error_rate))
        plt.clf()
        train_result.precision_recall_curve(plot=True,
                                            color=colors[0],
                                            style=styles[0],
                                            label='TRAIN')
        val_result.precision_recall_curve(plot=True,
                                          color=colors[1],
                                          style=styles[1],
                                          label='VAL')
        plt.title('Precision Recall Curves', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(model_output_dir, 'precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)

        plt.clf()
        train_result.roc_curve(plot=True,
                               color=colors[0],
                               style=styles[0],
                               label='TRAIN')
        val_result.roc_curve(plot=True,
                             color=colors[1],
                             style=styles[1],
                             label='VAL')
        plt.title('Reciever Operating Characteristic', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(model_output_dir, 'roc.png')
        plt.savefig(figname, dpi=self.dpi)
        
        # plot histogram of prediction errors
        num_bins = min(self.num_bins, train_result.num_datapoints)
                
        # train positives
        pos_ind = np.where(train_result.labels == 1)[0]
        diffs = np.abs(train_result.labels[pos_ind] - train_result.pred_probs[pos_ind])
        plt.figure()
        utils.histogram(diffs,
                        num_bins,
                        bounds=(0,1),
                        normalized=False,
                        plot=True)
        plt.title('Error on Positive Training Examples', fontsize=self.font_size)
        plt.xlabel('Abs Prediction Error', fontsize=self.font_size)
        plt.ylabel('Count', fontsize=self.font_size)
        figname = os.path.join(model_output_dir, '%s_pos_train_errors_histogram.png' %(model_name))
        plt.savefig(figname, dpi=self.dpi)

        # train negatives
        neg_ind = np.where(train_result.labels == 0)[0]
        diffs = np.abs(train_result.labels[neg_ind] - train_result.pred_probs[neg_ind])
        plt.figure()
        utils.histogram(diffs,
                        num_bins,
                        bounds=(0,1),
                        normalized=False,
                        plot=True)
        plt.title('Error on Negative Training Examples', fontsize=self.font_size)
        plt.xlabel('Abs Prediction Error', fontsize=self.font_size)
        plt.ylabel('Count', fontsize=self.font_size)
        figname = os.path.join(model_output_dir, '%s_neg_train_errors_histogram.png' %(model_name))
        plt.savefig(figname, dpi=self.dpi)

        # histogram of validation errors
        num_bins = min(self.num_bins, val_result.num_datapoints)

        # val positives
        pos_ind = np.where(val_result.labels == 1)[0]
        diffs = np.abs(val_result.labels[pos_ind] - val_result.pred_probs[pos_ind])
        plt.figure()
        utils.histogram(diffs,
                        num_bins,
                        bounds=(0,1),
                        normalized=False,
                        plot=True)
        plt.title('Error on Positive Validation Examples', fontsize=self.font_size)
        plt.xlabel('Abs Prediction Error', fontsize=self.font_size)
        plt.ylabel('Count', fontsize=self.font_size)
        figname = os.path.join(model_output_dir, '%s_pos_val_errors_histogram.png' %(model_name))
        plt.savefig(figname, dpi=self.dpi)

        # val negatives
        neg_ind = np.where(val_result.labels == 0)[0]
        diffs = np.abs(val_result.labels[neg_ind] - val_result.pred_probs[neg_ind])
        plt.figure()
        utils.histogram(diffs,
                        num_bins,
                        bounds=(0,1),
                        normalized=False,
                        plot=True)
        plt.title('Error on Negative Validation Examples', fontsize=self.font_size)
        plt.xlabel('Abs Prediction Error', fontsize=self.font_size)
        plt.ylabel('Count', fontsize=self.font_size)
        figname = os.path.join(model_output_dir, '%s_neg_val_errors_histogram.png' %(model_name))
        plt.savefig(figname, dpi=self.dpi)
