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
from gqcnn import GQCNN

from optimizer_constants import InputDataMode, ImageMode, ImageFileTemplates

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

    def analyze(self, output_dir):
        """ Analyzes a GQCNN model """
        # setup for analysis
        self._setup()

        # run predictions
        self._run_predictions(output_dir)

        # finally plot curves
        self._plot(output_dir)

    def _setup(self):
        """ Setup for analysis """
        # setup logger
        logging.getLogger().setLevel(logging.INFO)
        logging.info('Setting up for analysis')

        # read config
        self.font_size = self.cfg['font_size']
        self.dpi = self.cfg['dpi']
        self.vis_histograms = self.cfg['vis_histograms']
        self.model_dir = self.cfg['model_dir']
        self.models = self.cfg['models']

    def _run_prediction_single_model(self, model_output_dir, model_name='gqcnn', split_type='image_wise', model_tag="GQ-CNN", vis_conv=True, model_type = "gqcnn"):
        from dexnet.learning import BinaryClassificationResult
        logging.info('Analyzing model %s' %(model_name))

        # read in model config
        model_subdir = os.path.join(self.model_dir, model_name)
        model_config_filename = os.path.join(model_subdir, 'config.json')
        with open(model_config_filename) as data_file:
            model_config = json.load(data_file)
        model_type = self.models[model_name]['type']
        model_tag = self.models[model_name]['tag']
        split_type = self.models[model_name]['split_type']

        # create output dir
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        # load
        logging.info('Loading model %s' %(model_name))
        model = GQCNN.load(model_subdir)
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
                filt = sm.imresize(filt, 5.0, interp='nearest', mode='F')
                plt.subplot(d,d,k+1)
                plt.imshow(filt, cmap=plt.cm.gray, interpolation='nearest')
                plt.axis('off')
                figname = os.path.join(model_output_dir, 'conv1_filters.pdf')
                plt.savefig(figname, dpi=self.dpi)

        # read in training params
        model_training_dataset_dir = model_config['dataset_dir']
        model_image_mode = model_config['image_mode']
        model_target_metric = model_config['target_metric_name']
        model_metric_thresh = model_config['metric_thresh']
        model_input_data_mode = model_config['input_data_mode']

        # read in training, val indices
        train_indices = pkl.load(open(train_indices_filename, 'r'))
        val_indices = pkl.load(open(val_indices_filename, 'r'))

        # get filenames
        filenames = os.listdir(model_training_dataset_dir)
        if model_image_mode == ImageMode.BINARY_TF:
            im_filenames = [f for f in filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
        elif model_image_mode == ImageMode.DEPTH_TF:
            im_filenames = [f for f in filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
        elif model_image_mode == ImageMode.DEPTH_TF_TABLE:
            im_filenames = [f for f in filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
        elif model_image_mode == ImageMode.TF_DEPTH_IMS:
            im_filenames = [f for f in filenames if f.find(ImageFileTemplates.tf_depth_ims_tensor_template) > -1]
        else:
            raise ValueError('Model image mode %s not recognized' %(model_image_mode))

        new_im_filenames = []
        pose_filenames = []
        metric_filenames = []
        for im_filename in im_filenames:
            im_num = int(im_filename[-9:-4])
            pose_filename = '%s_%05d.npz' %(ImageFileTemplates.hand_poses_template, im_num)
            if not os.path.exists(os.path.join(model_training_dataset_dir, pose_filename)):
                pose_filename = '%s_%05d.npz' %(ImageFileTemplates.grasps_template, im_num)                
            metric_filename = '%s_%05d.npz' %(model_target_metric, im_num)
            if os.path.exists(os.path.join(model_training_dataset_dir, im_filename)) and os.path.exists(os.path.join(model_training_dataset_dir, pose_filename)) and os.path.exists(os.path.join(model_training_dataset_dir, metric_filename)):
                new_im_filenames.append(im_filename)
                pose_filenames.append(pose_filename)
                metric_filenames.append(metric_filename)
        im_filenames = new_im_filenames
                
        # sort filenames for consistency
        im_filenames.sort(key = lambda x: int(x[-9:-4]))
        pose_filenames.sort(key = lambda x: int(x[-9:-4]))
        metric_filenames.sort(key = lambda x: int(x[-9:-4]))

        im_filenames = [os.path.join(model_training_dataset_dir, f) for f in im_filenames]
        pose_filenames = [os.path.join(model_training_dataset_dir, f) for f in pose_filenames]
        metric_filenames = [os.path.join(model_training_dataset_dir, f) for f in metric_filenames]
            
        num_files = len(im_filenames)
        cur_file_num = 0
        evaluation_time = 0

        # aggregate training and validation true labels and predicted probabilities
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []
        for im_filename, pose_filename, metric_filename in zip(im_filenames, pose_filenames, metric_filenames):
            if cur_file_num % self.cfg['out_rate'] == 0:
                logging.info('Reading file %d of %d' %(cur_file_num+1, num_files))
                
            # read data
            image_arr = np.load(im_filename)['arr_0']            
            pose_arr = np.load(pose_filename)['arr_0']
            metric_arr = np.load(metric_filename)['arr_0']
            labels_arr = 1 * (metric_arr > model_metric_thresh)
            num_datapoints = image_arr.shape[0]

            # slice correct part of pose_arr corresponding to input_data_mode used for training model
            if model_input_data_mode == InputDataMode.PARALLEL_JAW:
                pose_arr = pose_arr[:,2:3]
            elif model_input_data_mode == InputDataMode.SUCTION:
                pose_arr = np.c_[pose_arr[:,2], pose_arr[:,4]]
            elif model_input_data_mode == InputDataMode.TF_IMAGE:
                pose_arr = pose_arr[:,2:3]
            elif model_input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
                pose_arr = np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
            elif model_input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
                pose_arr = pose_arr[:,2:4]
            elif model_input_data_mode == InputDataMode.RAW_IMAGE:
                pose_arr = pose_arr[:,:4]
            elif model_input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
                pose_arr = pose_arr[:,:6]
            else:
                raise ValueError('Input data mode %s not supported' %(model_input_data_mode))
                    
            # predict
            pred_start = time.time()
            pred_arr = model.predict(image_arr, pose_arr)
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

            train_preds.extend(pred_arr[train_indices[index_im_filename]].tolist())
            train_labels.extend(labels_arr[train_indices[index_im_filename]].tolist())
            val_preds.extend(pred_arr[val_indices[index_im_filename]].tolist())
            val_labels.extend(labels_arr[val_indices[index_im_filename]].tolist())
                
            cur_file_num += 1
            
        # aggregate results
        train_class_result = BinaryClassificationResult(np.array(train_preds)[:,1], np.array(train_labels))
        val_class_result = BinaryClassificationResult(np.array(val_preds)[:,1], np.array(val_labels))

        self.train_class_results[model_tag] = train_class_result
        self.val_class_results[model_tag] = val_class_result

        train_class_result.save(os.path.join(model_output_dir, 'train_class_result.cres'))
        val_class_result.save(os.path.join(model_output_dir, 'val_class_result.cres'))

        model.close_session()

        return train_class_result, val_class_result

    def _run_predictions(self, output_dir):
        """ Run predictions to use for plotting """
        logging.info('Running Predictions')
        from dexnet.learning import BinaryClassificationResult
        self.train_class_results = {}
        self.val_class_results = {}
        self.results = {}

        for model_name, model_data in self.models.iteritems():
            logging.info('Analyzing model: %s' %(model_name))
            model_output_dir = os.path.join(output_dir, model_name)
            train_result, val_result = self._run_prediction_single_model(model_output_dir,
                                                                         model_name=model_name,
                                                                         split_type=model_data['split_type'],
                                                                         model_tag=model_data['tag'],
                                                                         vis_conv=model_data['vis_conv'],
                                                                         model_type=model_data['type'])
            BinaryClassificationResult.make_summary_table(train_result, val_result, plot=False,
                                                          save_dir=model_output_dir, save=True)

            logging.info("Finished evaluating model: %s" %(model_name))
    
    def _plot(self, output_dir):
        """ Plot analysis curves """
        logging.info('Beginning Plotting')

        colors = ['g', 'b', 'c', 'y', 'm', 'r']
        styles = ['-', '--', '-.', ':', '-'] 
        num_colors = len(colors)
        num_styles = len(styles)
        
        # get stats, plot curves
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            logging.info('Model %s training error rate: %.3f' %(model_name, train_class_result.error_rate))
            train_class_result.precision_recall_curve(plot=True, color=colors[i%num_colors],
                                                      style=styles[i%num_styles], label=model_tag)
            i += 1
        plt.title('Training Precision Recall Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(output_dir, 'train_precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            train_class_result.roc_curve(plot=True, color=colors[i%num_colors],
                                         style=styles[i%num_styles], label=model_tag)
            i += 1
        plt.title('Training ROC Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(output_dir, 'train_roc.png')
        plt.savefig(figname, dpi=self.dpi)

        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            logging.info('Model %s validation error rate: %.3f' %(model_name, val_class_result.error_rate))
            val_class_result.precision_recall_curve(plot=True, color=colors[i%num_colors],
                                                      style=styles[i%num_styles], label=model_tag)
            i += 1
        plt.title('Validation Precision Recall Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(output_dir, 'val_precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)

        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            val_class_result.roc_curve(plot=True, color=colors[i%num_colors],
                                         style=styles[i%num_styles], label=model_tag)
            i += 1
        plt.title('Validation ROC Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(output_dir, 'val_roc.png')
        plt.savefig(figname, dpi=self.dpi)

        # combined training and validation precision-recall curves plot
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            if model_tag is None:
                train_class_result.precision_recall_curve(plot=True, color=colors[i%num_colors],
                                                      style=styles[i%num_styles], label='Training')
            else:
                train_class_result.precision_recall_curve(plot=True, color=colors[i%num_colors],
                                      style=styles[i%num_styles], label='Training ' + model_tag)
            i += 1
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            if model_tag is None:
                val_class_result.precision_recall_curve(plot=True, color=colors[i%num_colors],
                                                          style=styles[i%num_styles], label='Validation ')
            else:
                val_class_result.precision_recall_curve(plot=True, color=colors[i%num_colors],
                                          style=styles[i%num_styles], label='Validation ' + model_tag)
            i += 1
        
        plt.title('Precision Recall Curves', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(output_dir, 'precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)


        # combined training and validation roc curves plot
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            if model_tag is None:
                train_class_result.roc_curve(plot=True, color=colors[i%num_colors],
                                         style=styles[i%num_styles], label='Training')
            else:
                train_class_result.roc_curve(plot=True, color=colors[i%num_colors],
                                         style=styles[i%num_styles], label='Training' + model_tag)
            i += 1
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            if model_tag is None:
                val_class_result.roc_curve(plot=True, color=colors[i%num_colors],
                                         style=styles[i%num_styles], label='Validation')
            else:
                val_class_result.roc_curve(plot=True, color=colors[i%num_colors],
                             style=styles[i%num_styles], label='Validation' + model_tag)
            i += 1

        plt.title('ROC Curves', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(output_dir, 'ROC.png')
        plt.savefig(figname, dpi=self.dpi)

        # plot histogram of prediction errors
        if self.vis_histograms:
            for model_name in self.models.keys():
                model_output_dir = os.path.join(output_dir, model_name)
                train_result = self.train_class_results[model_tag]
                val_result = self.val_class_results[model_tag]

                # histogram of training errors
                num_bins = min(self.cfg['num_bins'], train_result.num_datapoints)
                
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

                # histogram of training errors
                num_bins = min(self.cfg['num_bins'], val_result.num_datapoints)

                # train positives
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

                # train negatives
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
                
