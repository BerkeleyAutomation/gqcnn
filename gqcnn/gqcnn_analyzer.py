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

from gqcnn import GQCNN, ClassificationResult

from optimizer_constants import InputDataMode, ImageMode, ImageFileTemplates

binary_im_tf_tensor_template = 'binary_ims_tf'
depth_im_tf_tensor_template = 'depth_ims_tf'
depth_im_tf_table_tensor_template = 'depth_ims_tf_table'
hand_poses_template = 'hand_poses'

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
        else:
            shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)



        self.font_size = self.cfg['font_size']
        self.dpi = self.cfg['dpi']

        self.models = self.cfg['models']

    def _run_prediction_single_model(self, model_subdir, model_name=None, split_type='image_wise', model_tag="GQ-CNN", vis_conv=True, model_type = "gqcnn"):

        if model_name is None:
            model_name = os.path.split(model_subdir)[1]


        # add to the model dict
        self.models[model_name] = {'tag': model_tag, 'split_type': split_type,
                                   'model_type': model_type}


        # read in model config
        model_subdir = os.path.join(self.model_dir, model_subdir)
        model_config_filename = os.path.join(model_subdir, 'config.json')
        with open(model_config_filename) as data_file:
            model_config = json.load(data_file)


        split_type = model_config['data_split_mode']

        # create output dir
        model_output_dir = os.path.join(self.output_dir, model_name)
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        # load
        logging.info('Loading model %s' % (model_name))
        if model_type == 'gqcnn':
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
            if vis_conv:
                conv1_filters = model.filters

                num_filt = conv1_filters.shape[3]
                d = int(np.ceil(np.sqrt(num_filt)))

                plt.clf()
                for k in range(num_filt):
                    filt = conv1_filters[:, :, 0, k]
                    filt = sm.imresize(filt, 5.0, interp='bilinear', mode='F')
                    plt.subplot(d, d, k + 1)
                    plt.imshow(filt, cmap=plt.cm.gray)
                    plt.axis('off')
                figname = os.path.join(model_output_dir, 'conv1_filters.png')
                plt.savefig(figname, dpi=self.dpi)
        else:
            model = pkl.load(open(os.path.join(model_subdir, 'model.pkl')))
            train_indices_filename = os.path.join(model_subdir, 'train_index_map.pkl')
            val_indices_filename = os.path.join(model_subdir, 'val_index_map.pkl')
            image_mean = np.load(os.path.join(model_subdir, 'mean.npy'))
            image_std = np.load(os.path.join(model_subdir, 'std.npy'))
            pose_mean = np.load(os.path.join(model_subdir, 'pose_mean.npy'))[2:3]
            pose_std = np.load(os.path.join(model_subdir, 'pose_std.npy'))[2:3]

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
        filenames = [os.path.join(model_training_dataset_dir, f) for f in os.listdir(model_training_dataset_dir)]
        if model_image_mode == ImageMode.BINARY_TF:
            im_filenames = [f for f in filenames if f.find(binary_im_tf_tensor_template) > -1]
        elif model_image_mode == ImageMode.DEPTH_TF:
            im_filenames = [f for f in filenames if f.find(depth_im_tf_tensor_template) > -1]
        elif model_image_mode == ImageMode.DEPTH_TF_TABLE:
            im_filenames = [f for f in filenames if f.find(depth_im_tf_table_tensor_template) > -1]
        elif model_image_mode == ImageMode.TF_DEPTH_IMS:
            file_template = ImageFileTemplates.tf_depth_ims_tensor_template
            im_filenames = [f for f in filenames if f.find(file_template) > -1]
        else:
            raise ValueError('Model image mode %s not recognized' % (model_image_mode))


        pose_filenames = [f for f in filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]

        logging.info("model_training_dataset_dir %s" %(model_training_dataset_dir))

        if len(pose_filenames) == 0:
            file_template = ImageFileTemplates.grasps_template
            # logging.info("pose filenames template: %s" %(file_template))
            # logging.info("len(filenames) %d" %(len(filenames)))
            pose_filenames = [f for f in filenames if f.find("grasps_") > -1]
            # logging.info("len(pose_filenames): %d" % (len(pose_filenames)))

        metric_filenames = [f for f in filenames if f.find(model_target_metric) > -1]

        # sort filenames for consistency
        im_filenames.sort(key=lambda x: int(x[-9:-4]))
        pose_filenames.sort(key=lambda x: int(x[-9:-4]))
        metric_filenames.sort(key=lambda x: int(x[-9:-4]))


        num_files = len(im_filenames)

        logging.info("len(im_filenames): %d" %(num_files))
        logging.info("len(pose_filenames): %d" %(len(pose_filenames)))
        logging.info("len(metric_filenames): %d" %(len(metric_filenames)))

        cur_file_num = 0
        evaluation_time = 0

        # aggregate training and validation true labels and predicted probabilities
        train_preds = []
        train_labels = []
        val_preds = []
        val_labels = []
        for im_filename, pose_filename, metric_filename in zip(im_filenames, pose_filenames, metric_filenames):

            if cur_file_num % self.cfg['out_rate'] == 0:
                logging.info('Reading file %d of %d' % (cur_file_num + 1, num_files))

            # read data
            image_arr = np.load(im_filename)['arr_0']
            pose_arr = np.load(pose_filename)['arr_0']
            metric_arr = np.load(metric_filename)['arr_0']
            labels_arr = 1 * (metric_arr > model_metric_thresh)
            num_datapoints = image_arr.shape[0]

            if model_type == 'gqcnn':
                # slice correct part of pose_arr corresponding to input_data_mode used for training model
                if model_input_data_mode == InputDataMode.PARALLEL_JAW:
                    pose_arr = pose_arr[:, 2:3]
                elif model_input_data_mode == InputDataMode.SUCTION:
                    pose_arr = np.c_[pose_arr[:, 2], pose_arr[:, 4]]
                elif model_input_data_mode == InputDataMode.TF_IMAGE:
                    pose_arr = pose_arr[:, 2:3]
                elif model_input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
                    pose_arr = np.c_[pose_arr[:, 2:3], pose_arr[:, 4:6]]
                elif model_input_data_mode == InputDataMode.TF_IMAGE_SUCTION:
                    pose_arr = pose_arr[:, 2:4]
                else:
                    raise ValueError('Input data mode %s not supported' % (model_input_data_mode))

            # print "pose_arr shape = ", np.shape(pose_arr)
            # print "model.pose_shape = ", model.pose_shape
            # print "model.get_pose_mean() = ", model.get_pose_mean()

            # predict
            pred_start = time.time()
            if model_type == 'gqcnn':
                pred_arr = model.predict(image_arr, pose_arr)
            else:
                pose_arr = (pose_arr - pose_mean) / pose_std

                if 'use_hog' in model_config.keys() and model_config['use_hog']:
                    feature_arr = None
                    for i in range(num_datapoints):
                        image = image_arr[i, :, :, 0]
                        feature_descriptor = hog(image, orientations=model_config['hog_num_orientations'],
                                                 pixels_per_cell=(model_config['hog_pixels_per_cell'],
                                                                  model_config['hog_pixels_per_cell']),
                                                 cells_per_block=(model_config['hog_cells_per_block'],
                                                                  model_config['hog_cells_per_block']))
                    feature_dim = feature_descriptor.shape[0]

                    if feature_arr is None:
                        feature_arr = np.zeros([num_datapoints, feature_dim + 1])
                    feature_arr[i, :] = np.r_[feature_descriptor, pose_arr[i]]
                else:
                    feature_arr = np.c_[((image_arr - image_mean) / image_std).reshape(num_datapoints, -1),
                                        (pose_arr - pose_mean) / pose_std]

                if model_type == 'rf':
                    pred_arr = model.predict_proba(feature_arr)
                elif model_type == 'svm':
                    pred_arr = model.decision_function(feature_arr)
                    pred_arr = pred_arr / (2 * np.max(np.abs(pred_arr)))
                    pred_arr = pred_arr - np.min(pred_arr)
                    pred_arr = np.c_[1 - pred_arr, pred_arr]
                else:
                    raise ValueError('Model type %s not supported' % (model_type))

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

            # IPython.embed()
            train_preds.append(pred_arr[train_indices[index_im_filename]])
            train_labels.append(labels_arr[train_indices[index_im_filename]])
            val_preds.append(pred_arr[val_indices[index_im_filename]])
            val_labels.append(labels_arr[val_indices[index_im_filename]])

            cur_file_num += 1

        # aggregate results
        train_class_result = ClassificationResult(copy.copy(train_preds), copy.copy(train_labels))
        val_class_result = ClassificationResult(copy.copy(val_preds), copy.copy(val_labels))

        self.train_class_results[model_tag] = train_class_result
        self.val_class_results[model_tag] = val_class_result

        train_class_result.save(os.path.join(model_output_dir, 'train_class_result.cres'))
        val_class_result.save(os.path.join(model_output_dir, 'val_class_result.cres'))

        if model_type == 'gqcnn':
            model.close_session()

        logging.info('Total evaluation time: %.3f sec' % (evaluation_time))




        return train_class_result, val_class_result

    def _run_predictions(self):
        """ Run predictions to use for plotting """
        logging.info('Running Predictions')

        self.train_class_results = {}
        self.val_class_results = {}
        self.results = {}

        for model_name, model_data in self.models.iteritems():
            logging.info('Analyzing model: %s' %(model_name))
            model_subdir = os.path.join(self.cfg['model_dir'], model_data['model_subdir'])
            train_result, val_result = self._run_prediction_single_model(model_subdir, model_name=model_name, split_type=model_data['split_type'], model_tag=model_data['tag'], vis_conv=model_data['vis_conv'], model_type=model_data['type'])


            save_dir = os.path.join(model_subdir, 'analysis')

            ClassificationResult.make_summary_table(train_result, val_result, plot=False,
                                                    save_dir=save_dir, save=True)


            logging.info("Finished evaluating model: %s" %(model_name))



    def _plot(self):
        """ Plot analysis curves """
        logging.info('Beginning Plotting')

        colors = ['g', 'b', 'c', 'y', 'm', 'r']
        styles = ['-', '--', '-.', ':', '-'] 

        # get stats, plot curves
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            logging.info('Model %s training error rate: %.3f' %(model_name, train_class_result.error_rate))
            train_class_result.precision_recall_curve(plot=True, color=colors[i],
                                                      style=styles[i], label=model_tag)
            i += 1
        plt.title('Training Precision Recall Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(self.output_dir, 'train_precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            train_class_result.roc_curve(plot=True, color=colors[i],
                                         style=styles[i], label=model_tag)
            i += 1
        plt.title('Training ROC Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(self.output_dir, 'train_roc.png')
        plt.savefig(figname, dpi=self.dpi)

        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            logging.info('Model %s validation error rate: %.3f' %(model_name, val_class_result.error_rate))
            val_class_result.precision_recall_curve(plot=True, color=colors[i],
                                                      style=styles[i], label=model_tag)
            i += 1
        plt.title('Validation Precision Recall Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(self.output_dir, 'val_precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)

        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            val_class_result.roc_curve(plot=True, color=colors[i],
                                         style=styles[i], label=model_tag)
            i += 1
        plt.title('Validation ROC Curve', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(self.output_dir, 'val_roc.png')
        plt.savefig(figname, dpi=self.dpi)

        # combined training and validation precision-recall curves plot
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            if model_tag is None:
                train_class_result.precision_recall_curve(plot=True, color=colors[i],
                                                      style=styles[i], label='Training')
            else:
                train_class_result.precision_recall_curve(plot=True, color=colors[i],
                                      style=styles[i], label='Training ' + model_tag)
            i += 1
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            if model_tag is None:
                val_class_result.precision_recall_curve(plot=True, color=colors[i],
                                                          style=styles[i], label='Validation ')
            else:
                val_class_result.precision_recall_curve(plot=True, color=colors[i],
                                          style=styles[i], label='Validation ' + model_tag)
            i += 1
        
        plt.title('Precision Recall Curves', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(self.output_dir, 'precision_recall.png')
        plt.savefig(figname, dpi=self.dpi)


        # combined training and validation roc curves plot
        plt.clf()
        i = 0
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            train_class_result = self.train_class_results[model_tag]
            if model_tag is None:
                train_class_result.roc_curve(plot=True, color=colors[i],
                                         style=styles[i], label='Training')
            else:
                train_class_result.roc_curve(plot=True, color=colors[i],
                                         style=styles[i], label='Training' + model_tag)
            i += 1
        for model_name in self.models.keys():
            model_tag = self.models[model_name]['tag']
            val_class_result = self.val_class_results[model_tag]
            if model_tag is None:
                val_class_result.roc_curve(plot=True, color=colors[i],
                                         style=styles[i], label='Validation')
            else:
                val_class_result.roc_curve(plot=True, color=colors[i],
                             style=styles[i], label='Validation' + model_tag)
            i += 1

        plt.title('ROC Curves', fontsize=self.font_size)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        figname = os.path.join(self.output_dir, 'ROC.png')
        plt.savefig(figname, dpi=self.dpi)
