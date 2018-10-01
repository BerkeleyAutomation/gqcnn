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
Compare a set of GQ-CNN models

Author
------
Jeff Mahler
"""
import datetime
import json
import logging
import os
import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np

import autolab_core.utils as utils
from autolab_core import YamlConfig

PCT_POS_VAL_FILENAME = 'pct_pos_val.npy'
TRAIN_LOSS_FILENAME = 'train_losses.npy'
TRAIN_ERRORS_FILENAME = 'train_errors.npy'
VAL_ERRORS_FILENAME = 'val_errors.npy'
TRAIN_ITERS_FILENAME = 'train_eval_iters.npy'
VAL_ITERS_FILENAME = 'val_eval_iters.npy'
WINDOW = 1000

IGNORE_MODELS = ['GQ-Adv', 'GQ-Image-Wise', 'GQ-Suction']

class GQCnnAnalysis(object):
    def __init__(self, config,
                 train_errors,
                 val_errors,
                 train_iters,
                 val_iters,
                 train_losses,
                 pct_pos_val,
                 train_time):
        self.config = config
        self.train_errors = train_errors
        self.val_errors = val_errors
        self.train_iters = train_iters
        self.val_iters = val_iters
        self.train_losses = train_losses
        self.pct_pos_val = pct_pos_val
        self.train_time = train_time
        
        if val_iters[0] != 0:
            self.val_errors = np.r_[self.pct_pos_val, self.val_errors]
            self.val_iters = np.r_[0, self.val_iters]
        
    @property
    def gqcnn_config(self):
        return self.config['gqcnn_config']
        
    @property
    def architecture(self):
        return self.gqcnn_config['architecture']
        
    @property
    def init_val_error(self):
        return self.val_errors[0]

    @property
    def final_val_error(self):
        return self.val_errors[-1]

    @property
    def final_norm_val_error(self):
        return self.norm_val_errors[-1]
    
    @property
    def norm_train_errors(self):
        if self.pct_pos_val > 0:
            return self.train_errors / self.pct_pos_val
        return self.train_errors / self.init_val_error

    @property
    def norm_val_errors(self):
        if self.pct_pos_val > 0:
            return self.val_errors / self.pct_pos_val
        return self.val_errors / self.init_val_error

    def plot_train_errors(self, linewidth=4, color='b', label=''):
        plt.plot(self.train_iters, self.train_errors, linewidth=linewidth, color=color, label=label)

    def plot_val_errors(self, linewidth=4, color='g', label=''):
        plt.plot(self.val_iters, self.val_errors, linewidth=linewidth, color=color, label=label)        

    def plot_norm_train_errors(self, linewidth=4, color='b', label=''):
        plt.plot(self.train_iters, self.norm_train_errors, linewidth=linewidth, color=color, label=label)

    def plot_norm_val_errors(self, linewidth=4, color='g', label=''):
        plt.plot(self.val_iters, self.norm_val_errors, linewidth=linewidth, color=color, label=label)        
            
def analyze_model(model_dir):
    """ Extract key stats for a model. """
    # form filenames
    train_errors_filename = os.path.join(model_dir, TRAIN_ERRORS_FILENAME)
    val_errors_filename = os.path.join(model_dir, VAL_ERRORS_FILENAME)
    train_iters_filename = os.path.join(model_dir, TRAIN_ITERS_FILENAME)
    val_iters_filename = os.path.join(model_dir, VAL_ITERS_FILENAME)
    pct_pos_val_filename = os.path.join(model_dir, PCT_POS_VAL_FILENAME)
    train_losses_filename = os.path.join(model_dir, TRAIN_LOSS_FILENAME)
    config_filename = os.path.join(model_dir, 'config.json')
    
    # read data
    raw_train_errors = np.load(train_errors_filename)
    val_errors = np.load(val_errors_filename)
    raw_train_iters = np.load(train_iters_filename)
    val_iters = np.load(val_iters_filename)
    pct_pos_val = float(val_errors[0])
    if os.path.exists(pct_pos_val_filename):
        pct_pos_val = 100.0 * np.load(pct_pos_val_filename)
    raw_train_losses = np.load(train_losses_filename)
    config = json.load(open(config_filename, 'r'))
    
    # window the training error
    i = 0
    train_errors = []
    train_losses = []
    train_iters = []
    while i < raw_train_errors.shape[0]:
        train_errors.append(np.mean(raw_train_errors[i:i+WINDOW]))
        train_losses.append(np.mean(raw_train_losses[i:i+WINDOW]))
        train_iters.append(i)
        i += WINDOW
    train_errors = np.array(train_errors)
    train_losses = np.array(train_losses)
    train_iters = np.array(train_iters)

    # estimate training time
    start_stats = os.stat(pct_pos_val_filename)
    end_stats = os.stat(train_errors_filename)
    train_time = end_stats.st_mtime - start_stats.st_mtime
    
    return GQCnnAnalysis(config,
                         train_errors,
                         val_errors,
                         train_iters,
                         val_iters,
                         train_losses,
                         pct_pos_val,
                         train_time)
                         
if __name__ == '__main__':
    # initialize logging
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('model_dir', type=str, default=None, help='directory containing the models to compare')
    parser.add_argument('--config_filename', type=str, default='cfg/compare_models.yaml', help='configuration file to use')
    args = parser.parse_args()
    model_dir = args.model_dir
    config_filename = args.config_filename

    # read config
    config = YamlConfig(config_filename)
    
    # read all models
    model_subdirs = utils.filenames(model_dir)

    # remove models outside of the time scope
    new_model_subdirs = []
    for model_subdir in model_subdirs:
        stats = os.stat(model_subdir)
        model_path, model_ext = os.path.splitext(model_subdir)
        if model_ext != '':
            continue
        model_path, model_root = os.path.split(model_subdir)
        if model_root in IGNORE_MODELS:
            continue
        dt = datetime.datetime.fromtimestamp(stats.st_mtime)
        if dt.year >= config['earliest']['year'] and \
           dt.month >= config['earliest']['month'] and \
           dt.day >= config['earliest']['day']:
            new_model_subdirs.append(model_subdir)
    model_subdirs = new_model_subdirs
            
    # analyze each model
    analyses = {}
    for model_subdir in model_subdirs:
        _, model_name = os.path.split(model_subdir)
        logging.info('Analyzing %s' %(model_name))
        try:
            analysis = analyze_model(model_subdir)
            analyses[model_name] = analysis            
            logging.info('Conv1_1: {}'.format(analysis.architecture['conv1_1']))
            logging.info('Conv2_1: {}'.format(analysis.architecture['conv2_1']['pool_size']))
            logging.info('Conv2_2: {}'.format(analysis.architecture['conv2_2']['pool_size']))
            logging.info('FC: {}'.format(analysis.architecture['fc3']))
        except:
            pass

    # comparison
    i = 0
    cm = plt.get_cmap('gist_rainbow')
    
    plt.figure(figsize=(8,8))
    compare_config = config['compare']
    num_colors = len(compare_config.values())
    for model_prefix, label in compare_config.iteritems():
        logging.info('Plotting %s' %(model_prefix))
        color = cm(1.*i / num_colors)
        analysis = None
        for model_name, model_analysis in analyses.iteritems():
            if model_name.startswith(model_prefix):
                analysis = model_analysis
                break
        logging.info('Train time %.3f' %(analysis.train_time))
        analysis.plot_norm_val_errors(color=color,
                                      label=str(label))
        i += 1
    plt.ylim(0, 2.0)
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normalized Validation Error', fontsize=15)
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, plt_labels, loc='best', fontsize=10)
    plt.show()        
    
    # plot
    i = 0
    num_colors = len(analyses.values())
    cm = plt.get_cmap('gist_rainbow')
    
    plt.figure(figsize=(12,12))
    for model_name, analysis in analyses.iteritems():
        logging.info('Plotting %s' %(model_name))
        color = cm(1.*i / num_colors)
        analysis.plot_norm_val_errors(color=color,
                                      label=model_name)
        i += 1
    plt.ylim(0, 2.0)
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normalized Validation Error', fontsize=15)
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, plt_labels, loc='best', fontsize=10)
    plt.show()
