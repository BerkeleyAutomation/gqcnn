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
Script for visualizing the predictions made by a given Grasp Quality Neural Network(GQ-CNN) model 
on a given dataset. Allows for the visualization of true-positives, true-negatives, false-positives, and
false negatives.

Author
------
Vishal Satish

YAML Configuration File Parameters
----------------------------------
dataset_dir : str
	the path to the dataset to use for visualization, ex. /path/to/dataset
image_mode : str
	the type of the input image datapoints, please refer to the README for the dataset for the possible options
data_format : str
	the format to use for the input pose data, please refer to the README for the dataset for the possible options 
metric_name : str
	the name of the target metric to use when training, please refer to the README for the dataset for the possible options
metric_thresh : float
	the threshold to use when converting the target metric into a binary metric
gripper_width_m : float
	the robot gripper width in meters (used only for visualization)
model_dir : str
	the path to the GQ-CNN model to use for predicting from the specified dataset, ex. /path/to/your/model
datapoint_type : str
	which datapoints to visualize, options: 1) false_positive 2) false_negative 3) true_positive 4) true_negative
display_image_type : str
	the type of image to display the grasps on during visualization, options: 1) depth 2) color 3) grayscale 4) rgbd 5) gd 6) segmask
font_size : int
	the font size to use for visualization
samples_per_object : int
	the number of predictions to display per object
"""
import argparse
import logging
import os
import sys

from autolab_core import YamlConfig
from gqcnn import GQCNNPredictionVisualizer

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Train a Grasp Quality Convolutional Neural Network with TensorFlow')
    parser.add_argument('--config_filename', type=str, default=None, help='path to the configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename

    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/gqcnn_prediction_visualizer.yaml')
    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)
    
    # load a valid config
    visualization_config = YamlConfig(config_filename)

    logging.info('Beginning Visualization')
    visualizer = GQCNNPredictionVisualizer(visualization_config['dataset_path'],
                                           visualization_config)
    visualizer.visualize()
