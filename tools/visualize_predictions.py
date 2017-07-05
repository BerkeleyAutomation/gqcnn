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

import logging

from autolab_core import YamlConfig
from gqcnn import GQCNNPredictionVisualizer

if __name__ == 'main':
	# setup logger
	logging.getLogger().setLevel(logging.INFO)

	# load a valid config
	visualization_config = YamlConfig('cfg/tools/gqcnn_prediction_visualizer.yaml')

	logging.info('Beginning Visualization')
	visualizer = GQCNNPredictionVisualizer(visualization_config)
	visualizer.visualize()