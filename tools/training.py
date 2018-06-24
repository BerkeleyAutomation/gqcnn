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
Script with examples for: 
1) Training Grasp Quality Neural Networks(GQ-CNN's) using Stochastic Gradient Descent 
2) Predicting probability of grasp success from images in batches using a pre-trained GQ-CNN model
3) Fine-tuning a GQ-CNN model
4) Analyzing a GQC-NN model

Author
------
Vishal Satish

YAML Configuration File Parameters
----------------------------------

Training/Fine-Tuning Configuration
``````````````````````````````````
dataset_dir : str
	the location of the dataset to use for training ex. /path/to/your/dataset
output_dir : str
	the location to save the trained model, which will be saved as model_XXXXX where XXXXX will
	be a randomly generated string of characters. Note: if the debug flag is on this model name will stay the same so the same model directory
	will continuosuly be over-written. If debug is off, a new name and thus directrory will be generated every time.
model_dir : str
	if the fine-tune flag is on this is the model that will be loaded for training ex. /path/to/pre-trained/model

train_batch_size : int
	the number of datapoints to process during each iteration of training
val_batch_size : int
	the number of datapoints to process during each iteration of validation
num_epochs : int
	the number of epoches to train for
eval_frequency : int
	the validation error will be calculated after this many iterations
save_frequency : int
	the model will be saved after this many iterations
vis_frequency : int
	if the visualization flag is turned on visualization will occur after this many iterations
log_frequency : int
	training metrics will be logged after this many iterations
show_filters : int
	flag (0 or 1) whether or not to show network filters during training

queue_capacity : int
	total capacity of data prefetch queue
queue_sleep : float
	how long to sleep between data prefetches

data_split_mode : str
	how to split up the data into training and validation, options are 1) image wise-randomly shuffle and split images 2) stable_pose_wise-randomly shuffle all valid stable
	poses of objects and then split so that all datapoints of a certain stable pose are entirely in either training or validation 3) object_wise-randomly shuffle all objects and split into training and validation
	so that all datapoints of a certain object are entirely in training or validation
train_pct : float
	percentage of datapoints to use for training
val_pct : float
	percentage of datapoints to use for validation
total_pct : float
	percentage of total datapoints to use
eval_total_train_error : int
	flag (0 or 1) whether or not to evaluate the total training error and save it at the end of optimization

loss : str
	the loss function to use, currently supported options are 1) l2-l2 loss 2) sparse-sparse softmax cross entropy with logits loss
optimizer : str
	the optimizer to use, currently supported options are 1) momentum 2) adam 3) rmsprop
train_l2_regularizer : float
	factor to multiple summed regularizers by before adding to loss:
		loss = loss + train_l2_regularizer * regularizers
base_lr : float
	base learning rate
decay_step_multiplier : float
	number of times to go through entire training datapoints before stepping down decay rate
decay_rate : float
	decay rate during optimization
momentum_rate : float
	momentume rate if using momentum optimizer
max_training_examples_per_load : int
	maximum number of training datapoints to use in each batch

fine_tune : int
	flag (0 or 1) whether or not fine-tuning
update_fc_only : int
	flag (0 or 1) used during fine-tuning to indicate whether or not to only update the first fully-connected layer
update_conv0_only : int
	flag (0 or 1) used during fine-tuning to indicate whether or not to only update the first convolution layer  
reinit_pc1 : int
	flag (0 or 1) used during fine-tuning to indicate whether or not to re-initialize the weights for the first pose layer
reinit_fc3 : int
	flag (0 or 1) used during fine-tuning to indicate whether or not to re-initialize the weights for the third fully-connected layer
reinit_fc4 : int
	flag (0 or 1) used during fine-tuning to indicate whether or not to re-initialize the weights for the fourth fully-connected layer
reinit_fc5 : int
	flag (0 or 1) used during fine-tuning to indicate whether or not to re-initialize the weights for the fifth fully-connected layer

image_mode : str 
	the type of the input image datapoints, please refer to the README for the dataset for the possible options
training_mode : str
	1) classification or 2) regression
preproc_mode : str
	the data pre-processing mode for use during regression, options are: 1) normalized 2) none
input_data_mode : str
	the format to use for the input pose data, please refer to the README for the dataset for the possible options 
num_tensor_channels : int
	the number of image channels 
num_random_files : int
	the sub-sample size when calculating dataset metrics such as image_mean, pose_mean, image_std, pose_std

target_metric_name : str
	the name of the target metric to use when training, please refer to the README for the dataset for the possible options
metric_thresh : float
	the threshold to use when converting the grasp probability metric predicted by the network into a binary metric

multiplicative_denoising : int
	flag (0 or 1) whether or not to apply multiplicative denoising to the images
gamma_shape : float
	gamma shape to use for multiplicative_denoising

symmetrize : int
	flag (0 or 1) whether or not to symmetrize images by randomly rotating and reflecting

morphological : int
	flag (0 or 1) whether or not to apply morphological filters to images
morph_open_rate : float
	open rate for morphological filter
morph_poisson_mean : float
	poisson mean to use for morphological filter

image_dropout : int 
	flag (0 or 1) whether or not to randomly dropout regions of the images for robustness
image_dropout_rate : float
	rate at which specific images are chosen to have regions dropped from them
dropout_poisson_mean : float
	poisson mean to use when dropping regions from image
dropout_radius_shape : float
	shape of dropout radius
dropout_radius_scale : float
	scale fo dropout radius

gradient_dropout : int
	flag (0 or 1) whether or not to drop out a region around the areas of the images with high gradient
gradient_dropout_rate : float
	rate at which specific images are chosen to have gradients dropped out
gradient_dropout_sigma : float
	sigma value for gradient dropout
gradient_dropout_shape : float
	shape of gradient dropout filter
gradient_dropout_scale : float
	scale of gradient dropout filter

gaussian_process_denoising : int
	flag (0 or 1) whether or not to add correlated gaussian noise to images
gaussian_process_rate : float
	rate at which specific images are chosen to have added correlated gaussian noise
gaussian_process_scaling_factor : float
	scaling factor to use for added correlated gaussian noise
gaussian_process_sigma : float
	sigma to use for added correlated gaussian noise

border_distortion : int
	flag (0 or 1) whether or not to randomly dropout regions of the image borders for robustness
border_grad_sigma : float
	sigma for gaussian gradient magnitude calculation
border_grad_thresh : float
	threshold for finding high gradient pixels in the image
border_poisson_mean : float
	poisson mean for calculating the number of dropout regions
border_radius_shape : float
	shape of radius of filter for border dropouts
border_radius_scale : float
	scale of radius of filter for border dropouts

background_denoising : int
	flag (0 or 1) whether or not to apply background denoising to images
background_rate : float
	rate at which specific images are chosen to have background denoising applied
background_min_depth : float
	minimum depth in meters for background denoising
background_max_depth : float
	maximum depth in meters for background denoising

debug : int
	flag (0 or 1) whether or not to use debug mode, in debug mode the same model name is used for saving between multiple
	optimizations and the model directory is thus overwritten. Also the number of datapoints used for training is limited by the size
	of the following debug_num_file parameter
debug_num_files : int
	the max number of files from the dataset to use for training and validation

gqcnn_config/im_height : int 
	the height of the input image data in pixels
gqcnn_config/im_width : int 
	the width of the input image data in pixels
gqcnn_config/im_channels : int
	the number of channels in the input images
gqcnn_config/input_data_mode : str
	the type of the input image datapoints, please refer to the README for the dataset for the possible options, NOTE: if the network is being fine-tuned
	then this parameter must match what was used for training as otherwise the pose tensor dimensions will not match
gqcnn_config/batch_size : int
	the batch size to use for prediction from this network, in training using the SGDOptimizer this will be overridden by the val_batch_size parameter
	in the training config
gqcnn_config/architecture : dict
	this section outlines the architecture of the GQ-CNN. Convolutional layers are denoted by the notation convX_Y where X is the group that the layer belongs to
	and Y is the individual layer id. Ex. conv1_1 and conv1_2 are the first and second convolutional layers of the first group of convolutional layers. Layers that process pose 
	data are denoted by pcY where Y is the layer id. Fully-connected layers are denoted by fcY where Y is the layer id. Underneath each layer are its vairous properties such as filter dimensions,
	number of filters, pooling size, normalization type and output_size. Please see the actual yaml file for an example architecture definition.

radius : float
	the network normalization radius
alpha : float
	the network normalization alpha
beta : float
	the network normalization beta
bias : float
	the network normalization bias

Analysis Configuration
``````````````````````
model_dir : str 
	the outer directory containing the models to be analyzed, ex. if the path to one of the models to be analyzed is 
	/home/user/models/grasp_quality/model_XXXX then the model_dir is /home/user/models/grasp_quality/
output_dir : str
	the output directory to save the output figures and/or filters from analysis, ex. /home/users/analyses/gqcnn_training_performance
out_rate : int
	the frequency with which to use dataset files for analyses, ex. if 1 then every single file is used, if 2 then every other file is used
font_size : int
	the font size to use in figures
dpi : int
	the dpi to use when plotting figures
models : dict
	a dictionary of models to analyze along with their analyzation parameters. The following documentation assumes we have
	a sample moduled named model_XXXX and describes the various model parameters
models/model_XXXX/tag : str
	the tag to be prepended before the curve titles
models/model_XXXX/type : str
	the type of model, options are: 1) gqcnn 2) rf 3) svm
models/model_XXXX/split_type : str
	how the data was split into training and validation, options are 1) image wise-randomly shuffle and split images 2) stable_pose_wise-randomly shuffle all valid stable
	poses of objects and then split so that all datapoints of a certain stable pose are entirely in either training or validation 3) object_wise-randomly shuffle all objects and split into training and validation
	so that all datapoints of a certain object are entirely in training or validation. This will determine which sets of indices are read in from the model directory
models/model_XXXX/vis_conv : int
	flag (0 or 1) whether or not to display and save convolution filters from the model directory	
"""
import argparse
import logging
import time
import os

from autolab_core import YamlConfig
from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer

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
                                               'cfg/tools/training.yaml')

        # open config file
	train_config = YamlConfig(config_filename)
	gqcnn_config = train_config['gqcnn_config']

	def get_elapsed_time(time_in_seconds):
		""" Helper function to get elapsed time """
		if time_in_seconds < 60:
			return '%.1f seconds' % (time_in_seconds)
		elif time_in_seconds < 3600:
			return '%.1f minutes' % (time_in_seconds / 60)
		else:
			return '%.1f hours' % (time_in_seconds / 3600)

	###Possible Use-Cases###

	# Training from Scratch
	start_time = time.time()
	gqcnn = GQCNN(gqcnn_config)
	sgdOptimizer = SGDOptimizer(gqcnn, train_config)
	with gqcnn.get_tf_graph().as_default():
	    sgdOptimizer.optimize()
	logging.info('Total Training Time:' + str(get_elapsed_time(time.time() - start_time))) 

	# Prediction
	"""
	start_time = time.time()
	model_dir = '/home/user/Data/models/grasp_quality/model_ewlohgukns'
	gqcnn = GQCNN.load(model_dir)
	output = gqcnn.predict(images, poses)
	pred_p_success = output[:,1]
	logging.info('Total Prediction Time:' + str(get_elapsed_time(time.time() - start_time)))
	"""

	# Analysis
	"""
	start_time = time.time()
	analysis_config = YamlConfig('cfg/tools/analyze_gqcnn_performance.yaml')
	analyzer = GQCNNAnalyzer(analysis_config)
	analyzer.analyze()
	logging.info('Total Analysis Time:' + str(get_elapsed_time(time.time() - start_time)))
	"""

	# Fine-Tuning
	"""
	start_time = time.time()
	model_dir = train_config['model_dir']
	gqcnn = GQCNN.load(model_dir)
	sgdOptimizer = SGDOptimizer(gqcnn, train_config)
	with gqcnn._graph.as_default():
	        sgdOptimizer.optimize()
	logging.info('Total Fine Tuning Time:' + str(get_elapsed_time(time.time() - start_time)))
	"""
