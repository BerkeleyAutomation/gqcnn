"""
Visualize the predictions of a GQCNN on a dataset Visualizes TP, TN, FP, FN..
Author: Vishal Satish 
"""
import copy
import logging
import numpy as np
import os
import sys
from random import shuffle

import core.utils as utils
from core import YamlConfig, Point
from perception import BinaryImage, ColorImage, DepthImage, GdImage, GrayscaleImage, RgbdImage, RenderMode

from gqcnn import Grasp2D, GQCNN, ClassificationResult, InputDataMode, ImageMode, ImageFileTemplates
from gqcnn import Visualizer as vis2d

class GQCNNPredictionVisualizer(object):
    """ Class to visualize predictions of GQCNN on a specified dataset. Visualizes TP, TN, FP, FN. """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            dictionary of configuration parameters
        """
        # setup config
    	self.cfg = config

    	# setup for visualization
    	self._setup()

    def visualize(self):
        """ Visualize predictions """

        logging.info('Visualizing.')

        # iterate through shuffled file indices
        for i in self.indices:
            im_filename = self.im_filenames[i]
            pose_filename = self.pose_filenames[i]
            label_filename = self.label_filenames[i]

            logging.info('Loading Image File: ' + im_filename + ' Pose File: ' + pose_filename + ' Label File: ' + label_filename)

            # load tensors from files
            image_tensor = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
            hand_poses_tensor = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
            metric_tensor = np.load(os.path.join(self.data_dir, label_filename))['arr_0']
            label_tensor = 1 * (metric_tensor > self.metric_thresh)

            pose_tensor = self._read_pose_data(hand_poses_tensor, self.input_data_mode)

            # score with neural network
            pred_p_success_tensor = self._gqcnn.predict(image_tensor, pose_tensor)

            # compute results
            classification_result = ClassificationResult([pred_p_success_tensor],
                                                         [label_tensor])

            logging.info('Error rate on files: %.3f' %(classification_result.error_rate))
            logging.info('Precision on files: %.3f' %(classification_result.precision))
            logging.info('Recall on files: %.3f' %(classification_result.recall))
            mispred_ind = classification_result.mispredicted_indices()
            correct_ind = classification_result.correct_indices()


            if self.datapoint_type == 'true_positive' or self.datapoint_type == 'true_negative':
                vis_ind = correct_ind
            else:
                vis_ind = mispred_ind

            # visualize
            for ind in vis_ind:
                # don't visualize the datapoints that we don't want
                if self.datapoint_type == 'true_positive':
                    if classification_result.labels[ind] == 0:
                        continue
                elif self.datapoint_type == 'true_negative':
                    if classification_result.labels[ind] == 1:
                        continue
                elif self.datapoint_type == 'false_positive':
                    if classification_result.labels[ind] == 1:
                        continue
                elif self.datapoint_type == 'false_negative':
                    if classification_result.labels[ind] == 0:
                        continue

                logging.info('Datapoint %d of files for %s' %(ind, im_filename))
                logging.info('Depth: %.3f' %(hand_poses_tensor[ind, 2]))

                data = image_tensor[ind,...]
                if self.display_image_type == RenderMode.SEGMASK:
                    image = BinaryImage(data)
                elif self.display_image_type == RenderMode.GRAYSCALE:
                    image = GrayscaleImage(data)
                elif self.display_image_type == RenderMode.COLOR:
                    image = ColorImage(data)
                elif self.display_image_type == RenderMode.DEPTH:
                    image = DepthImage(data)
                elif self.display_image_type == RenderMode.RGBD:
                    image = RgbdImage(data)
                elif self.display_image_type == RenderMode.GD:
                    image = GdImage(data)

                vis2d.figure()

                if self.display_image_type == RenderMode.RGBD:
                    vis2d.subplot(1,2,1)
                    vis2d.imshow(image.color)
                    grasp = Grasp2D(Point(image.center, 'img'), 0, hand_poses_tensor[ind, 2], self.gripper_width_m)
                    grasp.camera_intr = grasp.camera_intr.resize(1.0 / 3.0)
                    vis2d.grasp(grasp)
                    vis2d.subplot(1,2,2)
                    vis2d.imshow(image.depth)
                    vis2d.grasp(grasp)
                elif self.display_image_type == RenderMode.GD:
                    vis2d.subplot(1,2,1)
                    vis2d.imshow(image.gray)
                    grasp = Grasp2D(Point(image.center, 'img'), 0, hand_poses_tensor[ind, 2], self.gripper_width_m)
                    grasp.camera_intr = grasp.camera_intr.resize(1.0 / 3.0)
                    vis2d.grasp(grasp)
                    vis2d.subplot(1,2,2)
                    vis2d.imshow(image.depth)
                    vis2d.grasp(grasp)
                else:
                    vis2d.imshow(image)
                    grasp = Grasp2D(Point(image.center, 'img'), 0, hand_poses_tensor[ind, 2], self.gripper_width_m)
                    grasp.camera_intr = grasp.camera_intr.resize(1.0 / 3.0)
                    vis2d.grasp(grasp)
                vis2d.title('Datapoint %d: Pred: %.3f Label: %.3f' %(ind,
                                                                     classification_result.pred_probs[ind,1],
                                                                     classification_result.labels[ind]))
                vis2d.show()

        # cleanup
        self._cleanup()

    def _cleanup(self):
        """ Close GQCNN TF session"""
    	self._gqcnn.close_session()

    def _setup(self):
        """ Setup for visualization """
    	# setup logger
    	logging.getLogger().setLevel(logging.INFO)	
        logging.info('Setting up for visualization.')

    	#### read config params ###

    	# dataset directory
    	self.data_dir = self.cfg['dataset_dir']

    	# datatpoints params
        self.display_image_type = self.cfg['display_image_type']
        self.font_size = self.cfg['font_size']

        # analysis params
        self.datapoint_type = self.cfg['datapoint_type']
        self.image_mode = self.cfg['image_mode']
        self.input_data_mode = self.cfg['data_format']
        self.target_metric_name = self.cfg['metric_name']
        self.metric_thresh = self.cfg['metric_thresh']
        self.gripper_width_m = self.cfg['gripper_width_m']

        # setup data filenames
        self._setup_data_filenames()

        # setup shuffled file indices
        self._compute_indices()

        # load gqcnn
        logging.info('Loading GQ-CNN')
        self.model_dir = self.cfg['model_dir']
        self._gqcnn = GQCNN.load(self.model_dir)
        self._gqcnn.open_session()

    def _setup_data_filenames(self):
        """ Setup image and pose data filenames, subsample files, check validity of filenames/image mode """

        # read in filenames of training data(poses, images, labels)
        logging.info('Reading filenames')
        all_filenames = os.listdir(self.data_dir)
        if self.image_mode== ImageMode.BINARY:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tensor_template) > -1]
        elif self.image_mode== ImageMode.DEPTH:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tensor_template) > -1]
        elif self.image_mode== ImageMode.BINARY_TF:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.COLOR_TF:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.color_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.GRAY_TF:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.gray_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.DEPTH_TF:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.DEPTH_TF_TABLE:
            self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
        else:
            raise ValueError('Image mode %s not supported.' %(self.image_mode))

        self.pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]
        self.label_filenames = [f for f in all_filenames if f.find(self.target_metric_name) > -1]

        self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.label_filenames.sort(key = lambda x: int(x[-9:-4]))

        # check that all file categories were found
        if len(self.im_filenames) == 0 or len(self.label_filenames) == 0 or len(self.label_filenames) == 0:
            raise ValueError('1 or more required training files could not be found')

    def _compute_indices(self):
        """ Generate random file index so visualization starts from a 
            different random file everytime """
        self.indices = np.arange(len(self.im_filenames))
        np.random.shuffle(self.indices)

    def _read_pose_data(self, pose_arr, input_data_mode):
        """ Read the pose data and slice it according to the specified input_data_mode

        Parameters
        ----------
        pose_arr: :obj:`ndArray`
            full pose data array read in from file
        input_data_mode: :obj:`InputDataMode`
            enum for input data mode, see optimizer_constants.py for all
            possible input data modes 

        Returns
        -------
        :obj:`ndArray`
            sliced pose_data corresponding to input data mode
        """
        if input_data_mode == InputDataMode.TF_IMAGE:
            return pose_arr[:,2:3]
        elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
            return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
        elif input_data_mode == InputDataMode.RAW_IMAGE:
            return pose_arr[:,:4]
        elif input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
            return pose_arr[:,:6]
        elif input_data_mode == InputDataMode.REGRASPING:
            # depth, approach angle, and delta angle for reorientation
            return np.c_[pose_arr[:,2:3], pose_arr[:,4:5], pose_arr[:,6:7]]
        else:
            raise ValueError('Input data mode %s not supported' %(input_data_mode))
