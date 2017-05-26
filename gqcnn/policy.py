"""
Grasping policies
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time

import core.utils as utils
from core import Point
from perception import DepthImage

from gqcnn import Grasp2D, RobotGripper, ImageGraspSamplerFactory, GQCNN, InputDataMode
from gqcnn import Visualizer as vis

class RgbdImageState(object):
    """ State to encapsulate RGB-D images.

    Attributes
    ----------
    rgbd_im : :obj:`perception.RgbdImage`
        an RGB-D image to plan grasps on
    camera_intr : :obj:`perception.CameraIntrinsics`
        intrinsics of the RGB-D camera
    segmask : :obj:`perception.BinaryImage`
        segmentation mask for the binary image
    """
    def __init__(self, rgbd_im, camera_intr, segmask=None):
        self.rgbd_im = rgbd_im
        self.camera_intr = camera_intr
        self.segmask = segmask

class ParallelJawGrasp(object):
    """ Action to encapsulate parallel jaw grasps.
    """
    def __init__(self, grasp, p_success, image, pose):
        self.grasp = grasp
        self.p_success = p_success,
        self.image = image
        self.pose = pose

class Policy(object):
    """ Abstract policy class. """
    __metaclass__ = ABCMeta

    def __call__(self, state):
        return self.action(state)

    @abstractmethod
    def action(self, state):
        """ Returns an action for a given state.
        """
        pass

class GraspingPolicy(Policy):
    """ Policy for robust grasping with Grasp Quality Convolutional Neural Networks (GQ-CNN).

    Attributes
    ----------
    config : dict
        dictionary of parameters for the policy

    Notes
    -----
    Required configuration paramters are specified in Other Parameters

    Other Parameters
    ----------------
    sampling : dict
        dictionary of parameters for grasp sampling, see gqcnn/image_grasp_sampler.py
    gqcnn_model : str
        string path to a trained GQ-CNN model see gqcnn/neural_networks.py
    """
    def __init__(self, config):
        # store parameters
        self._config = config
        self._gripper_width = config['gripper_width']
        self._crop_height = config['crop_height']
        self._crop_width = config['crop_width']
        self._sampling_config = config['sampling']
        self._gqcnn_model_dir = config['gqcnn_model']
        sampler_type = self._sampling_config['type']
        
        # init grasp sampler
        self._grasp_sampler = ImageGraspSamplerFactory.sampler(sampler_type,
                                                               self._sampling_config,
                                                               self._gripper_width)
        
        # init GQ-CNN
        self._gqcnn = GQCNN.load(self._gqcnn_model_dir)
        
    @property
    def config(self):
        """ Returns the policy parameters. """
        return self._config

    @property
    def grasp_sampler(self):
        """ Returns the grasp sampler. """
        return self._grasp_sampler

    @property
    def gqcnn(self):
        """ Returns the GQ-CNN. """
        return self._gqcnn

    @abstractmethod
    def action(self, state):
        """ Returns an action for a given state.
        """
        pass

    def grasps_to_tensors(self, grasps, state):
        """ Converts a list of grasps to an image and pose tensor.

        Attributes
        ----------
        grasps : :obj:`list` of :obj:`Grasp2D`
            list of image grassps to convert
        state : :obj:`RgbdImageState`
            RGB-D image to plan grasps on

        Returns
        -------
        image_arr : :obj:`numpy.ndarray`
            4D Tensor of image to be predicted
        pose_arr : :obj:`numpy.ndarray`
            2D Tensor of depth values
        """
        # parse params
        gqcnn_im_height = self.gqcnn.im_height
        gqcnn_im_width = self.gqcnn.im_width
        gqcnn_num_channels = self.gqcnn.num_channels
        gqcnn_pose_dim = self.gqcnn.pose_dim
        input_data_mode = self.gqcnn.input_data_mode
        num_grasps = len(grasps)
        depth_im = state.rgbd_im.depth

        # allocate tensors
        tensor_start = time()
        image_tensor = np.zeros([num_grasps, gqcnn_im_height, gqcnn_im_width, gqcnn_num_channels])
        pose_tensor = np.zeros([num_grasps, gqcnn_pose_dim])
        for i, grasp in enumerate(grasps):
            translation = np.array([depth_im.center[0] - grasp.center.data[1],
                                    depth_im.center[1] - grasp.center.data[0]])
            im_tf = depth_im.transform(translation, grasp.angle)
            im_tf = im_tf.crop(self._crop_height, self._crop_width)
            im_tf = im_tf.resize((gqcnn_im_height, gqcnn_im_width))
            image_tensor[i,...] = im_tf.raw_data
            
            if input_data_mode == InputDataMode.TF_IMAGE:
                pose_tensor[i] = grasp.depth
            elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.center.x, grasp.center.y])
            else:
                raise ValueError('Input data mode %s not supported' %(input_data_mode))
        logging.debug('Tensor conversion took %.3f sec' %(time()-tensor_start))
        return image_tensor, pose_tensor

class AntipodalGraspingPolicy(GraspingPolicy):
    """ Samples a set of antipodal grasp candidates in image space,
    ranks the grasps by the predicted probability of success from a GQ-CNN,
    and returns the grasp with the highest probability of success.

    Notes
    -----
    Required configuration paramters are specified in Other Parameters

    Other Parameters
    ----------------
    num_grasp_samples : int
        number of grasps to sample
    gripper_width : float, optional
        width of the gripper in meters
    gripper_name : str, optional
        name of the gripper
    """
    def __init__(self, config):
        GraspingPolicy.__init__(self, config)

        self._parse_config()

    def _parse_config(self):
        """ Parses the parameters of the policy. """
        self._num_grasp_samples = self.config['num_grasp_samples']
        self._gripper_width = np.inf
        if 'gripper_width' in self.config.keys():
            self._gripper_width = self.config['gripper_width']
        self._gripper = None
        if 'gripper_name' in self.config.keys():
            self._gripper = RobotGripper.load(self.config['gripper_name'])
            self._gripper_width = self._gripper.max_width

    def select(self, grasps, p_success):
        """ Selects the grasp with the highest probability of success.
        Can override for alternate policies (e.g. epsilon greedy).
        """
        # sort
        num_grasps = len(grasps)
        grasps_and_predictions = zip(np.arange(num_grasps), p_success)
        grasps_and_predictions.sort(key = lambda x : x[1], reverse=True)
        return grasps_and_predictions[0][0]

    def action(self, state):
        """ Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            image to plan grasps on

        Returns
        -------
        :obj:`ParallelJawGrasp`
            grasp to execute
        """
        # check valid input
        if not isinstance(state, RgbdImageState):
            raise ValueError('Must provide an RGB-D image state.')

        # parse state
        rgbd_im = state.rgbd_im
        camera_intr = state.camera_intr
        segmask = state.segmask

        # sample grasps
        grasps = self._grasp_sampler.sample(rgbd_im, camera_intr,
                                            self._num_grasp_samples,
                                            segmask=segmask,
                                            visualize=self.config['vis']['sampling'],
                                            seed=999)
        num_grasps = len(grasps)

        # form tensors
        image_tensor, pose_tensor = self.grasps_to_tensors(grasps, state)
        if self.config['vis']['tf']:
            d = utils.sqrt_ceil(num_grasps)
            vis.figure()
            for i, image_tf in enumerate(image_tensor):
                vis.subplot(d,d,i+1)
                vis.imshow(DepthImage(image_tf))
            vis.show()

        # predict grasps
        predict_start = time()
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
        p_successes = output_arr[:,1]
        logging.debug('Prediction took %.3f sec' %(time()-predict_start))
        if self.config['vis']['candidates']:
            vis.figure()
            vis.imshow(rgbd_im.depth)
            for grasp, q in zip(grasps, p_successes):
                vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title('Sampled grasps')
            vis.show()

        # select grasp
        index = self.select(grasps, p_successes)
        grasp = grasps[index]
        p_success = p_successes[index]
        image = DepthImage(image_tensor[index,...])
        pose = pose_tensor[index,...]
        if self.config['vis']['plan']:
            scale_factor = float(self.gqcnn.im_width) / float(self._crop_width)
            scaled_camera_intr = camera_intr.resize(scale_factor)
            grasp = Grasp2D(Point(image.center), 0.0, pose[0],
                            width=self._gripper_width,
                            camera_intr=scaled_camera_intr)
            vis.figure()
            vis.imshow(image)
            vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(pose[0], p_success))
            vis.show()

        # return action
        return ParallelJawGrasp(grasp, p_success, image, pose)
        
