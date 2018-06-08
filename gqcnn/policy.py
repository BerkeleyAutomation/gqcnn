"""
Grasping policies
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time
import math

from sklearn.mixture import GaussianMixture

import autolab_core.utils as utils
from autolab_core import Point
from perception import DepthImage, RgbdImage, ColorImage

from gqcnn import Grasp2D, ImageGraspSamplerFactory, GraspQualityFunctionFactory
from gqcnn.utils.enums import InputPoseMode
from gqcnn.model import get_gqcnn_model
from gqcnn import Visualizer as vis
from gqcnn.utils.policy_exceptions import NoValidGraspsException

# from dexnet.visualization import DexNetVisualizer3D as vis3d
from dexnet.grasping import ParallelJawPtGrasp3D
from dexnet.visualization import DexNetVisualizer3D as vis3d

# declare any enums or constants
PI = math.pi
FIGSIZE = 16
SEED = 5234709

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
    full_observed : :obj:`object`
        representation of the fully observed state
    """
    def __init__(self, rgbd_im, camera_intr, segmask=None, obj_segmask=None,
                 fully_observed=None):
        self.rgbd_im = rgbd_im
        self.camera_intr = camera_intr
        self.segmask = segmask
        self.obj_segmask = obj_segmask
        self.fully_observed = fully_observed

class ParallelJawGrasp(object):
    """ Action to encapsulate parallel jaw grasps.
    """
    def __init__(self, grasp, q_value, image):
        self.grasp = grasp
        self.q_value = q_value
        self.image = image

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
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    sampling : dict
        dictionary of parameters for grasp sampling, see gqcnn/image_grasp_sampler.py
    gqcnn_model : str
        string path to a trained GQ-CNN model see gqcnn/neural_networks.py
    """
    def __init__(self, config):
        # store parameters
#        self._config = config
#        self._gripper_width = config['gripper_width']
#        self._crop_height = config['crop_height']
#        self._crop_width = config['crop_width']
#        self._sampling_config = config['sampling']
#        self._gqcnn_model_dir = config['gqcnn_model']
#        self._gqcnn_backend = config['gqcnn_backend']
#        sampler_type = self._sampling_config['type']
        
        # init grasp sampler
#        self._grasp_sampler = ImageGraspSamplerFactory.sampler(sampler_type,
#                                                               self._sampling_config,
#                                                               self._gripper_width)
        
        # init GQ-CNN
#        self._gqcnn = get_gqcnn_model(self._gqcnn_backend).load(self._gqcnn_model_dir)

        # open tensorflow session for gqcnn
#        self._gqcnn.open_session()

        # store parameters
        self._config = config
        self._gripper_width = np.inf
        if 'gripper_width' in config.keys():
            self._gripper_width = config['gripper_width']

        # init grasp sampler
        self._sampling_config = config['sampling']
        self._sampling_config['gripper_width'] = self._gripper_width
        sampler_type = self._sampling_config['type']
        self._grasp_sampler = ImageGraspSamplerFactory.sampler(sampler_type,
                                                               self._sampling_config)

        # init grasp quality function
        self._metric_config = config['metric']
        metric_type = self._metric_config['type']
        self._grasp_quality_fn = GraspQualityFunctionFactory.quality_function(metric_type,
                                                                              self._metric_config)

    def __del__(self):
        try:
            self._gqcnn.close_session()
        except:
            pass
        del self

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
            
            if input_data_mode == InputPoseMode.TF_IMAGE:
                pose_tensor[i] = grasp.depth
            elif input_data_mode == InputPoseMode.TF_IMAGE_PERSPECTIVE:
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
    Required configuration parameters are specified in Other Parameters

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
        self._num_grasp_samples = self.config['sampling']['num_grasp_samples']
        self._gripper_width = np.inf
        if 'gripper_width' in self.config.keys():
            self._gripper_width = self.config['gripper_width']

    def select(self, grasps, q_value):
        """ Selects the grasp with the highest probability of success.
        Can override for alternate policies (e.g. epsilon greedy).
        """
        # sort
        num_grasps = len(grasps)
        grasps_and_predictions = zip(np.arange(num_grasps), q_value)
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
                                            visualize=self.config['vis']['grasp_sampling'],
                                            seed=None)
        num_grasps = len(grasps)

        # form tensors
        image_tensor, pose_tensor = self.grasps_to_tensors(grasps, state)
        if self.config['vis']['tf_images']:
            # read vis params
            k = self.config['vis']['k']
            d = utils.sqrt_ceil(k)

            # display grasp transformed images
            vis.figure(size=(FIGSIZE,FIGSIZE))
            for i, image_tf in enumerate(image_tensor[:k,...]):
                depth = pose_tensor[i][0]
                vis.subplot(d,d,i+1)
                vis.imshow(DepthImage(image_tf))
                vis.title('Image %d: d=%.3f' %(i, depth))
            vis.show()

        # predict grasps
        predict_start = time()
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
        q_values = output_arr[:,-1]
        logging.debug('Prediction took %.3f sec' %(time()-predict_start))

        if self.config['vis']['grasp_candidates']:
            # display each grasp on the original image, colored by predicted success
            vis.figure(size=(FIGSIZE,FIGSIZE))
            vis.imshow(rgbd_im.depth)
            for grasp, q in zip(grasps, q_values):
                vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title('Sampled grasps')
            vis.show()

        if self.config['vis']['grasp_ranking']:
            # read vis params
            k = self.config['vis']['k']
            d = utils.sqrt_ceil(k)

            # form camera intr for the thumbnail (to compute gripper width)
            scale_factor = float(self.gqcnn.im_width) / float(self._crop_width)
            scaled_camera_intr = camera_intr.resize(scale_factor)

            # sort grasps
            q_values_and_indices = zip(q_values, np.arange(num_grasps))
            q_values_and_indices.sort(key = lambda x : x[0], reverse=True)

            vis.figure(size=(FIGSIZE,FIGSIZE))
            for i, p in enumerate(q_values_and_indices[:k]):
                # read stats for grasp
                q_value = p[0]
                ind = p[1]
                depth = pose_tensor[ind][0]
                image = DepthImage(image_tensor[ind,...])
                grasp = Grasp2D(Point(image.center), 0.0, depth,
                                width=self._gripper_width,
                                camera_intr=scaled_camera_intr)

                # plot
                vis.subplot(d,d,i+1)
                vis.imshow(image)
                vis.grasp(grasp, scale=1.5)
                vis.title('K=%d: d=%.3f, q=%.3f' %(i, depth, q_value))
            vis.show()

        # select grasp
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        image = DepthImage(image_tensor[index,...])
        pose = pose_tensor[index,...]
        depth = pose[0]
        if self.config['vis']['grasp_plan']:
            scale_factor = float(self.gqcnn.im_width) / float(self._crop_width)
            scaled_camera_intr = camera_intr.resize(scale_factor)
            grasp = Grasp2D(Point(image.center), 0.0, pose[0],
                            width=self._gripper_width,
                            camera_intr=scaled_camera_intr)
            vis.figure()
            vis.imshow(image)
            vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(depth, q_value))
            vis.show()

        # return action
        return ParallelJawGrasp(grasp, q_value, image)

class CrossEntropyAntipodalGraspingPolicy(GraspingPolicy):
    """ Optimizes a set of antipodal grasp candidates in image space using the 
    cross entropy method:
    (1) sample an initial set of candidates
    (2) sort the candidates
    (3) fit a GMM to the top P%
    (4) re-sample grasps from the distribution
    (5) repeat steps 2-4 for K iters
    (6) return the best candidate from the final sample set

    Notes
    -----
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    num_seed_samples : int
        number of candidate to sample in the initial set
    num_gmm_samples : int
        number of candidates to sample on each resampling from the GMMs
    num_iters : int
        number of sample-and-refit iterations of CEM
    gmm_refit_p : float
        top p-% of grasps used for refitting
    gmm_component_frac : float
        percentage of the elite set size used to determine number of GMM components
    gmm_reg_covar : float
        regularization parameters for GMM covariance matrix, enforces diversity of fitted distributions
    deterministic : bool, optional
        whether to set the random seed to enforce deterministic behavior
    gripper_width : float, optional
        width of the gripper in meters
    gripper_name : str, optional
        name of the gripper
    """
    def __init__(self, config):
        GraspingPolicy.__init__(self, config)
        CrossEntropyAntipodalGraspingPolicy._parse_config(self)

    def _parse_config(self):
        """ Parses the parameters of the policy. """
        # cross entropy method parameters
        self._num_seed_samples = self.config['num_seed_samples']
        self._num_gmm_samples = self.config['num_gmm_samples']
        self._num_iters = self.config['num_iters']
        self._gmm_refit_p = self.config['gmm_refit_p']
        self._gmm_component_frac = self.config['gmm_component_frac']
        self._gmm_reg_covar = self.config['gmm_reg_covar']

        # gripper parameters
        self._seed = None
        if self.config['deterministic']:
            self._seed = SEED
        self._gripper_width = np.inf
        if 'gripper_width' in self.config.keys():
            self._gripper_width = self.config['gripper_width']

    def select(self, grasps, q_value):
        """ Selects the grasp with the highest probability of success.
        Can override for alternate policies (e.g. epsilon greedy).
        """
        # sort
        num_grasps = len(grasps)
        if num_grasps == 0:
            raise ValueError('Zero grasps')
        grasps_and_predictions = zip(np.arange(num_grasps), q_value)
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
                                            self._num_seed_samples,
                                            segmask=segmask,
                                            visualize=self.config['vis']['grasp_sampling'],
                                            seed=self._seed)
        num_grasps = len(grasps)

        if num_grasps == 0:
            logging.warning('No valid grasps could be found')
            raise NoValidGraspsException()

        # form tensors
        image_tensor, pose_tensor = self.grasps_to_tensors(grasps, state)
        if self.config['vis']['tf_images']:
            # read vis params
            k = self.config['vis']['k']
            d = utils.sqrt_ceil(k)

            # display grasp transformed images
            vis.figure(size=(FIGSIZE,FIGSIZE))
            for i, image_tf in enumerate(image_tensor[:k,...]):
                depth = pose_tensor[i][0]
                vis.subplot(d,d,i+1)
                vis.imshow(DepthImage(image_tf))
                vis.title('Image %d: d=%.3f' %(i, depth))

            # display grasp transformed images
            vis.figure(size=(FIGSIZE,FIGSIZE))
            for i in range(d):
                image_tf = image_tensor[i,...]
                depth = pose_tensor[i][0]
                grasp = grasps[i]

                vis.subplot(d,2,2*i+1)
                vis.imshow(rgbd_im.depth)
                vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
                vis.title('Grasp %d: d=%.3f' %(i, depth))

                vis.subplot(d,2,2*i+2)
                vis.imshow(DepthImage(image_tf))
                vis.title('TF image %d: d=%.3f' %(i, depth))
            vis.show()

        # iteratively refit and sample
        for j in range(self._num_iters):
            logging.debug('CEM iter %d' %(j))

            # predict grasps
            predict_start = time()
            output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
            q_values = output_arr[:,-1]
            logging.debug('Prediction took %.3f sec' %(time()-predict_start))

            # sort grasps
            q_values_and_indices = zip(q_values, np.arange(num_grasps))
            q_values_and_indices.sort(key = lambda x : x[0], reverse=True)

            if self.config['vis']['grasp_candidates']:
                # display each grasp on the original image, colored by predicted success
                vis.figure(size=(FIGSIZE,FIGSIZE))
                vis.imshow(rgbd_im.depth)
                for grasp, q in zip(grasps, q_values):
                    vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title('Sampled grasps iter %d' %(j))
                vis.show()

            if self.config['vis']['grasp_ranking']:
                # read vis params
                k = self.config['vis']['k']
                d = utils.sqrt_ceil(k)

                # form camera intr for the thumbnail (to compute gripper width)
                scale_factor = float(self.gqcnn.im_width) / float(self._crop_width)
                scaled_camera_intr = camera_intr.resize(scale_factor)

                vis.figure(size=(FIGSIZE,FIGSIZE))
                for i, p in enumerate(q_values_and_indices[:k]):
                    # read stats for grasp
                    q_value = p[0]
                    ind = p[1]
                    depth = pose_tensor[ind][0]
                    image = DepthImage(image_tensor[ind,...])
                    grasp = Grasp2D(Point(image.center), 0.0, depth,
                                    width=self._gripper_width,
                                    camera_intr=scaled_camera_intr)

                    # plot
                    vis.subplot(d,d,i+1)
                    vis.imshow(image)
                    vis.grasp(grasp, scale=1.5)
                    vis.title('K=%d: d=%.3f, q=%.3f' %(i, depth, q_value))
                vis.show()

            # fit elite set
            num_refit = max(int(np.ceil(self._gmm_refit_p * num_grasps)), 1)
            elite_q_values = [i[0] for i in q_values_and_indices[:num_refit]]
            elite_grasp_indices = [i[1] for i in q_values_and_indices[:num_refit]]
            elite_grasps = [grasps[i] for i in elite_grasp_indices]
            elite_grasp_arr = np.array([g.feature_vec for g in elite_grasps])


            if self.config['vis']['elite_grasps']:
                # display each grasp on the original image, colored by predicted success
                vis.figure(size=(FIGSIZE,FIGSIZE))
                vis.imshow(rgbd_im.depth)
                for grasp, q in zip(elite_grasps, elite_q_values):
                    vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title('Elite grasps iter %d' %(j))
                vis.show()

            # normalize elite set
            elite_grasp_mean = np.mean(elite_grasp_arr, axis=0)
            elite_grasp_std = np.std(elite_grasp_arr, axis=0)
            elite_grasp_std[elite_grasp_std == 0] = 1.0
            elite_grasp_arr = (elite_grasp_arr - elite_grasp_mean) / elite_grasp_std

            # fit a GMM to the top samples
            num_components = max(int(np.ceil(self._gmm_component_frac * num_refit)), 1)
            uniform_weights = (1.0 / num_components) * np.ones(num_components)
            gmm = GaussianMixture(n_components=num_components,
                                  weights_init=uniform_weights,
                                  reg_covar=self._gmm_reg_covar)
            train_start = time()

            gmm.fit(elite_grasp_arr)
            train_duration = time() - train_start
            logging.debug('GMM fitting with %d components took %.3f sec' %(num_components, train_duration))

            # sample the next grasps
            sample_start = time()
            grasp_vecs, _ = gmm.sample(n_samples=self._num_gmm_samples)
            grasp_vecs = elite_grasp_std * grasp_vecs + elite_grasp_mean
            sample_duration = time() - sample_start
            logging.debug('GMM sampling took %.3f sec' %(sample_duration))

            # convert features to grasps
            grasps = []
            for grasp_vec in grasp_vecs:
                grasps.append(Grasp2D.from_feature_vec(grasp_vec,
                                                       width=self._gripper_width,
                                                       camera_intr=camera_intr))
            num_grasps = len(grasps)
            if num_grasps == 0:
                logging.warning('No valid grasps could be found')
                raise NoValidGraspsException()

            # form tensors
            image_tensor, pose_tensor = self.grasps_to_tensors(grasps, state)
            if self.config['vis']['tf_images']:
                # read vis params
                k = self.config['vis']['k']
                d = utils.sqrt_ceil(k)

                # display grasp transformed images
                vis.figure(size=(FIGSIZE,FIGSIZE))
                for i, image_tf in enumerate(image_tensor[:k,...]):
                    depth = pose_tensor[i][0]
                    vis.subplot(d,d,i+1)
                    vis.imshow(DepthImage(image_tf))
                    vis.title('Image %d: d=%.3f' %(i, depth))
                vis.show()
          
        # predict final set of grasps
        predict_start = time()
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
        q_values = output_arr[:,-1]
        logging.debug('Final prediction took %.3f sec' %(time()-predict_start))

        if self.config['vis']['grasp_candidates']:
            # display each grasp on the original image, colored by predicted success
            vis.figure(size=(FIGSIZE,FIGSIZE))
            vis.imshow(rgbd_im.depth)
            for grasp, q in zip(grasps, q_values):
                vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title('Final sampled grasps')
            vis.show()

        # select grasp
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        image = DepthImage(image_tensor[index,...])
        pose = pose_tensor[index,...]
        depth = pose[0]
        if self.config['vis']['grasp_plan']:
            scale_factor = float(self.gqcnn.im_width) / float(self._crop_width)
            scaled_camera_intr = camera_intr.resize(scale_factor)
            grasp = Grasp2D(Point(image.center), 0.0, pose[0],
                            width=self._gripper_width,
                            camera_intr=scaled_camera_intr)
            vis.figure()
            vis.imshow(image)
            vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(depth, q_value))
            vis.show()

        # return action
        return ParallelJawGrasp(grasp, q_value, image)
        
class QFunctionAntipodalGraspingPolicy(CrossEntropyAntipodalGraspingPolicy):
    """ Optimizes a set of antipodal grasp candidates in image space using the 
    cross entropy method with a GQ-CNN that estimates the Q-function
    for use in Q-learning.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    reinit_pc1 : bool
        whether or not to reinitialize the pc1 layer of the GQ-CNN
    reinit_fc3: bool
        whether or not to reinitialize the fc3 layer of the GQ-CNN
    reinit_fc4: bool
        whether or not to reinitialize the fc4 layer of the GQ-CNN
    reinit_fc5: bool
        whether or not to reinitialize the fc5 layer of the GQ-CNN
    num_seed_samples : int
        number of candidate to sample in the initial set
    num_gmm_samples : int
        number of candidates to sample on each resampling from the GMMs
    num_iters : int
        number of sample-and-refit iterations of CEM
    gmm_refit_p : float
        top p-% of grasps used for refitting
    gmm_component_frac : float
        percentage of the elite set size used to determine number of GMM components
    gmm_reg_covar : float
        regularization parameters for GMM covariance matrix, enforces diversity of fitted distributions
    deterministic : bool, optional
        whether to set the random seed to enforce deterministic behavior
    gripper_width : float, optional
        width of the gripper in meters
    gripper_name : str, optional
        name of the gripper
    """
    def __init__(self, config):
        CrossEntropyAntipodalGraspingPolicy.__init__(self, config)
        QFunctionAntipodalGraspingPolicy._parse_config(self)
        self._setup_gqcnn()

    def _parse_config(self):
        """ Parses the parameters of the policy. """
        self._reinit_pc1 = self.config['reinit_pc1']
        self._reinit_fc3 = self.config['reinit_fc3']
        self._reinit_fc4 = self.config['reinit_fc4']
        self._reinit_fc5 = self.config['reinit_fc5']

    def _setup_gqcnn(self):
        """ Sets up the GQ-CNN. """
        # close existing session (from superclass initializer)
        self.gqcnn.close_session()

        # check valid output size
        if self.gqcnn.fc5_out_size != 1 and not self._reinit_fc5:
            raise ValueError('Q function must return scalar values')

        # reinitialize layers
        if self._reinit_fc5:
            self.gqcnn.fc5_out_size = 1

        # TODO: implement reinitialization of pc0
        self.gqcnn.reinitialize_layers(self._reinit_fc3,
                                       self._reinit_fc4,
                                       self._reinit_fc5)
        self.gqcnn.initialize_network(add_softmax=False)
        
class EpsilonGreedyQFunctionAntipodalGraspingPolicy(QFunctionAntipodalGraspingPolicy):
    """ Optimizes a set of antipodal grasp candidates in image space 
    using the cross entropy method with a GQ-CNN that estimates the
    Q-function for use in Q-learning, and chooses a random antipodal
    grasp with probability epsilon.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    epsilon : float
    """
    def __init__(self, config):
        QFunctionAntipodalGraspingPolicy.__init__(self, config)
        self._parse_config()

    def _parse_config(self):
        """ Parses the parameters of the policy. """
        self._epsilon = self.config['epsilon']

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = val

    def greedy_action(self, state):
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
        return CrossEntropyAntipodalGraspingPolicy.action(self, state)
    
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
        # take the greedy action with prob 1 - epsilon
        if np.random.rand() > self.epsilon:
            logging.debug('Taking greedy action')
            return CrossEntropyAntipodalGraspingPolicy.action(self, state)

        # otherwise take a random action
        logging.debug('Taking random action')

        # check valid input
        if not isinstance(state, RgbdImageState):
            raise ValueError('Must provide an RGB-D image state.')

        # parse state
        rgbd_im = state.rgbd_im
        camera_intr = state.camera_intr
        segmask = state.segmask

        # sample random antipodal grasps
        grasps = self._grasp_sampler.sample(rgbd_im, camera_intr,
                                            self._num_seed_samples,
                                            segmask=segmask,
                                            visualize=self.config['vis']['grasp_sampling'],
                                            seed=self._seed)
        
        num_grasps = len(grasps)
        if num_grasps == 0:
            logging.warning('No valid grasps could be found')
            raise NoValidGraspsException()

        # choose a grasp uniformly at random
        grasp_ind = np.random.choice(num_grasps, size=1)[0]
        grasp = grasps[grasp_ind]
        depth = grasp.depth

        # create transformed image
        image_tensor, pose_tensor = self.grasps_to_tensors([grasp], state)
        image = DepthImage(image_tensor[0,...])

        # predict prob success
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
        q_value = output_arr[0,-1]
        
        # visualize planned grasp
        if self.config['vis']['grasp_plan']:
            scale_factor = float(self.gqcnn.im_width) / float(self._crop_width)
            scaled_camera_intr = camera_intr.resize(scale_factor)
            vis_grasp = Grasp2D(Point(image.center), 0.0, depth,
                                width=self._gripper_width,
                                camera_intr=scaled_camera_intr)
            vis.figure()
            vis.imshow(image)
            vis.grasp(vis_grasp, scale=1.5, show_center=False, show_axis=True)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(depth, q_value))
            vis.show()

        # return action
        return ParallelJawGrasp(grasp, q_value, image)

class FullyConvolutionalAngularPolicyTopK(object):
    ''' Grasp sampling policy using full-convolutional angular GQ-CNN network '''
    def __init__(self, cfg):
        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._width = self._cfg['width']
        self._use_segmask = self._cfg['use_segmask']

        self._gqcnn_dir = self._cfg['gqcnn_model']
        self._gqcnn_backend = self._cfg['gqcnn_backend']
        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']
        self._fully_conv_config = self._cfg['fully_conv_gqcnn_config']

        self._vis_config = self._cfg['policy_vis']
        self._top_k_to_vis = self._vis_config['top_k']
        self._vis_top_k = self._vis_config['vis_top_k']
        self._vis_3d = self._vis_config['vis_3d']
               
        # initialize gqcnn
        self._gqcnn = get_gqcnn_model(backend=self._gqcnn_backend).load(self._gqcnn_dir, fully_conv_config=self._fully_conv_config)
        self._gqcnn.open_session()

    def __del__(self):
        # close gqcnn session
        try:
            self._gqcnn.close_session()
        except:
            pass

    def __call__(self, state):
        return self._action(state)

    def action(self, state):
        return self._action(state)

    def _action(self, state, k=1):
        # extract raw depth data matrix
        rgbd_im = state.rgbd_im
        d_im = rgbd_im.depth
        raw_d = d_im._data # TODO: Access this properly

        policy_start_time = time()
        
        # sample depths
        max_d = np.max(raw_d)
        min_d = np.min(raw_d)
        depth_bin_width = (max_d - min_d) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_d + (i * depth_bin_width + depth_bin_width / 2)

        # predict
        images = np.tile(np.asarray([raw_d]), (self._num_depth_bins, 1, 1, 1))
        use_opt = self._num_depth_bins > self._gqcnn.batch_size
        if use_opt:
            unique_im_map = np.zeros((self._num_depth_bins,), dtype=np.int32)
            preds = self._gqcnn.predict(images, depths, unique_im_map=unique_im_map)
        else:
            pred_start_time = time()
            preds = self._gqcnn.predict(images, depths)
            logging.info('Inference took {} seconds'.format(time() - pred_start_time))
        success_ind = np.arange(1, preds.shape[-1], 2)

        # if we want to visualize the top k, then extract those indices, else just get the number specified by the policy
        top_k = self._top_k_to_vis if self._vis_top_k else k

        # get indices of top k predictions
        preds_success_only = preds[:, :, :, success_ind]

#        for i in range(self._num_depth_bins):
#            logging.info('Max at slice {} is {}'.format(i, np.max(preds_success_only[i])))
#            logging.info('Depth at slice {} is {}'.format(i, depths[i, 0]))

        if self._use_segmask:
            raw_segmask = state.segmask.raw_data
            preds_success_only_new = np.zeros_like(preds_success_only)
            raw_segmask_cropped = raw_segmask[self._gqcnn_recep_h / 2:raw_segmask.shape[0] - self._gqcnn_recep_h / 2, self._gqcnn_recep_w / 2:raw_segmask.shape[1] - self._gqcnn_recep_w / 2, 0]
            raw_segmask_downsamp = raw_segmask_cropped[::self._gqcnn_stride, ::self._gqcnn_stride]
            if raw_segmask_downsamp.shape[0] != preds_success_only.shape[1]:
                raw_segmask_downsamp_new = np.zeros(preds_success_only.shape[1:3])
                raw_segmask_downsamp_new[:raw_segmask_downsamp.shape[0], :raw_segmask_downsamp.shape[1]] = raw_segmask_downsamp
                raw_segmask_downsamp = raw_segmask_downsamp_new
            nonzero_mask_ind = np.where(raw_segmask_downsamp > 0)
            preds_success_only_new[:, nonzero_mask_ind[0], nonzero_mask_ind[1]] = preds_success_only[:, nonzero_mask_ind[0], nonzero_mask_ind[1]]
            preds_success_only = preds_success_only_new       

        preds_success_only_flat = np.ravel(preds_success_only)
        top_k_pred_ind_flat = np.argpartition(preds_success_only_flat, -1 * top_k)[-1 * top_k:]
        top_k_pred_ind = np.zeros((top_k, len(preds.shape)), dtype=np.int32)
        im_width = preds_success_only.shape[2]
        im_height = preds_success_only.shape[1]
        num_angular_bins = preds_success_only.shape[3]
        for idx in range(top_k):
            top_k_pred_ind[idx, 0] = top_k_pred_ind_flat[idx] // (im_width * im_height * num_angular_bins) 
            top_k_pred_ind[idx, 1] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
            top_k_pred_ind[idx, 2] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) // num_angular_bins
            top_k_pred_ind[idx, 3] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) % num_angular_bins

#        for i in range(self._num_depth_bins):
#            h = top_k_pred_ind[i, 1]
#            w = top_k_pred_ind[i, 2]
#            logging.info('Original depth at point ({}, {}) is {}'.format(w, h, images[i, h, w]))

        # generate grasps
        grasps = []
        ang_bin_width = PI / preds_success_only.shape[-1]
        for i in range(top_k):
            im_idx = top_k_pred_ind[i, 0]
            h_idx = top_k_pred_ind[i, 1]
            w_idx = top_k_pred_ind[i, 2]
            ang_idx = top_k_pred_ind[i, 3]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            ang = PI / 2 - (ang_idx * ang_bin_width + ang_bin_width / 2)
            depth = depths[im_idx]
            grasp = Grasp2D(center, ang, depth, width=self._width, camera_intr=state.camera_intr)
            pj_grasp = ParallelJawGrasp(grasp, preds_success_only[im_idx, h_idx, w_idx, ang_idx], DepthImage(images[im_idx]))
            grasps.append(pj_grasp)

        logging.info('Policy took {} seconds'.format(time() - policy_start_time))            
            
        if self._vis_top_k:
            # visualize 3D
            if self._vis_3d:
                logging.info('Generating 3D Visualization...')
                vis3d.figure()
                vis3d.points(state.camera_intr.deproject(d_im),
                             scale=0.001)
                for i in range(top_k):
                    logging.info('Visualizing top k grasp {} of {}'.format(i, top_k))
                    logging.info('Depth: {}'.format(grasps[i].grasp.depth))
                    vis3d.grasp(ParallelJawPtGrasp3D.grasp_from_pose(grasps[i].grasp.pose()), color=(1, 0, 0))
                vis3d.show()
                
            #visualize 2D
            logging.info('Generating 2D visualization...')
            vis.figure()
            vis.imshow(d_im)
            im_tensor = np.zeros((top_k, 96, 96, 1))
            pose_tensor = np.zeros((top_k, 6))
            metric_tensor = np.zeros((top_k,))
            for i in range(top_k):
                logging.info('Depth: {}'.format(grasps[i].grasp.depth))
                vis.grasp(grasps[i].grasp, scale=self._vis_config['scale'], show_axis=self._vis_config['show_axis'], color=plt.cm.RdYlGn(grasps[i].q_value))
                im_tensor[i] = d_im.align(1.0, [grasps[i].grasp.center.x, grasps[i].grasp.center.y], 0.0, 96, 96)._data
                pose_tensor[i] = np.asarray([0, 0, grasps[i].grasp.depth, 0, 0, 0])
                metric_tensor[i] = 0.0
            vis.show()
            """
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/depth_ims_tf_table_00000', im_tensor)
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/hand_poses_00000', pose_tensor)
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/robust_wrench_resistance_00000', metric_tensor)
            """
        return grasps[-1] if k == 1 else grasps[-(k+1):]

    def action_set(self, state, num_actions):
        return [pj_grasp.grasp for pj_grasp in self._action(state, k=num_actions)]

class FullyConvolutionalAngularPolicyImportance(object):
    ''' Grasp sampling policy using full-convolutional angular GQ-CNN network '''
    def __init__(self, cfg):
        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._width = self._cfg['width']
        self._use_segmask = self._cfg['use_segmask']

        self._gqcnn_dir = self._cfg['gqcnn_model']
        self._gqcnn_backend = self._cfg['gqcnn_backend']
        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']
        self._fully_conv_config = self._cfg['fully_conv_gqcnn_config']

        self._vis_config = self._cfg['policy_vis']
        self._top_k_to_vis = self._vis_config['top_k']
        self._vis_top_k = self._vis_config['vis_top_k']
        self._vis_3d = self._vis_config['vis_3d']
        self._vis_segmask_crop = self._vis_config['vis_segmask_crop']
               
        # initialize gqcnn
        self._gqcnn = get_gqcnn_model(backend=self._gqcnn_backend).load(self._gqcnn_dir, fully_conv_config=self._fully_conv_config)
        self._gqcnn.open_session()

    def __del__(self):
        # close gqcnn session
        try:
            self._gqcnn.close_session()
        except:
            pass

    def __call__(self, state):
        return self._action(state)

    def action(self, state):
        return self._action(state)

    def _action(self, state, k=1):
        # extract raw depth data matrix
        rgbd_im = state.rgbd_im
        d_im = rgbd_im.depth
        raw_d = d_im._data # TODO: Access this properly

#        if self._use_segmask:
#            segmask = state.segmask
#            raw_segmask = segmask.raw_data
#            nonzero_indices = np.where(raw_segmask > 0)
#            crop_min_x = np.min(nonzero_indices[1])
#            crop_min_y = np.min(nonzero_indices[0])
#            crop_max_x = np.max(nonzero_indices[1])
#            crop_max_y = np.max(nonzero_indices[0])
#            if self._vis_segmask_crop:
#                vis.figure()
#                vis.subplot(131)
#                vis.imshow(d_im)
#                vis.subplot(132)
#                vis.imshow(segmask)
#                vis.subplot(133)
#                vis.imshow(d_im.crop(crop_max_y - crop_min_y, crop_max_x - crop_min_x, center_i=((crop_min_y + crop_max_y) / 2), center_j=((crop_min_x + crop_max_x) / 2)))
#                vis.show()
#            crop_min_x_scaled = (crop_min_x - self._gqcnn_recep_w / 2) / self._gqcnn_stride
#            crop_min_y_scaled = (crop_min_y - self._gqcnn_recep_h / 2) / self._gqcnn_stride
#            crop_max_x_scaled = (crop_max_x - self._gqcnn_recep_w / 2) / self._gqcnn_stride
#            crop_max_y_scaled = (crop_max_y - self._gqcnn_recep_h / 2) / self._gqcnn_stride
         
        # sample depths
        max_d = np.max(raw_d)
        min_d = np.min(raw_d)
        depth_bin_width = (max_d - min_d) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_d + (i * depth_bin_width + depth_bin_width / 2)

        # predict
        images = np.tile(np.asarray([raw_d]), (self._num_depth_bins, 1, 1, 1))
        use_opt = self._num_depth_bins > self._gqcnn.batch_size
        if use_opt:
            unique_im_map = np.zeros((self._num_depth_bins,), dtype=np.int32)
            preds = self._gqcnn.predict(images, depths, unique_im_map=unique_im_map)
        else:
            pred_start_time = time()
            preds = self._gqcnn.predict(images, depths)
            logging.info('Inference took {} seconds'.format(time() - pred_start_time))
        success_ind = np.arange(1, preds.shape[-1], 2)

        # if we want to visualize the top k, then extract those indices, else just get the number specified by the policy
        top_k = self._top_k_to_vis if self._vis_top_k else k

        # get indices of top k predictions
        preds_success_only = preds[:, :, :, success_ind]

#        for i in range(self._num_depth_bins):
#            logging.info('Max at slice {} is {}'.format(i, np.max(preds_success_only[i])))
#            logging.info('Depth at slice {} is {}'.format(i, depths[i, 0]))

        if self._use_segmask:
            raw_segmask = state.segmask.raw_data
            preds_success_only_new = np.zeros_like(preds_success_only)
            raw_segmask_cropped = raw_segmask[self._gqcnn_recep_h / 2:raw_segmask.shape[0] - self._gqcnn_recep_h / 2, self._gqcnn_recep_w / 2:raw_segmask.shape[1] - self._gqcnn_recep_w / 2, 0]
            raw_segmask_downsamp = raw_segmask_cropped[::self._gqcnn_stride, ::self._gqcnn_stride]
            if raw_segmask_downsamp.shape[0] != preds_success_only.shape[1]:
                raw_segmask_downsamp_new = np.zeros(preds_success_only.shape[1:3])
                raw_segmask_downsamp_new[:raw_segmask_downsamp.shape[0], :raw_segmask_downsamp.shape[1]] = raw_segmask_downsamp
                raw_segmask_downsamp = raw_segmask_downsamp_new
            nonzero_mask_ind = np.where(raw_segmask_downsamp > 0)
            preds_success_only_new[:, nonzero_mask_ind[0], nonzero_mask_ind[1]] = preds_success_only[:, nonzero_mask_ind[0], nonzero_mask_ind[1]]
            preds_success_only = preds_success_only_new       

        preds_success_only_flat = np.ravel(preds_success_only)
        p_importance = preds_success_only_flat / np.sum(preds_success_only_flat)
        x = np.random.multinomial(top_k, p_importance)
        top_k_pred_ind_flat_rolled = np.where(x > 0)[0]
        top_k_pred_ind_flat = np.repeat(top_k_pred_ind_flat_rolled, x[top_k_pred_ind_flat_rolled])
        top_k_pred_ind = np.zeros((top_k, len(preds.shape)), dtype=np.int32)
        im_width = preds_success_only.shape[2]
        im_height = preds_success_only.shape[1]
        num_angular_bins = preds_success_only.shape[3]
        for idx in range(top_k):
            top_k_pred_ind[idx, 0] = top_k_pred_ind_flat[idx] // (im_width * im_height * num_angular_bins) 
            top_k_pred_ind[idx, 1] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
            top_k_pred_ind[idx, 2] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) // num_angular_bins
            top_k_pred_ind[idx, 3] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) % num_angular_bins

#        for i in range(self._num_depth_bins):
#            h = top_k_pred_ind[i, 1]
#            w = top_k_pred_ind[i, 2]
#            logging.info('Original depth at point ({}, {}) is {}'.format(w, h, images[i, h, w]))

        # generate grasps
        grasps = []
        ang_bin_width = PI / preds_success_only.shape[-1]
        for i in range(top_k):
            im_idx = top_k_pred_ind[i, 0]
            h_idx = top_k_pred_ind[i, 1]
            w_idx = top_k_pred_ind[i, 2]
            ang_idx = top_k_pred_ind[i, 3]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            ang = PI / 2 - (ang_idx * ang_bin_width + ang_bin_width / 2)
            depth = depths[im_idx]
            grasp = Grasp2D(center, ang, depth, width=self._width, camera_intr=state.camera_intr)
            pj_grasp = ParallelJawGrasp(grasp, preds_success_only[im_idx, h_idx, w_idx, ang_idx], DepthImage(images[im_idx]))
            grasps.append(pj_grasp)

        if self._vis_top_k:
            # visualize 3D
            if self._vis_3d:
                logging.info('Generating 3D Visualization...')
                vis3d.figure()
                for i in range(top_k):
                    logging.info('Visualizing top k grasp {} of {}'.format(i, top_k))
                    vis3d.clf()
                    vis3d.points(state.camera_intr.deproject(d_im))
                    logging.info('Depth: {}'.format(grasps[i].grasp.depth))
                    vis3d.grasp(ParallelJawPtGrasp3D.grasp_from_pose(grasps[i].grasp.pose()), color=(1, 0, 0))
                    vis3d.show()
                
            #visualize 2D
            logging.info('Generating 2D visualization...')
            vis.figure()
            vis.imshow(d_im)
            im_tensor = np.zeros((50, 96, 96, 1))
            pose_tensor = np.zeros((50, 6))
            metric_tensor = np.zeros((50,))
            for i in range(top_k):
                logging.info('Depth: {}'.format(grasps[i].grasp.depth))
                vis.grasp(grasps[i].grasp, scale=self._vis_config['scale'], show_axis=self._vis_config['show_axis'], color=plt.cm.RdYlGn(grasps[i].q_value))
                im_tensor[i] = d_im.align(1.0, [grasps[i].grasp.center.x, grasps[i].grasp.center.y], 0.0, 96, 96)._data
                pose_tensor[i] = np.asarray([0, 0, grasps[i].grasp.depth, 0, 0, 0])
                metric_tensor[i] = 0.0
            vis.show()
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/depth_ims_tf_table_00000', im_tensor)
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/hand_poses_00000', pose_tensor)
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/robust_wrench_resistance_00000', metric_tensor)
        return grasps[-1] if k == 1 else grasps[-(k+1):]

    def action_set(self, state, num_actions):
        return [pj_grasp.grasp for pj_grasp in self._action(state, k=num_actions)]

class FullyConvolutionalAngularPolicyUniform(object):
    ''' Grasp sampling policy using fully-convolutional angular GQ-CNN network '''
    def __init__(self, cfg):
        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._width = self._cfg['width']
        self._use_segmask = self._cfg['use_segmask']

        self._gqcnn_dir = self._cfg['gqcnn_model']
        self._gqcnn_backend = self._cfg['gqcnn_backend']
        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']
        self._fully_conv_config = self._cfg['fully_conv_gqcnn_config']

        self._vis_config = self._cfg['policy_vis']
        self._vis_depth_im = False
        if 'vis_depth_im' in self._vis_config.keys():
            self._vis_depth_im = self._vis_config['vis_depth_im']
        self._top_k_to_vis = self._vis_config['top_k']
        self._vis_top_k = self._vis_config['vis_top_k']
        self._vis_3d = self._vis_config['vis_3d']
        self._vis_segmask_crop = self._vis_config['vis_segmask_crop']
               
        # initialize gqcnn
        self._gqcnn = get_gqcnn_model(backend=self._gqcnn_backend).load(self._gqcnn_dir, fully_conv_config=self._fully_conv_config)
        self._gqcnn.open_session()

    def __del__(self):
        # close gqcnn session
        try:
            self._gqcnn.close_session()
        except:
            pass

    def __call__(self, state):
        return self._action(state)

    def action(self, state):
        return self._action(state)

    def _action(self, state, k=1):
        # extract raw depth data matrix
        rgbd_im = state.rgbd_im
        d_im = rgbd_im.depth
        raw_d = d_im._data # TODO: Access this properly

        # visualize depth image
        if self._vis_depth_im:
            vis.figure()
            vis.imshow(d_im)
            vis.show()

#        if self._use_segmask:
#            segmask = state.segmask
#            raw_segmask = segmask.raw_data
#            nonzero_indices = np.where(raw_segmask > 0)
#            crop_min_x = np.min(nonzero_indices[1])
#            crop_min_y = np.min(nonzero_indices[0])
#            crop_max_x = np.max(nonzero_indices[1])
#            crop_max_y = np.max(nonzero_indices[0])
#            if self._vis_segmask_crop:
#                vis.figure()
#                vis.subplot(131)
#                vis.imshow(d_im)
#                vis.subplot(132)
#                vis.imshow(segmask)
#                vis.subplot(133)
#                vis.imshow(d_im.crop(crop_max_y - crop_min_y, crop_max_x - crop_min_x, center_i=((crop_min_y + crop_max_y) / 2), center_j=((crop_min_x + crop_max_x) / 2)))
#                vis.show()
#            crop_min_x_scaled = (crop_min_x - self._gqcnn_recep_w / 2) / self._gqcnn_stride
#            crop_min_y_scaled = (crop_min_y - self._gqcnn_recep_h / 2) / self._gqcnn_stride
#            crop_max_x_scaled = (crop_max_x - self._gqcnn_recep_w / 2) / self._gqcnn_stride
#            crop_max_y_scaled = (crop_max_y - self._gqcnn_recep_h / 2) / self._gqcnn_stride
         
        # sample depths
        max_d = np.max(raw_d)
        raw_d_seg_min = np.ones_like(raw_d)
        raw_d_seg_min[np.where(state.segmask.raw_data > 0)] = raw_d[np.where(state.segmask.raw_data > 0)] # remove the bin from the depth image
        min_d = np.min(raw_d_seg_min)
        depth_bin_width = (max_d - min_d) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_d + (i * depth_bin_width + depth_bin_width / 2)

        # predict
        images = np.tile(np.asarray([raw_d]), (self._num_depth_bins, 1, 1, 1))
        use_opt = self._num_depth_bins > self._gqcnn.batch_size
        if use_opt:
            unique_im_map = np.zeros((self._num_depth_bins,), dtype=np.int32)
            preds = self._gqcnn.predict(images, depths, unique_im_map=unique_im_map)
        else:
            pred_start_time = time()
            preds = self._gqcnn.predict(images, depths)
            logging.info('Inference took {} seconds'.format(time() - pred_start_time))
        success_ind = np.arange(1, preds.shape[-1], 2)

        # if we want to visualize the top k, then extract those indices, else just get the number specified by the policy
        top_k = self._top_k_to_vis if self._vis_top_k else k

        # get indices of top k predictions
        preds_success_only = preds[:, :, :, success_ind]

#        for i in range(self._num_depth_bins):
#            logging.info('Max at slice {} is {}'.format(i, np.max(preds_success_only[i])))
#            logging.info('Depth at slice {} is {}'.format(i, depths[i, 0]))

        if self._use_segmask:
            raw_segmask = state.segmask.raw_data
            preds_success_only_new = np.zeros_like(preds_success_only)
            raw_segmask_cropped = raw_segmask[self._gqcnn_recep_h / 2:raw_segmask.shape[0] - self._gqcnn_recep_h / 2, self._gqcnn_recep_w / 2:raw_segmask.shape[1] - self._gqcnn_recep_w / 2, 0]
            raw_segmask_downsamp = raw_segmask_cropped[::self._gqcnn_stride, ::self._gqcnn_stride]
            if raw_segmask_downsamp.shape[0] != preds_success_only.shape[1]:
                raw_segmask_downsamp_new = np.zeros(preds_success_only.shape[1:3])
                raw_segmask_downsamp_new[:raw_segmask_downsamp.shape[0], :raw_segmask_downsamp.shape[1]] = raw_segmask_downsamp
                raw_segmask_downsamp = raw_segmask_downsamp_new
            nonzero_mask_ind = np.where(raw_segmask_downsamp > 0)
            preds_success_only_new[:, nonzero_mask_ind[0], nonzero_mask_ind[1]] = preds_success_only[:, nonzero_mask_ind[0], nonzero_mask_ind[1]]
            preds_success_only = preds_success_only_new       

        preds_success_only_flat = np.ravel(preds_success_only)
        top_k_pred_ind_flat = np.random.choice(np.where(preds_success_only_flat > 0)[0], size=top_k)
        top_k_pred_ind = np.zeros((top_k, len(preds.shape)), dtype=np.int32)
        im_width = preds_success_only.shape[2]
        im_height = preds_success_only.shape[1]
        num_angular_bins = preds_success_only.shape[3]
        for idx in range(top_k):
            top_k_pred_ind[idx, 0] = top_k_pred_ind_flat[idx] // (im_width * im_height * num_angular_bins) 
            top_k_pred_ind[idx, 1] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
            top_k_pred_ind[idx, 2] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) // num_angular_bins
            top_k_pred_ind[idx, 3] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) % num_angular_bins

#        for i in range(self._num_depth_bins):
#            h = top_k_pred_ind[i, 1]
#            w = top_k_pred_ind[i, 2]
#            logging.info('Original depth at point ({}, {}) is {}'.format(w, h, images[i, h, w]))

        # generate grasps
        grasps = []
        ang_bin_width = PI / preds_success_only.shape[-1]
        for i in range(top_k):
            im_idx = top_k_pred_ind[i, 0]
            h_idx = top_k_pred_ind[i, 1]
            w_idx = top_k_pred_ind[i, 2]
            ang_idx = top_k_pred_ind[i, 3]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            ang = PI / 2 - (ang_idx * ang_bin_width + ang_bin_width / 2)
            depth = depths[im_idx]
            grasp = Grasp2D(center, ang, depth, width=self._width, camera_intr=state.camera_intr)
            pj_grasp = ParallelJawGrasp(grasp, preds_success_only[im_idx, h_idx, w_idx, ang_idx], DepthImage(images[im_idx]))
            grasps.append(pj_grasp)

        if self._vis_top_k:
            # visualize 3D
            if self._vis_3d:
                logging.info('Generating 3D Visualization...')
                vis3d.figure()
                for i in range(top_k):
                    logging.info('Visualizing top k grasp {} of {}'.format(i, top_k))
                    vis3d.clf()
                    vis3d.points(state.camera_intr.deproject(d_im))
                    logging.info('Depth: {}'.format(grasps[i].grasp.depth))
                    vis3d.grasp(ParallelJawPtGrasp3D.grasp_from_pose(grasps[i].grasp.pose()), color=(1, 0, 0))
                    vis3d.show()
                
            #visualize 2D
            logging.info('Generating 2D visualization...')
            vis.figure()
            vis.imshow(d_im)
            im_tensor = np.zeros((50, 96, 96, 1))
            pose_tensor = np.zeros((50, 6))
            metric_tensor = np.zeros((50,))
            for i in range(top_k):
                logging.info('Depth: {}'.format(grasps[i].grasp.depth))
                vis.grasp(grasps[i].grasp, scale=self._vis_config['scale'], show_axis=self._vis_config['show_axis'], color=plt.cm.RdYlGn(grasps[i].q_value))
                im_tensor[i] = d_im.align(1.0, [grasps[i].grasp.center.x, grasps[i].grasp.center.y], 0.0, 96, 96)._data
                pose_tensor[i] = np.asarray([0, 0, grasps[i].grasp.depth, 0, 0, 0])
                metric_tensor[i] = 0.0
            vis.show()
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/depth_ims_tf_table_00000', im_tensor)
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/hand_poses_00000', pose_tensor)
            np.savez_compressed('/home/vsatish/Workspace/dev/gqcnn/test_dump/robust_wrench_resistance_00000', metric_tensor)
        return grasps[-1] if k == 1 else grasps[-(k+1):]

    def action_set(self, state, num_actions):
        return [pj_grasp.grasp for pj_grasp in self._action(state, k=num_actions)]


class GraspAction(object):
    """ Action to encapsulate parallel jaw grasps.
    """
    def __init__(self, grasp, q_value, image):
        self.grasp = grasp
        self.q_value = q_value
        self.image = image

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        grasp_filename = os.path.join(save_dir, 'grasp.pkl')
        q_value_filename = os.path.join(save_dir, 'pred_robustness.pkl')
        image_filename = os.path.join(save_dir, 'tf_image.npy')
        pkl.dump(self.grasp, open(grasp_filename, 'wb'))
        pkl.dump(self.q_value, open(q_value_filename, 'wb'))
        self.image.save(image_filename)

class UniformRandomGraspingPolicy(GraspingPolicy):
    """ Returns a grasp uniformly at random. """
    def __init__(self, config):
        GraspingPolicy.__init__(self, config)
        self._num_grasp_samples = 1

    def action(self, state):
        """ Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            image to plan grasps on

        Returns
        -------
        :obj:`GraspAction`
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
                                            visualize=self.config['vis']['grasp_sampling'],
                                            seed=None)
        num_grasps = len(grasps)
        if num_grasps == 0:
            logging.warning('No valid grasps could be found')
            raise NoValidGraspsException()

        # set grasp
        grasp = grasps[0]

        # form tensors
        return GraspAction(grasp, 0.0, state.rgbd_im.depth)

class CrossEntropyRobustGraspingPolicy(GraspingPolicy):
    """ Optimizes a set of grasp candidates in image space using the 
    cross entropy method:
    (1) sample an initial set of candidates
    (2) sort the candidates
    (3) fit a GMM to the top P%
    (4) re-sample grasps from the distribution
    (5) repeat steps 2-4 for K iters
    (6) return the best candidate from the final sample set

    Parameters
    ----------
    filters : :obj:`dict` mapping names to functions
        list of functions to apply to filter invalid grasps

    Notes
    -----
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    num_seed_samples : int
        number of candidate to sample in the initial set
    num_gmm_samples : int
        number of candidates to sample on each resampling from the GMMs
    num_iters : int
        number of sample-and-refit iterations of CEM
    gmm_refit_p : float
        top p-% of grasps used for refitting
    gmm_component_frac : float
        percentage of the elite set size used to determine number of GMM components
    gmm_reg_covar : float
        regularization parameters for GMM covariance matrix, enforces diversity of fitted distributions
    deterministic : bool, optional
        whether to set the random seed to enforce deterministic behavior
    gripper_width : float, optional
        width of the gripper in meters
    """
    def __init__(self, config, filters=None):
        GraspingPolicy.__init__(self, config)
        self._parse_config()
        self._filters = filters
        
    def _parse_config(self):
        """ Parses the parameters of the policy. """
        # cross entropy method parameters
        self._num_seed_samples = self.config['num_seed_samples']
        self._num_gmm_samples = self.config['num_gmm_samples']
        self._num_iters = self.config['num_iters']
        self._gmm_refit_p = self.config['gmm_refit_p']
        self._gmm_component_frac = self.config['gmm_component_frac']
        self._gmm_reg_covar = self.config['gmm_reg_covar']

        self._max_grasps_filter = 1
        if 'max_grasps_filter' in self.config.keys():
            self._max_grasps_filter = self.config['max_grasps_filter']

        self._max_resamples_per_iteration = 100
        if 'max_resamples_per_iteration' in self.config.keys():
            self._max_resamples_per_iteration = self.config['max_resamples_per_iteration']

        self._max_approach_angle = np.inf
        if 'max_approach_angle' in self.config.keys():
            self._max_approach_angle = np.deg2rad(self.config['max_approach_angle'])
            
        # gripper parameters
        self._seed = None
        if self.config['deterministic']:
            self._seed = SEED
        self._gripper_width = np.inf
        if 'gripper_width' in self.config.keys():
            self._gripper_width = self.config['gripper_width']

        # optional, logging dir
        self._logging_dir = None
        if 'logging_dir' in self.config.keys():
            self._logging_dir = self.config['logging_dir']
            if not os.path.exists(self._logging_dir):
                os.mkdir(self._logging_dir)
            
    def select(self, grasps, q_values):
        """ Selects the grasp with the highest probability of success.
        Can override for alternate policies (e.g. epsilon greedy).
        """ 
        # sort
        logging.info('Sorting grasps')
        num_grasps = len(grasps)
        if num_grasps == 0:
            raise NoValidGraspsException('Zero grasps')
        grasps_and_predictions = zip(np.arange(num_grasps), q_values)
        grasps_and_predictions.sort(key = lambda x : x[1], reverse=True)

        # return top grasps
        if self._filters is None:
            return grasps_and_predictions[0][0]
        
        # filter grasps
        logging.info('Filtering grasps')
        i = 0
        while i < self._max_grasps_filter and i < len(grasps_and_predictions):
            index = grasps_and_predictions[i][0]
            grasp = grasps[index]
            valid = True
            for filter_name, is_valid in self._filters.iteritems():
                valid = is_valid(grasp) 
                logging.debug('Grasp {} filter {} valid: {}'.format(i, filter_name, valid))
                if not valid:
                    valid = False
                    break
            if valid:
                return index
            i += 1
        raise NoValidGraspsException('No grasps satisfied filters')

    def _action(self, state):
        """ Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            image to plan grasps on

        Returns
        -------
        :obj:`GraspAction`
            grasp to execute
        """
        # check valid input
        if not isinstance(state, RgbdImageState):
            raise ValueError('Must provide an RGB-D image state.')

        if self._logging_dir is not None:
            policy_id = utils.gen_experiment_id()
            policy_dir = os.path.join(self._logging_dir, 'policy_output_%s' %(policy_id))
            while os.path.exists(policy_dir):
                policy_id = utils.gen_experiment_id()
                policy_dir = os.path.join(self._logging_dir, 'policy_output_%s' %(policy_id))
            os.mkdir(policy_dir)
            state_dir = os.path.join(policy_dir, 'state')
            state.save(state_dir)
        
        # parse state
        seed_set_start = time()
        rgbd_im = state.rgbd_im
        depth_im = rgbd_im.depth
        camera_intr = state.camera_intr
        segmask = state.segmask
        point_cloud_im = camera_intr.deproject_to_image(depth_im)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        
        # sample grasps
        grasps = self._grasp_sampler.sample(rgbd_im, camera_intr,
                                            self._num_seed_samples,
                                            segmask=segmask,
                                            visualize=self.config['vis']['grasp_sampling'],
                                            seed=self._seed)
        num_grasps = len(grasps)
        if num_grasps == 0:
            logging.warning('No valid grasps could be found')
            raise NoValidGraspsException()

        grasp_type = 'parallel_jaw'
        if isinstance(grasps[0], SuctionPoint2D):
            grasp_type = 'suction'

        logging.info('Sampled %d grasps' %(len(grasps)))
        logging.info('Computing the seed set took %.3f sec' %(time() - seed_set_start))

        # iteratively refit and sample
        for j in range(self._num_iters):
            logging.info('CEM iter %d' %(j))

            # predict grasps
            predict_start = time()
            q_values = self._grasp_quality_fn(state, grasps, params=self._config)
            logging.info('Prediction took %.3f sec' %(time()-predict_start))

            # sort grasps
            resample_start = time()
            q_values_and_indices = zip(q_values, np.arange(num_grasps))
            q_values_and_indices.sort(key = lambda x : x[0], reverse=True)

            if self.config['vis']['grasp_candidates']:
                # display each grasp on the original image, colored by predicted success
                norm_q_values = q_values #(q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
                vis.figure(size=(FIGSIZE,FIGSIZE))
                vis.imshow(rgbd_im.depth,
                           vmin=self.config['vis']['vmin'],
                           vmax=self.config['vis']['vmax'])
                for grasp, q in zip(grasps, norm_q_values):
                    vis.grasp(grasp, scale=2.0,
                              jaw_width=2.0,
                              show_center=False,
                              show_axis=True,
                              color=plt.cm.RdYlGn(q))
                vis.title('Sampled grasps iter %d' %(j))
                filename = None
                if self._logging_dir is not None:
                    filename = os.path.join(self._logging_dir, 'cem_iter_%d.png' %(j))
                vis.show(filename)
                
            # fit elite set
            elite_start = time()
            num_refit = max(int(np.ceil(self._gmm_refit_p * num_grasps)), 1)
            elite_q_values = [i[0] for i in q_values_and_indices[:num_refit]]
            elite_grasp_indices = [i[1] for i in q_values_and_indices[:num_refit]]
            elite_grasps = [grasps[i] for i in elite_grasp_indices]
            elite_grasp_arr = np.array([g.feature_vec for g in elite_grasps])

            if self.config['vis']['elite_grasps']:
                # display each grasp on the original image, colored by predicted success
                norm_q_values = (elite_q_values - np.min(elite_q_values)) / (np.max(elite_q_values) - np.min(elite_q_values))
                vis.figure(size=(FIGSIZE,FIGSIZE))
                vis.imshow(rgbd_im.depth,
                           vmin=self.config['vis']['vmin'],
                           vmax=self.config['vis']['vmax'])
                for grasp, q in zip(elite_grasps, norm_q_values):
                    vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                              color=plt.cm.RdYlGn(q))
                vis.title('Elite grasps iter %d' %(j))
                filename = None
                if self._logging_dir is not None:
                    filename = os.path.join(self._logging_dir, 'elite_set_iter_%d.png' %(j))
                vis.show(filename)
                    
            # normalize elite set
            elite_grasp_mean = np.mean(elite_grasp_arr, axis=0)
            elite_grasp_std = np.std(elite_grasp_arr, axis=0)
            elite_grasp_std[elite_grasp_std == 0] = 1.0
            elite_grasp_arr = (elite_grasp_arr - elite_grasp_mean) / elite_grasp_std
            logging.info('Elite set computation took %.3f sec' %(time()-elite_start))

            # fit a GMM to the top samples
            num_components = max(int(np.ceil(self._gmm_component_frac * num_refit)), 1)
            uniform_weights = (1.0 / num_components) * np.ones(num_components)
            gmm = GaussianMixture(n_components=num_components,
                                  weights_init=uniform_weights,
                                  reg_covar=self._gmm_reg_covar)
            train_start = time()
            gmm.fit(elite_grasp_arr)
            logging.info('GMM fitting with %d components took %.3f sec' %(num_components, time()-train_start))

            # sample the next grasps
            grasps = []
            loop_start = time()
            num_tries = 0
            while len(grasps) < self._num_gmm_samples and num_tries < self._max_resamples_per_iteration:
                # sample from GMM
                sample_start = time()
                grasp_vecs, _ = gmm.sample(n_samples=self._num_gmm_samples)
                grasp_vecs = elite_grasp_std * grasp_vecs + elite_grasp_mean
                logging.info('GMM sampling took %.3f sec' %(time()-sample_start))

                # convert features to grasps and store if in segmask
                for k, grasp_vec in enumerate(grasp_vecs):
                    feature_start = time()
                    if grasp_type == 'parallel_jaw':
                        # form grasp object
                        grasp = Grasp2D.from_feature_vec(grasp_vec,
                                                         width=self._gripper_width,
                                                         camera_intr=camera_intr)
                    elif grasp_type == 'suction':
                        # read depth and approach axis
                        u = int(min(max(grasp_vec[1], 0), depth_im.height-1))
                        v = int(min(max(grasp_vec[0], 0), depth_im.width-1))
                        grasp_depth = depth_im[u, v]

                        # approach_axis
                        grasp_axis = -normal_cloud_im[u, v]
                        
                        # form grasp object
                        grasp = SuctionPoint2D.from_feature_vec(grasp_vec,
                                                                camera_intr=camera_intr,
                                                                depth=grasp_depth,
                                                                axis=grasp_axis)
                    logging.debug('Feature vec took %.5f sec' %(time()-feature_start))

                        
                    bounds_start = time()
                    # check in bounds
                    if state.segmask is None or \
                        (grasp.center.y >= 0 and grasp.center.y < state.segmask.height and \
                         grasp.center.x >= 0 and grasp.center.x < state.segmask.width and \
                         np.any(state.segmask[int(grasp.center.y), int(grasp.center.x)] != 0) and \
                         grasp.approach_angle < self._max_approach_angle):

                        # check validity according to filters
                        valid = True
                        if self._filters is not None:
                            for filter_name, is_valid in self._filters.iteritems():
                                valid = is_valid(grasp) 
                                logging.debug('Grasp {} filter {} valid: {}'.format(k, filter_name, valid))
                                if not valid:
                                    valid = False
                                    break
                        if valid:
                            grasps.append(grasp)
                    logging.debug('Bounds took %.5f sec' %(time()-bounds_start))
                    num_tries += 1
                    
            # check num grasps
            num_grasps = len(grasps)
            if num_grasps == 0:
                logging.warning('No valid grasps could be found')
                raise NoValidGraspsException()
            logging.info('Resample loop took %.3f sec' %(time()-loop_start))
            logging.info('Resampling took %.3f sec' %(time()-resample_start))

        # predict final set of grasps
        predict_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        logging.info('Final prediction took %.3f sec' %(time()-predict_start))

        if self.config['vis']['grasp_candidates']:
            # display each grasp on the original image, colored by predicted success
            norm_q_values = q_values #(q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
            vis.figure(size=(FIGSIZE,FIGSIZE))
            vis.imshow(rgbd_im.depth,
                       vmin=self.config['vis']['vmin'],
                       vmax=self.config['vis']['vmax'])
            for grasp, q in zip(grasps, norm_q_values):
                vis.grasp(grasp, scale=2.0,
                          jaw_width=2.0,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlGn(q))
            vis.title('Final sampled grasps')
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir, 'final_grasps.png')
            vis.show(filename)

        # select grasp
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config['vis']['grasp_plan']:
            vis.figure()
            vis.imshow(rgbd_im.depth,
                       vmin=self.config['vis']['vmin'],
                       vmax=self.config['vis']['vmax'])
            vis.grasp(grasp, scale=5.0, show_center=False, show_axis=True, jaw_width=1.0, grasp_axis_width=0.2)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(grasp.depth,
                                                     q_value))
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir, 'planned_grasp.png')
            vis.show(filename)

        # form return image
        image = state.rgbd_im.depth
        if isinstance(self._grasp_quality_fn, GQCnnQualityFunction):
            image_arr, _ = self._grasp_quality_fn.grasps_to_tensors([grasp], state)
            image = DepthImage(image_arr[0,...],
                               frame=state.rgbd_im.frame)

        # return action
        action = GraspAction(grasp, q_value, image)

        if self._logging_dir is not None:
            action_dir = os.path.join(policy_dir, 'action')
            action.save(action_dir)

        return action
