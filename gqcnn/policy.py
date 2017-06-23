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

from sklearn.mixture import GaussianMixture

import autolab_core.utils as utils
from autolab_core import Point
from perception import DepthImage

from gqcnn import Grasp2D, ImageGraspSamplerFactory, GQCNN, InputDataMode
from gqcnn import Visualizer as vis
from gqcnn import NoValidGraspsException

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
    def __init__(self, rgbd_im, camera_intr, segmask=None,
                 fully_observed=None):
        self.rgbd_im = rgbd_im
        self.camera_intr = camera_intr
        self.segmask = segmask
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

        # open tensorflow session for gqcnn
        self._gqcnn.open_session()

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

