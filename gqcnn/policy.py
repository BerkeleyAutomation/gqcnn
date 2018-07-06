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
Grasping policies
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import cPickle as pkl
import logging
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys
from time import time

from sklearn.mixture import GaussianMixture

import autolab_core.utils as utils
from autolab_core import Point
from perception import BinaryImage, ColorImage, DepthImage, RgbdImage, SegmentationImage, CameraIntrinsics
from visualization import Visualizer2D as vis

from . import Grasp2D, SuctionPoint2D, ImageGraspSamplerFactory, GraspQualityFunctionFactory, GQCnnQualityFunction
from .utils import GripperMode, NoValidGraspsException

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
        segmentation mask for the image
    obj_segmask : :obj:`perception.SegmentationImage`
        segmentation mask for the different objects in the image
    full_observed : :obj:`object`
        representation of the fully observed state
    """
    def __init__(self, rgbd_im, camera_intr,
                 segmask=None,
                 obj_segmask=None,
                 fully_observed=None):
        self.rgbd_im = rgbd_im
        self.camera_intr = camera_intr
        self.segmask = segmask
        self.obj_segmask = obj_segmask
        self.fully_observed = fully_observed

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        color_image_filename = os.path.join(save_dir, 'color.png')
        depth_image_filename = os.path.join(save_dir, 'depth.npy')
        camera_intr_filename = os.path.join(save_dir, 'camera.intr')
        segmask_filename = os.path.join(save_dir, 'segmask.npy')
        obj_segmask_filename = os.path.join(save_dir, 'obj_segmask.npy')
        state_filename = os.path.join(save_dir, 'state.pkl')
        self.rgbd_im.color.save(color_image_filename)
        self.rgbd_im.depth.save(depth_image_filename)
        self.camera_intr.save(camera_intr_filename)
        if self.segmask is not None:
            self.segmask.save(segmask_filename)
        if self.obj_segmask is not None:
            self.obj_segmask.save(obj_segmask_filename)
        if self.fully_observed is not None:
            pkl.dump(self.fully_observed, open(state_filename, 'wb'))

    @staticmethod
    def load(save_dir):
        if not os.path.exists(save_dir):
            raise ValueError('Directory %s does not exist!' %(save_dir))
        color_image_filename = os.path.join(save_dir, 'color.png')
        depth_image_filename = os.path.join(save_dir, 'depth.npy')
        camera_intr_filename = os.path.join(save_dir, 'camera.intr')
        segmask_filename = os.path.join(save_dir, 'segmask.npy')
        obj_segmask_filename = os.path.join(save_dir, 'obj_segmask.npy')
        state_filename = os.path.join(save_dir, 'state.pkl')
        camera_intr = CameraIntrinsics.load(camera_intr_filename)
        color = ColorImage.open(color_image_filename, frame=camera_intr.frame)
        depth = DepthImage.open(depth_image_filename, frame=camera_intr.frame)
        segmask = None
        if os.path.exists(segmask_filename):
            segmask = BinaryImage.open(segmask_filename, frame=camera_intr.frame)
        obj_segmask = None
        if os.path.exists(obj_segmask_filename):
            obj_segmask = SegmentationImage.open(obj_segmask_filename, frame=camera_intr.frame)
        fully_observed = None    
        if os.path.exists(state_filename):
            fully_observed = pkl.load(open(state_filename, 'rb'))
        return RgbdImageState(RgbdImage.from_color_and_depth(color, depth),
                              camera_intr,
                              segmask=segmask,
                              obj_segmask=obj_segmask,
                              fully_observed=fully_observed)
            
class GraspAction(object):
    """ Action to encapsulate grasps.
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

    @staticmethod
    def load(save_dir):
        if not os.path.exists(save_dir):
            raise ValueError('Directory %s does not exist!' %(save_dir))
        grasp_filename = os.path.join(save_dir, 'grasp.pkl')
        q_value_filename = os.path.join(save_dir, 'pred_robustness.pkl')
        image_filename = os.path.join(save_dir, 'tf_image.npy')
        grasp = pkl.load(open(grasp_filename, 'rb'))
        q_value = pkl.load(open(q_value_filename, 'rb'))
        image = DepthImage.open(image_filename)
        return GraspAction(grasp, q_value, image)
        
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
        self._gripper_width = 0.05
        if 'gripper_width' in config.keys():
            self._gripper_width = config['gripper_width']

        # set the logging dir
        self._logging_dir = None
        if 'logging_dir' in self.config.keys():
            self._logging_dir = self.config['logging_dir']
            
        # init grasp sampler
        self._sampling_config = config['sampling']
        self._sampling_config['gripper_width'] = self._gripper_width
        if 'crop_width' in config['metric'].keys() and 'crop_height' in config['metric'].keys():
            pad = max(
                math.ceil(np.sqrt(2) * (float(config['metric']['crop_width']) / 2)),
                math.ceil(np.sqrt(2) * (float(config['metric']['crop_height']) / 2))
            )
            self._sampling_config['min_dist_from_boundary'] = pad
        self._sampling_config['gripper_width'] = self._gripper_width
        sampler_type = self._sampling_config['type']
        self._grasp_sampler = ImageGraspSamplerFactory.sampler(sampler_type,
                                                               self._sampling_config)

        # init grasp quality function
        self._metric_config = config['metric']
        metric_type = self._metric_config['type']
        self._grasp_quality_fn = GraspQualityFunctionFactory.quality_function(metric_type,
                                                                              self._metric_config)

    @property
    def config(self):
        """ Returns the policy parameters. """
        return self._config

    @property
    def grasp_sampler(self):
        """ Returns the grasp sampler. """
        return self._grasp_sampler

    @property
    def grasp_quality_fn(self):
        """ Returns the grasp sampler. """
        return self._grasp_quality_fn

    @property
    def gqcnn(self):
        """ Returns the GQ-CNN. """
        return self._gqcnn

    def action(self, state):
        """ Returns an action for a given state.
        Public handle to function.
        """
        # save state
        if self._logging_dir is not None:
            policy_id = utils.gen_experiment_id()
            self._policy_dir = os.path.join(self._logging_dir, 'policy_output_%s' %(policy_id))
            while os.path.exists(self._policy_dir):
                policy_id = utils.gen_experiment_id()
            self._policy_dir = os.path.join(self._logging_dir, 'policy_output_%s' %(policy_id))
            os.mkdir(self._policy_dir)
            state_dir = os.path.join(self._policy_dir, 'state')
            state.save(state_dir)

        # plan action
        action = self._action(state)

        # save action
        if self._logging_dir is not None:
            action_dir = os.path.join(self._policy_dir, 'action')
            action.save(action_dir)
        return action
        
    @abstractmethod
    def _action(self, state):
        """ Returns an action for a given state.
        """
        pass
    
    def show(self, filename=None, dpi=100):
        """ Show a figure. """
        if self._logging_dir is None:
            vis.show()
        else:
            filename = os.path.join(self._policy_dir, filename)
            vis.savefig(filename, dpi=dpi)

class UniformRandomGraspingPolicy(GraspingPolicy):
    """ Returns a grasp uniformly at random. """
    def __init__(self, config):
        GraspingPolicy.__init__(self, config)
        self._num_grasp_samples = 1

        self._grasp_center_std = 0.0
        if 'grasp_center_std' in config.keys():
            self._grasp_center_std = config['grasp_center_std']
        
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
        
        # perturb grasp
        if self._grasp_center_std > 0.0:
            grasp_center_rv = ss.multivariate_normal(grasp.center.data, cov=self._grasp_center_std**2)
            grasp.center.data = grasp_center_rv.rvs(size=1)[0]
        
        # form tensors
        return GraspAction(grasp, 0.0, state.rgbd_im.depth)

class RobustGraspingPolicy(GraspingPolicy):
    """ Samples a set of grasp candidates in image space,
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
    logging_dir : str, optional
        directory in which to save the sampled grasps and input images
    """
    def __init__(self, config, filters=None):
        GraspingPolicy.__init__(self, config)
        self._parse_config()
        self._filters = filters

    def _parse_config(self):
        """ Parses the parameters of the policy. """
        self._num_grasp_samples = self.config['sampling']['num_grasp_samples']
        self._max_grasps_filter = 1
        if 'max_grasps_filter' in self.config.keys():
            self._max_grasps_filter = self.config['max_grasps_filter']
        self._gripper_width = np.inf
        if 'gripper_width' in self.config.keys():
            self._gripper_width = self.config['gripper_width']

    def select(self, grasps, q_value):
        """ Selects the grasp with the highest probability of success.
        Can override for alternate policies (e.g. epsilon greedy).
        """
        # sort grasps
        num_grasps = len(grasps)
        grasps_and_predictions = zip(np.arange(num_grasps), q_value)
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
        
        # compute grasp quality
        compute_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        logging.debug('Grasp evaluation took %.3f sec' %(time()-compute_start))
        
        if self.config['vis']['grasp_candidates']:
            # display each grasp on the original image, colored by predicted success
            norm_q_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
            vis.figure(size=(FIGSIZE,FIGSIZE))
            vis.imshow(rgbd_im.depth,
                       vmin=self.config['vis']['vmin'],
                       vmax=self.config['vis']['vmax'])
            for grasp, q in zip(grasps, norm_q_values):
                vis.grasp(grasp, scale=1.0,
                          grasp_center_size=10,
                          grasp_center_thickness=2.5,
                          jaw_width=2.5,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlGn(q))
            vis.title('Sampled grasps')
            self.show('grasp_candidates.png')

        # select grasp
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config['vis']['grasp_plan']:
            vis.figure()
            vis.imshow(rgbd_im.depth,
                       vmin=self.config['vis']['vmin'],
                       vmax=self.config['vis']['vmax'])
            vis.grasp(grasp, scale=2.0, show_axis=True)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(grasp.depth, q_value))
            vis.show()

        return GraspAction(grasp, q_value, state.rgbd_im.depth)

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
            elite_grasp_std[elite_grasp_std == 0] = 1e-6
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
        return action
        
class QFunctionRobustGraspingPolicy(CrossEntropyRobustGraspingPolicy):
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
    """
    def __init__(self, config):
        CrossEntropyRobustGraspingPolicy.__init__(self, config)
        QFunctionRobustGraspingPolicy._parse_config(self)
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
        
class EpsilonGreedyQFunctionRobustGraspingPolicy(QFunctionRobustGraspingPolicy):
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
        QFunctionRobustGraspingPolicy.__init__(self, config)
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
        :obj:`GraspAction`
            grasp to execute
        """
        return CrossEntropyRobustGraspingPolicy.action(self, state)
    
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
        # take the greedy action with prob 1 - epsilon
        if np.random.rand() > self.epsilon:
            logging.debug('Taking greedy action')
            return CrossEntropyRobustGraspingPolicy.action(self, state)

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
        return GraspAction(grasp, q_value, image)

