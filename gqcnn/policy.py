"""
Grasping policies
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import cPickle as pkl
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

from gqcnn import Grasp2D, SuctionPoint2D, ImageGraspSamplerFactory, GQCNN, InputDataMode, GraspQualityFunctionFactory, GQCnnQualityFunction, NoValidGraspsException
from gqcnn import Visualizer as vis

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

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        color_image_filename = os.path.join(save_dir, 'color.png')
        depth_image_filename = os.path.join(save_dir, 'depth.npy')
        camera_intr_filename = os.path.join(save_dir, 'camera.intr')
        segmask_filename = os.path.join(save_dir, 'segmask.npy')
        state_filename = os.path.join(save_dir, 'state.pkl')
        self.rgbd_im.color.save(color_image_filename)
        self.rgbd_im.depth.save(depth_image_filename)
        self.camera_intr.save(camera_intr_filename)
        if self.segmask is not None:
            self.segmask.save(segmask_filename)
        if self.fully_observed is not None:
            pkl.dump(self.fully_observed, state_filename)
        
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

    @abstractmethod
    def action(self, state):
        """ Returns an action for a given state.
        """
        pass

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
            return None

        # set grasp
        grasp = grasps[0]

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
    def __init__(self, config):
        GraspingPolicy.__init__(self, config)

        self._parse_config()

    def _parse_config(self):
        """ Parses the parameters of the policy. """
        self._num_grasp_samples = self.config['sampling']['num_grasp_samples']
        self._gripper_width = np.inf
        if 'gripper_width' in self.config.keys():
            self._gripper_width = self.config['gripper_width']
        self._logging_dir = None
        if 'logging_dir' in self.config.keys():
            self._logging_dir = self.config['logging_dir']

    def select(self, grasps, q_value):
        """ Selects the grasp with the highest probability of success.
        Can override for alternate policies (e.g. epsilon greedy).
        """
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

        # save if specified
        if self._logging_dir is not None:
            if not os.path.exists(self._logging_dir):
                raise ValueError('Logging directory %s does not exist' %(self._logging_dir))

            # create output dir
            test_case_id = utils.gen_experiment_id()
            test_case_dir = os.path.join(self._logging_dir, 'test_case_%s' %(test_case_id))

            # re-sample if the test case dir exists
            while os.path.exists(test_case_dir):
                test_case_id = utils.gen_experiment_id()
                test_case_dir = os.path.join(self._logging_dir, 'test_case_%s' %(test_case_id))
                
            # create the directory and save
            os.mkdir(test_case_dir)
            candidate_actions_filename = os.path.join(test_case_dir, 'actions.pkl')
            pkl.dump(grasps, open(candidate_actions_filename, 'wb'))
            image_state_filename = os.path.join(test_case_dir, 'state.pkl')
            pkl.dump(state, open(image_state_filename, 'wb'))

        # compute grasp quality
        compute_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        logging.debug('Grasp evaluation took %.3f sec' %(time()-compute_start))
        
        if self.config['vis']['grasp_candidates']:
            # display each grasp on the original image, colored by predicted success
            norm_q_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
            vis.figure(size=(FIGSIZE,FIGSIZE))
            vis.imshow(rgbd_im.depth)
            for grasp, q in zip(grasps, norm_q_values):
                vis.grasp(grasp, scale=1.0, show_center=False, show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title('Sampled grasps')
            vis.show()

        # select grasp
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config['vis']['grasp_plan']:
            vis.figure()
            vis.imshow(state.rgbd_im.depth)
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
    def __init__(self, config):
        GraspingPolicy.__init__(self, config)
        CrossEntropyRobustGraspingPolicy._parse_config(self)

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

        # optional, logging dir
        self._logging_dir = None
        if 'logging_dir' in self.config.keys():
            self._logging_dir = self.config['logging_dir']
            if not os.path.exists(self._logging_dir):
                os.mkdir(self._logging_dir)
            
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
        rgbd_im = state.rgbd_im
        camera_intr = state.camera_intr
        segmask = state.segmask

        # sample grasps
        seed_set_start = time()
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

        logging.info('Computing the seed set took %.3f sec' %(time() - seed_set_start))

        # iteratively refit and sample
        for j in range(self._num_iters):
            logging.info('CEM iter %d' %(j))

            # predict grasps
            predict_start = time()
            q_values = self._grasp_quality_fn(state, grasps, params=self._config)
            logging.debug('Prediction took %.3f sec' %(time()-predict_start))

            # sort grasps
            q_values_and_indices = zip(q_values, np.arange(num_grasps))
            q_values_and_indices.sort(key = lambda x : x[0], reverse=True)

            if self.config['vis']['grasp_candidates']:
                # display each grasp on the original image, colored by predicted success
                norm_q_values = q_values #(q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
                vis.figure(size=(FIGSIZE,FIGSIZE))
                vis.imshow(rgbd_im.depth)
                for grasp, q in zip(grasps, norm_q_values):
                    vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title('Sampled grasps iter %d' %(j))
                vis.show()

            # fit elite set
            num_refit = max(int(np.ceil(self._gmm_refit_p * num_grasps)), 1)
            elite_q_values = [i[0] for i in q_values_and_indices[:num_refit]]
            elite_grasp_indices = [i[1] for i in q_values_and_indices[:num_refit]]
            elite_grasps = [grasps[i] for i in elite_grasp_indices]
            elite_grasp_arr = np.array([g.feature_vec for g in elite_grasps])


            if self.config['vis']['elite_grasps']:
                # display each grasp on the original image, colored by predicted success
                norm_q_values = (elite_q_values - np.min(elite_q_values)) / (np.max(elite_q_values) - np.min(elite_q_values))
                vis.figure(size=(FIGSIZE,FIGSIZE))
                vis.imshow(rgbd_im.depth)
                for grasp, q in zip(elite_grasps, norm_q_values):
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
            logging.info('GMM fitting with %d components took %.3f sec' %(num_components, train_duration))

            # sample the next grasps
            grasps = []
            while len(grasps) < self._num_gmm_samples:
                # sample from GMM
                sample_start = time()
                grasp_vecs, _ = gmm.sample(n_samples=self._num_gmm_samples)
                grasp_vecs = elite_grasp_std * grasp_vecs + elite_grasp_mean
                sample_duration = time() - sample_start
                logging.debug('GMM sampling took %.3f sec' %(sample_duration))

                # convert features to grasps and store if in segmask
                for grasp_vec in grasp_vecs:
                    if grasp_type == 'parallel_jaw':
                        grasp = Grasp2D.from_feature_vec(grasp_vec,
                                                         width=self._gripper_width,
                                                         camera_intr=camera_intr)
                    elif grasp_type == 'suction':
                        grasp = SuctionPoint2D.from_feature_vec(grasp_vec,
                                                                camera_intr=camera_intr)
                    if state.segmask is None or \
                       np.any(state.segmask[int(grasp.center.y), int(grasp.center.x)] != 0):
                        grasps.append(grasp)

            # check num grasps
            num_grasps = len(grasps)
            if num_grasps == 0:
                logging.warning('No valid grasps could be found')
                raise NoValidGraspsException()

        # predict final set of grasps
        predict_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        logging.debug('Final prediction took %.3f sec' %(time()-predict_start))

        if self.config['vis']['grasp_candidates']:
            # display each grasp on the original image, colored by predicted success
            norm_q_values = q_values #(q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
            vis.figure(size=(FIGSIZE,FIGSIZE))
            vis.imshow(rgbd_im.depth)
            for grasp, q in zip(grasps, norm_q_values):
                vis.grasp(grasp, scale=1.0, show_center=False, show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title('Final sampled grasps')
            vis.show()

        # select grasp
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config['vis']['grasp_plan']:
            vis.figure()
            vis.imshow(rgbd_im.depth)
            vis.grasp(grasp, scale=1.0, show_center=True, show_axis=True)
            vis.title('Best Grasp: d=%.3f, q=%.3f' %(grasp.depth,
                                                     q_value))
            vis.show()

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

