# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Grasping policies.

Author
------
Jeff Mahler
"""
from abc import ABC, abstractmethod
import copy
import pickle as pkl
import math
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as snf
import scipy.stats as ss
from sklearn.mixture import GaussianMixture

import autolab_core.utils as utils
from autolab_core import Point, Logger
from perception import (BinaryImage, ColorImage, DepthImage, RgbdImage,
                        SegmentationImage, CameraIntrinsics)
from visualization import Visualizer2D as vis

from ..constraint_fn import GraspConstraintFnFactory
from ..grasp import Grasp2D, SuctionPoint2D, MultiSuctionPoint2D
from ..grasp_quality_function import (GraspQualityFunctionFactory,
                                      GQCnnQualityFunction)
from ..image_grasp_sampler import ImageGraspSamplerFactory
from ...utils import GeneralConstants, NoValidGraspsException


class RgbdImageState(object):
    """State to encapsulate RGB-D images."""

    def __init__(self,
                 rgbd_im,
                 camera_intr,
                 segmask=None,
                 obj_segmask=None,
                 fully_observed=None):
        """
        Parameters
        ----------
        rgbd_im : :obj:`perception.RgbdImage`
            An RGB-D image to plan grasps on.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the RGB-D camera.
        segmask : :obj:`perception.BinaryImage`
            Segmentation mask for the image.
        obj_segmask : :obj:`perception.SegmentationImage`
            Segmentation mask for the different objects in the image.
        full_observed : :obj:`object`
            Representation of the fully observed state.
        """
        self.rgbd_im = rgbd_im
        self.camera_intr = camera_intr
        self.segmask = segmask
        self.obj_segmask = obj_segmask
        self.fully_observed = fully_observed

    def save(self, save_dir):
        """Save to a directory.

        Parameters
        ----------
        save_dir : str
            The directory to save to.
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        color_image_filename = os.path.join(save_dir, "color.png")
        depth_image_filename = os.path.join(save_dir, "depth.npy")
        camera_intr_filename = os.path.join(save_dir, "camera.intr")
        segmask_filename = os.path.join(save_dir, "segmask.npy")
        obj_segmask_filename = os.path.join(save_dir, "obj_segmask.npy")
        state_filename = os.path.join(save_dir, "state.pkl")
        self.rgbd_im.color.save(color_image_filename)
        self.rgbd_im.depth.save(depth_image_filename)
        self.camera_intr.save(camera_intr_filename)
        if self.segmask is not None:
            self.segmask.save(segmask_filename)
        if self.obj_segmask is not None:
            self.obj_segmask.save(obj_segmask_filename)
        if self.fully_observed is not None:
            pkl.dump(self.fully_observed, open(state_filename, "wb"))

    @staticmethod
    def load(save_dir):
        """Load an :obj:`RGBDImageState`.

        Parameters
        ----------
        save_dir : str
            The directory to load from.
        """
        if not os.path.exists(save_dir):
            raise ValueError("Directory %s does not exist!" % (save_dir))
        color_image_filename = os.path.join(save_dir, "color.png")
        depth_image_filename = os.path.join(save_dir, "depth.npy")
        camera_intr_filename = os.path.join(save_dir, "camera.intr")
        segmask_filename = os.path.join(save_dir, "segmask.npy")
        obj_segmask_filename = os.path.join(save_dir, "obj_segmask.npy")
        state_filename = os.path.join(save_dir, "state.pkl")
        camera_intr = CameraIntrinsics.load(camera_intr_filename)
        color = ColorImage.open(color_image_filename, frame=camera_intr.frame)
        depth = DepthImage.open(depth_image_filename, frame=camera_intr.frame)
        segmask = None
        if os.path.exists(segmask_filename):
            segmask = BinaryImage.open(segmask_filename,
                                       frame=camera_intr.frame)
        obj_segmask = None
        if os.path.exists(obj_segmask_filename):
            obj_segmask = SegmentationImage.open(obj_segmask_filename,
                                                 frame=camera_intr.frame)
        fully_observed = None
        if os.path.exists(state_filename):
            fully_observed = pkl.load(open(state_filename, "rb"))
        return RgbdImageState(RgbdImage.from_color_and_depth(color, depth),
                              camera_intr,
                              segmask=segmask,
                              obj_segmask=obj_segmask,
                              fully_observed=fully_observed)


class GraspAction(object):
    """Action to encapsulate grasps.
    """

    def __init__(self, grasp, q_value, image=None, policy_name=None):
        """
        Parameters
        ----------
        grasp : :obj`Grasp2D` or :obj:`SuctionPoint2D`
            2D grasp to wrap.
        q_value : float
            Grasp quality.
        image : :obj:`perception.DepthImage`
            Depth image corresponding to grasp.
        policy_name : str
            Policy name.
        """
        self.grasp = grasp
        self.q_value = q_value
        self.image = image
        self.policy_name = policy_name

    def save(self, save_dir):
        """Save grasp action.

        Parameters
        ----------
        save_dir : str
            Directory to save the grasp action to.
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        grasp_filename = os.path.join(save_dir, "grasp.pkl")
        q_value_filename = os.path.join(save_dir, "pred_robustness.pkl")
        image_filename = os.path.join(save_dir, "tf_image.npy")
        pkl.dump(self.grasp, open(grasp_filename, "wb"))
        pkl.dump(self.q_value, open(q_value_filename, "wb"))
        if self.image is not None:
            self.image.save(image_filename)

    @staticmethod
    def load(save_dir):
        """Load a saved grasp action.

        Parameters
        ----------
        save_dir : str
            Directory of the saved grasp action.

        Returns
        -------
        :obj:`GraspAction`
            Loaded grasp action.
        """
        if not os.path.exists(save_dir):
            raise ValueError("Directory %s does not exist!" % (save_dir))
        grasp_filename = os.path.join(save_dir, "grasp.pkl")
        q_value_filename = os.path.join(save_dir, "pred_robustness.pkl")
        image_filename = os.path.join(save_dir, "tf_image.npy")
        grasp = pkl.load(open(grasp_filename, "rb"))
        q_value = pkl.load(open(q_value_filename, "rb"))
        image = None
        if os.path.exists(image_filename):
            image = DepthImage.open(image_filename)
        return GraspAction(grasp, q_value, image)


class Policy(ABC):
    """Abstract policy class."""

    def __call__(self, state):
        """Execute the policy on a state."""
        return self.action(state)

    @abstractmethod
    def action(self, state):
        """Returns an action for a given state."""
        pass


class GraspingPolicy(Policy):
    """Policy for robust grasping with Grasp Quality Convolutional Neural
    Networks (GQ-CNN)."""

    def __init__(self, config, init_sampler=True):
        """
        Parameters
        ----------
        config : dict
            Python dictionary of parameters for the policy.
        init_sampler : bool
            Whether or not to initialize the grasp sampler.

        Notes
        -----
        Required configuration parameters are specified in Other Parameters.

        Other Parameters
        ----------------
        sampling : dict
            Dictionary of parameters for grasp sampling, see
            `gqcnn/grasping/image_grasp_sampler.py`.
        gqcnn_model : str
            String path to a trained GQ-CNN model.
        """
        # Store parameters.
        self._config = config
        self._gripper_width = 0.05
        if "gripper_width" in config:
            self._gripper_width = config["gripper_width"]

        # Set the logging dir and possibly log file.
        self._logging_dir = None
        log_file = None
        if "logging_dir" in self.config:
            self._logging_dir = self.config["logging_dir"]
            if not os.path.exists(self._logging_dir):
                os.makedirs(self._logging_dir)
            log_file = os.path.join(self._logging_dir, "policy.log")

        # Setup logger.
        self._logger = Logger.get_logger(self.__class__.__name__,
                                         log_file=log_file,
                                         global_log_file=True)

        # Init grasp sampler.
        if init_sampler:
            self._sampling_config = config["sampling"]
            self._sampling_config["gripper_width"] = self._gripper_width
            if "crop_width" in config["metric"] and "crop_height" in config[
                    "metric"]:
                pad = max(
                    math.ceil(
                        np.sqrt(2) * (config["metric"]["crop_width"] / 2)),
                    math.ceil(
                        np.sqrt(2) * (config["metric"]["crop_height"] / 2)))
                self._sampling_config["min_dist_from_boundary"] = pad
            self._sampling_config["gripper_width"] = self._gripper_width
            sampler_type = self._sampling_config["type"]
            self._grasp_sampler = ImageGraspSamplerFactory.sampler(
                sampler_type, self._sampling_config)

        # Init constraint function.
        self._grasp_constraint_fn = None
        if "constraints" in self._config:
            self._constraint_config = self._config["constraints"]
            constraint_type = self._constraint_config["type"]
            self._grasp_constraint_fn = GraspConstraintFnFactory.constraint_fn(
                constraint_type, self._constraint_config)

        # Init grasp quality function.
        self._metric_config = config["metric"]
        metric_type = self._metric_config["type"]
        self._grasp_quality_fn = GraspQualityFunctionFactory.quality_function(
            metric_type, self._metric_config)

    @property
    def config(self):
        """Returns the policy configuration parameters.

        Returns
        -------
        dict
            Python dictionary of the policy configuration parameters.
        """
        return self._config

    @property
    def grasp_sampler(self):
        """Returns the grasp sampler.

        Returns
        -------
        :obj:`gqcnn.grasping.image_grasp_sampler.ImageGraspSampler`
            The grasp sampler.
        """
        return self._grasp_sampler

    @property
    def grasp_quality_fn(self):
        """Returns the grasp quality function.

        Returns
        -------
        :obj:`gqcnn.grasping.grasp_quality_function.GraspQualityFunction`
            The grasp quality function.
        """
        return self._grasp_quality_fn

    @property
    def grasp_constraint_fn(self):
        """Returns the grasp constraint function.

        Returns
        -------
        :obj:`gqcnn.grasping.constraint_fn.GraspConstraintFn`
            The grasp contraint function.
        """
        return self._grasp_constraint_fn

    @property
    def gqcnn(self):
        """Returns the GQ-CNN.

        Returns
        -------
        :obj:`gqcnn.model.tf.GQCNNTF`
            The GQ-CNN model.
        """
        return self._gqcnn

    def set_constraint_fn(self, constraint_fn):
        """Sets the grasp constraint function.

        Parameters
        ----------
        constraint_fn : :obj`gqcnn.grasping.constraint_fn.GraspConstraintFn`
            The grasp contraint function.
        """
        self._grasp_constraint_fn = constraint_fn

    def action(self, state):
        """Returns an action for a given state.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            The RGB-D image state to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            The planned grasp action.
        """
        # Save state.
        if self._logging_dir is not None:
            policy_id = utils.gen_experiment_id()
            policy_dir = os.path.join(self._logging_dir,
                                      "policy_output_%s" % (policy_id))
            while os.path.exists(policy_dir):
                policy_id = utils.gen_experiment_id()
                policy_dir = os.path.join(self._logging_dir,
                                          "policy_output_%s" % (policy_id))
            self._policy_dir = policy_dir
            os.mkdir(self._policy_dir)
            state_dir = os.path.join(self._policy_dir, "state")
            state.save(state_dir)

        # Plan action.
        action = self._action(state)

        # Save action.
        if self._logging_dir is not None:
            action_dir = os.path.join(self._policy_dir, "action")
            action.save(action_dir)
        return action

    @abstractmethod
    def _action(self, state):
        """Returns an action for a given state.
        """
        pass

    def show(self, filename=None, dpi=100):
        """Show a figure.

        Parameters
        ----------
        filename : str
            File to save figure to.
        dpi : int
            Dpi of figure.
        """
        if self._logging_dir is None:
            vis.show()
        else:
            filename = os.path.join(self._policy_dir, filename)
            vis.savefig(filename, dpi=dpi)


class UniformRandomGraspingPolicy(GraspingPolicy):
    """Returns a grasp uniformly at random."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Python dictionary of policy configuration parameters.
        filters : dict
            Python dictionary of functions to apply to filter invalid grasps.
        """
        GraspingPolicy.__init__(self, config)
        self._num_grasp_samples = 1

        self._grasp_center_std = 0.0
        if "grasp_center_std" in config:
            self._grasp_center_std = config["grasp_center_std"]

    def _action(self, state):
        """Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            Grasp to execute.
        """
        # Check valid input.
        if not isinstance(state, RgbdImageState):
            raise ValueError("Must provide an RGB-D image state.")

        # Parse state.
        rgbd_im = state.rgbd_im
        camera_intr = state.camera_intr
        segmask = state.segmask

        # Sample grasps.
        grasps = self._grasp_sampler.sample(
            rgbd_im,
            camera_intr,
            self._num_grasp_samples,
            segmask=segmask,
            visualize=self.config["vis"]["grasp_sampling"],
            constraint_fn=self._grasp_constraint_fn,
            seed=None)
        num_grasps = len(grasps)
        if num_grasps == 0:
            self._logger.warning("No valid grasps could be found")
            raise NoValidGraspsException()

        # Set grasp.
        grasp = grasps[0]

        # Perturb grasp.
        if self._grasp_center_std > 0.0:
            grasp_center_rv = ss.multivariate_normal(
                grasp.center.data, cov=self._grasp_center_std**2)
            grasp.center.data = grasp_center_rv.rvs(size=1)[0]

        # Form tensors.
        return GraspAction(grasp, 0.0, state.rgbd_im.depth)


class RobustGraspingPolicy(GraspingPolicy):
    """Samples a set of grasp candidates in image space,
    ranks the grasps by the predicted probability of success from a GQ-CNN
    and returns the grasp with the highest probability of success."""

    def __init__(self, config, filters=None):
        """
        Parameters
        ----------
        config : dict
            Python dictionary of policy configuration parameters.
        filters : dict
            Python dictionary of functions to apply to filter invalid grasps.

        Notes
        -----
        Required configuration dictionary parameters are specified in
        Other Parameters.

        Other Parameters
        ----------------
        num_grasp_samples : int
            Number of grasps to sample.
        gripper_width : float, optional
            Width of the gripper in meters.
        logging_dir : str, optional
            Directory in which to save the sampled grasps and input images.
        """
        GraspingPolicy.__init__(self, config)
        self._parse_config()
        self._filters = filters

    def _parse_config(self):
        """Parses the parameters of the policy."""
        self._num_grasp_samples = self.config["sampling"]["num_grasp_samples"]
        self._max_grasps_filter = 1
        if "max_grasps_filter" in self.config:
            self._max_grasps_filter = self.config["max_grasps_filter"]
        self._gripper_width = np.inf
        if "gripper_width" in self.config:
            self._gripper_width = self.config["gripper_width"]

    def select(self, grasps, q_value):
        """Selects the grasp with the highest probability of success.

        Can override for alternate policies (e.g. epsilon greedy).

        Parameters
        ----------
        grasps : list
            Python list of :obj:`gqcnn.grasping.Grasp2D` or
            :obj:`gqcnn.grasping.SuctionPoint2D` grasps to select from.
        q_values : list
            Python list of associated q-values.

        Returns
        -------
        :obj:`gqcnn.grasping.Grasp2D` or :obj:`gqcnn.grasping.SuctionPoint2D`
            Grasp with highest probability of success .
        """
        # Sort grasps.
        num_grasps = len(grasps)
        grasps_and_predictions = zip(np.arange(num_grasps), q_value)
        grasps_and_predictions = sorted(grasps_and_predictions,
                                        key=lambda x: x[1],
                                        reverse=True)

        # Return top grasps.
        if self._filters is None:
            return grasps_and_predictions[0][0]

        # Filter grasps.
        self._logger.info("Filtering grasps")
        i = 0
        while i < self._max_grasps_filter and i < len(grasps_and_predictions):
            index = grasps_and_predictions[i][0]
            grasp = grasps[index]
            valid = True
            for filter_name, is_valid in self._filters.items():
                valid = is_valid(grasp)
                self._logger.debug("Grasp {} filter {} valid: {}".format(
                    i, filter_name, valid))
                if not valid:
                    valid = False
                    break
            if valid:
                return index
            i += 1
        raise NoValidGraspsException("No grasps satisfied filters")

    def _action(self, state):
        """Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            Grasp to execute.
        """
        # Check valid input.
        if not isinstance(state, RgbdImageState):
            raise ValueError("Must provide an RGB-D image state.")

        # Parse state.
        rgbd_im = state.rgbd_im
        camera_intr = state.camera_intr
        segmask = state.segmask

        # Sample grasps.
        grasps = self._grasp_sampler.sample(
            rgbd_im,
            camera_intr,
            self._num_grasp_samples,
            segmask=segmask,
            visualize=self.config["vis"]["grasp_sampling"],
            constraint_fn=self._grasp_constraint_fn,
            seed=None)
        num_grasps = len(grasps)
        if num_grasps == 0:
            self._logger.warning("No valid grasps could be found")
            raise NoValidGraspsException()

        # Compute grasp quality.
        compute_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        self._logger.debug("Grasp evaluation took %.3f sec" %
                           (time() - compute_start))

        if self.config["vis"]["grasp_candidates"]:
            # Display each grasp on the original image, colored by predicted
            # success.
            norm_q_values = (q_values - np.min(q_values)) / (np.max(q_values) -
                                                             np.min(q_values))
            vis.figure(size=(GeneralConstants.FIGSIZE,
                             GeneralConstants.FIGSIZE))
            vis.imshow(rgbd_im.depth,
                       vmin=self.config["vis"]["vmin"],
                       vmax=self.config["vis"]["vmax"])
            for grasp, q in zip(grasps, norm_q_values):
                vis.grasp(grasp,
                          scale=1.0,
                          grasp_center_size=10,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlBu(q))
                vis.title("Sampled grasps")
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir,
                                        "grasp_candidates.png")
            vis.show(filename)

        # Select grasp.
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config["vis"]["grasp_plan"]:
            vis.figure()
            vis.imshow(rgbd_im.depth,
                       vmin=self.config["vis"]["vmin"],
                       vmax=self.config["vis"]["vmax"])
            vis.grasp(grasp, scale=2.0, show_axis=True)
            vis.title("Best Grasp: d=%.3f, q=%.3f" % (grasp.depth, q_value))
            vis.show()

        return GraspAction(grasp, q_value, state.rgbd_im.depth)


class CrossEntropyRobustGraspingPolicy(GraspingPolicy):
    """Optimizes a set of grasp candidates in image space using the
    cross entropy method.

    Cross entropy method (CEM):
    (1) sample an initial set of candidates
    (2) sort the candidates
    (3) fit a GMM to the top P%
    (4) re-sample grasps from the distribution
    (5) repeat steps 2-4 for K iters
    (6) return the best candidate from the final sample set
    """

    def __init__(self, config, filters=None):
        """
        Parameters
        ----------
        config : dict
            Python dictionary of policy configuration parameters.
        filters : dict
            Python dictionary of functions to apply to filter invalid grasps.

        Notes
        -----
        Required configuration dictionary parameters are specified in Other
        Parameters.

        Other Parameters
        ----------------
        num_seed_samples : int
            Number of candidate to sample in the initial set.
        num_gmm_samples : int
            Number of candidates to sample on each resampling from the GMMs.
        num_iters : int
            Number of sample-and-refit iterations of CEM.
        gmm_refit_p : float
            Top p-% of grasps used for refitting.
        gmm_component_frac : float
            Percentage of the elite set size used to determine number of GMM
            components.
        gmm_reg_covar : float
            Regularization parameters for GMM covariance matrix, enforces
            diversity of fitted distributions.
        deterministic : bool, optional
            Whether to set the random seed to enforce deterministic behavior.
        gripper_width : float, optional
            Width of the gripper in meters.
        """
        GraspingPolicy.__init__(self, config)
        self._parse_config()
        self._filters = filters

        self._case_counter = 0

    def _parse_config(self):
        """Parses the parameters of the policy."""
        # Cross entropy method parameters.
        self._num_seed_samples = self.config["num_seed_samples"]
        self._num_gmm_samples = self.config["num_gmm_samples"]
        self._num_iters = self.config["num_iters"]
        self._gmm_refit_p = self.config["gmm_refit_p"]
        self._gmm_component_frac = self.config["gmm_component_frac"]
        self._gmm_reg_covar = self.config["gmm_reg_covar"]

        self._depth_gaussian_sigma = 0.0
        if "depth_gaussian_sigma" in self.config:
            self._depth_gaussian_sigma = self.config["depth_gaussian_sigma"]

        self._max_grasps_filter = 1
        if "max_grasps_filter" in self.config:
            self._max_grasps_filter = self.config["max_grasps_filter"]

        self._max_resamples_per_iteration = 100
        if "max_resamples_per_iteration" in self.config:
            self._max_resamples_per_iteration = self.config[
                "max_resamples_per_iteration"]

        self._max_approach_angle = np.inf
        if "max_approach_angle" in self.config:
            self._max_approach_angle = np.deg2rad(
                self.config["max_approach_angle"])

        # Gripper parameters.
        self._seed = None
        if self.config["deterministic"]:
            self._seed = GeneralConstants.SEED
        self._gripper_width = np.inf
        if "gripper_width" in self.config:
            self._gripper_width = self.config["gripper_width"]

        # Affordance map visualization.
        self._vis_grasp_affordance_map = False
        if "grasp_affordance_map" in self.config["vis"]:
            self._vis_grasp_affordance_map = self.config["vis"][
                "grasp_affordance_map"]

        self._state_counter = 0  # Used for logging state data.

    def select(self, grasps, q_values):
        """Selects the grasp with the highest probability of success.

        Can override for alternate policies (e.g. epsilon greedy).

        Parameters
        ----------
        grasps : list
            Python list of :obj:`gqcnn.grasping.Grasp2D` or
            :obj:`gqcnn.grasping.SuctionPoint2D` grasps to select from.
        q_values : list
            Python list of associated q-values.

        Returns
        -------
        :obj:`gqcnn.grasping.Grasp2D` or :obj:`gqcnn.grasping.SuctionPoint2D`
            Grasp with highest probability of success.
        """
        # Sort.
        self._logger.info("Sorting grasps")
        num_grasps = len(grasps)
        if num_grasps == 0:
            raise NoValidGraspsException("Zero grasps")
        grasps_and_predictions = zip(np.arange(num_grasps), q_values)
        grasps_and_predictions = sorted(grasps_and_predictions,
                                        key=lambda x: x[1],
                                        reverse=True)

        # Return top grasps.
        if self._filters is None:
            return grasps_and_predictions[0][0]

        # Filter grasps.
        self._logger.info("Filtering grasps")
        i = 0
        while i < self._max_grasps_filter and i < len(grasps_and_predictions):
            index = grasps_and_predictions[i][0]
            grasp = grasps[index]
            valid = True
            for filter_name, is_valid in self._filters.items():
                valid = is_valid(grasp)
                self._logger.debug("Grasp {} filter {} valid: {}".format(
                    i, filter_name, valid))
                if not valid:
                    valid = False
                    break
            if valid:
                return index
            i += 1
        raise NoValidGraspsException("No grasps satisfied filters")

    def _mask_predictions(self, pred_map, segmask):
        self._logger.info("Masking predictions...")
        seg_pred_mismatch_msg = ("Prediction map shape {} does not match shape"
                                 " of segmask {}.")
        assert pred_map.shape == segmask.shape, seg_pred_mismatch_msg.format(
            pred_map.shape, segmask.shape)
        preds_masked = np.zeros_like(pred_map)
        nonzero_ind = np.where(segmask > 0)
        preds_masked[nonzero_ind] = pred_map[nonzero_ind]
        return preds_masked

    def _gen_grasp_affordance_map(self, state, stride=1):
        self._logger.info("Generating grasp affordance map...")

        # Generate grasps at points to evaluate (this is just the interface to
        # `GraspQualityFunction`).
        crop_candidate_start_time = time()
        point_cloud_im = state.camera_intr.deproject_to_image(
            state.rgbd_im.depth)
        normal_cloud_im = point_cloud_im.normal_cloud_im()

        q_vals = []
        gqcnn_recep_h_half = self._grasp_quality_fn.gqcnn_recep_height // 2
        gqcnn_recep_w_half = self._grasp_quality_fn.gqcnn_recep_width // 2
        im_h = state.rgbd_im.height
        im_w = state.rgbd_im.width
        for i in range(gqcnn_recep_h_half - 1, im_h - gqcnn_recep_h_half,
                       stride):
            grasps = []
            for j in range(gqcnn_recep_w_half - 1, im_w - gqcnn_recep_w_half,
                           stride):
                # TODO(vsatish): Find a better way to find policy type.
                if self.config["sampling"]["type"] == "suction":
                    grasps.append(
                        SuctionPoint2D(Point(np.array([j, i])),
                                       axis=-normal_cloud_im[i, j],
                                       depth=state.rgbd_im.depth[i, j],
                                       camera_intr=state.camera_intr))
                else:
                    raise NotImplementedError(
                        "Parallel Jaw Grasp Affordance Maps Not Supported!")
            q_vals.extend(self._grasp_quality_fn(state, grasps))
        self._logger.info(
            "Generating crop grasp candidates took {} sec.".format(
                time() - crop_candidate_start_time))

        # Mask out predictions not in the segmask (we don't really care about
        # them).
        pred_map = np.array(q_vals).reshape(
            (im_h - gqcnn_recep_h_half * 2) // stride + 1,
            (im_w - gqcnn_recep_w_half * 2) // stride + 1)
        # TODO(vsatish): Don't access the raw data like this!
        tf_segmask = state.segmask.crop(im_h - gqcnn_recep_h_half * 2,
                                        im_w - gqcnn_recep_w_half * 2).resize(
                                            1.0 / stride,
                                            interp="nearest")._data.squeeze()
        if tf_segmask.shape != pred_map.shape:
            new_tf_segmask = np.zeros_like(pred_map)
            smaller_i = min(pred_map.shape[0], tf_segmask.shape[0])
            smaller_j = min(pred_map.shape[1], tf_segmask.shape[1])
            new_tf_segmask[:smaller_i, :smaller_j] = tf_segmask[:smaller_i, :
                                                                smaller_j]
            tf_segmask = new_tf_segmask
        pred_map_masked = self._mask_predictions(pred_map, tf_segmask)
        return pred_map_masked

    def _plot_grasp_affordance_map(self,
                                   state,
                                   affordance_map,
                                   stride=1,
                                   grasps=None,
                                   q_values=None,
                                   plot_max=True,
                                   title=None,
                                   scale=1.0,
                                   save_fname=None,
                                   save_path=None):
        gqcnn_recep_h_half = self._grasp_quality_fn.gqcnn_recep_height // 2
        gqcnn_recep_w_half = self._grasp_quality_fn.gqcnn_recep_width // 2
        im_h = state.rgbd_im.height
        im_w = state.rgbd_im.width

        # Plot.
        vis.figure()
        tf_depth_im = state.rgbd_im.depth.crop(
            im_h - gqcnn_recep_h_half * 2,
            im_w - gqcnn_recep_w_half * 2).resize(1.0 / stride,
                                                  interp="nearest")
        vis.imshow(tf_depth_im)
        plt.imshow(affordance_map, cmap=plt.cm.RdYlGn, alpha=0.3)
        if grasps is not None:
            grasps = copy.deepcopy(grasps)
            for grasp, q in zip(grasps, q_values):
                grasp.center.data[0] -= gqcnn_recep_w_half
                grasp.center.data[1] -= gqcnn_recep_h_half
                vis.grasp(grasp,
                          scale=scale,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlGn(q))
        if plot_max:
            affordance_argmax = np.unravel_index(np.argmax(affordance_map),
                                                 affordance_map.shape)
            plt.scatter(affordance_argmax[1],
                        affordance_argmax[0],
                        c="black",
                        marker=".",
                        s=scale * 25)
        if title is not None:
            vis.title(title)
        if save_path is not None:
            save_path = os.path.join(save_path, save_fname)
        vis.show(save_path)

    def action_set(self, state):
        """Plan a set of grasps with the highest probability of success on
        the given RGB-D image.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        python list of :obj:`gqcnn.grasping.Grasp2D` or :obj:`gqcnn.grasping.SuctionPoint2D`  # noqa E501
            Grasps to execute.
        """
        # Check valid input.
        if not isinstance(state, RgbdImageState):
            raise ValueError("Must provide an RGB-D image state.")

        state_output_dir = None
        if self._logging_dir is not None:
            state_output_dir = os.path.join(
                self._logging_dir,
                "state_{}".format(str(self._state_counter).zfill(5)))
            if not os.path.exists(state_output_dir):
                os.makedirs(state_output_dir)
            self._state_counter += 1

        # Parse state.
        seed_set_start = time()
        rgbd_im = state.rgbd_im
        depth_im = rgbd_im.depth
        camera_intr = state.camera_intr
        segmask = state.segmask

        if self._depth_gaussian_sigma > 0:
            depth_im_filtered = depth_im.apply(
                snf.gaussian_filter, sigma=self._depth_gaussian_sigma)
        else:
            depth_im_filtered = depth_im
        point_cloud_im = camera_intr.deproject_to_image(depth_im_filtered)
        normal_cloud_im = point_cloud_im.normal_cloud_im()

        # Visualize grasp affordance map.
        if self._vis_grasp_affordance_map:
            grasp_affordance_map = self._gen_grasp_affordance_map(state)
            self._plot_grasp_affordance_map(state,
                                            grasp_affordance_map,
                                            title="Grasp Affordance Map",
                                            save_fname="affordance_map.png",
                                            save_path=state_output_dir)

        if "input_images" in self.config["vis"] and self.config["vis"][
                "input_images"]:
            vis.figure()
            vis.subplot(1, 2, 1)
            vis.imshow(depth_im)
            vis.title("Depth")
            vis.subplot(1, 2, 2)
            vis.imshow(segmask)
            vis.title("Segmask")
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir, "input_images.png")
            vis.show(filename)

        # Sample grasps.
        self._logger.info("Sampling seed set")
        grasps = self._grasp_sampler.sample(
            rgbd_im,
            camera_intr,
            self._num_seed_samples,
            segmask=segmask,
            visualize=self.config["vis"]["grasp_sampling"],
            constraint_fn=self._grasp_constraint_fn,
            seed=self._seed)
        num_grasps = len(grasps)
        if num_grasps == 0:
            self._logger.warning("No valid grasps could be found")
            raise NoValidGraspsException()

        grasp_type = "parallel_jaw"
        if isinstance(grasps[0], SuctionPoint2D):
            grasp_type = "suction"
        elif isinstance(grasps[0], MultiSuctionPoint2D):
            grasp_type = "multi_suction"

        self._logger.info("Sampled %d grasps" % (len(grasps)))
        self._logger.info("Computing the seed set took %.3f sec" %
                          (time() - seed_set_start))

        # Iteratively refit and sample.
        for j in range(self._num_iters):
            self._logger.info("CEM iter %d" % (j))

            # Predict grasps.
            predict_start = time()
            q_values = self._grasp_quality_fn(state,
                                              grasps,
                                              params=self._config)
            self._logger.info("Prediction took %.3f sec" %
                              (time() - predict_start))

            # Sort grasps.
            resample_start = time()
            q_values_and_indices = zip(q_values, np.arange(num_grasps))
            q_values_and_indices = sorted(q_values_and_indices,
                                          key=lambda x: x[0],
                                          reverse=True)

            if self.config["vis"]["grasp_candidates"]:
                # Display each grasp on the original image, colored by
                # predicted success.
                # norm_q_values = ((q_values - np.min(q_values)) /
                #                   (np.max(q_values) - np.min(q_values)))
                norm_q_values = q_values
                title = "Sampled Grasps Iter %d" % (j)
                if self._vis_grasp_affordance_map:
                    self._plot_grasp_affordance_map(
                        state,
                        grasp_affordance_map,
                        grasps=grasps,
                        q_values=norm_q_values,
                        scale=2.0,
                        title=title,
                        save_fname="cem_iter_{}.png".format(j),
                        save_path=state_output_dir)
                display_grasps_and_q_values = zip(grasps, q_values)
                display_grasps_and_q_values = sorted(
                    display_grasps_and_q_values, key=lambda x: x[1])
                vis.figure(size=(GeneralConstants.FIGSIZE,
                                 GeneralConstants.FIGSIZE))
                vis.imshow(rgbd_im.depth,
                           vmin=self.config["vis"]["vmin"],
                           vmax=self.config["vis"]["vmax"])
                for grasp, q in display_grasps_and_q_values:
                    vis.grasp(grasp,
                              scale=2.0,
                              jaw_width=2.0,
                              show_center=False,
                              show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title("Sampled grasps iter %d" % (j))
                filename = None
                if self._logging_dir is not None:
                    filename = os.path.join(self._logging_dir,
                                            "cem_iter_%d.png" % (j))
                vis.show(filename)

            # Fit elite set.
            elite_start = time()
            num_refit = max(int(np.ceil(self._gmm_refit_p * num_grasps)), 1)
            elite_q_values = [i[0] for i in q_values_and_indices[:num_refit]]
            elite_grasp_indices = [
                i[1] for i in q_values_and_indices[:num_refit]
            ]
            elite_grasps = [grasps[i] for i in elite_grasp_indices]
            elite_grasp_arr = np.array([g.feature_vec for g in elite_grasps])

            if self.config["vis"]["elite_grasps"]:
                # Display each grasp on the original image, colored by
                # predicted success.
                norm_q_values = (elite_q_values - np.min(elite_q_values)) / (
                    np.max(elite_q_values) - np.min(elite_q_values))
                vis.figure(size=(GeneralConstants.FIGSIZE,
                                 GeneralConstants.FIGSIZE))
                vis.imshow(rgbd_im.depth,
                           vmin=self.config["vis"]["vmin"],
                           vmax=self.config["vis"]["vmax"])
                for grasp, q in zip(elite_grasps, norm_q_values):
                    vis.grasp(grasp,
                              scale=1.5,
                              show_center=False,
                              show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title("Elite grasps iter %d" % (j))
                filename = None
                if self._logging_dir is not None:
                    filename = os.path.join(self._logging_dir,
                                            "elite_set_iter_%d.png" % (j))
                vis.show(filename)

            # Normalize elite set.
            elite_grasp_mean = np.mean(elite_grasp_arr, axis=0)
            elite_grasp_std = np.std(elite_grasp_arr, axis=0)
            elite_grasp_std[elite_grasp_std == 0] = 1e-6
            elite_grasp_arr = (elite_grasp_arr -
                               elite_grasp_mean) / elite_grasp_std
            self._logger.info("Elite set computation took %.3f sec" %
                              (time() - elite_start))

            # Fit a GMM to the top samples.
            num_components = max(
                int(np.ceil(self._gmm_component_frac * num_refit)), 1)
            uniform_weights = (1.0 / num_components) * np.ones(num_components)
            gmm = GaussianMixture(n_components=num_components,
                                  weights_init=uniform_weights,
                                  reg_covar=self._gmm_reg_covar)
            train_start = time()
            gmm.fit(elite_grasp_arr)
            self._logger.info("GMM fitting with %d components took %.3f sec" %
                              (num_components, time() - train_start))

            # Sample the next grasps.
            grasps = []
            loop_start = time()
            num_tries = 0
            while (len(grasps) < self._num_gmm_samples
                   and num_tries < self._max_resamples_per_iteration):
                # Sample from GMM.
                sample_start = time()
                grasp_vecs, _ = gmm.sample(n_samples=self._num_gmm_samples)
                grasp_vecs = elite_grasp_std * grasp_vecs + elite_grasp_mean
                self._logger.info("GMM sampling took %.3f sec" %
                                  (time() - sample_start))

                # Convert features to grasps and store if in segmask.
                for k, grasp_vec in enumerate(grasp_vecs):
                    feature_start = time()
                    if grasp_type == "parallel_jaw":
                        # Form grasp object.
                        grasp = Grasp2D.from_feature_vec(
                            grasp_vec,
                            width=self._gripper_width,
                            camera_intr=camera_intr)
                    elif grasp_type == "suction":
                        # Read depth and approach axis.
                        u = int(min(max(grasp_vec[1], 0), depth_im.height - 1))
                        v = int(min(max(grasp_vec[0], 0), depth_im.width - 1))
                        grasp_depth = depth_im[u, v, 0]

                        # Approach axis.
                        grasp_axis = -normal_cloud_im[u, v]

                        # Form grasp object.
                        grasp = SuctionPoint2D.from_feature_vec(
                            grasp_vec,
                            camera_intr=camera_intr,
                            depth=grasp_depth,
                            axis=grasp_axis)
                    elif grasp_type == "multi_suction":
                        # Read depth and approach axis.
                        u = int(min(max(grasp_vec[1], 0), depth_im.height - 1))
                        v = int(min(max(grasp_vec[0], 0), depth_im.width - 1))
                        grasp_depth = depth_im[u, v]

                        # Approach_axis.
                        grasp_axis = -normal_cloud_im[u, v]

                        # Form grasp object.
                        grasp = MultiSuctionPoint2D.from_feature_vec(
                            grasp_vec,
                            camera_intr=camera_intr,
                            depth=grasp_depth,
                            axis=grasp_axis)
                    self._logger.debug("Feature vec took %.5f sec" %
                                       (time() - feature_start))

                    bounds_start = time()
                    # Check in bounds.
                    if (state.segmask is None or
                        (grasp.center.y >= 0
                         and grasp.center.y < state.segmask.height
                         and grasp.center.x >= 0
                         and grasp.center.x < state.segmask.width
                         and np.any(state.segmask[int(grasp.center.y),
                                                  int(grasp.center.x)] != 0)
                         and grasp.approach_angle < self._max_approach_angle)
                            and (self._grasp_constraint_fn is None
                                 or self._grasp_constraint_fn(grasp))):

                        # Check validity according to filters.
                        grasps.append(grasp)
                    self._logger.debug("Bounds took %.5f sec" %
                                       (time() - bounds_start))
                    num_tries += 1

            # Check num grasps.
            num_grasps = len(grasps)
            if num_grasps == 0:
                self._logger.warning("No valid grasps could be found")
                raise NoValidGraspsException()
            self._logger.info("Resample loop took %.3f sec" %
                              (time() - loop_start))
            self._logger.info("Resampling took %.3f sec" %
                              (time() - resample_start))

        # Predict final set of grasps.
        predict_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        self._logger.info("Final prediction took %.3f sec" %
                          (time() - predict_start))

        if self.config["vis"]["grasp_candidates"]:
            # Display each grasp on the original image, colored by predicted
            # success.
            # norm_q_values = ((q_values - np.min(q_values)) /
            #                   (np.max(q_values) - np.min(q_values)))
            norm_q_values = q_values
            title = "Final Sampled Grasps"
            if self._vis_grasp_affordance_map:
                self._plot_grasp_affordance_map(
                    state,
                    grasp_affordance_map,
                    grasps=grasps,
                    q_values=norm_q_values,
                    scale=2.0,
                    title=title,
                    save_fname="final_sampled_grasps.png".format(j),
                    save_path=state_output_dir)
            display_grasps_and_q_values = zip(grasps, q_values)
            display_grasps_and_q_values = sorted(display_grasps_and_q_values,
                                                 key=lambda x: x[1])
            vis.figure(size=(GeneralConstants.FIGSIZE,
                             GeneralConstants.FIGSIZE))
            vis.imshow(rgbd_im.depth,
                       vmin=self.config["vis"]["vmin"],
                       vmax=self.config["vis"]["vmax"])
            for grasp, q in display_grasps_and_q_values:
                vis.grasp(grasp,
                          scale=2.0,
                          jaw_width=2.0,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title("Sampled grasps iter %d" % (j))
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir,
                                        "cem_iter_%d.png" % (j))
            vis.show(filename)

        return grasps, q_values

    def _action(self, state):
        """Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            Grasp to execute.
        """
        # Parse state.
        rgbd_im = state.rgbd_im
        depth_im = rgbd_im.depth
        #        camera_intr = state.camera_intr
        #        segmask = state.segmask

        # Plan grasps.
        grasps, q_values = self.action_set(state)

        # Select grasp.
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config["vis"]["grasp_plan"]:
            title = "Best Grasp: d=%.3f, q=%.3f" % (grasp.depth, q_value)
            vis.figure()
            vis.imshow(depth_im,
                       vmin=self.config["vis"]["vmin"],
                       vmax=self.config["vis"]["vmax"])
            vis.grasp(grasp,
                      scale=5.0,
                      show_center=False,
                      show_axis=True,
                      jaw_width=1.0,
                      grasp_axis_width=0.2)
            vis.title(title)
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir, "planned_grasp.png")
            vis.show(filename)

        # Form return image.
        image = depth_im
        if isinstance(self._grasp_quality_fn, GQCnnQualityFunction):
            image_arr, _ = self._grasp_quality_fn.grasps_to_tensors([grasp],
                                                                    state)
            image = DepthImage(image_arr[0, ...], frame=rgbd_im.frame)

        # Return action.
        action = GraspAction(grasp, q_value, image)
        return action


class QFunctionRobustGraspingPolicy(CrossEntropyRobustGraspingPolicy):
    """Optimizes a set of antipodal grasp candidates in image space using the
    cross entropy method with a GQ-CNN that estimates the Q-function
    for use in Q-learning.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters.

    Other Parameters
    ----------------
    reinit_pc1 : bool
        Whether or not to reinitialize the pc1 layer of the GQ-CNN.
    reinit_fc3: bool
        Whether or not to reinitialize the fc3 layer of the GQ-CNN.
    reinit_fc4: bool
        Whether or not to reinitialize the fc4 layer of the GQ-CNN.
    reinit_fc5: bool
        Whether or not to reinitialize the fc5 layer of the GQ-CNN.
    num_seed_samples : int
        Number of candidate to sample in the initial set.
    num_gmm_samples : int
        Number of candidates to sample on each resampling from the GMMs.
    num_iters : int
        Number of sample-and-refit iterations of CEM.
    gmm_refit_p : float
        Top p-% of grasps used for refitting.
    gmm_component_frac : float
        Percentage of the elite set size used to determine number of GMM
        components.
    gmm_reg_covar : float
        Regularization parameters for GMM covariance matrix, enforces diversity
        of fitted distributions.
    deterministic : bool, optional
        Whether to set the random seed to enforce deterministic behavior.
    gripper_width : float, optional
        Width of the gripper in meters.
    """

    def __init__(self, config):
        CrossEntropyRobustGraspingPolicy.__init__(self, config)
        QFunctionRobustGraspingPolicy._parse_config(self)
        self._setup_gqcnn()

    def _parse_config(self):
        """Parses the parameters of the policy."""
        self._reinit_pc1 = self.config["reinit_pc1"]
        self._reinit_fc3 = self.config["reinit_fc3"]
        self._reinit_fc4 = self.config["reinit_fc4"]
        self._reinit_fc5 = self.config["reinit_fc5"]

    def _setup_gqcnn(self):
        """Sets up the GQ-CNN."""
        # Close existing session (from superclass initializer).
        self.gqcnn.close_session()

        # Check valid output size.
        if self.gqcnn.fc5_out_size != 1 and not self._reinit_fc5:
            raise ValueError("Q function must return scalar values")

        # Reinitialize layers
        if self._reinit_fc5:
            self.gqcnn.fc5_out_size = 1

        # TODO(jmahler): Implement reinitialization of pc0.
        self.gqcnn.reinitialize_layers(self._reinit_fc3, self._reinit_fc4,
                                       self._reinit_fc5)
        self.gqcnn.initialize_network(add_softmax=False)


class EpsilonGreedyQFunctionRobustGraspingPolicy(QFunctionRobustGraspingPolicy
                                                 ):
    """Optimizes a set of antipodal grasp candidates in image space
    using the cross entropy method with a GQ-CNN that estimates the
    Q-function for use in Q-learning, and chooses a random antipodal
    grasp with probability epsilon.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters.

    Other Parameters
    ----------------
    epsilon : float
    """

    def __init__(self, config):
        QFunctionRobustGraspingPolicy.__init__(self, config)
        self._parse_config()

    def _parse_config(self):
        """Parses the parameters of the policy."""
        self._epsilon = self.config["epsilon"]

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = val

    def greedy_action(self, state):
        """Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            Grasp to execute.
        """
        return CrossEntropyRobustGraspingPolicy.action(self, state)

    def _action(self, state):
        """Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            Grasp to execute.
        """
        # Take the greedy action with prob 1-epsilon.
        if np.random.rand() > self.epsilon:
            self._logger.debug("Taking greedy action")
            return CrossEntropyRobustGraspingPolicy.action(self, state)

        # Otherwise take a random action.
        self._logger.debug("Taking random action")

        # Check valid input.
        if not isinstance(state, RgbdImageState):
            raise ValueError("Must provide an RGB-D image state.")

        # Parse state.
        rgbd_im = state.rgbd_im
        camera_intr = state.camera_intr
        segmask = state.segmask

        # Sample random antipodal grasps.
        grasps = self._grasp_sampler.sample(
            rgbd_im,
            camera_intr,
            self._num_seed_samples,
            segmask=segmask,
            visualize=self.config["vis"]["grasp_sampling"],
            constraint_fn=self._grasp_constraint_fn,
            seed=self._seed)

        num_grasps = len(grasps)
        if num_grasps == 0:
            self._logger.warning("No valid grasps could be found")
            raise NoValidGraspsException()

        # Choose a grasp uniformly at random.
        grasp_ind = np.random.choice(num_grasps, size=1)[0]
        grasp = grasps[grasp_ind]
        depth = grasp.depth

        # Create transformed image.
        image_tensor, pose_tensor = self.grasps_to_tensors([grasp], state)
        image = DepthImage(image_tensor[0, ...])

        # Predict prob success.
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)
        q_value = output_arr[0, -1]

        # Visualize planned grasp.
        if self.config["vis"]["grasp_plan"]:
            scale_factor = self.gqcnn.im_width / self._crop_width
            scaled_camera_intr = camera_intr.resize(scale_factor)
            vis_grasp = Grasp2D(Point(image.center),
                                0.0,
                                depth,
                                width=self._gripper_width,
                                camera_intr=scaled_camera_intr)
            vis.figure()
            vis.imshow(image)
            vis.grasp(vis_grasp, scale=1.5, show_center=False, show_axis=True)
            vis.title("Best Grasp: d=%.3f, q=%.3f" % (depth, q_value))
            vis.show()

        # Return action.
        return GraspAction(grasp, q_value, image)


class CompositeGraspingPolicy(Policy):
    """Grasping policy composed of multiple sub-policies.

    Attributes
    ----------
    policies : dict mapping str to `gqcnn.GraspingPolicy`
        Key-value dict mapping policy names to grasping policies.
    """

    def __init__(self, policies):
        self._policies = policies
        self._logger = Logger.get_logger(self.__class__.__name__,
                                         log_file=None,
                                         global_log_file=True)

    @property
    def policies(self):
        return self._policies

    def subpolicy(self, name):
        return self._policies[name]

    def set_constraint_fn(self, constraint_fn):
        for policy in self._policies:
            policy.set_constraint_fn(constraint_fn)


class PriorityCompositeGraspingPolicy(CompositeGraspingPolicy):
    def __init__(self, policies, priority_list):
        # Check validity.
        for name in priority_list:
            if str(name) not in policies:
                raise ValueError(
                    "Policy named %s is not in the list of policies!" % (name))

        self._priority_list = priority_list
        CompositeGraspingPolicy.__init__(self, policies)

    @property
    def priority_list(self):
        return self._priority_list

    def action(self, state, policy_subset=None, min_q_value=-1.0):
        """Returns an action for a given state.
        """
        action = None
        i = 0
        max_q = min_q_value

        while action is None or (max_q <= min_q_value
                                 and i < len(self._priority_list)):
            name = self._priority_list[i]
            if policy_subset is not None and name not in policy_subset:
                i += 1
                continue
            self._logger.info("Planning action for sub-policy {}".format(name))
            try:
                action = self.policies[name].action(state)
                action.policy_name = name
                max_q = action.q_value
            except NoValidGraspsException:
                pass
            i += 1
        if action is None:
            raise NoValidGraspsException()
        return action

    def action_set(self, state, policy_subset=None, min_q_value=-1.0):
        """Returns an action for a given state.
        """
        actions = None
        q_values = None
        i = 0
        max_q = min_q_value
        while actions is None or (max_q <= min_q_value
                                  and i < len(self._priority_list)):
            name = self._priority_list[i]
            if policy_subset is not None and name not in policy_subset:
                i += 1
                continue
            self._logger.info(
                "Planning action set for sub-policy {}".format(name))
            try:
                actions, q_values = self.policies[name].action_set(state)
                for action in actions:
                    action.policy_name = name
                max_q = np.max(q_values)
            except NoValidGraspsException:
                pass
            i += 1
        if actions is None:
            raise NoValidGraspsException()
        return actions, q_values


class GreedyCompositeGraspingPolicy(CompositeGraspingPolicy):
    def __init__(self, policies):
        CompositeGraspingPolicy.__init__(self, policies)

    def action(self, state, policy_subset=None, min_q_value=-1.0):
        """Returns an action for a given state."""
        # Compute all possible actions.
        actions = []
        for name, policy in self.policies.items():
            if policy_subset is not None and name not in policy_subset:
                continue
            try:
                action = policy.action(state)
                action.policy_name = name
                actions.append()
            except NoValidGraspsException:
                pass

        if len(actions) == 0:
            raise NoValidGraspsException()

        # Rank based on q value.
        actions = sorted(actions, key=lambda x: x.q_value, reverse=True)
        return actions[0]

    def action_set(self, state, policy_subset=None, min_q_value=-1.0):
        """Returns an action for a given state."""
        actions = []
        q_values = []
        for name, policy in self.policies.items():
            if policy_subset is not None and name not in policy_subset:
                continue
            try:
                action_set, q_vals = self.policies[name].action_set(state)
                for action in action_set:
                    action.policy_name = name
                actions.extend(action_set)
                q_values.extend(q_vals)
            except NoValidGraspsException:
                continue
        if actions is None:
            raise NoValidGraspsException()
        return actions, q_values
