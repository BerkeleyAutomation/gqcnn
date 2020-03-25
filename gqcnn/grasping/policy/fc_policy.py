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

Fully-Convolutional GQ-CNN grasping policies.

Author
------
Vishal Satish
"""
from abc import abstractmethod
import os

import matplotlib.pyplot as plt
import numpy as np

from autolab_core import Point, Logger
from perception import DepthImage
from visualization import Visualizer2D as vis

from ...grasping import Grasp2D, SuctionPoint2D
from ...utils import GeneralConstants, NoValidGraspsException

from .enums import SamplingMethod
from .policy import GraspingPolicy, GraspAction


class FullyConvolutionalGraspingPolicy(GraspingPolicy):
    """Abstract grasp sampling policy class using Fully-Convolutional GQ-CNN
    network."""

    def __init__(self, cfg, filters=None):
        """
        Parameters
        ----------
        cfg : dict
            Python dictionary of policy configuration parameters.
        filters : dict
            Python dictionary of kinematic filters to apply.
        """
        GraspingPolicy.__init__(self, cfg, init_sampler=False)

        # Init logger.
        self._logger = Logger.get_logger(self.__class__.__name__)

        self._cfg = cfg
        self._sampling_method = self._cfg["sampling_method"]

        # GQ-CNN parameters.
        self._gqcnn_stride = self._cfg["gqcnn_stride"]
        self._gqcnn_recep_h = self._cfg["gqcnn_recep_h"]
        self._gqcnn_recep_w = self._cfg["gqcnn_recep_w"]

        # Grasp filtering.
        self._filters = filters
        self._max_grasps_to_filter = self._cfg["max_grasps_to_filter"]
        self._filter_grasps = self._cfg["filter_grasps"]

        # Visualization parameters.
        self._vis_config = self._cfg["policy_vis"]
        self._vis_scale = self._vis_config["scale"]
        self._vis_show_axis = self._vis_config["show_axis"]

        self._num_vis_samples = self._vis_config["num_samples"]
        self._vis_actions_2d = self._vis_config["actions_2d"]
        self._vis_actions_3d = self._vis_config["actions_3d"]

        self._vis_affordance_map = self._vis_config["affordance_map"]

        self._vis_output_dir = None
        # If this exists in the config then all visualizations will be logged
        # here instead of displayed.
        if "output_dir" in self._vis_config:
            self._vis_output_dir = self._vis_config["output_dir"]
            self._state_counter = 0

    def _unpack_state(self, state):
        """Unpack information from the provided `RgbdImageState`."""
        # TODO(vsatish): Don't access raw depth data like this.
        return (state.rgbd_im.depth, state.rgbd_im.depth._data,
                state.segmask.raw_data, state.camera_intr)

    def _mask_predictions(self, preds, raw_segmask):
        """Mask the given predictions with the given segmask, setting the rest
        to 0.0."""
        preds_masked = np.zeros_like(preds)
        raw_segmask_cropped = raw_segmask[self._gqcnn_recep_h //
                                          2:raw_segmask.shape[0] -
                                          self._gqcnn_recep_h //
                                          2, self._gqcnn_recep_w //
                                          2:raw_segmask.shape[1] -
                                          self._gqcnn_recep_w // 2, 0]
        raw_segmask_downsampled = raw_segmask_cropped[::self._gqcnn_stride, ::
                                                      self._gqcnn_stride]
        if raw_segmask_downsampled.shape[0] != preds.shape[1]:
            raw_segmask_downsampled_new = np.zeros(preds.shape[1:3])
            raw_segmask_downsampled_new[:raw_segmask_downsampled.
                                        shape[0], :raw_segmask_downsampled.
                                        shape[1]] = raw_segmask_downsampled
            raw_segmask_downsampled = raw_segmask_downsampled_new
        nonzero_mask_ind = np.where(raw_segmask_downsampled > 0)
        preds_masked[:, nonzero_mask_ind[0],
                     nonzero_mask_ind[1]] = preds[:, nonzero_mask_ind[0],
                                                  nonzero_mask_ind[1]]
        return preds_masked

    def _sample_predictions(self, preds, num_actions):
        """Sample predictions."""
        dim2 = preds.shape[2]
        dim1 = preds.shape[1]
        dim3 = preds.shape[3]
        preds_flat = np.ravel(preds)
        pred_ind_flat = self._sample_predictions_flat(preds_flat, num_actions)
        pred_ind = np.zeros((num_actions, len(preds.shape)), dtype=np.int32)
        for idx in range(num_actions):
            pred_ind[idx, 0] = pred_ind_flat[idx] // (dim2 * dim1 * dim3)
            pred_ind[idx, 1] = (pred_ind_flat[idx] -
                                (pred_ind[idx, 0] *
                                 (dim2 * dim1 * dim3))) // (dim2 * dim3)
            pred_ind[idx, 2] = (pred_ind_flat[idx] - (pred_ind[idx, 0] *
                                                      (dim2 * dim1 * dim3)) -
                                (pred_ind[idx, 1] * (dim2 * dim3))) // dim3
            pred_ind[idx, 3] = (pred_ind_flat[idx] - (pred_ind[idx, 0] *
                                                      (dim2 * dim1 * dim3)) -
                                (pred_ind[idx, 1] * (dim2 * dim3))) % dim3
        return pred_ind

    def _sample_predictions_flat(self, preds_flat, num_samples):
        """Helper function to do the actual sampling."""
        if num_samples == 1:
            # `argmax` is faster than `argpartition` for special case of single
            # sample.
            if self._sampling_method == SamplingMethod.TOP_K:
                return [np.argmax(preds_flat)]
            elif self._sampling_method == SamplingMethod.UNIFORM:
                nonzero_ind = np.where(preds_flat > 0)[0]
                return np.random.choice(nonzero_ind)
            else:
                raise ValueError("Invalid sampling method: {}".format(
                    self._sampling_method))
        else:
            if self._sampling_method == "top_k":
                return np.argpartition(preds_flat,
                                       -1 * num_samples)[-1 * num_samples:]
            elif self._sampling_method == "uniform":
                nonzero_ind = np.where(preds_flat > 0)[0]
                if nonzero_ind.shape[0] == 0:
                    raise NoValidGraspsException(
                        "No grasps with nonzero quality")
                return np.random.choice(nonzero_ind, size=num_samples)
            else:
                raise ValueError("Invalid sampling method: {}".format(
                    self._sampling_method))

    @abstractmethod
    def _get_actions(self, preds, ind, images, depths, camera_intr,
                     num_actions):
        """Generate the actions to be returned."""
        pass

    @abstractmethod
    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr,
                      num_actions):
        """Visualize the actions in 3D."""
        pass

    @abstractmethod
    def _visualize_affordance_map(self,
                                  preds,
                                  depth_im,
                                  scale,
                                  plot_max=True,
                                  output_dir=None):
        """Visualize an affordance map of the network predictions overlayed on
        the depth image."""
        pass

    def _visualize_2d(self,
                      actions,
                      preds,
                      wrapped_depth_im,
                      num_actions,
                      scale,
                      show_axis,
                      output_dir=None):
        """Visualize the actions in 2D."""
        self._logger.info("Visualizing actions in 2d...")

        # Plot actions in 2D.
        vis.figure()
        vis.imshow(wrapped_depth_im)
        for i in range(num_actions):
            vis.grasp(actions[i].grasp,
                      scale=scale,
                      show_axis=show_axis,
                      color=plt.cm.RdYlGn(actions[i].q_value))
        vis.title("Top {} Grasps".format(num_actions))
        if output_dir is not None:
            vis.savefig(os.path.join(output_dir, "top_grasps.png"))
        else:
            vis.show()

    def _filter(self, actions):
        """Filter actions."""
        for action in actions:
            valid = True
            for filter_name, is_valid in self._filters.items():
                if not is_valid(action.grasp):
                    self._logger.info(
                        "Grasp {} is not valid with filter {}".format(
                            action.grasp, filter_name))
                    valid = False
                    break
            if valid:
                return action
        raise NoValidGraspsException("No grasps found after filtering!")

    @abstractmethod
    def _gen_images_and_depths(self, depth, segmask):
        """Generate inputs for the grasp quality function."""
        pass

    def _action(self, state, num_actions=1):
        """Plan action(s)."""
        if self._filter_grasps:
            assert self._filters is not None, ("Trying to filter grasps but no"
                                               " filters were provided!")
            assert num_actions == 1, ("Filtering support is only implemented"
                                      " for single actions!")
            num_actions = self._max_grasps_to_filter

        # Set up log dir for state visualizations.
        state_output_dir = None
        if self._vis_output_dir is not None:
            state_output_dir = os.path.join(
                self._vis_output_dir,
                "state_{}".format(str(self._state_counter).zfill(5)))
            if not os.path.exists(state_output_dir):
                os.makedirs(state_output_dir)
            self._state_counter += 1

        # Unpack the `RgbdImageState`.
        wrapped_depth, raw_depth, raw_seg, camera_intr = self._unpack_state(
            state)

        # Predict.
        images, depths = self._gen_images_and_depths(raw_depth, raw_seg)
        preds = self._grasp_quality_fn.quality(images, depths)

        # Get success probablility predictions only (this is needed because the
        # output of the network is pairs of (p_failure, p_success)).
        preds_success_only = preds[:, :, :, 1::2]

        # Mask predicted success probabilities with the cropped and downsampled
        # object segmask so we only sample grasps on the objects.
        preds_success_only = self._mask_predictions(preds_success_only,
                                                    raw_seg)

        # If we want to visualize more than one action, we have to sample more.
        # TODO(vsatish): If this is used with the "top_k" sampling method, the
        # final returned action is not the best because the argpartition does
        # not sort the partitioned indices. Fix this.
        num_actions_to_sample = self._num_vis_samples if (
            self._vis_actions_2d or self._vis_actions_3d) else num_actions

        if (self._sampling_method == SamplingMethod.TOP_K
                and self._num_vis_samples):
            self._logger.warning("FINAL GRASP RETURNED IS NOT THE BEST!")

        # Sample num_actions_to_sample indices from the success predictions.
        sampled_ind = self._sample_predictions(preds_success_only,
                                               num_actions_to_sample)

        # Wrap actions to be returned.
        actions = self._get_actions(preds_success_only, sampled_ind, images,
                                    depths, camera_intr, num_actions_to_sample)

        # Filter grasps.
        if self._filter_grasps:
            actions = sorted(actions,
                             reverse=True,
                             key=lambda action: action.q_value)
            actions = [self._filter(actions)]

        # Visualize.
        if self._vis_actions_3d:
            self._logger.info("Generating 3D Visualization...")
            self._visualize_3d(actions, wrapped_depth, camera_intr,
                               num_actions_to_sample)
        if self._vis_actions_2d:
            self._logger.info("Generating 2D visualization...")
            self._visualize_2d(actions,
                               preds_success_only,
                               wrapped_depth,
                               num_actions_to_sample,
                               self._vis_scale,
                               self._vis_show_axis,
                               output_dir=state_output_dir)
        if self._vis_affordance_map:
            self._visualize_affordance_map(preds_success_only,
                                           wrapped_depth,
                                           self._vis_scale,
                                           output_dir=state_output_dir)

        return actions[-1] if (self._filter_grasps or num_actions == 1
                               ) else actions[-(num_actions + 1):]

    def action_set(self, state, num_actions):
        """Plan a set of actions.

        Parameters
        ----------
        state : :obj:`gqcnn.RgbdImageState`
            The RGBD image state.
        num_actions : int
            The number of actions to plan.

        Returns
        ------
        list of :obj:`gqcnn.GraspAction`
            The planned grasps.
        """
        return [
            action.grasp
            for action in self._action(state, num_actions=num_actions)
        ]


class FullyConvolutionalGraspingPolicyParallelJaw(
        FullyConvolutionalGraspingPolicy):
    """Parallel jaw grasp sampling policy using the FC-GQ-CNN."""

    def __init__(self, cfg, filters=None):
        """
        Parameters
        ----------
        cfg : dict
            Python dictionary of policy configuration parameters.
        filters : dict
            Python dictionary of functions to apply to filter invalid grasps.
        """
        FullyConvolutionalGraspingPolicy.__init__(self, cfg, filters=filters)

        self._gripper_width = self._cfg["gripper_width"]

        # Depth sampling parameters.
        self._num_depth_bins = self._cfg["num_depth_bins"]
        self._depth_offset = 0.0  # Hard offset from the workspace.
        if "depth_offset" in self._cfg:
            self._depth_offset = self._cfg["depth_offset"]

    def _sample_depths(self, raw_depth_im, raw_seg):
        """Sample depths from the raw depth image."""
        max_depth = np.max(raw_depth_im) + self._depth_offset

        # For sampling the min depth, we only sample from the portion of the
        # depth image in the object segmask because sometimes the rim of the
        # bin is not properly subtracted out of the depth image.
        raw_depth_im_segmented = np.ones_like(raw_depth_im)
        raw_depth_im_segmented[np.where(raw_seg > 0)] = raw_depth_im[np.where(
            raw_seg > 0)]
        min_depth = np.min(raw_depth_im_segmented) + self._depth_offset

        depth_bin_width = (max_depth - min_depth) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1))
        for i in range(self._num_depth_bins):
            depths[i][0] = min_depth + (i * depth_bin_width +
                                        depth_bin_width / 2)
        return depths

    def _get_actions(self, preds, ind, images, depths, camera_intr,
                     num_actions):
        """Generate the actions to be returned."""
        actions = []
        # TODO(vsatish): These should use the max angle instead.
        ang_bin_width = GeneralConstants.PI / preds.shape[-1]
        for i in range(num_actions):
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            ang_idx = ind[i, 3]
            center = Point(
                np.asarray([
                    w_idx * self._gqcnn_stride + self._gqcnn_recep_w // 2,
                    h_idx * self._gqcnn_stride + self._gqcnn_recep_h // 2
                ]))
            ang = GeneralConstants.PI / 2 - (ang_idx * ang_bin_width +
                                             ang_bin_width / 2)
            depth = depths[im_idx, 0]
            grasp = Grasp2D(center,
                            ang,
                            depth,
                            width=self._gripper_width,
                            camera_intr=camera_intr)
            grasp_action = GraspAction(grasp,
                                       preds[im_idx, h_idx, w_idx, ang_idx],
                                       DepthImage(images[im_idx]))
            actions.append(grasp_action)
        return actions

    def _gen_images_and_depths(self, depth, segmask):
        """Replicate the depth image and sample corresponding depths."""
        depths = self._sample_depths(depth, segmask)
        images = np.tile(np.asarray([depth]), (self._num_depth_bins, 1, 1, 1))
        return images, depths

    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr,
                      num_actions):
        """Visualize the actions in 3D."""
        raise NotImplementedError

    def _visualize_affordance_map(self, preds, depth_im):
        """Visualize an affordance map of the network predictions overlayed on
        the depth image."""
        raise NotImplementedError


class FullyConvolutionalGraspingPolicySuction(FullyConvolutionalGraspingPolicy
                                              ):
    """Suction grasp sampling policy using the FC-GQ-CNN."""

    def _get_actions(self, preds, ind, images, depths, camera_intr,
                     num_actions):
        """Generate the actions to be returned."""
        depth_im = DepthImage(images[0], frame=camera_intr.frame)
        point_cloud_im = camera_intr.deproject_to_image(depth_im)
        normal_cloud_im = point_cloud_im.normal_cloud_im()

        actions = []
        for i in range(num_actions):
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            center = Point(
                np.asarray([
                    w_idx * self._gqcnn_stride + self._gqcnn_recep_w // 2,
                    h_idx * self._gqcnn_stride + self._gqcnn_recep_h // 2
                ]))
            axis = -normal_cloud_im[center.y, center.x]
            if np.linalg.norm(axis) == 0:
                continue
            depth = depth_im[center.y, center.x, 0]
            if depth == 0.0:
                continue
            grasp = SuctionPoint2D(center,
                                   axis=axis,
                                   depth=depth,
                                   camera_intr=camera_intr)
            grasp_action = GraspAction(grasp, preds[im_idx, h_idx, w_idx, 0],
                                       DepthImage(images[im_idx]))
            actions.append(grasp_action)
        return actions

    def _visualize_affordance_map(self,
                                  preds,
                                  depth_im,
                                  scale,
                                  plot_max=True,
                                  output_dir=None):
        """Visualize an affordance map of the network predictions overlayed on
        the depth image."""
        self._logger.info("Visualizing affordance map...")

        affordance_map = preds[0, ..., 0]
        tf_depth_im = depth_im.crop(
            depth_im.shape[0] - self._gqcnn_recep_h, depth_im.shape[1] -
            self._gqcnn_recep_w).resize(1.0 / self._gqcnn_stride)

        # Plot.
        vis.figure()
        vis.imshow(tf_depth_im)
        plt.imshow(affordance_map,
                   cmap=plt.cm.RdYlGn,
                   alpha=0.3,
                   vmin=0.0,
                   vmax=1.0)
        if plot_max:
            affordance_argmax = np.unravel_index(np.argmax(affordance_map),
                                                 affordance_map.shape)
            plt.scatter(affordance_argmax[1],
                        affordance_argmax[0],
                        c="black",
                        marker=".",
                        s=scale * 25)
        vis.title("Grasp Affordance Map")
        if output_dir is not None:
            vis.savefig(os.path.join(output_dir, "grasp_affordance_map.png"))
        else:
            vis.show()

    def _gen_images_and_depths(self, depth, segmask):
        """Extend the image to a 4D tensor."""
        # TODO(vsatish): Depth should be made an optional input to the network.
        return np.expand_dims(depth, 0), np.array([-1])

    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr,
                      num_actions):
        """Visualize the actions in 3D."""
        raise NotImplementedError
