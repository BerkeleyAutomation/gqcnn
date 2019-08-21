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
Fully-Convolutional GQ-CNN grasping policies.
Author: Vishal Satish
"""
import math
from abc import abstractmethod, ABCMeta
import os
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from ambicore import DepthImage, Transform, Visualizer2D as vis
from gqcnn.grasping import Grasp2D, SuctionPoint2D, MultiSuctionPoint2D
from gqcnn.utils import NoValidGraspsException, GripperMode

from .enums import SamplingMethod
from .policy import GraspingPolicy, GraspAction

MASKED_FLAG = -1
DEFAULT_RESCALE_FACTOR = 1.0
DEFAULT_KERNEL_SIZE = 5

class FullyConvolutionalGraspingPolicy(GraspingPolicy):
    """Abstract grasp sampling policy class using Fully-Convolutional GQ-CNN network."""
    __metaclass__ = ABCMeta

    def __init__(self, cfg, filters=None):
        """
        Parameters
        ----------
        cfg : dict
            python dictionary of policy configuration parameters
        filters : dict
            python dictionary of kinematic filters to apply 
        """
        GraspingPolicy.__init__(self, cfg, init_sampler=False)

        # init logger
        self._logger = logging.getLogger(self.__class__.__name__)

        self._cfg = cfg
        self._sampling_method = self._cfg['sampling_method']

        # gqcnn parameters
        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']

        # grasp filtering
        self._filters = filters
        self._max_grasps_to_filter = self._cfg['max_grasps_to_filter']
        self._filter_grasps = self._cfg['filter_grasps']

        # visualization parameters
        self._vis_config = self._cfg['policy_vis']
        self._vis_scale = self._vis_config['scale']
        self._vis_show_axis = self._vis_config['show_axis']
        
        self._num_vis_samples = self._vis_config['num_samples']
        self._vis_actions_2d = self._vis_config['actions_2d']
        self._vis_actions_3d = self._vis_config['actions_3d']

        self._vis_affordance_map = self._vis_config['affordance_map']

        self._vis_output_dir = None
        if 'output_dir' in self._vis_config: # if this exists in the config then all visualizations will be logged here instead of displayed
            self._vis_output_dir = self._vis_config['output_dir']
            self._state_counter = 0

    def _unpack_state(self, state):
        """Unpack information from the RgbdImageState"""
        depth = state.rgbd_im.depth
        depth_data = state.rgbd_im.depth.data
        if depth_data.ndim < 3:
            depth_data = depth_data[:,:,np.newaxis]
        segmask_data = state.segmask.data
        if segmask_data.ndim < 3:
            segmask_data = segmask_data[:,:,np.newaxis]
        return depth, depth_data, segmask_data, state.camera_intr #TODO: @Vishal don't access raw depth data like this
       
    def _mask_predictions(self, preds, raw_segmask):
        """Mask the given predictions with the given segmask, setting the rest to 0.0."""
        preds_masked = MASKED_FLAG * np.ones_like(preds)
        raw_segmask_cropped = raw_segmask[self._gqcnn_recep_h // 2:raw_segmask.shape[0] - self._gqcnn_recep_h // 2, self._gqcnn_recep_w // 2:raw_segmask.shape[1] - self._gqcnn_recep_w // 2, 0]
        raw_segmask_downsampled = raw_segmask_cropped[::self._gqcnn_stride, ::self._gqcnn_stride]

        if raw_segmask_downsampled.shape[0] != preds.shape[1]:
            raw_segmask_downsampled_new = np.zeros(preds.shape[1:3])
            raw_segmask_downsampled_new[:raw_segmask_downsampled.shape[0], :raw_segmask_downsampled.shape[1]] = raw_segmask_downsampled
            raw_segmask_downsampled = raw_segmask_downsampled_new
        nonzero_mask_ind = np.where(raw_segmask_downsampled > 0)
        preds_masked[:, nonzero_mask_ind[0], nonzero_mask_ind[1]] = preds[:, nonzero_mask_ind[0], nonzero_mask_ind[1]]
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
            pred_ind[idx, 1] = (pred_ind_flat[idx] - (pred_ind[idx, 0] * (dim2 * dim1 * dim3))) // (dim2 * dim3)
            pred_ind[idx, 2] = (pred_ind_flat[idx] - (pred_ind[idx, 0] * (dim2 * dim1 * dim3)) - (pred_ind[idx, 1] * (dim2 * dim3))) // dim3
            pred_ind[idx, 3] = (pred_ind_flat[idx] - (pred_ind[idx, 0] * (dim2 * dim1 * dim3)) - (pred_ind[idx, 1] * (dim2 * dim3))) % dim3
        return pred_ind

    def _sample_predictions_flat(self, preds_flat, num_samples):
        """Helper function to do the actual sampling."""
        if num_samples == 1: # argmax() is faster than argpartition() for special case of single sample
            if self._sampling_method == SamplingMethod.TOP_K:
                return [np.argmax(preds_flat)]
            elif self._sampling_method == SamplingMethod.UNIFORM:
                nonzero_ind = np.where(preds_flat > 0)[0] 
                return np.random.choice(nonzero_ind)
            else:
                raise ValueError('Invalid sampling method: {}'.format(self._sampling_method))
        else:
            if self._sampling_method == 'top_k':
                return np.argpartition(preds_flat, -1 * num_samples)[-1 * num_samples:]
            elif self._sampling_method == 'uniform':
                nonzero_ind = np.where(preds_flat >= 0)[0]
                if nonzero_ind.shape[0] == 0:
                    raise NoValidGraspsException('No grasps with nonzero quality')
                return np.random.choice(nonzero_ind, size=num_samples)
            else:
                raise ValueError('Invalid sampling method: {}'.format(self._sampling_method))

    @abstractmethod
    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions):
        """Generate the actions to be returned."""
        pass

    @abstractmethod
    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """Visualize the actions in 3D."""
        pass

    @abstractmethod
    def _visualize_affordance_map(self, preds, depth_im, scale, plot_max=True, output_dir=None):
        """Visualize an affordance map of the network predictions overlayed on the depth image."""
        pass

    def _visualize_2d(self, actions, preds, wrapped_depth_im, num_actions, scale, show_axis, output_dir=None):
        """Visualize the actions in 2D."""
        self._logger.info('Visualizing actions in 2d...')

        # plot actions in 2D
        vis.figure()
        vis.imshow(DepthImage(wrapped_depth_im.data))
        actions.sort(key = lambda x: x.q_value)
        for i in range(len(actions)):
            vis.grasp(actions[i].grasp, scale=scale, show_axis=show_axis, color=plt.cm.RdYlGn(actions[i].q_value))
        vis.title('Top {} Grasps'.format(num_actions))
        if output_dir is not None:
            vis.savefig(os.path.join(output_dir, 'top_grasps.png'))
        else:
            vis.show()

    def _filter(self, actions):
        """Filter actions."""
        for action in actions:
            valid = True
            for filter_name, is_valid in self._filters.items():
                if not is_valid(action.grasp):
                    self._logger.info('Grasp {} is not valid with filter {}'.format(action.grasp, filter_name))
                    valid = False
                    break
            if valid:
                return action
        raise NoValidGraspsException('No grasps found after filtering!')

    @abstractmethod
    def _gen_images_and_depths(self, depth, segmask):
        """Generate inputs for the grasp quality function."""
        pass 

    def _action(self, state, num_actions=1, save_inputs=False):
        """Plan action(s)."""
        if self._filter_grasps:
            assert self._filters is not None, 'Trying to filter grasps but no filters were provided!'
            assert num_actions == 1, 'Filtering support is only implemented for single actions!'
            num_actions = self._max_grasps_to_filter

        # set up log dir for state visualizations
        state_output_dir = None
        if self._vis_output_dir is not None:
            state_output_dir = os.path.join(self._vis_output_dir, 'state_{}'.format(str(self._state_counter).zfill(5)))
            if not os.path.exists(state_output_dir):
                os.makedirs(state_output_dir)
            self._state_counter += 1

        # unpack the RgbdImageState
        wrapped_depth, raw_depth, raw_seg, camera_intr = self._unpack_state(state)

        # save raw inputs and outputs
        if save_inputs:
            state.rgbd_im.save('rgbd.npy')
            state.rgbd_im.depth.save('depth.npy')
            state.rgbd_im.color.save('color.png')
            state.camera_intr.save('camera.intr')
            state.segmask.save('segmask.png')

        # predict
        images, depths = self._gen_images_and_depths(raw_depth, raw_seg)

        pred_start = time.time()
        preds = self._grasp_quality_fn.quality(images, depths)
        pred_stop = time.time()
        self._logger.debug('Prediction took %.3f sec' %(pred_stop-pred_start))

        # get success probablility predictions only (this is needed because the output of the net is pairs of (p_failure, p_success))
        preds_success_only = preds[:, :, :, 1::2]
        
        # mask predicted success probabilities with the cropped and downsampled object segmask so we only sample grasps on the objects
        preds_success_only = self._mask_predictions(preds_success_only, raw_seg) 

        # if we want to visualize more than one action, we have to sample more
        num_actions_to_sample = self._num_vis_samples if (self._vis_actions_2d or self._vis_actions_3d) else num_actions #TODO: @Vishal if this is used with the 'top_k' sampling method, the final returned action is not the best because the argpartition does not sort the partitioned indices 

        # sample num_actions_to_sample indices from the success predictions
        sampled_ind = self._sample_predictions(preds_success_only, num_actions_to_sample)
        
        # wrap actions to be returned
        actions_start = time.time()
        actions = self._get_actions(preds_success_only, sampled_ind, images, depths, camera_intr, num_actions_to_sample)
        actions_stop = time.time()
        self._logger.debug('Action postprocessing took %.3f sec' %(actions_stop-actions_start))
        
        # filter grasps
        if self._filter_grasps:
            actions.sort(reverse=True, key=lambda action: action.q_value)
            actions = [self._filter(actions)]

        # visualize
        if self._vis_actions_3d:
            self._logger.info('Generating 3D Visualization...')
            self._visualize_3d(actions, wrapped_depth, camera_intr, num_actions_to_sample)
        if self._vis_actions_2d:
            self._logger.info('Generating 2D visualization...')
            self._visualize_2d(actions, preds_success_only, wrapped_depth, num_actions_to_sample, self._vis_scale, self._vis_show_axis, output_dir=state_output_dir)
        if self._vis_affordance_map:
            self._visualize_affordance_map(preds_success_only, wrapped_depth, self._vis_scale, output_dir=state_output_dir)

        actions.sort(reverse=True, key=lambda action: action.q_value)
        return actions[0] if (self._filter_grasps or num_actions == 1) else actions[:num_actions]

    def action_set(self, state, num_actions, policy_subset=None, min_q_value=-1):
        """ Plan a set of actions.

        Parameters
        ----------
        state : :obj:`gqcnn.RgbdImageState`
            the RGBD Image State
        num_actions : int
            the number of actions to plan

        Returns
        ------
        list of :obj:`gqcnn.GraspAction`
            the planned grasps
        """
        return [action for action in self._action(state, num_actions=num_actions)]

class FullyConvolutionalGraspingPolicyParallelJaw(FullyConvolutionalGraspingPolicy):
    """Parallel jaw grasp sampling policy using Fully-Convolutional GQ-CNN network."""
    def __init__(self, cfg, filters=None):
        """
        Parameters
        ----------
        cfg : dict
            python dictionary of policy configuration parameters
        filters : dict
            python dictionary of kinematic filters to apply 
        """
        FullyConvolutionalGraspingPolicy.__init__(self, cfg, filters=filters)

        self._gripper_width = self._cfg['gripper_width']

        # depth sampling parameters
        self._num_depth_bins = self._cfg['num_depth_bins']

        #TODO: ask Jeff what this is for again
        self._depth_offset = 0.0
        if 'depth_offset' in self._cfg.keys():
            self._depth_offset = self._cfg['depth_offset']

    def _sample_depths(self, raw_depth_im, raw_seg):
        """Sample depths from the raw depth image."""
        max_depth = np.percentile(raw_depth_im, 95) + self._depth_offset

        # for sampling the min depth, we only sample from the portion of the depth image in the object segmask because sometimes the rim of the bin is not properly subtracted out of the depth image
        raw_depth_im_segmented = np.zeros(raw_depth_im.shape)
        raw_depth_im_segmented[raw_seg > 0] = raw_depth_im[raw_seg > 0]

        min_depth = self._depth_offset
        if np.any(raw_seg > 0):
            min_depth = np.percentile(raw_depth_im_segmented[raw_seg > 0], 5) + self._depth_offset
        
        depth_bin_width = (max_depth - min_depth) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_depth + (i * depth_bin_width + depth_bin_width / 2)

        return depths

    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions):
        """Generate the actions to be returned."""
        actions = []

        max_angle = math.pi
        bin_width = max_angle / preds.shape[-1]

        for i in range(num_actions):
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            ang_idx = ind[i, 3]
            center = np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2])
            ang = ang_idx * bin_width + bin_width / 2
            depth = depths[im_idx, 0]
            grasp = Grasp2D(center, ang, depth, width=self._gripper_width, camera_intr=camera_intr)
            q_value = preds[im_idx, h_idx, w_idx, ang_idx]
            if q_value == MASKED_FLAG:
                continue

            grasp_action = GraspAction(grasp,
                                       q_value,
                                       DepthImage(images[im_idx]))
            actions.append(grasp_action)
        return actions

    def _gen_images_and_depths(self, depth, segmask):
        """Replicate the depth image and sample corresponding depths."""
        depths = self._sample_depths(depth, segmask)
        images = np.tile(np.asarray([depth]), (self._num_depth_bins, 1, 1, 1))
        return images, depths

    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """Visualize the actions in 3D."""
        raise NotImplementedError

    def _visualize_affordance_map(self, preds, depth_im):
        """Visualize an affordance map of the network predictions overlayed on the depth image."""
        raise NotImplementedError

class FullyConvolutionalGraspingPolicySuction(FullyConvolutionalGraspingPolicy):
    """Suction grasp sampling policy using Fully-Convolutional GQ-CNN network."""
    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions,
                     rescale_factor=DEFAULT_RESCALE_FACTOR, kernel_size=DEFAULT_KERNEL_SIZE):
        """Generate the actions to be returned."""
        depth_im = DepthImage(images[0], frame=camera_intr.frame)

        rescaled_depth_im = depth_im.rescale(rescale_factor, interp='nearest')
        rescaled_camera_intr = camera_intr.rescale(rescale_factor)
        point_cloud_im = rescaled_camera_intr.deproject_to_image(rescaled_depth_im)
        normal_cloud_im = point_cloud_im.normal_cloud_im(ksize=kernel_size)
        
        max_angle = 2 * math.pi
        bin_width = max_angle / preds.shape[-1]
        num_angles = 2 * preds.shape[-1]

        ang_idx = 0
        ang = 0.0
        
        actions = []
        for i in range(num_actions):
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            if ind.shape[1] > 3:
                ang_idx = ind[i, 3]
                ang = ang_idx * bin_width + bin_width / 2
            center = np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w // 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h // 2])
            rescaled_center = (rescale_factor * np.array([center[1], center[0]])).astype(np.uint32)
            axis = -normal_cloud_im[rescaled_center[0], rescaled_center[1]]
            if np.linalg.norm(axis) == 0:
                continue
            depth = depth_im[center[1], center[0]]
            if depth == 0.0:
                continue
            grasp = SuctionPoint2D(center, axis=axis, depth=depth, camera_intr=camera_intr, angle=ang)
            q_value = preds[im_idx, h_idx, w_idx, ang_idx]
            if q_value == MASKED_FLAG:
                continue

            grasp_action = GraspAction(grasp,
                                       q_value,
                                       DepthImage(images[im_idx]))
            actions.append(grasp_action)
        return actions

    def _visualize_affordance_map(self, preds, depth_im, scale, plot_max=True, output_dir=None):
        """Visualize an affordance map of the network predictions overlayed on the depth image."""
        self._logger.info('Visualizing affordance map...')

        affordance_map = preds[0, ..., 0]
        tf_depth_im = depth_im.crop(depth_im.shape[0] - self._gqcnn_recep_h, depth_im.shape[1] - self._gqcnn_recep_w).rescale(1.0 / self._gqcnn_stride)

        # plot
        vis.figure()
        vis.imshow(tf_depth_im)
        plt.imshow(affordance_map, cmap=plt.cm.RdYlGn, alpha=0.3, vmin=0.0, vmax=1.0)
        if plot_max:
            affordance_argmax = np.unravel_index(np.argmax(affordance_map), affordance_map.shape)
            plt.scatter(affordance_argmax[1], affordance_argmax[0], c='black', marker='.', s=scale*25)
        vis.title('Grasp Affordance Map')
        if output_dir is not None:
            vis.savefig(os.path.join(output_dir, 'grasp_affordance_map.png'))
        else:
            vis.show()
 
    def _gen_images_and_depths(self, depth, segmask):
        """Extend the image to a 4D tensor."""
        return np.expand_dims(depth, 0), np.array([-1]) #TODO: @Vishal depth should really be optional to the network...
   
    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """Visualize the actions in 3D."""
        raise NotImplementedError

class FullyConvolutionalGraspingPolicyMultiSuction(FullyConvolutionalGraspingPolicy):
    """Multi suction grasp sampling policy using Fully-Convolutional GQ-CNN network."""
    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions,
                     rescale_factor=DEFAULT_RESCALE_FACTOR, kernel_size=DEFAULT_KERNEL_SIZE):
        """Generate the actions to be returned."""
        # compute point cloud and normals
        depth_im = DepthImage(images[0], frame=camera_intr.frame)

        rescaled_depth_im = depth_im.rescale(rescale_factor, interp='nearest')
        rescaled_camera_intr = camera_intr.rescale(rescale_factor)
        point_cloud_im = rescaled_camera_intr.deproject_to_image(rescaled_depth_im)
        normal_cloud_im = point_cloud_im.normal_cloud_im(ksize=kernel_size)

        # set angle params
        max_angle = 2 * math.pi
        bin_width = max_angle / preds.shape[-1]
        num_angles = 2 * preds.shape[-1]

        actions = []
        for i in range(num_actions):
            # read index
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            ang_idx = ind[i, 3]

            # read center, axis and depth
            center =(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w // 2,
                                       h_idx * self._gqcnn_stride + self._gqcnn_recep_h // 2]))
            rescaled_center = (rescale_factor * np.array([center[1], center[0]])).astype(np.uint32)
            axis = -normal_cloud_im[rescaled_center[0], rescaled_center[1]]
            if np.linalg.norm(axis) == 0:
                axis = np.array([0,0,1])
            ang = ang_idx * bin_width + bin_width / 2
            depth = depth_im[center[1], center[0]]
            if depth == 0.0:
                continue

            # determine basis axes
            x_axis = axis
            y_axis = np.array([axis[1], -axis[0], 0])
            if np.linalg.norm(y_axis) == 0:
                y_axis = np.array([1,0,0])
            y_axis_im = np.array([np.cos(ang), np.sin(ang), 0])
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)

            # find rotation that aligns with the image orientation
            R = np.array([x_axis, y_axis, z_axis]).T
            max_dot = -np.inf
            aligned_R = R.copy()
            for k in range(num_angles):
                theta = float(k * max_angle) / num_angles
                R_tf = R.dot(Transform.x_axis_rotation(theta).R)
                dot = R_tf[:,1].dot(y_axis_im)
                if dot > max_dot:
                    max_dot = dot
                    aligned_R = R_tf.copy()

            # define multi cup suction point by the aligned pose
            t = camera_intr.deproject_pixel(depth, center)
            T = Transform(rotation=aligned_R,
                               translation=t,
                               from_frame='grasp',
                               to_frame=camera_intr.frame)

            # create grasp action
            grasp = MultiSuctionPoint2D(T, camera_intr=camera_intr)
            q_value = preds[im_idx, h_idx, w_idx, ang_idx]
            if q_value == MASKED_FLAG:
                continue

            grasp_action = GraspAction(grasp,
                                       q_value,
                                       DepthImage(images[im_idx]))
            grasp_action.gripper_name = self.name
            actions.append(grasp_action)
        return actions

    def _visualize_affordance_map(self, preds, depth_im, scale, plot_max=True, output_dir=None):
        """Visualize an affordance map of the network predictions overlayed on the depth image."""
        pass
        
    def _gen_images_and_depths(self, depth, segmask):
        """Extend the image to a 4D tensor."""
        return np.expand_dims(depth, 0), np.array([-1]) #TODO: @Vishal depth should really be optional to the network...
   
    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """Visualize the actions in 3D."""
        raise NotImplementedError
    
class FullyConvolutionalGraspingPolicyMultiGripper(FullyConvolutionalGraspingPolicy):
    """Suction grasp sampling policy using Fully-Convolutional GQ-CNN network."""
    def __init__(self, cfg, filters=None):
        FullyConvolutionalGraspingPolicy.__init__(self, cfg, filters=filters)

        # read the multi gripper indices
        self._gripper_types = self.grasp_quality_fn.gqcnn.gripper_types
        self._gripper_names = self.grasp_quality_fn.gqcnn.gripper_names
        self._tool_configs = self.grasp_quality_fn.gqcnn.tool_configs
        self._gripper_start_indices = self.grasp_quality_fn.gqcnn.gripper_start_indices
        self._gripper_max_angles = self.grasp_quality_fn.gqcnn.gripper_max_angles
        self._gripper_bin_widths = self.grasp_quality_fn.gqcnn.gripper_bin_widths
        self._gripper_num_angular_bins = self.grasp_quality_fn.gqcnn.gripper_num_angular_bins

        # read gripper params
        self._gripper_width = 0
        if 'gripper_width' in self._cfg.keys():
            self._gripper_width = self._cfg['gripper_width']

        # depth sampling parameters
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._depth_offset = 0.0
        if 'depth_offset' in self._cfg.keys():
            self._depth_offset = self._cfg['depth_offset']

    def _sample_depths(self, raw_depth_im, raw_seg):
        """Sample depths from the raw depth image."""
        max_depth = np.max(raw_depth_im) + self._depth_offset

        # for sampling the min depth, we only sample from the portion of the depth image in the object segmask because sometimes the rim of the bin is not properly subtracted out of the depth image
        raw_depth_im_segmented = np.ones_like(raw_depth_im)
        raw_depth_im_segmented[np.where(raw_seg > 0)] = raw_depth_im[np.where(raw_seg > 0)]
        min_depth = np.min(raw_depth_im_segmented) + self._depth_offset

        depth_bin_width = (max_depth - min_depth) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_depth + (i * depth_bin_width + depth_bin_width / 2)
        return depths

    def _gen_images_and_depths(self, depth, segmask):
        """Replicate the depth image and sample corresponding depths."""
        depths = self._sample_depths(depth, segmask)
        images = np.tile(np.asarray([depth]), (self._num_depth_bins, 1, 1, 1))
        return images, depths
    
    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions,
                     rescale_factor=DEFAULT_RESCALE_FACTOR, kernel_size=DEFAULT_KERNEL_SIZE):
        """Generate the actions to be returned."""
        depth_im = DepthImage(images[0], frame=camera_intr.frame)

        rescaled_depth_im = depth_im.rescale(rescale_factor, interp='nearest')
        rescaled_camera_intr = camera_intr.rescale(rescale_factor)
        point_cloud_im = rescaled_camera_intr.deproject_to_image(rescaled_depth_im)
        normal_cloud_im = point_cloud_im.normal_cloud_im(ksize=kernel_size)

        num_grippers = len(self._gripper_types.keys())
        
        actions = []
        for i in range(num_actions):
            # read index
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            g_idx = ind[i, 3]

            # determine gripper
            gripper_id = None
            g_start_idx = -1
            for j in range(num_grippers):
                candidate_gripper_id = list(self._gripper_types.keys())[j]
                start_idx = self._gripper_start_indices[candidate_gripper_id]
                if start_idx <= g_idx and start_idx > g_start_idx:
                    g_start_idx = start_idx
                    gripper_id = candidate_gripper_id
                j += 1

            if gripper_id is None:
                raise ValueError('Predicted gripper index %d is invalid' %(g_idx))

            gripper_type = self._gripper_types[gripper_id]
            max_angle = 0.0
            bin_width = 0.0
            if self._gripper_max_angles is not None:
                max_angle = self._gripper_max_angles[gripper_id]
                bin_width = self._gripper_bin_widths[gripper_id]
            num_angles = 2 * int(max_angle / bin_width)
            gripper_name = None
            tool_config = None
            if self._gripper_names is not None:
                gripper_name = self._gripper_names[gripper_id]
            if self._tool_configs is not None:
                tool_config = self._tool_configs[gripper_id]
                
            # determine grasp pose
            center = np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w // 2,
                                       h_idx * self._gqcnn_stride + self._gqcnn_recep_h // 2])
            if gripper_type == GripperMode.SUCTION or gripper_type == GripperMode.LEGACY_SUCTION:
                # read axis and depth from the images
                rescaled_center = (rescale_factor * np.array([center[1], center[0]])).astype(np.uint32)
                axis = -normal_cloud_im[rescaled_center[0], rescaled_center[1]]
                if np.linalg.norm(axis) == 0:
                    axis = np.array([0,0,1])
                depth = depth_im[center[1], center[0]]
                if depth == 0.0:
                    continue

                # read angle
                ang_idx = g_idx - g_start_idx
                ang = ang_idx * bin_width + bin_width / 2
                
                # create grasp
                grasp = SuctionPoint2D(center, axis=axis, depth=depth, camera_intr=camera_intr, angle=ang)

            elif gripper_type == GripperMode.PARALLEL_JAW or gripper_type == GripperMode.LEGACY_PARALLEL_JAW:
                # read angle and depth
                ang_idx = g_idx - g_start_idx
                ang = ang_idx * bin_width + bin_width / 2
                depth = depths[im_idx, 0]

                # create grasp
                grasp = Grasp2D(center, ang, depth, width=self._gripper_width, camera_intr=camera_intr)                

            elif gripper_type == GripperMode.MULTI_SUCTION:
                # read axis, angle, and depth
                ang_idx = g_idx - g_start_idx
                ang = ang_idx * bin_width + bin_width / 2
                rescaled_center = (rescale_factor * np.array([center[1], center[0]])).astype(np.uint32)
                axis = -normal_cloud_im[rescaled_center[0], rescaled_center[1]]
                if np.linalg.norm(axis) == 0:
                    axis = np.array([0,0,1])
                depth = depth_im[center[1], center[0]]
                if depth == 0.0:
                    continue

                # determine basis axes
                x_axis = axis
                y_axis = np.array([axis[1], -axis[0], 0])
                if np.linalg.norm(y_axis) == 0:
                    y_axis = np.array([1,0,0])
                y_axis_im = np.array([np.cos(ang), np.sin(ang), 0])
                y_axis = y_axis / np.linalg.norm(y_axis)
                z_axis = np.cross(x_axis, y_axis)

                # find rotation that aligns with the image orientation
                R = np.array([x_axis, y_axis, z_axis]).T
                max_dot = -np.inf
                aligned_R = R.copy()
                for k in range(num_angles):
                    theta = float(k * max_angle) / num_angles
                    R_tf = R.dot(Transform.x_axis_rotation(theta).R)
                    dot = R_tf[:,1].dot(y_axis_im)
                    if dot > max_dot:
                        max_dot = dot
                        aligned_R = R_tf.copy()

                # define multi cup suction point by the aligned pose
                t = camera_intr.deproject_pixel(depth, center)
                T = Transform(rotation=aligned_R,
                                   translation=t,
                                   from_frame='grasp',
                                   to_frame=camera_intr.frame)

                # create grasp
                grasp = MultiSuctionPoint2D(T, camera_intr=camera_intr)

            # create grasp action
            q_value = preds[im_idx, h_idx, w_idx, g_idx]
            if q_value == MASKED_FLAG:
                continue

            grasp_action = GraspAction(grasp,
                                       q_value,
                                       DepthImage(images[im_idx]),
                                       gripper_name=gripper_name,
                                       tool_config=tool_config)

            actions.append(grasp_action)
        return actions
        
    def _visualize_affordance_map(self, preds, depth_im, scale, plot_max=True, output_dir=None):
        """Visualize an affordance map of the network predictions overlayed on the depth image."""
        self._logger.info('Visualizing affordance map...')

        ind = []
        for gripper_id, gripper_type in self._gripper_types.iteritems():
            if gripper_type == 'suction':
                ind.append(self._gripper_start_indices[gripper_id])
        
        for i in ind:
            affordance_map = preds[0, ..., i]
            tf_depth_im = depth_im.crop(depth_im.shape[0] - self._gqcnn_recep_h, depth_im.shape[1] - self._gqcnn_recep_w).rescale(1.0 / self._gqcnn_stride)

            # plot
            vis.figure()
            vis.imshow(tf_depth_im)
            plt.imshow(affordance_map, cmap=plt.cm.RdYlGn, alpha=0.3, vmin=0.0, vmax=1.0)
            if plot_max:
                affordance_argmax = np.unravel_index(np.argmax(affordance_map), affordance_map.shape)
                plt.scatter(affordance_argmax[1], affordance_argmax[0], c='black', marker='.', s=scale*25)
            vis.title('Grasp Affordance Map')
            if output_dir is not None:
                vis.savefig(os.path.join(output_dir, 'grasp_affordance_map_%03d.png' %(i)))
            else:
                vis.show()
 
    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """Visualize the actions in 3D."""
        raise NotImplementedError
    
