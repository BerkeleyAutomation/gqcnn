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
Fully-Convolutional GQ-CNN grasping policies
Author: Vishal Satish
"""
import numpy as np
import math
import logging
from abc import abstractmethod, ABCMeta

import matplotlib.pyplot as plt

from autolab_core import Point
from gqcnn import Grasp2D, SuctionPoint2D
from perception import DepthImage
from visualization import Visualizer2D as vis
from policy import GraspingPolicy, GraspAction

class FullyConvolutionalGraspingPolicy(GraspingPolicy):
    """ Abstract grasp sampling policy class using fully-convolutional GQ-CNN network """
    __metaclass__ = ABCMeta

    def __init__(self, cfg):
        GraspingPolicy.__init__(self, cfg, init_sampler=False)

        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._sampling_method = self._cfg['sampling_method']

        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']

        self._vis_config = self._cfg['policy_vis']
        self._num_vis_samples = self._vis_config['num_samples']
        self._vis = self._vis_config['vis']
        self._vis_3d = self._vis_config['vis_3d']
        
    def _unpack_state(self, state):
        """ Unpack information from the RgbdImageState """
        return state.rgbd_im.depth, state.rgbd_im.depth._data, state.segmask.raw_data, state.camera_intr #TODO: don't access raw depth data like this
       
    def _sample_depths(self, raw_depth_im, raw_seg):
        """ Sample depths from the raw depth image  """
        max_depth = np.max(raw_depth_im)

        # for sampling the min depth, we only sample from the portion of the depth image in the object segmask because sometimes the rim of the bin is not properly subtracted out of the depth image
        raw_depth_im_segmented = np.ones_like(raw_depth_im)
        raw_depth_im_segmented[np.where(raw_seg > 0)] = raw_depth_im[np.where(raw_seg > 0)]
        min_depth = np.min(raw_depth_im_segmented)

        depth_bin_width = (max_depth - min_depth) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_depth + (i * depth_bin_width + depth_bin_width / 2)
        return depths

    def _mask_predictions(self, preds, raw_segmask):
        """ Mask the given predictions with the given segmask, setting the rest to 0.0  """
        preds_masked = np.zeros_like(preds)
        raw_segmask_cropped = raw_segmask[self._gqcnn_recep_h / 2:raw_segmask.shape[0] - self._gqcnn_recep_h / 2, self._gqcnn_recep_w / 2:raw_segmask.shape[1] - self._gqcnn_recep_w / 2, 0]
        raw_segmask_downsampled = raw_segmask_cropped[::self._gqcnn_stride, ::self._gqcnn_stride]
        if raw_segmask_downsampled.shape[0] != preds.shape[1]:
            raw_segmask_downsampled_new = np.zeros(preds.shape[1:3])
            raw_segmask_downsampled_new[:raw_segmask_downsampled.shape[0], :raw_segmask_downsampled.shape[1]] = raw_segmask_downsampled
            raw_segmask_downsampled = raw_segmask_downsampled_new
        nonzero_mask_ind = np.where(raw_segmask_downsampled > 0)
        preds_masked[:, nonzero_mask_ind[0], nonzero_mask_ind[1]] = preds[:, nonzero_mask_ind[0], nonzero_mask_ind[1]]
        return preds_masked

    def _sample_predictions(self, preds, num_actions):
        """ Sample predictions using the specified sampling method  """
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
        """ Helper function to do the actual sampling  """
        if num_samples == 1: # argmax() is faster than argpartition() for special case of single sample
            if self._sampling_method == 'top_k':
                return [np.argmax(preds_flat)]
            elif self._sampling_method == 'uniform':
                nonzero_ind = np.where(preds_flat > 0)[0]
                import IPython
                IPython.embed() #TODO: make sure a list/array is returned 
                return np.random.choice(nonzero_ind)
            else:
                raise ValueError('Invalid sampling method: {}'.format(self._sampling_method))
        else:
            if self._sampling_method == 'top_k':
                return np.argpartition(preds_flat, -1 * num_samples)[-1 * num_samples:]
            elif self._sampling_method == 'uniform':
                nonzero_ind = np.where(preds_flat > 0)[0]
                return np.random.choice(nonzero_ind, size=num_samples)
            else:
                raise ValueError('Invalid sampling method: {}'.format(self._sampling_method))

    @abstractmethod
    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions):
        """ Generate the actions to be returned  """
        pass

    @abstractmethod
    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """ Visualize the actions in 3D  """
        pass

    def _visualize_2d(self, actions, wrapped_depth_im, num_actions, scale, show_axis):
        """ Visualize the actions in 2D  """
        vis.figure()
        vis.imshow(wrapped_depth_im)
        for i in range(num_actions):
            vis.grasp(actions[i].grasp, scale=scale, show_axis=show_axis, color=plt.cm.RdYlGn(actions[i].q_value))
        vis.show()

    def _action(self, state, num_actions=1):
        """ Plan action(s)  """
        # unpack the RgbdImageState
        wrapped_depth, raw_depth, raw_seg, camera_intr = self._unpack_state(state)

        # sample depths to evaluate from the depth image
        depths = self._sample_depths(raw_depth, raw_seg)

        # predict
        images = np.tile(np.asarray([raw_depth]), (self._num_depth_bins, 1, 1, 1))
        preds = self._grasp_quality_fn.quality(images, depths)

        # get success probablility predictions only (this is needed because the output of the net is pairs of (p_failure, p_success))
        preds_success_only = preds[:, :, :, 1::2]
        
        # mask predicted success probabilities with the cropped and downsampled object segmask so we only sample grasps on the objects
        preds_success_only = self._mask_predictions(preds_success_only, raw_seg) 

        # if we want to visualize more than one action, we have to sample more
        num_actions_to_sample = self._num_vis_samples if self._vis else num_actions #TODO: If this is used with the 'top_k' sampling method, the final returned action is not the best because the argpartition does not sort the partitioned indices 

        # sample num_actions_to_sample indices from the success predictions
        sampled_ind = self._sample_predictions(preds_success_only, num_actions_to_sample)

        # wrap actions to be returned
        actions = self._get_actions(preds_success_only, sampled_ind, images, depths, camera_intr, num_actions_to_sample)

        if self._vis:
            # visualize 3D
            if self._vis_3d:
                logging.info('Generating 3D Visualization...')
                self._visualize_3d(actions, wrapped_depth, camera_intr, num_actions_to_sample)
            # visualize 2D
            logging.info('Generating 2D visualization...')
            self._visualize_2d(actions, wrapped_depth, num_actions_to_sample, self._vis_config['scale'], self._vis_config['show_axis'])

        return actions[-1] if num_actions == 1 else actions[-(num_actions+1):]

    def action_set(self, state, num_actions):
        """ Plan a set of actions  """
        return [action.grasp for action in self._action(state, num_actions=num_actions)]

class FullyConvolutionalGraspingPolicyParallelJaw(FullyConvolutionalGraspingPolicy):
    """ Parallel jaw grasp sampling policy using fully-convolutional GQ-CNN network """
    def __init__(self, cfg):
        FullyConvolutionalGraspingPolicy.__init__(self, cfg)
        self._width = self._cfg['gripper_width']

    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions):
        """ Generate the actions to be returned  """
        actions = []
        ang_bin_width = math.pi / preds.shape[-1]
        for i in range(num_actions):
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            ang_idx = ind[i, 3]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            ang = math.pi / 2 - (ang_idx * ang_bin_width + ang_bin_width / 2)
            depth = depths[im_idx]
            grasp = Grasp2D(center, ang, depth, width=self._width, camera_intr=camera_intr)
            grasp_action = GraspAction(grasp, preds[im_idx, h_idx, w_idx, ang_idx], DepthImage(images[im_idx]))
            actions.append(grasp_action)
        return actions

    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """ Visualize the actions in 3D  """
        raise NotImplementedError

class FullyConvolutionalGraspingPolicySuction(FullyConvolutionalGraspingPolicy):
    """ Suction grasp sampling policy using fully-convolutional GQ-CNN network """
    def _get_actions(self, preds, ind, images, depths, camera_intr, num_actions):
        """ Generate the actions to be returned  """
        actions = []
        for i in range(num_actions):
            im_idx = ind[i, 0]
            h_idx = ind[i, 1]
            w_idx = ind[i, 2]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            depth = depths[im_idx]
            grasp = SuctionPoint2D(center, depth=depth, camera_intr=camera_intr)
            grasp_action = GraspAction(grasp, preds[im_idx, h_idx, w_idx, 0], DepthImage(images[im_idx]))
            actions.append(grasp_action)
        return actions

    def _visualize_3d(self, actions, wrapped_depth_im, camera_intr, num_actions):
        """ Visualize the actions in 3D  """
        raise NotImplementedError
