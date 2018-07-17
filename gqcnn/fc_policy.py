"""
Fully-Convolutional GQ-CNN grasping policies
Author: Vishal Satish
"""
import numpy as np
import math
import logging

import matplotlib.pyplot as plt

from autolab_core import Point
from gqcnn import Grasp2D
from perception import DepthImage
from visualization import Visualizer2D as vis
from policy import GraspingPolicy, GraspAction

class FullyConvolutionalAngularPolicyTopK(GraspingPolicy):
    """ Grasp sampling policy using full-convolutional angular GQ-CNN network that returns
        the top k predictions
    """
    def __init__(self, cfg):
        GraspingPolicy.__init__(self, cfg, init_sampler=False)

        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._width = self._cfg['gripper_width']

        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']

        self._vis_config = self._cfg['policy_vis']
        self._top_k_to_vis = self._vis_config['top_k']
        self._vis_top_k = self._vis_config['vis_top_k']
        self._vis_3d = self._vis_config['vis_3d']
               
    def _action(self, state, k=1):
        # extract raw depth data matrix
        rgbd_im = state.rgbd_im
        d_im = rgbd_im.depth
        raw_d = d_im._data # TODO: Access this properly

        # sample depths
        max_d = np.max(raw_d)
        min_d = np.min(raw_d)
        depth_bin_width = (max_d - min_d) / self._num_depth_bins
        depths = np.zeros((self._num_depth_bins, 1)) 
        for i in range(self._num_depth_bins):
            depths[i][0] = min_d + (i * depth_bin_width + depth_bin_width / 2)

        # predict
        images = np.tile(np.asarray([raw_d]), (self._num_depth_bins, 1, 1, 1))
        preds = self._grasp_quality_fn.quality(images, depths)
        success_ind = np.arange(1, preds.shape[-1], 2)

        # if we want to visualize the top k, then extract those indices, else just get the number specified by the policy
        top_k = self._top_k_to_vis if self._vis_top_k else k

        # get indices of top k predictions
        preds_success_only = preds[:, :, :, 1::2]
        
        # only take predictions from the area of the image corresponding to the segmask
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

        im_width = preds_success_only.shape[2]
        im_height = preds_success_only.shape[1]
        num_angular_bins = preds_success_only.shape[3]
        if top_k == 1:
            preds_success_only_flat = np.ravel(preds_success_only)
            best_ind = np.argmax(preds_success_only_flat)
            top_k_pred_ind = np.zeros((1, len(preds.shape)), dtype=np.int32)
            top_k_pred_ind[0,0] = best_ind // (im_width * im_height * num_angular_bins) 
            top_k_pred_ind[0,1] = (best_ind - (top_k_pred_ind[0,0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
            top_k_pred_ind[0,2] = (best_ind - (top_k_pred_ind[0,0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[0,1] * (im_width * num_angular_bins))) // num_angular_bins
            top_k_pred_ind[0,3] = (best_ind - (top_k_pred_ind[0,0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[0,1] * (im_width * num_angular_bins))) % num_angular_bins
        else:
            preds_success_only_flat = np.ravel(preds_success_only)
            top_k_pred_ind_flat = np.argpartition(preds_success_only_flat, -1 * top_k)[-1 * top_k:]
            top_k_pred_ind = np.zeros((top_k, len(preds.shape)), dtype=np.int32)
            for idx in range(top_k):
                top_k_pred_ind[idx, 0] = top_k_pred_ind_flat[idx] // (im_width * im_height * num_angular_bins) 
                top_k_pred_ind[idx, 1] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
                top_k_pred_ind[idx, 2] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) // num_angular_bins
                top_k_pred_ind[idx, 3] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) % num_angular_bins

        # generate grasps
        grasps = []
        ang_bin_width = math.pi / preds_success_only.shape[-1]
        for i in range(top_k):
            im_idx = top_k_pred_ind[i, 0]
            h_idx = top_k_pred_ind[i, 1]
            w_idx = top_k_pred_ind[i, 2]
            ang_idx = top_k_pred_ind[i, 3]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            ang = math.pi / 2 - (ang_idx * ang_bin_width + ang_bin_width / 2)
            depth = depths[im_idx]
            grasp = Grasp2D(center, ang, depth, width=self._width, camera_intr=state.camera_intr)
            pj_grasp = GraspAction(grasp, preds_success_only[im_idx, h_idx, w_idx, ang_idx], DepthImage(images[im_idx]))
            grasps.append(pj_grasp)

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
            for i in range(top_k):
                vis.grasp(grasps[i].grasp, scale=self._vis_config['scale'], show_axis=self._vis_config['show_axis'], color=plt.cm.RdYlGn(grasps[i].q_value))
            vis.show()
        return grasps[-1] if k == 1 else grasps[-(k+1):]

    def action_set(self, state, num_actions):
        return [pj_grasp.grasp for pj_grasp in self._action(state, k=num_actions)]

class FullyConvolutionalAngularPolicyUniform(GraspingPolicy):
    """ Grasp sampling policy using full-convolutional angular GQ-CNN network that returns
        the uniform predictions
    """
    def __init__(self, cfg):
        GraspingPolicy.__init__(self, cfg, init_sampler=False)

        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._width = self._cfg['gripper_width']

        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']

        self._vis_config = self._cfg['policy_vis']
        self._top_k_to_vis = self._vis_config['top_k']
        self._vis_top_k = self._vis_config['vis_top_k']
        self._vis_3d = self._vis_config['vis_3d']
 
    def _action(self, state, k=1):
        # extract raw depth data matrix
        rgbd_im = state.rgbd_im
        d_im = rgbd_im.depth
        raw_d = d_im._data # TODO: Access this properly

#        vis.figure()
#        vis.imshow(d_im)
#        vis.show()

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
        preds = self._grasp_quality_fn.quality(images, depths)
        success_ind = np.arange(1, preds.shape[-1], 2)

        # if we want to visualize the top k, then extract those indices, else just get the number specified by the policy
        top_k = self._top_k_to_vis if self._vis_top_k else k

        # get indices of top k predictions
        preds_success_only = preds[:, :, :, success_ind]

        # only take predictions from the area of the image corresponding to the segmask
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
        nonzero_ind = np.where(preds_success_only_flat > 0)[0]
        if nonzero_ind.shape[0] == 0:
            return []
        top_k_pred_ind_flat = np.random.choice(nonzero_ind, size=top_k)
        top_k_pred_ind = np.zeros((top_k, len(preds.shape)), dtype=np.int32)
        im_width = preds_success_only.shape[2]
        im_height = preds_success_only.shape[1]
        num_angular_bins = preds_success_only.shape[3]
        for idx in range(top_k):
            top_k_pred_ind[idx, 0] = top_k_pred_ind_flat[idx] // (im_width * im_height * num_angular_bins) 
            top_k_pred_ind[idx, 1] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
            top_k_pred_ind[idx, 2] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) // num_angular_bins
            top_k_pred_ind[idx, 3] = (top_k_pred_ind_flat[idx] - (top_k_pred_ind[idx, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[idx, 1] * (im_width * num_angular_bins))) % num_angular_bins

        # generate grasps
        grasps = []
        ang_bin_width = math.pi / preds_success_only.shape[-1]
        for i in range(top_k):
            im_idx = top_k_pred_ind[i, 0]
            h_idx = top_k_pred_ind[i, 1]
            w_idx = top_k_pred_ind[i, 2]
            ang_idx = top_k_pred_ind[i, 3]
            center = Point(np.asarray([w_idx * self._gqcnn_stride + self._gqcnn_recep_w / 2, h_idx * self._gqcnn_stride + self._gqcnn_recep_h / 2]))
            ang = math.pi / 2 - (ang_idx * ang_bin_width + ang_bin_width / 2)
            depth = depths[im_idx]
            grasp = Grasp2D(center, ang, depth, width=self._width, camera_intr=state.camera_intr)
            pj_grasp = GraspAction(grasp, preds_success_only[im_idx, h_idx, w_idx, ang_idx], DepthImage(images[im_idx]))
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
            for i in range(top_k):
                vis.grasp(grasps[i].grasp, scale=self._vis_config['scale'], show_axis=self._vis_config['show_axis'], color=plt.cm.RdYlGn(grasps[i].q_value))
            vis.show()
        return grasps[-1] if k == 1 else grasps[-(k+1):]

    def action_set(self, state, num_actions):
        actions = self._action(state, k=num_actions)
#        depths = {}
#        ds = []
#        for a in actions:
#            if str(a.grasp.depth) not in depths.keys():
#                depths[str(a.grasp.depth)] = 1
#                ds.append(a.grasp.depth)
#            else:
#                depths[str(a.grasp.depth)] += 1
#        ds = sorted(ds)
#        chosen_ds = ds[-3:]
#        for d in ds:
#            print(d, depths[str(d)])
#        return [pj_grasp.grasp for pj_grasp in actions if pj_grasp.grasp.depth in chosen_ds]
        return [pj_grasp.grasp for pj_grasp in actions]
