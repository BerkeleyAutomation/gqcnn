"""
Grasping policy using fully-convolutional angular gqcnn.
Author: Vishal Satish
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import logging

from autolab_core import Point
from gqcnn import Grasp2D, ParallelJawGrasp, Visualizer as vis
from gqcnn.model import get_gqcnn_model
from perception import DepthImage

# declare any enums or constants
PI = math.pi

class FullyConvolutionalAngularPolicy(object):
    ''' Grasp sampling policy using full-convolutional angular GQ-CNN network '''
    def __init__(self, cfg):
        # parse config
        self._cfg = cfg
        self._num_depth_bins = self._cfg['num_depth_bins']
        self._width = self._cfg['width']

        self._gqcnn_dir = self._cfg['gqcnn_model']
        self._gqcnn_backend = self._cfg['gqcnn_backend']
        self._gqcnn_stride = self._cfg['gqcnn_stride']
        self._gqcnn_recep_h = self._cfg['gqcnn_recep_h']
        self._gqcnn_recep_w = self._cfg['gqcnn_recep_w']
        self._fully_conv_config = self._cfg['fully_conv_gqcnn_config']

        self._vis_config = self._cfg['vis']
        self._top_k_to_vis = self._vis_config['top_k']
        self._vis_top_k = self._vis_config['vis_top_k']
               
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

    def _action(self, state):
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
        use_opt = self._num_depth_bins > self._gqcnn.batch_size
        if use_opt:
            unique_im_map = np.zeros((self._num_depth_bins,), dtype=np.int32)
            preds = self._gqcnn.predict(images, depths, unique_im_map=unique_im_map)
        else:
            preds = self._gqcnn.predict(images, depths)
        success_ind = np.arange(1, preds.shape[-1], 2)

        # if we want to visualize the top k, then extract those indices, else just get the top 1
        top_k = self._top_k_to_vis if self._vis_top_k else 1

        # get indices of top k predictions
        preds_success_only = preds[:, :, :, success_ind]
        preds_success_only_flat = np.ravel(preds_success_only)
        top_k_pred_ind_flat = np.argpartition(preds_success_only_flat, -1 * top_k)[-1 * top_k:]
        top_k_pred_ind = np.zeros((top_k, len(preds.shape)), dtype=np.int32)
        im_width = preds_success_only.shape[2]
        im_height = preds_success_only.shape[1]
        num_angular_bins = preds_success_only.shape[3]
        for k in range(top_k):
            top_k_pred_ind[k, 0] = top_k_pred_ind_flat[k] // (im_width * im_height * num_angular_bins) 
            top_k_pred_ind[k, 1] = (top_k_pred_ind_flat[k] - (top_k_pred_ind[k, 0] * (im_width * im_height * num_angular_bins))) // (im_width * num_angular_bins)
            top_k_pred_ind[k, 2] = (top_k_pred_ind_flat[k] - (top_k_pred_ind[k, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[k, 1] * (im_width * num_angular_bins))) // num_angular_bins
            top_k_pred_ind[k, 3] = (top_k_pred_ind_flat[k] - (top_k_pred_ind[k, 0] * (im_width * im_height * num_angular_bins)) - (top_k_pred_ind[k, 1] * (im_width * num_angular_bins))) % num_angular_bins

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

        # visualize
        if self._vis_top_k:
            vis.figure((15, 15))
            vis.imshow(d_im)
            for i in range(top_k):
                vis.grasp(grasps[i].grasp, scale=self._vis_config['scale'], show_axis=self._vis_config['show_axis'], color=plt.cm.RdYlGn(grasps[i].q_value))
            vis.show() 

        return grasps[-1]
