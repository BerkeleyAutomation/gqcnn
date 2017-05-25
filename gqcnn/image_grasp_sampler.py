"""
Classes for sampling a set of grasps directly from images to generate data for a neural network
Author: Jeff Mahler, Sherdil Niyaz
"""
from abc import ABCMeta, abstractmethod

import copy
import cv2
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
import sys
from time import sleep, time

import scipy.spatial.distance as ssd
import scipy.ndimage.filters as snf
import scipy.stats as ss
import sklearn.mixture

from core import Point, RigidTransform
from perception import BinaryImage, ColorImage, DepthImage, RgbdImage

from gqcnn import Grasp2D
from gqcnn import Visualizer as vis

def force_closure(p1, p2, n1, n2, mu):
    """ Computes whether or not the point and normal pairs are in force closure. """
    # line between the contacts 
    v = p2 - p1
    v = v / np.linalg.norm(v)
    
    # compute cone membership
    alpha = np.arctan(mu)
    in_cone_1 = (np.arccos(n1.dot(-v)) < alpha)
    in_cone_2 = (np.arccos(n2.dot(v)) < alpha)
    return (in_cone_1 and in_cone_2)

class ImageGraspSampler(object):
    """
    Wraps image to crane grasp candidate generation for easy deployment of GQ-CNN.

    Attributes
    ----------
    config : :obj:`core.YamlConfig`
        a dictionary-like object containing the parameters of the sampler
    gripper_width : float
        width of the gripper in 3D space

    Notes
    -----
    Required configuration paramters are specified in Other Parameters

    Other Parameters
    ----------------
    min_depth_offset : float
        offset from the minimum depth at the grasp center pixel to use in depth sampling
    max_depth : float
        maximum possible depth used in sampling
    depth_sample_win_height : float
        height of a window around the grasp center pixel used to determine min depth
    depth_sample_win_height : float
        width of a window around the grasp center pixel used to determine min depth
    """
    __metaclass__ = ABCMeta

    def __init__(self, config, gripper_width=np.inf):
        # set params
        self._config = config
        self._gripper_width = gripper_width

    def sample(self, rgbd_im, camera_intr, num_samples,
               segmask=None, seed=None, visualize=False):
        """
        Samples a set of 2D grasps from a given RGB-D image.
        
        Parameters
        ----------
        rgbd_im : :obj:`perception.RgbdImage`
            RGB-D image to sample from
        camera_intr : :obj:`perception.CameraIntrinsics`
            intrinsics of the camera that captured the images
        num_samples : int
            number of grasps to sample
        segmask : :obj:`perception.BinaryImage`
            binary image segmenting out the object of interest
        seed : int
            number to use in random seed (None if no seed)
        visualize : bool
            whether or not to show intermediate samples (for debugging)

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            the list of grasps in image space
        """
        # set random seed for determinism
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # sample an initial set of grasps (without depth)
        logging.debug('Sampling 2d candidates')
        sampling_start = time()
        grasps = self._sample(rgbd_im, camera_intr, num_samples,
                              segmask=segmask, visualize=visualize)
        sampling_stop = time()
        logging.debug('Sampled %d grasps from image' %(len(grasps)))
        logging.debug('Sampling grasps took %.3f sec' %(sampling_stop - sampling_start))
        return grasps

    @abstractmethod
    def _sample(self, rgbd_im, camera_intr, num_samples, segmask=None,
                visualize=False):
        """
        Sample a set of 2D grasp candidates from a depth image.
        Subclasses must override.

        Parameters
        ----------
        rgbd_im : :obj:`perception.RgbdImage`
            RGB-D image to sample from
        camera_intr : :obj:`perception.CameraIntrinsics`
            intrinsics of the camera that captured the images
        num_samples : int
            number of grasps to sample
        segmask : :obj:`perception.BinaryImage`
            binary image segmenting out the object of interest
        visualize : bool
            whether or not to show intermediate samples (for debugging)
 
        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            list of 2D grasp candidates
        """
        pass
        
class AntipodalDepthImageGraspSampler(ImageGraspSampler):
    """ Grasp sampler for antipodal point pairs from depth image gradients. """

    def __init__(self, config, gripper_width=np.inf):
        # init superclass
        ImageGraspSampler.__init__(self, config, gripper_width)

        # antipodality params
        self._friction_coef = self._config['friction_coef']
        self._depth_grad_thresh = self._config['depth_grad_thresh']
        self._depth_grad_gaussian_sigma = self._config['depth_grad_gaussian_sigma']
        self._downsample_rate = self._config['downsample_rate']
        self._rescale_factor = 1.0 / self._downsample_rate
        self._max_num_attempts = self._config['max_num_attempts']

        # distance thresholds for rejection sampling
        self._max_dist_from_center = self._config['max_dist_from_center']
        self._min_grasp_dist = self._config['min_grasp_dist']
        self._angle_dist_weight = self._config['angle_dist_weight']

        # depth sampling params
        self._depth_samples_per_grasp = max(self._config['depth_samples_per_grasp'], 1)
        self._min_depth_offset = self._config['min_depth_offset']
        self._max_depth = self._config['max_depth']
        self._h = self._config['depth_sample_win_height']
        self._w = self._config['depth_sample_win_width']

    def _surface_normals(self, depth_im, edge_pixels):
        """ Return an array of the surface normals at the edge pixels. """
        # compute the gradients
        grad = np.gradient(depth_im.data.astype(np.float32))

        # compute surface normals
        normals = np.zeros([edge_pixels.shape[0], 2])
        for i, pix in enumerate(edge_pixels):
            dx = grad[1][pix[0], pix[1]]
            dy = grad[0][pix[0], pix[1]]
            normal_vec = np.array([dy, dx])
            if np.linalg.norm(normal_vec) > 0:
                normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i,:] = normal_vec

        return normals

    def _sample(self, rgbd_im, camera_intr, num_samples, segmask=None,
                visualize=False):
        """
        Sample a set of 2D grasp candidates from a depth image.
        Subclasses must override.

        Parameters
        ----------
        rgbd_im : :obj:`perception.RgbdImage`
            RGB-D image to sample from
        camera_intr : :obj:`perception.CameraIntrinsics`
            intrinsics of the camera that captured the images
        num_samples : int
            number of grasps to sample
        segmask : :obj:`perception.BinaryImage`
            binary image segmenting out the object of interest
        visualize : bool
            whether or not to show intermediate samples (for debugging)
 
        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            list of 2D grasp candidates
        """
        # sample antipodal pairs in image space
        grasps = self._sample_antipodal_grasps(rgbd_im, camera_intr, num_samples,
                                               segmask=segmask, visualize=visualize)
        return grasps

    def _sample_antipodal_grasps(self, rgbd_im, camera_intr, num_samples,
                                 segmask=None, visualize=False):
        # compute edge pixels
        edge_start = time()
        depth_im = rgbd_im.depth
        depth_im = depth_im.apply(snf.gaussian_filter,
                                  sigma=self._depth_grad_gaussian_sigma)
        depth_im_downsampled = depth_im.resize(self._rescale_factor)
        depth_im_threshed = depth_im_downsampled.threshold_gradients(self._depth_grad_thresh)
        edge_pixels = self._downsample_rate * depth_im_threshed.zero_pixels()
        if segmask is not None:
            edge_pixels = np.array([p for p in edge_pixels if np.any(segmask[p[0], p[1]] > 0)])
        num_pixels = edge_pixels.shape[0]
        logging.debug('Depth edge detection took %.3f sec' %(time() - edge_start))
        logging.debug('Found %d edge pixels' %(num_pixels))

        # compute surface normals
        normal_start = time()
        edge_normals = self._surface_normals(depth_im, edge_pixels)
        logging.info('Normal computation took %.3f sec' %(time() - normal_start))

        if visualize:
            vis.figure()
            vis.subplot(1,2,1)            
            vis.imshow(depth_im)
            if num_pixels > 0:
                vis.scatter(edge_pixels[:,1], edge_pixels[:,0], s=10, c='b')

            X = [pix[1] for pix in edge_pixels]
            Y = [pix[0] for pix in edge_pixels]
            U = [10*pix[1] for pix in edge_normals]
            V = [-10*pix[0] for pix in edge_normals]
            plt.quiver(X, Y, U, V, units='x', scale=1, zorder=2, color='g')
            vis.title('Edge pixels and normals')

            vis.subplot(1,2,2)
            vis.imshow(depth_im_threshed)
            vis.title('Edge map')
            vis.show()

        # form set of valid candidate point pairs
        sample_start = time()
        max_grasp_width_px = Grasp2D(Point(np.zeros(2)), 0.0, self._max_depth,
                                     width = self._gripper_width,
                                     camera_intr=camera_intr).width_px
        normal_ip = edge_normals.dot(edge_normals.T)
        dists = ssd.squareform(ssd.pdist(edge_pixels))
        valid_indices = np.where((normal_ip < -np.cos(np.arctan(self._friction_coef))) & (dists < max_grasp_width_px) & (dists > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        num_pairs = valid_indices.shape[0]
        logging.info('Normal pruning %.3f sec' %(time() - sample_start))

        # iteratively sample grasps
        k = 0
        grasps = []
        candidate_pair_indices = np.random.choice(num_pairs, size=self._max_num_attempts,
                                                  replace=False)
        while k < self._max_num_attempts and len(grasps) < num_samples:
            # sample a random pair without replacement
            j = candidate_pair_indices[k]
            pair_ind = valid_indices[j,:]
            p1 = edge_pixels[pair_ind[0],:]
            p2 = edge_pixels[pair_ind[1],:]
            n1 = edge_normals[pair_ind[0],:]
            n2 = edge_normals[pair_ind[1],:]
            width = np.linalg.norm(p1 - p2)
            k += 1

            # check force closure
            if force_closure(p1, p2, n1, n2, self._friction_coef):
                # compute grasp parameters
                grasp_center = (p1 + p2) / 2
                grasp_axis = p2 - p1
                grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
                grasp_theta = np.arctan(grasp_axis[0] / grasp_axis[1])
                    
                # compute distance from image center
                dist_from_center = np.linalg.norm(grasp_center - depth_im.center)
                if dist_from_center < self._max_dist_from_center:
                    # form grasp object
                    grasp_center_pt = Point(np.array([grasp_center[1], grasp_center[0]]))
                    grasp = Grasp2D(grasp_center_pt, grasp_theta, 0.0)
                    
                    # check grasp dists
                    grasp_dists = [Grasp2D.image_dist(grasp, candidate, alpha=self._angle_dist_weight) for candidate in grasps]
                    if len(grasps) == 0 or np.min(grasp_dists) > self._min_grasp_dist:

                        if visualize:
                            vis.figure()
                            vis.imshow(depth_im)
                            vis.scatter(p1[1],p1[0])
                            vis.scatter(p2[1],p2[0])
                            vis.title('Grasp candidate %d' %(len(grasps)))
                            vis.show()

                        # sample depths
                        for i in range(self._depth_samples_per_grasp):
                            min_depth = np.min(depth_im.data[grasp_center[0]-self._h:grasp_center[0]+self._h, grasp_center[1]-self._w:grasp_center[1]+self._w]) + self._min_depth_offset
                            sample_depth = min_depth + (self._max_depth - min_depth) * np.random.rand()
                            candidate_grasp = Grasp2D(grasp_center_pt,
                                                      grasp_theta,
                                                      sample_depth,
                                                      width=self._gripper_width,
                                                      camera_intr=camera_intr)
                            grasps.append(candidate_grasp)
        return grasps
