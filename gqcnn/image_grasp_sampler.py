"""
Classes for sampling a set of grasps directly from images to generate data for a neural network
Author: Jeff Mahler, Sherdil Niyaz
"""
from abc import ABCMeta, abstractmethod

import copy
import cv2
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

from autolab_core import Point, RigidTransform
from perception import BinaryImage, ColorImage, DepthImage, RgbdImage

from gqcnn import Grasp2D, SuctionPoint2D
from gqcnn import Visualizer as vis

from gqcnn import NoAntipodalPairsFoundException

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

class DepthSamplingMode(object):
    """ Modes for sampling grasp depth. """
    UNIFORM = 'uniform'
    MIN = 'min'
    MAX = 'max'

class ImageGraspSampler(object):
    """
    Wraps image to crane grasp candidate generation for easy deployment of GQ-CNN.

    Attributes
    ----------
    config : :obj:`autolab_core.YamlConfig`
        a dictionary-like object containing the parameters of the sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, config):
        # set params
        self._config = config

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
    """ Grasp sampler for antipodal point pairs from depth image gradients.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    gripper_width : float
        width of the gripper, in meters
    friction_coef : float
        friction coefficient for 2D force closure
    depth_grad_thresh : float
        threshold for depth image gradients to determine edge points for sampling
    depth_grad_gaussian_sigma : float
        sigma used for pre-smoothing the depth image for better gradients
    downsample_rate : float
        factor to downsample the depth image by before sampling grasps
    max_rejection_samples : int
        ceiling on the number of grasps to check in antipodal grasp rejection sampling
    max_dist_from_center : int
        maximum allowable distance of a grasp from the image center
    min_grasp_dist : float
        threshold on the grasp distance
    angle_dist_weight : float
        amount to weight the angle difference in grasp distance computation
    depth_samples_per_grasp : int
        number of depth samples to take per grasp
    min_depth_offset : float
        offset from the minimum depth at the grasp center pixel to use in depth sampling
    max_depth_offset : float
        offset from the maximum depth across all edges
    depth_sample_win_height : float
        height of a window around the grasp center pixel used to determine min depth
    depth_sample_win_height : float
        width of a window around the grasp center pixel used to determine min depth
    depth_sampling_mode : str
        name of depth sampling mode (uniform, min, max)
    """
    def __init__(self, config, gripper_width=np.inf):
        # init superclass
        ImageGraspSampler.__init__(self, config)

        # antipodality params
        self._gripper_width = self._config['gripper_width']
        self._friction_coef = self._config['friction_coef']
        self._depth_grad_thresh = self._config['depth_grad_thresh']
        self._depth_grad_gaussian_sigma = self._config['depth_grad_gaussian_sigma']
        self._downsample_rate = self._config['downsample_rate']
        self._rescale_factor = 1.0 / self._downsample_rate
        self._max_rejection_samples = self._config['max_rejection_samples']

        # distance thresholds for rejection sampling
        self._max_dist_from_center = self._config['max_dist_from_center']
        self._min_dist_from_boundary = self._config['min_dist_from_boundary']
        self._min_grasp_dist = self._config['min_grasp_dist']
        self._angle_dist_weight = self._config['angle_dist_weight']

        # depth sampling params
        self._depth_samples_per_grasp = max(self._config['depth_samples_per_grasp'], 1)
        self._min_depth_offset = self._config['min_depth_offset']
        self._max_depth_offset = self._config['max_depth_offset']
        self._h = self._config['depth_sample_win_height']
        self._w = self._config['depth_sample_win_width']
        self._depth_sampling_mode = self._config['depth_sampling_mode']

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
            if np.linalg.norm(normal_vec) == 0:
                normal_vec = np.array([1,0])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i,:] = normal_vec

        return normals

    def _sample_depth(self, min_depth, max_depth):
        """ Samples a depth value between the min and max. """
        depth_sample = max_depth
        if self._depth_sampling_mode == DepthSamplingMode.UNIFORM:
            depth_sample = min_depth + (max_depth - min_depth) * np.random.rand()
        elif self._depth_sampling_mode == DepthSamplingMode.MIN:
            depth_sample = min_depth
        return depth_sample

    def _sample(self, rgbd_im, camera_intr, num_samples, segmask=None,
                visualize=False):
        """
        Sample a set of 2D grasp candidates from a depth image.

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
        """
        Sample a set of 2D grasp candidates from a depth image by finding depth
        edges, then uniformly sampling point pairs and keeping only antipodal
        grasps with width less than the maximum allowable.

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
        # compute edge pixels
        edge_start = time()
        depth_im = rgbd_im.depth
        depth_im = depth_im.apply(snf.gaussian_filter,
                                  sigma=self._depth_grad_gaussian_sigma)
        depth_im_downsampled = depth_im.resize(self._rescale_factor)
        depth_im_threshed = depth_im_downsampled.threshold_gradients(self._depth_grad_thresh)
        edge_pixels = self._downsample_rate * depth_im_threshed.zero_pixels()
        depth_im_mask = depth_im.copy()
        if segmask is not None:
            edge_pixels = np.array([p for p in edge_pixels if np.any(segmask[p[0], p[1]] > 0)])
            depth_im_mask = depth_im.mask_binary(segmask)
        num_pixels = edge_pixels.shape[0]
        logging.debug('Depth edge detection took %.3f sec' %(time() - edge_start))
        logging.debug('Found %d edge pixels' %(num_pixels))

        # exit if no edge pixels
        if num_pixels == 0:
            return []

        # compute_max_depth
        min_depth = np.min(depth_im_mask.data[depth_im_mask.data > 0]) + self._min_depth_offset
        max_depth = np.max(depth_im_mask.data[depth_im_mask.data > 0]) + self._max_depth_offset

        # compute surface normals
        normal_start = time()
        edge_normals = self._surface_normals(depth_im, edge_pixels)
        logging.debug('Normal computation took %.3f sec' %(time() - normal_start))

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
        max_grasp_width_px = Grasp2D(Point(np.zeros(2)), 0.0, min_depth,
                                     width = self._gripper_width,
                                     camera_intr=camera_intr).width_px
        normal_ip = edge_normals.dot(edge_normals.T)
        dists = ssd.squareform(ssd.pdist(edge_pixels))
        valid_indices = np.where((normal_ip < -np.cos(np.arctan(self._friction_coef))) & (dists < max_grasp_width_px) & (dists > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        num_pairs = valid_indices.shape[0]
        logging.debug('Normal pruning %.3f sec' %(time() - sample_start))

        # raise exception if no antipodal pairs
        if num_pairs == 0:
            return []

        # iteratively sample grasps
        k = 0
        grasps = []
        sample_size = min(self._max_rejection_samples, num_pairs)
        candidate_pair_indices = np.random.choice(num_pairs, size=sample_size,
                                                  replace=False)
        while k < sample_size and len(grasps) < num_samples:
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
                grasp_theta = 0
                if grasp_axis[1] != 0:
                    grasp_theta = np.arctan(grasp_axis[0] / grasp_axis[1])
                    
                # compute distance from image center
                dist_from_center = np.linalg.norm(grasp_center - depth_im.center)
                dist_from_boundary = min(np.abs(depth_im.height - grasp_center[0]),
                                         np.abs(depth_im.width - grasp_center[1]),
                                         grasp_center[0],
                                         grasp_center[1])
                if dist_from_center < self._max_dist_from_center and \
                   dist_from_boundary > self._min_dist_from_boundary:
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
                            # get depth in the neighborhood of the center pixel
                            depth_win = depth_im.data[grasp_center[0]-self._h:grasp_center[0]+self._h, grasp_center[1]-self._w:grasp_center[1]+self._w]
                            center_depth = np.min(depth_win)
                            if center_depth == 0 or np.isnan(center_depth):
                                continue

                            # sample depth between the min and max
                            min_depth = np.min(center_depth) + self._min_depth_offset
                            max_depth = np.max(center_depth) + self._max_depth_offset
                            sample_depth = min_depth + (max_depth - min_depth) * np.random.rand()
                            candidate_grasp = Grasp2D(grasp_center_pt,
                                                      grasp_theta,
                                                      sample_depth,
                                                      width=self._gripper_width,
                                                      camera_intr=camera_intr)
                            grasps.append(candidate_grasp)

        # return sampled grasps
        return grasps

class DepthImageSuctionPointSampler(ImageGraspSampler):
    """ Grasp sampler for suction points from depth images.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters

    Other Parameters
    ----------------
    max_suction_dir_optical_axis_angle : float
        maximum angle, in degrees, between the suction approach axis and the camera optical axis
    delta_theta : float
        maximum deviation from zero for the aziumth angle of a rotational perturbation to the surface normal (for sample diversity)
    delta_phi : float
        maximum deviation from zero for the elevation angle of a rotational perturbation to the surface normal (for sample diversity)
    sigma_depth : float
        standard deviation for a normal distribution over depth values (for sample diversity)
    min_suction_dist : float
        minimum admissible distance between suction points (for sample diversity)
    angle_dist_weight : float
        amount to weight the angle difference in suction point distance computation
    depth_gaussian_sigma : float
        sigma used for pre-smoothing the depth image for better gradients
    """
    def __init__(self, config):
        # init superclass
        ImageGraspSampler.__init__(self, config)

        # read params
        self._max_suction_dir_optical_axis_angle = np.deg2rad(self._config['max_suction_dir_optical_axis_angle'])

        self._min_theta = -np.deg2rad(self._config['delta_theta'])
        self._max_theta = np.deg2rad(self._config['delta_theta'])
        self._theta_rv = ss.uniform(loc=self._min_theta,
                                    scale=self._max_theta-self._min_theta)

        self._min_phi = -np.deg2rad(self._config['delta_phi'])
        self._max_phi = np.deg2rad(self._config['delta_phi'])
        self._phi_rv = ss.uniform(loc=self._min_phi,
                                  scale=self._max_phi-self._min_phi)

        self._sigma_depth = self._config['sigma_depth']
        self._depth_rv = ss.norm(0.0, self._sigma_depth**2)

        self._min_suction_dist = self._config['min_suction_dist']
        self._angle_dist_weight = self._config['angle_dist_weight']
        self._depth_gaussian_sigma = self._config['depth_gaussian_sigma']
 
    def _sample(self, rgbd_im, camera_intr, num_samples, segmask=None,
                visualize=False):
        """
        Sample a set of 2D grasp candidates from a depth image.

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
        grasps = self._sample_suction_points(rgbd_im, camera_intr, num_samples,
                                             segmask=segmask, visualize=visualize)
        return grasps

    def _sample_suction_points(self, rgbd_im, camera_intr, num_samples,
                               segmask=None, visualize=False):
        """
        Sample a set of 2D suction point candidates from a depth image by
        choosing points on an object surface uniformly at random
        and then sampling around the surface normal

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
        :obj:`list` of :obj:`SuctionPoint2D`
            list of 2D suction point candidates
        """
        # compute edge pixels
        depth_im = rgbd_im.depth
        depth_im = depth_im.apply(snf.gaussian_filter,
                                  sigma=self._depth_gaussian_sigma)
        depth_im_mask = depth_im.copy()
        if segmask is not None:
            depth_im_mask = depth_im.mask_binary(segmask)
            
        if visualize:
            vis.figure()
            vis.subplot(1,2,1)
            vis.imshow(depth_im)
            vis.subplot(1,2,2)
            vis.imshow(depth_im_mask)
            vis.show()

            """
            from visualization import Visualizer3D as vis3d
            point_cloud_im = camera_intr.deproject_to_image(depth_im)
            normal_cloud_im = point_cloud_im.normal_cloud_im()
            point_cloud = point_cloud_im.to_point_cloud()
            normal_cloud = normal_cloud_im.to_normal_cloud()
            vis3d.figure()
            vis3d.normals(normal_cloud, point_cloud,
                          scale=0.05, subsample=20)
            vis3d.points(point_cloud, subsample=20, scale=0.01, color=(0,0,1))
            vis3d.show()
            """

        # project to get the point cloud
        point_cloud_im = camera_intr.deproject_to_image(depth_im_mask)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        nonzero_px = depth_im_mask.nonzero_pixels()
        num_nonzero_px = nonzero_px.shape[0]
        
        # randomly sample points and add to image
        suction_points = []
        num_tries = 0
        while len(suction_points) < num_samples and num_tries < 10000:
            # update number of tries
            num_tries += 1

            # sample a point uniformly at random 
            ind = np.random.choice(num_nonzero_px, size=1, replace=False)[0]
            center_px = np.array([nonzero_px[ind,1], nonzero_px[ind,0]])
            center = Point(center_px, frame=camera_intr.frame)
            axis = -normal_cloud_im[center.y, center.x]
            depth = point_cloud_im[center.y, center.x][2]

            # perturb axis
            delta_theta = self._theta_rv.rvs(size=1)[0]
            delta_phi = self._phi_rv.rvs(size=1)[0]
            delta_T = RigidTransform.sph_coords_to_pose(delta_theta,
                                                        delta_phi)
            #axis = delta_T.rotation.dot(axis)

            # perturb depth
            delta_depth = self._depth_rv.rvs(size=1)[0]
            depth = depth + delta_depth

            # keep if the angle between the camera optical axis and the suction direction is less than a threshold
            psi = np.arccos(axis.dot(np.array([0,0,1])))
            if psi < self._max_suction_dir_optical_axis_angle:

                # check distance to ensure sample diversity
                candidate = SuctionPoint2D(center, axis, depth, camera_intr=camera_intr)
                suction_point_dists = [SuctionPoint2D.image_dist(suction_point, candidate, alpha=self._angle_dist_weight) for suction_point in suction_points]
                if len(suction_points) == 0 or np.min(suction_point_dists) > self._min_suction_dist:

                    if visualize:
                        vis.figure()
                        vis.imshow(depth_im)
                        vis.scatter(center.x, center.y)
                        vis.show()

                    suction_points.append(candidate)

        return suction_points
        
class ImageGraspSamplerFactory(object):
    """ Factory for image grasp samplers. """
    @staticmethod
    def sampler(sampler_type, config):
        if sampler_type == 'antipodal_depth':
            return AntipodalDepthImageGraspSampler(config)
        elif sampler_type == 'suction':
            return DepthImageSuctionPointSampler(config)
        else:
            raise ValueError('Image grasp sampler type %s not supported!' %(sampler_type))
