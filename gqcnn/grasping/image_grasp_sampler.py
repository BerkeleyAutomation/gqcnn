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

Classes for sampling a set of grasps directly from images to generate data for
a neural network.

Author
------
Jeff Mahler & Sherdil Niyaz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import random
from time import time

from future.utils import with_metaclass
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as snf
import scipy.spatial.distance as ssd
import scipy.stats as ss

from autolab_core import Point, RigidTransform, Logger
from perception import (DepthImage, RgbdImage, GdImage)
from visualization import Visualizer2D as vis

from .grasp import Grasp2D, SuctionPoint2D, MultiSuctionPoint2D


def force_closure(p1, p2, n1, n2, mu):
    """Computes whether or not the point and normal pairs are in force
    closure."""
    # Line between the contacts.
    v = p2 - p1
    v = v / np.linalg.norm(v)

    # Compute cone membership.
    alpha = np.arctan(mu)
    dot_1 = max(min(n1.dot(-v), 1.0), -1.0)
    dot_2 = max(min(n2.dot(v), 1.0), -1.0)
    in_cone_1 = (np.arccos(dot_1) < alpha)
    in_cone_2 = (np.arccos(dot_2) < alpha)
    return (in_cone_1 and in_cone_2)


class DepthSamplingMode(object):
    """Modes for sampling grasp depth."""
    UNIFORM = "uniform"
    MIN = "min"
    MAX = "max"


class ImageGraspSampler(with_metaclass(ABCMeta, object)):
    """Wraps image to crane grasp candidate generation for easy deployment of
    GQ-CNN.

    Attributes
    ----------
    config : :obj:`autolab_core.YamlConfig`
        A dictionary-like object containing the parameters of the sampler.
    """

    def __init__(self, config):
        # Set params.
        self._config = config

        # Setup logger.
        self._logger = Logger.get_logger(self.__class__.__name__)

    def sample(self,
               rgbd_im,
               camera_intr,
               num_samples,
               segmask=None,
               seed=None,
               visualize=False,
               constraint_fn=None):
        """Samples a set of 2D grasps from a given RGB-D image.

        Parameters
        ----------
        rgbd_im : :obj:`perception.RgbdImage`
            RGB-D image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        seed : int
            Number to use in random seed (`None` if no seed).
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            The list of grasps in image space.
        """
        # Set random seed for determinism.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Sample an initial set of grasps (without depth).
        self._logger.debug("Sampling 2d candidates")
        sampling_start = time()
        grasps = self._sample(rgbd_im,
                              camera_intr,
                              num_samples,
                              segmask=segmask,
                              visualize=visualize,
                              constraint_fn=constraint_fn)
        sampling_stop = time()
        self._logger.debug("Sampled %d grasps from image" % (len(grasps)))
        self._logger.debug("Sampling grasps took %.3f sec" %
                           (sampling_stop - sampling_start))
        return grasps

    @abstractmethod
    def _sample(self,
                rgbd_im,
                camera_intr,
                num_samples,
                segmask=None,
                visualize=False,
                constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image.

        Subclasses must override.

        Parameters
        ----------
        rgbd_im : :obj:`perception.RgbdImage`
            RGB-D image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        pass


class AntipodalDepthImageGraspSampler(ImageGraspSampler):
    """Grasp sampler for antipodal point pairs from depth image gradients.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters.

    Other Parameters
    ----------------
    gripper_width : float
        Width of the gripper, in meters.
    friction_coef : float
        Friction coefficient for 2D force closure.
    depth_grad_thresh : float
        Threshold for depth image gradients to determine edge points for
        sampling.
    depth_grad_gaussian_sigma : float
        Sigma used for pre-smoothing the depth image for better gradients.
    downsample_rate : float
        Factor to downsample the depth image by before sampling grasps.
    max_rejection_samples : int
        Ceiling on the number of grasps to check in antipodal grasp rejection
        sampling.
    max_dist_from_center : int
        Maximum allowable distance of a grasp from the image center.
    min_grasp_dist : float
        Threshold on the grasp distance.
    angle_dist_weight : float
        Amount to weight the angle difference in grasp distance computation.
    depth_samples_per_grasp : int
        Number of depth samples to take per grasp.
    min_depth_offset : float
        Offset from the minimum depth at the grasp center pixel to use in depth
        sampling.
    max_depth_offset : float
        Offset from the maximum depth across all edges.
    depth_sample_win_height : float
        Height of a window around the grasp center pixel used to determine min
        depth.
    depth_sample_win_height : float
        Width of a window around the grasp center pixel used to determine min
        depth.
    depth_sampling_mode : str
        Name of depth sampling mode (uniform, min, max).
    """

    def __init__(self, config, gripper_width=np.inf):
        # Init superclass.
        ImageGraspSampler.__init__(self, config)

        # Antipodality params.
        self._gripper_width = self._config["gripper_width"]
        self._friction_coef = self._config["friction_coef"]
        self._depth_grad_thresh = self._config["depth_grad_thresh"]
        self._depth_grad_gaussian_sigma = self._config[
            "depth_grad_gaussian_sigma"]
        self._downsample_rate = self._config["downsample_rate"]
        self._rescale_factor = 1.0 / self._downsample_rate
        self._max_rejection_samples = self._config["max_rejection_samples"]

        self._min_num_edge_pixels = 0
        if "min_num_edge_pixels" in self._config:
            self._min_num_edge_pixels = self._config["min_num_edge_pixels"]

        # Distance thresholds for rejection sampling.
        self._max_dist_from_center = self._config["max_dist_from_center"]
        self._min_dist_from_boundary = self._config["min_dist_from_boundary"]
        self._min_grasp_dist = self._config["min_grasp_dist"]
        self._angle_dist_weight = self._config["angle_dist_weight"]

        # Depth sampling params.
        self._depth_samples_per_grasp = max(
            self._config["depth_samples_per_grasp"], 1)
        self._min_depth_offset = self._config["min_depth_offset"]
        self._max_depth_offset = self._config["max_depth_offset"]
        self._h = self._config["depth_sample_win_height"]
        self._w = self._config["depth_sample_win_width"]
        self._depth_sampling_mode = self._config["depth_sampling_mode"]

        # Perturbation.
        self._grasp_center_sigma = 0.0
        if "grasp_center_sigma" in self._config:
            self._grasp_center_sigma = self._config["grasp_center_sigma"]
        self._grasp_angle_sigma = 0.0
        if "grasp_angle_sigma" in self._config:
            self._grasp_angle_sigma = np.deg2rad(
                self._config["grasp_angle_sigma"])

    def _surface_normals(self, depth_im, edge_pixels):
        """Return an array of the surface normals at the edge pixels."""
        # Compute the gradients.
        grad = np.gradient(depth_im.data.astype(np.float32))

        # Compute surface normals.
        normals = np.zeros([edge_pixels.shape[0], 2])
        for i, pix in enumerate(edge_pixels):
            dx = grad[1][pix[0], pix[1]]
            dy = grad[0][pix[0], pix[1]]
            normal_vec = np.array([dy, dx])
            if np.linalg.norm(normal_vec) == 0:
                normal_vec = np.array([1, 0])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i, :] = normal_vec

        return normals

    def _sample_depth(self, min_depth, max_depth):
        """Samples a depth value between the min and max."""
        depth_sample = max_depth
        if self._depth_sampling_mode == DepthSamplingMode.UNIFORM:
            depth_sample = min_depth + (max_depth -
                                        min_depth) * np.random.rand()
        elif self._depth_sampling_mode == DepthSamplingMode.MIN:
            depth_sample = min_depth
        return depth_sample

    def _sample(self,
                image,
                camera_intr,
                num_samples,
                segmask=None,
                visualize=False,
                constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image.

        Parameters
        ----------
        image : :obj:`perception.RgbdImage` or :obj:`perception.DepthImage` or :obj:`perception.GdImage`  # noqa: E501
            RGB-D or Depth image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        if isinstance(image, RgbdImage) or isinstance(image, GdImage):
            depth_im = image.depth
        elif isinstance(image, DepthImage):
            depth_im = image
        else:
            raise ValueError(
                "image type must be one of [RgbdImage, DepthImage, GdImage]")

        # Sample antipodal pairs in image space.
        grasps = self._sample_antipodal_grasps(depth_im,
                                               camera_intr,
                                               num_samples,
                                               segmask=segmask,
                                               visualize=visualize,
                                               constraint_fn=constraint_fn)
        return grasps

    def _sample_antipodal_grasps(self,
                                 depth_im,
                                 camera_intr,
                                 num_samples,
                                 segmask=None,
                                 visualize=False,
                                 constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image by finding
        depth edges, then uniformly sampling point pairs and keeping only
        antipodal grasps with width less than the maximum allowable.

        Parameters
        ----------
        depth_im : :obj:"perception.DepthImage"
            Depth image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        # Compute edge pixels.
        edge_start = time()
        depth_im = depth_im.apply(snf.gaussian_filter,
                                  sigma=self._depth_grad_gaussian_sigma)
        scale_factor = self._rescale_factor
        depth_im_downsampled = depth_im.resize(scale_factor)
        depth_im_threshed = depth_im_downsampled.threshold_gradients(
            self._depth_grad_thresh)
        edge_pixels = (1.0 / scale_factor) * depth_im_threshed.zero_pixels()
        edge_pixels = edge_pixels.astype(np.int16)

        depth_im_mask = depth_im.copy()
        if segmask is not None:
            edge_pixels = np.array(
                [p for p in edge_pixels if np.any(segmask[p[0], p[1]] > 0)])
            depth_im_mask = depth_im.mask_binary(segmask)

        # Re-threshold edges if there are too few.
        if edge_pixels.shape[0] < self._min_num_edge_pixels:
            self._logger.info("Too few edge pixels!")
            depth_im_threshed = depth_im.threshold_gradients(
                self._depth_grad_thresh)
            edge_pixels = depth_im_threshed.zero_pixels()
            edge_pixels = edge_pixels.astype(np.int16)
            depth_im_mask = depth_im.copy()
            if segmask is not None:
                edge_pixels = np.array([
                    p for p in edge_pixels if np.any(segmask[p[0], p[1]] > 0)
                ])
                depth_im_mask = depth_im.mask_binary(segmask)

        num_pixels = edge_pixels.shape[0]
        self._logger.debug("Depth edge detection took %.3f sec" %
                           (time() - edge_start))
        self._logger.debug("Found %d edge pixels" % (num_pixels))

        # Compute point cloud.
        point_cloud_im = camera_intr.deproject_to_image(depth_im_mask)

        # Compute_max_depth.
        depth_data = depth_im_mask.data[depth_im_mask.data > 0]
        if depth_data.shape[0] == 0:
            return []

        min_depth = np.min(depth_data) + self._min_depth_offset
        max_depth = np.max(depth_data) + self._max_depth_offset

        # Compute surface normals.
        normal_start = time()
        edge_normals = self._surface_normals(depth_im, edge_pixels)
        self._logger.debug("Normal computation took %.3f sec" %
                           (time() - normal_start))

        if visualize:
            edge_pixels = edge_pixels[::2, :]
            edge_normals = edge_normals[::2, :]

            vis.figure()
            vis.subplot(1, 3, 1)
            vis.imshow(depth_im)
            if num_pixels > 0:
                vis.scatter(edge_pixels[:, 1], edge_pixels[:, 0], s=2, c="b")

            X = [pix[1] for pix in edge_pixels]
            Y = [pix[0] for pix in edge_pixels]
            U = [3 * pix[1] for pix in edge_normals]
            V = [-3 * pix[0] for pix in edge_normals]
            plt.quiver(X,
                       Y,
                       U,
                       V,
                       units="x",
                       scale=0.25,
                       width=0.5,
                       zorder=2,
                       color="r")
            vis.title("Edge pixels and normals")

            vis.subplot(1, 3, 2)
            vis.imshow(depth_im_threshed)
            vis.title("Edge map")

            vis.subplot(1, 3, 3)
            vis.imshow(segmask)
            vis.title("Segmask")
            vis.show()

        # Exit if no edge pixels.
        if num_pixels == 0:
            return []

        # Form set of valid candidate point pairs.
        pruning_start = time()
        max_grasp_width_px = Grasp2D(Point(np.zeros(2)),
                                     0.0,
                                     min_depth,
                                     width=self._gripper_width,
                                     camera_intr=camera_intr).width_px
        normal_ip = edge_normals.dot(edge_normals.T)
        dists = ssd.squareform(ssd.pdist(edge_pixels))
        valid_indices = np.where(
            (normal_ip < -np.cos(np.arctan(self._friction_coef)))
            & (dists < max_grasp_width_px) & (dists > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        self._logger.debug("Normal pruning %.3f sec" %
                           (time() - pruning_start))

        # Raise exception if no antipodal pairs.
        num_pairs = valid_indices.shape[0]
        if num_pairs == 0:
            return []

        # Prune out grasps.
        contact_points1 = edge_pixels[valid_indices[:, 0], :]
        contact_points2 = edge_pixels[valid_indices[:, 1], :]
        contact_normals1 = edge_normals[valid_indices[:, 0], :]
        contact_normals2 = edge_normals[valid_indices[:, 1], :]
        v = contact_points1 - contact_points2
        v_norm = np.linalg.norm(v, axis=1)
        v = v / np.tile(v_norm[:, np.newaxis], [1, 2])
        ip1 = np.sum(contact_normals1 * v, axis=1)
        ip2 = np.sum(contact_normals2 * (-v), axis=1)
        ip1[ip1 > 1.0] = 1.0
        ip1[ip1 < -1.0] = -1.0
        ip2[ip2 > 1.0] = 1.0
        ip2[ip2 < -1.0] = -1.0
        beta1 = np.arccos(ip1)
        beta2 = np.arccos(ip2)
        alpha = np.arctan(self._friction_coef)
        antipodal_indices = np.where((beta1 < alpha) & (beta2 < alpha))[0]

        # Raise exception if no antipodal pairs.
        num_pairs = antipodal_indices.shape[0]
        if num_pairs == 0:
            return []
        sample_size = min(self._max_rejection_samples, num_pairs)
        grasp_indices = np.random.choice(antipodal_indices,
                                         size=sample_size,
                                         replace=False)
        self._logger.debug("Grasp comp took %.3f sec" %
                           (time() - pruning_start))

        # Compute grasps.
        sample_start = time()
        k = 0
        grasps = []
        while k < sample_size and len(grasps) < num_samples:
            grasp_ind = grasp_indices[k]
            p1 = contact_points1[grasp_ind, :]
            p2 = contact_points2[grasp_ind, :]
            n1 = contact_normals1[grasp_ind, :]
            n2 = contact_normals2[grasp_ind, :]
            #            width = np.linalg.norm(p1 - p2)
            k += 1

            # Compute center and axis.
            grasp_center = (p1 + p2) // 2
            grasp_axis = p2 - p1
            grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
            grasp_theta = np.pi / 2
            if grasp_axis[1] != 0:
                grasp_theta = np.arctan2(grasp_axis[0], grasp_axis[1])
            grasp_center_pt = Point(np.array(
                [grasp_center[1], grasp_center[0]]),
                                    frame=camera_intr.frame)

            # Compute grasp points in 3D.
            x1 = point_cloud_im[p1[0], p1[1]]
            x2 = point_cloud_im[p2[0], p2[1]]
            if np.linalg.norm(x2 - x1) > self._gripper_width:
                continue

            # Perturb.
            if self._grasp_center_sigma > 0.0:
                grasp_center_pt = grasp_center_pt + ss.multivariate_normal.rvs(
                    cov=self._grasp_center_sigma * np.diag(np.ones(2)))
            if self._grasp_angle_sigma > 0.0:
                grasp_theta = grasp_theta + ss.norm.rvs(
                    scale=self._grasp_angle_sigma)

            # Check center px dist from boundary.
            if (grasp_center[0] < self._min_dist_from_boundary
                    or grasp_center[1] < self._min_dist_from_boundary
                    or grasp_center[0] >
                    depth_im.height - self._min_dist_from_boundary
                    or grasp_center[1] >
                    depth_im.width - self._min_dist_from_boundary):
                continue

            # Sample depths.
            for i in range(self._depth_samples_per_grasp):
                # Get depth in the neighborhood of the center pixel.
                depth_win = depth_im.data[grasp_center[0] -
                                          self._h:grasp_center[0] +
                                          self._h, grasp_center[1] -
                                          self._w:grasp_center[1] + self._w]
                center_depth = np.min(depth_win)
                if center_depth == 0 or np.isnan(center_depth):
                    continue

                # Sample depth between the min and max.
                min_depth = center_depth + self._min_depth_offset
                max_depth = center_depth + self._max_depth_offset
                sample_depth = min_depth + (max_depth -
                                            min_depth) * np.random.rand()
                candidate_grasp = Grasp2D(grasp_center_pt,
                                          grasp_theta,
                                          sample_depth,
                                          width=self._gripper_width,
                                          camera_intr=camera_intr,
                                          contact_points=[p1, p2],
                                          contact_normals=[n1, n2])

                if visualize:
                    vis.figure()
                    vis.imshow(depth_im)
                    vis.grasp(candidate_grasp)
                    vis.scatter(p1[1], p1[0], c="b", s=25)
                    vis.scatter(p2[1], p2[0], c="b", s=25)
                    vis.show()

                grasps.append(candidate_grasp)

        # Return sampled grasps.
        self._logger.debug("Loop took %.3f sec" % (time() - sample_start))
        return grasps


class DepthImageSuctionPointSampler(ImageGraspSampler):
    """Grasp sampler for suction points from depth images.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters.

    Other Parameters
    ----------------
    max_suction_dir_optical_axis_angle : float
        Maximum angle, in degrees, between the suction approach axis and the
        camera optical axis.
    delta_theta : float
        Maximum deviation from zero for the aziumth angle of a rotational
        perturbation to the surface normal (for sample diversity).
    delta_phi : float
        Maximum deviation from zero for the elevation angle of a rotational
        perturbation to the surface normal (for sample diversity).
    sigma_depth : float
        Standard deviation for a normal distribution over depth values (for
        sample diversity).
    min_suction_dist : float
        Minimum admissible distance between suction points (for sample
        diversity).
    angle_dist_weight : float
        Amount to weight the angle difference in suction point distance
        computation.
    depth_gaussian_sigma : float
        Sigma used for pre-smoothing the depth image for better gradients.
    """

    def __init__(self, config):
        # Init superclass.
        ImageGraspSampler.__init__(self, config)

        # Read params.
        self._max_suction_dir_optical_axis_angle = np.deg2rad(
            self._config["max_suction_dir_optical_axis_angle"])
        self._max_dist_from_center = self._config["max_dist_from_center"]
        self._min_dist_from_boundary = self._config["min_dist_from_boundary"]
        self._max_num_samples = self._config["max_num_samples"]

        self._min_theta = -np.deg2rad(self._config["delta_theta"])
        self._max_theta = np.deg2rad(self._config["delta_theta"])
        self._theta_rv = ss.uniform(loc=self._min_theta,
                                    scale=self._max_theta - self._min_theta)

        self._min_phi = -np.deg2rad(self._config["delta_phi"])
        self._max_phi = np.deg2rad(self._config["delta_phi"])
        self._phi_rv = ss.uniform(loc=self._min_phi,
                                  scale=self._max_phi - self._min_phi)

        self._mean_depth = 0.0
        if "mean_depth" in self._config:
            self._mean_depth = self._config["mean_depth"]
        self._sigma_depth = self._config["sigma_depth"]
        self._depth_rv = ss.norm(self._mean_depth, self._sigma_depth**2)

        self._min_suction_dist = self._config["min_suction_dist"]
        self._angle_dist_weight = self._config["angle_dist_weight"]
        self._depth_gaussian_sigma = self._config["depth_gaussian_sigma"]

    def _sample(self,
                image,
                camera_intr,
                num_samples,
                segmask=None,
                visualize=False,
                constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image.

        Parameters
        ----------
        image : :obj:`perception.RgbdImage` or "perception.DepthImage"
            RGB-D or D image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        if isinstance(image, RgbdImage) or isinstance(image, GdImage):
            depth_im = image.depth
        elif isinstance(image, DepthImage):
            depth_im = image
        else:
            raise ValueError(
                "image type must be one of [RgbdImage, DepthImage, GdImage]")

        # Sample antipodal pairs in image space.
        grasps = self._sample_suction_points(depth_im,
                                             camera_intr,
                                             num_samples,
                                             segmask=segmask,
                                             visualize=visualize,
                                             constraint_fn=constraint_fn)
        return grasps

    def _sample_suction_points(self,
                               depth_im,
                               camera_intr,
                               num_samples,
                               segmask=None,
                               visualize=False,
                               constraint_fn=None):
        """Sample a set of 2D suction point candidates from a depth image by
        choosing points on an object surface uniformly at random
        and then sampling around the surface normal.

        Parameters
        ----------
        depth_im : :obj:"perception.DepthImage"
            Depth image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).

        Returns
        -------
        :obj:`list` of :obj:`SuctionPoint2D`
            List of 2D suction point candidates.
        """
        # Compute edge pixels.
        filter_start = time()
        if self._depth_gaussian_sigma > 0:
            depth_im_mask = depth_im.apply(snf.gaussian_filter,
                                           sigma=self._depth_gaussian_sigma)
        else:
            depth_im_mask = depth_im.copy()
        if segmask is not None:
            depth_im_mask = depth_im.mask_binary(segmask)
        self._logger.debug("Filtering took %.3f sec" % (time() - filter_start))

        if visualize:
            vis.figure()
            vis.subplot(1, 2, 1)
            vis.imshow(depth_im)
            vis.subplot(1, 2, 2)
            vis.imshow(depth_im_mask)
            vis.show()

        # Project to get the point cloud.
        cloud_start = time()
        point_cloud_im = camera_intr.deproject_to_image(depth_im_mask)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        nonzero_px = depth_im_mask.nonzero_pixels()
        num_nonzero_px = nonzero_px.shape[0]
        if num_nonzero_px == 0:
            return []
        self._logger.debug("Normal cloud took %.3f sec" %
                           (time() - cloud_start))

        # Randomly sample points and add to image.
        sample_start = time()
        suction_points = []
        k = 0
        sample_size = min(self._max_num_samples, num_nonzero_px)
        indices = np.random.choice(num_nonzero_px,
                                   size=sample_size,
                                   replace=False)
        while k < sample_size and len(suction_points) < num_samples:
            # Sample a point uniformly at random.
            ind = indices[k]
            center_px = np.array([nonzero_px[ind, 1], nonzero_px[ind, 0]])
            center = Point(center_px, frame=camera_intr.frame)
            axis = -normal_cloud_im[center.y, center.x]
            depth = point_cloud_im[center.y, center.x][2]

            # Update number of tries.
            k += 1

            # Check center px dist from boundary.
            if (center_px[0] < self._min_dist_from_boundary
                    or center_px[1] < self._min_dist_from_boundary
                    or center_px[1] >
                    depth_im.height - self._min_dist_from_boundary
                    or center_px[0] >
                    depth_im.width - self._min_dist_from_boundary):
                continue

            # Perturb depth.
            delta_depth = self._depth_rv.rvs(size=1)[0]
            depth = depth + delta_depth

            # Keep if the angle between the camera optical axis and the suction
            # direction is less than a threshold.
            dot = max(min(axis.dot(np.array([0, 0, 1])), 1.0), -1.0)
            psi = np.arccos(dot)
            if psi < self._max_suction_dir_optical_axis_angle:

                # Create candidate grasp.
                candidate = SuctionPoint2D(center,
                                           axis,
                                           depth,
                                           camera_intr=camera_intr)

                # Check constraint satisfaction.
                if constraint_fn is None or constraint_fn(candidate):
                    if visualize:
                        vis.figure()
                        vis.imshow(depth_im)
                        vis.scatter(center.x, center.y)
                        vis.show()

                    suction_points.append(candidate)
        self._logger.debug("Loop took %.3f sec" % (time() - sample_start))
        return suction_points


class DepthImageMultiSuctionPointSampler(ImageGraspSampler):
    """Grasp sampler for suction points from depth images.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters.

    Other Parameters
    ----------------
    max_suction_dir_optical_axis_angle : float
        Maximum angle, in degrees, between the suction approach axis and the
        camera optical axis.
    delta_theta : float
        Maximum deviation from zero for the aziumth angle of a rotational
        perturbation to the surface normal (for sample diversity).
    delta_phi : float
        Maximum deviation from zero for the elevation angle of a rotational
        perturbation to the surface normal (for sample diversity).
    sigma_depth : float
        Standard deviation for a normal distribution over depth values (for
        sample diversity).
    min_suction_dist : float
        Minimum admissible distance between suction points (for sample
        diversity).
    angle_dist_weight : float
        Amount to weight the angle difference in suction point distance
        computation.
    depth_gaussian_sigma : float
        Sigma used for pre-smoothing the depth image for better gradients.
    """

    def __init__(self, config):
        # Init superclass.
        ImageGraspSampler.__init__(self, config)

        # Read params.
        self._max_suction_dir_optical_axis_angle = np.deg2rad(
            self._config["max_suction_dir_optical_axis_angle"])
        self._max_dist_from_center = self._config["max_dist_from_center"]
        self._min_dist_from_boundary = self._config["min_dist_from_boundary"]
        self._max_num_samples = self._config["max_num_samples"]

        self._min_theta = -np.deg2rad(self._config["delta_theta"])
        self._max_theta = np.deg2rad(self._config["delta_theta"])
        self._theta_rv = ss.uniform(loc=self._min_theta,
                                    scale=self._max_theta - self._min_theta)

        self._min_phi = -np.deg2rad(self._config["delta_phi"])
        self._max_phi = np.deg2rad(self._config["delta_phi"])
        self._phi_rv = ss.uniform(loc=self._min_phi,
                                  scale=self._max_phi - self._min_phi)

        self._mean_depth = 0.0
        if "mean_depth" in self._config:
            self._mean_depth = self._config["mean_depth"]
        self._sigma_depth = self._config["sigma_depth"]
        self._depth_rv = ss.norm(self._mean_depth, self._sigma_depth**2)

        self._min_suction_dist = self._config["min_suction_dist"]
        self._angle_dist_weight = self._config["angle_dist_weight"]
        self._depth_gaussian_sigma = self._config["depth_gaussian_sigma"]

    def _sample(self,
                image,
                camera_intr,
                num_samples,
                segmask=None,
                visualize=False,
                constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image.

        Parameters
        ----------
        image : :obj:`perception.RgbdImage` or `perception.DepthImage`
            RGB-D or D image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        if isinstance(image, RgbdImage) or isinstance(image, GdImage):
            depth_im = image.depth
        elif isinstance(image, DepthImage):
            depth_im = image
        else:
            raise ValueError(
                "image type must be one of [RgbdImage, DepthImage, GdImage]")

        # Sample antipodal pairs in image space.
        grasps = self._sample_suction_points(depth_im,
                                             camera_intr,
                                             num_samples,
                                             segmask=segmask,
                                             visualize=visualize,
                                             constraint_fn=constraint_fn)
        return grasps

    def _sample_suction_points(self,
                               depth_im,
                               camera_intr,
                               num_samples,
                               segmask=None,
                               visualize=False,
                               constraint_fn=None):
        """Sample a set of 2D suction point candidates from a depth image by
        choosing points on an object surface uniformly at random
        and then sampling around the surface normal.

        Parameters
        ----------
        depth_im : :obj:"perception.DepthImage"
            Depth image to sample from.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`perception.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`SuctionPoint2D`
            List of 2D suction point candidates.
        """
        # Compute edge pixels.
        filter_start = time()
        if self._depth_gaussian_sigma > 0:
            depth_im_mask = depth_im.apply(snf.gaussian_filter,
                                           sigma=self._depth_gaussian_sigma)
        else:
            depth_im_mask = depth_im.copy()
        if segmask is not None:
            depth_im_mask = depth_im.mask_binary(segmask)
        self._logger.debug("Filtering took %.3f sec" % (time() - filter_start))

        if visualize:
            vis.figure()
            vis.subplot(1, 2, 1)
            vis.imshow(depth_im)
            vis.subplot(1, 2, 2)
            vis.imshow(depth_im_mask)
            vis.show()

        # Project to get the point cloud.
        cloud_start = time()
        point_cloud_im = camera_intr.deproject_to_image(depth_im_mask)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        nonzero_px = depth_im_mask.nonzero_pixels()
        num_nonzero_px = nonzero_px.shape[0]
        if num_nonzero_px == 0:
            return []
        self._logger.debug("Normal cloud took %.3f sec" %
                           (time() - cloud_start))

        # Randomly sample points and add to image.
        sample_start = time()
        suction_points = []
        k = 0
        sample_size = min(self._max_num_samples, num_nonzero_px)
        indices = np.random.choice(num_nonzero_px,
                                   size=sample_size,
                                   replace=False)
        while k < sample_size and len(suction_points) < num_samples:
            # Sample a point uniformly at random.
            ind = indices[k]
            center_px = np.array([nonzero_px[ind, 1], nonzero_px[ind, 0]])
            center = Point(center_px, frame=camera_intr.frame)
            axis = -normal_cloud_im[center.y, center.x]
            #            depth = point_cloud_im[center.y, center.x][2]
            orientation = 2 * np.pi * np.random.rand()

            # Update number of tries.
            k += 1

            # Skip bad axes.
            if np.linalg.norm(axis) == 0:
                continue

            # Rotation matrix.
            x_axis = axis
            y_axis = np.array([axis[1], -axis[0], 0])
            if np.linalg.norm(y_axis) == 0:
                y_axis = np.array([1, 0, 0])
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            R = np.array([x_axis, y_axis, z_axis]).T
            #            R_orig = np.copy(R)
            R = R.dot(RigidTransform.x_axis_rotation(orientation))
            t = point_cloud_im[center.y, center.x]
            pose = RigidTransform(rotation=R,
                                  translation=t,
                                  from_frame="grasp",
                                  to_frame=camera_intr.frame)

            # Check center px dist from boundary.
            if (center_px[0] < self._min_dist_from_boundary
                    or center_px[1] < self._min_dist_from_boundary
                    or center_px[1] >
                    depth_im.height - self._min_dist_from_boundary
                    or center_px[0] >
                    depth_im.width - self._min_dist_from_boundary):
                continue

            # Keep if the angle between the camera optical axis and the suction
            # direction is less than a threshold.
            dot = max(min(axis.dot(np.array([0, 0, 1])), 1.0), -1.0)
            psi = np.arccos(dot)
            if psi < self._max_suction_dir_optical_axis_angle:

                # Check distance to ensure sample diversity.
                candidate = MultiSuctionPoint2D(pose, camera_intr=camera_intr)

                # Check constraint satisfaction.
                if constraint_fn is None or constraint_fn(candidate):
                    if visualize:
                        vis.figure()
                        vis.imshow(depth_im)
                        vis.scatter(center.x, center.y)
                        vis.show()

                    suction_points.append(candidate)
        self._logger.debug("Loop took %.3f sec" % (time() - sample_start))
        return suction_points


class ImageGraspSamplerFactory(object):
    """Factory for image grasp samplers."""

    @staticmethod
    def sampler(sampler_type, config):
        if sampler_type == "antipodal_depth":
            return AntipodalDepthImageGraspSampler(config)
        elif sampler_type == "suction":
            return DepthImageSuctionPointSampler(config)
        elif sampler_type == "multi_suction":
            return DepthImageMultiSuctionPointSampler(config)
        else:
            raise ValueError("Image grasp sampler type %s not supported!" %
                             (sampler_type))
