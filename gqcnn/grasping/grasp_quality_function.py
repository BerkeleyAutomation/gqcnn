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
Grasp quality functions: suction quality function and parallel jaw grasping quality fuction.
Author: Jason Liu and Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
from time import time
import logging

import scipy.ndimage.filters as snf
import cv2
import numpy as np
import matplotlib.pyplot as plt

import autolab_core.utils as utils
from ambicore import DepthImage, Transform
from gqcnn import get_gqcnn_model, get_fc_gqcnn_model
from gqcnn.grasping import Grasp2D, SuctionPoint2D
from gqcnn.utils import GripperMode

# constant for display
FIGSIZE = 16

class GraspQualityFunction(object):
    """Abstract grasp quality class. """
    __metaclass__ = ABCMeta

    def __init__(self):
        # set up logger
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, state, actions, params=None):
        """ Evaluates grasp quality for a set of actions given a state. """
        return self.quality(state, actions, params)

    @abstractmethod
    def quality(self, state, actions, params=None):
        """ Evaluates grasp quality for a set of actions given a state.

        Parameters
        ----------
        state : :obj:`object`
            state of the world e.g. image
        actions : :obj:`list`
            list of actions to evaluate e.g. parallel-jaw or suction grasps
        params : :obj:`dict`
            optional parameters for the evaluation

        Returns
        -------
        :obj:`numpy.ndarray`
            vector containing the real-valued grasp quality for each candidate
        """
        pass

class ZeroGraspQualityFunction(object):
    """ Null function. """
    def quality(self, state, actions, params=None):
        """ Returns zero for all grasps

        Parameters
        ----------
        state : :obj:`object`
            state of the world e.g. image
        actions : :obj:`list`
            list of actions to evaluate e.g. parallel-jaw or suction grasps
        params : :obj:`dict`
            optional parameters for the evaluation

        Returns
        -------
        :obj:`numpy.ndarray`
            vector containing the real-valued grasp quality for each candidate
        """
        return 0.0

class ParallelJawQualityFunction(GraspQualityFunction):
    """Abstract wrapper class for parallel jaw quality functions (only image based metrics for now). """
    def __init__(self, config):
        """Create a suction quality function. """
        GraspQualityFunction.__init__(self)

        # read parameters
        self._friction_coef = config['friction_coef']
        self._max_friction_cone_angle = np.arctan(self._friction_coef)
        
    def friction_cone_angle(self, action):
        """ Compute the angle between the axis and the boundaries of the friction cone. """
        if action.contact_points is None or action.contact_normals is None:
            raise ValueError('Cannot compute friction cone angle without precomputed contact points and normals')
        dot_prod1 = min(max(action.contact_normals[0].dot(-action.axis), -1.0), 1.0)
        angle1 = np.arccos(dot_prod1)
        dot_prod2 = min(max(action.contact_normals[1].dot(action.axis), -1.0), 1.0)
        angle2 = np.arccos(dot_prod2)
        return max(angle1, angle2)
        
    def force_closure(self, action):
        """ Determine if the grasp is in force closure. """
        return (self.friction_cone_angle(action) < self._max_friction_cone_angle)

class ComForceClosureParallelJawQualityFunction(ParallelJawQualityFunction):
    """ Measures the distance to the estimated center of mass for antipodal parallel-jaw grasps. """
    def __init__(self, config):
        """Create a best-fit planarity suction metric. """
        self._antipodality_pctile = config['antipodality_pctile']
        ParallelJawQualityFunction.__init__(self, config)

    def quality(self, state, actions, params=None): 
        """Given a parallel-jaw grasp, compute the distance to the center of mass of the grasped object

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`Grasp2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing quality. 

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        # compute antipodality
        antipodality_q = [ParallelJawQualityFunction.friction_cone_angle(self, action) for action in actions]
        
        # compute object centroid
        object_com = state.rgbd_im.center
        if state.segmask is not None:
            nonzero_px = state.segmask.nonzero_pixels()
            object_com = np.mean(nonzero_px, axis=0)
        
        # compute negative SSE from the best fit plane for each grasp
        antipodality_thresh = abs(np.percentile(antipodality_q, 100-self._antipodality_pctile))
        qualities = []
        max_q = max(state.rgbd_im.height, state.rgbd_im.width)
        for i, action in enumerate(actions):
            q = max_q
            friction_cone_angle = antipodality_q[i]
            force_closure = ParallelJawQualityFunction.force_closure(self, action)
            if force_closure or friction_cone_angle < antipodality_thresh:
                grasp_center = np.array([action.center[1], action.center[0]])

                if state.obj_segmask is not None:
                    grasp_obj_id = state.obj_segmask[grasp_center[0],
                                                     grasp_center[1]]
                    obj_mask = state.obj_segmask.segment_mask(grasp_obj_id)
                    nonzero_px = obj_mask.nonzero_pixels()
                    object_com = np.mean(nonzero_px, axis=0)            

                q = np.linalg.norm(grasp_center - object_com)

                if state.obj_segmask is not None and grasp_obj_id == 0:
                    q = max_q
                    
            q = (np.exp(-q/max_q) - np.exp(-1)) / (1 - np.exp(-1))
            qualities.append(q)

        return np.array(qualities)
            
class SuctionQualityFunction(GraspQualityFunction):
    """Abstract wrapper class for suction quality functions (only image based metrics for now). """
    def __init__(self, config):
        """Create a suction quality function. """
        GraspQualityFunction.__init__(self)

        # read parameters
        self._window_size = config['window_size']
        self._sample_rate = config['sample_rate']

    def _points_in_window(self, point_cloud_image, action, segmask=None):
        """Retrieve all points on the object in a box of size self._window_size. """
        # read indices
        im_shape = point_cloud_image.shape
        i_start = int(max(action.center[1]-self._window_size/2, 0))
        j_start = int(max(action.center[0]-self._window_size/2, 0))
        i_end = int(min(i_start+self._window_size, im_shape[0]))
        j_end = int(min(j_start+self._window_size, im_shape[1]))
        step = int(1 / self._sample_rate)

        # read 3D points in the window
        points = point_cloud_image[i_start:i_end:step, j_start:j_end:step]
        stacked_points = points.reshape(points.shape[0]*points.shape[1], -1)

        # form the matrices for plane-fitting
        return stacked_points

    def _points_to_matrices(self, points):
        """ Convert a set of 3D points to an A and b matrix for regression. """
        A = points[:, [0, 1]]
        ones = np.ones((A.shape[0], 1))
        A = np.hstack((A, ones))
        b = points[:, 2]
        return A, b

    def _best_fit_plane(self, A, b):
        """Find a best-fit plane of points. """
        try:
            w, _, _, _ = np.linalg.lstsq(A, b) 
        except np.linalg.LinAlgError:
            self._logger.warning('Could not find a best-fit plane!')
            raise
        return w

    def _sum_of_squared_residuals(self, w, A, z):
        """ Returns the sum of squared residuals from the plane. """
        if A.shape[0] == 0:
            return 0
        return (1.0 / A.shape[0]) * np.square(np.linalg.norm(np.dot(A, w) - z))

    def _residuals(self, w, A, z):
        """ Returns the sum of squared residuals from the plane. """
        return np.linalg.norm(np.dot(A, w) - z)
    
class BestFitPlanaritySuctionQualityFunction(SuctionQualityFunction):
    """A best-fit planarity suction metric. """

    def __init__(self, config):
        """Create a best-fit planarity suction metric. """
        SuctionQualityFunction.__init__(self, config)

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality. 

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w = self._best_fit_plane(A, b) # vector w w/ a bias term represents a best-fit plane.

            if params is not None and params['vis']['plane']:
                from ambicore import Visualizer2D as vis2d
                from ambicore import Visualizer3D as vis3d
                mid_i = A.shape[0] // 2
                pred_z = A.dot(w)
                p0 = np.array([A[mid_i,0], A[mid_i,1], pred_z[mid_i]])
                n = np.array([w[0], w[1], -1])
                n = n / np.linalg.norm(n)
                tx = np.array([n[1], -n[0], 0])
                tx = tx / np.linalg.norm(tx)
                ty = np.cross(n, tx)
                R = np.array([tx, ty, n]).T
                T_table_world = Transform(rotation=R,
                                               translation=p0,
                                               from_frame='patch',
                                               to_frame='world')
                
                vis3d.figure()
                vis3d.points(point_cloud_image.to_point_cloud(), scale=0.0025, subsample=10, random=True, color=(0,0,1))
                # vis3d.points(PointCloud(points.T), scale=0.0025, color=(1,0,0))
                vis3d.table(T_table_world, dim=0.01)
                vis3d.show()
                
                vis2d.figure()
                vis2d.imshow(state.rgbd_im.depth)
                vis2d.scatter(action.center[0], action.center[1], s=50, c='b')
                vis2d.show()

            quality = np.exp(-self._sum_of_squared_residuals(w, A, b)) # evaluate how well best-fit plane describles all points in window.            
            qualities.append(quality)

        return np.array(qualities)

class ApproachPlanaritySuctionQualityFunction(SuctionQualityFunction):
    """A approach planarity suction metric. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        SuctionQualityFunction.__init__(self, config)

    def _action_to_plane(self, point_cloud_image, action):
        """Convert a plane from point-normal form to general form. """
        x = int(action.center[0])
        y = int(action.center[1])
        p_0 = point_cloud_image[y, x]
        n = -action.axis
        w = np.array([-n[0]/n[2], -n[1]/n[2], np.dot(n,p_0)/n[2]])
        return w

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w = self._action_to_plane(point_cloud_image, action) # vector w w/ a bias term represents a best-fit plane.

            if params is not None and params['vis']['plane']:
                from ambicore import Visualizer2D as vis2d
                from ambicore import Visualizer3D as vis3d
                mid_i = A.shape[0] // 2
                pred_z = A.dot(w)
                p0 = np.array([A[mid_i,0], A[mid_i,1], pred_z[mid_i]])
                n = np.array([w[0], w[1], -1])
                n = n / np.linalg.norm(n)
                tx = np.array([n[1], -n[0], 0])
                tx = tx / np.linalg.norm(tx)
                ty = np.cross(n, tx)
                R = np.array([tx, ty, n]).T

                c = state.camera_intr.deproject_pixel(action.depth, action.center)
                d = c - 0.01*action.axis

                T_table_world = Transform(rotation=R,
                                               translation=p0,
                                               from_frame='patch',
                                               to_frame='world')
                
                vis3d.figure()
                vis3d.points(point_cloud_image.to_point_cloud(), scale=0.0025, subsample=10, random=True, color=(0,0,1))
                # vis3d.points(PointCloud(points.T), scale=0.0025, color=(1,0,0))
                vis3d.points(c, scale=0.005, color=(1,1,0))
                vis3d.points(d, scale=0.005, color=(1,1,0))
                vis3d.table(T_table_world, dim=0.01)
                vis3d.show()
                
                vis2d.figure()
                vis2d.imshow(state.rgbd_im.depth)
                vis2d.scatter(action.center[0], action.center[1], s=50, c='b')
                vis2d.show()

            quality = np.exp(-self._sum_of_squared_residuals(w, A, b)) # evaluate how well best-fit plane describles all points in window.            
            qualities.append(quality)

        return np.array(qualities)

class DiscApproachPlanaritySuctionQualityFunction(SuctionQualityFunction):
    """A approach planarity suction metric using a disc-shaped window. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        self._radius = config['radius']
        SuctionQualityFunction.__init__(self, config)

    def _action_to_plane(self, point_cloud_image, action):
        """Convert a plane from point-normal form to general form. """
        x = int(action.center[0])
        y = int(action.center[1])
        p_0 = point_cloud_image[y, x]
        n = -action.axis
        w = np.array([-n[0]/n[2], -n[1]/n[2], np.dot(n,p_0)/n[2]])
        return w, p_0

    def _points_in_window(self, point_cloud_image, action, segmask=None):
        """Retrieve all points on the object in a disc of size self._window_size. """
        # compute plane
        n = -action.axis
        U, _, _ = np.linalg.svd(n.reshape((3,1)))
        tangents = U[:,1:]        

        # read indices
        im_shape = point_cloud_image.shape
        i_start = int(max(action.center[1]-self._window_size/2, 0))
        j_start = int(max(action.center[0]-self._window_size/2, 0))
        i_end = int(min(i_start+self._window_size, im_shape[0]))
        j_end = int(min(j_start+self._window_size, im_shape[1]))
        step = int(1 / self._sample_rate)

        # read 3D points in the window
        points = point_cloud_image[i_start:i_end:step, j_start:j_end:step]
        stacked_points = points.reshape(points.shape[0]*points.shape[1], -1)
        
        # compute the center point
        contact_point = point_cloud_image[int(action.center[1]),
                                          int(action.center[0])]

        # project onto approach plane
        residuals = stacked_points - contact_point
        coords = residuals.dot(tangents)
        proj_residuals = coords.dot(tangents.T)

        # check distance from the center point along the approach plane
        dists = np.linalg.norm(proj_residuals, axis=1)
        stacked_points = stacked_points[dists <= self._radius]

        # form the matrices for plane-fitting
        return stacked_points

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w, p0 = self._action_to_plane(point_cloud_image, action) # vector w w/ a bias term represents a best-fit plane.
            #sse = self._sum_of_squared_residuals(w, A, b)
            sse = np.max(self._residuals(w, A, b))

            if params is not None and params['vis']['plane']:
                from ambicore import Visualizer2D as vis2d
                from ambicore import Visualizer3D as vis3d
                mid_i = A.shape[0] // 2
                pred_z = A.dot(w)
                n = -action.axis
                n = n / np.linalg.norm(n)
                tx = np.array([n[1], -n[0], 0])
                tx = tx / np.linalg.norm(tx)
                ty = np.cross(n, tx)
                R = np.array([tx, ty, n]).T

                c = state.camera_intr.deproject_pixel(action.depth, action.center)
                d = c - 0.01*action.axis

                T_table_world = Transform(rotation=R,
                                               translation=p0,
                                               from_frame='patch',
                                               to_frame='world')
                
                vis3d.figure()
                vis3d.points(point_cloud_image.to_point_cloud(), scale=0.001, subsample=5, random=True, color=(0,0,1))
                # vis3d.points(PointCloud(points.T), scale=0.0015, color=(1,0,0))
                vis3d.points(c, scale=0.002, color=(1,1,0))
                vis3d.points(d, scale=0.002, color=(1,1,0))
                vis3d.table(T_table_world, dim=0.01)
                vis3d.show()
                
                vis2d.figure()
                vis2d.imshow(state.rgbd_im.depth)
                vis2d.scatter(action.center[0], action.center[1], s=50, c='b')
                vis2d.show()

            quality = np.exp(-sse) # evaluate how well best-fit plane describles all points in window.            
            qualities.append(quality)

        return np.array(qualities)

class ComApproachPlanaritySuctionQualityFunction(ApproachPlanaritySuctionQualityFunction):
    """A approach planarity suction metric that ranks sufficiently planar points by their distance to the object COM. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        self._planarity_thresh = config['planarity_thresh']

        ApproachPlanaritySuctionQualityFunction.__init__(self, config)

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        # compute planarity
        sse = ApproachPlanaritySuctionQualityFunction.quality(self, state, actions, params=params)

        if params['vis']['hist']:
            plt.figure()
            utils.histogram(sse, 100, (np.min(sse), np.max(sse)), normalized=False, plot=True)
            plt.show()

        # compute object centroid
        object_com = state.rgbd_im.center
        if state.segmask is not None:
            nonzero_px = state.segmask.nonzero_pixels()
            object_com = np.mean(nonzero_px, axis=0)

        # threshold
        qualities = []
        for k, action in enumerate(actions):
            q = max(state.rgbd_im.height, state.rgbd_im.width)
            if np.abs(sse[k]) < self._planarity_thresh:
                grasp_center = np.array([action.center[1], action.center[0]])
                q = np.linalg.norm(grasp_center - object_com)

            qualities.append(np.exp(-q))

        return np.array(qualities)

class ComDiscApproachPlanaritySuctionQualityFunction(DiscApproachPlanaritySuctionQualityFunction):
    """A approach planarity suction metric that ranks sufficiently planar points by their distance to the object COM. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        self._planarity_pctile = config['planarity_pctile']
        self._planarity_abs_thresh = 0
        if 'planarity_abs_thresh' in config.keys():
            self._planarity_abs_thresh = np.exp(-config['planarity_abs_thresh'])

        DiscApproachPlanaritySuctionQualityFunction.__init__(self, config)

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        # compute planarity
        sse_q = DiscApproachPlanaritySuctionQualityFunction.quality(self, state, actions, params=params)

        if params['vis']['hist']:
            plt.figure()
            utils.histogram(sse_q, 100, (np.min(sse_q), np.max(sse_q)), normalized=False, plot=True)
            plt.show()

        # compute object centroid
        object_com = state.rgbd_im.center
        if state.segmask is not None:
            nonzero_px = state.segmask.nonzero_pixels()
            object_com = np.mean(nonzero_px, axis=0)

        # threshold
        planarity_thresh = abs(np.percentile(sse_q, 100-self._planarity_pctile))
        qualities = []
        max_q = max(state.rgbd_im.height, state.rgbd_im.width)
        for k, action in enumerate(actions):
            q = max_q
            if sse_q[k] > planarity_thresh or sse_q[k] > self._planarity_abs_thresh:
                grasp_center = np.array([action.center[1], action.center[0]])

                if state.obj_segmask is not None:
                    grasp_obj_id = state.obj_segmask[grasp_center[0],
                                                     grasp_center[1]]
                    obj_mask = state.obj_segmask.segment_mask(grasp_obj_id)
                    nonzero_px = obj_mask.nonzero_pixels()
                    object_com = np.mean(nonzero_px, axis=0)            

                q = np.linalg.norm(grasp_center - object_com)

            q = (np.exp(-q/max_q) - np.exp(-1)) / (1 - np.exp(-1))
            qualities.append(q)

        return np.array(qualities)

class GaussianCurvatureSuctionQualityFunction(SuctionQualityFunction):
    """A approach planarity suction metric. """

    def __init__(self, config):
        """Create approach planarity suction metric. """
        SuctionQualityFunction.__init__(self, config)

    def _points_to_matrices(self, points):
        """ Convert a set of 3D points to an A and b matrix for regression. """
        x = points[:,0]
        y = points[:,1]
        A = np.c_[x, y, x*x, x*y, y*y]
        ones = np.ones([A.shape[0], 1])
        A = np.c_[A, ones]
        b = points[:, 2]
        return A, b

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on a best-fit 3D plane of the neighboring points.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        qualities = []

        # deproject points
        point_cloud_image = state.camera_intr.deproject_to_image(state.rgbd_im.depth)

        # compute negative SSE from the best fit plane for each grasp
        for i, action in enumerate(actions):
            if not isinstance(action, SuctionPoint2D):
                raise ValueError('This function can only be used to evaluate suction quality')
            
            points = self._points_in_window(point_cloud_image, action, segmask=state.segmask) # x,y in matrix A and z is vector z.
            A, b = self._points_to_matrices(points)
            w = self._best_fit_plane(A, b) # vector w w/ a bias term represents a best-fit plane.
            
            # compute curvature
            fx = w[0]
            fy = w[1]
            fxx = 2 * w[2]
            fxy = w[3]
            fyy = 2 * w[4]
            curvature = (fxx * fyy - fxy**2) / ((1 + fx**2 + fy**2)**2)

            # store quality
            quality = np.exp(-np.abs(curvature))
            qualities.append(quality)

        return np.array(qualities)

class DiscCurvatureSuctionQualityFunction(GaussianCurvatureSuctionQualityFunction):
    def __init__(self, config):
        """Create approach planarity suction metric. """
        self._radius = config['radius']
        SuctionQualityFunction.__init__(self, config)

    def _points_in_window(self, point_cloud_image, action, segmask=None):
        """Retrieve all points on the object in a disc of size self._window_size. """
        # read indices
        im_shape = point_cloud_image.shape
        i_start = int(max(action.center[1]-self._window_size/2, 0))
        j_start = int(max(action.center[0]-self._window_size/2, 0))
        i_end = int(min(i_start+self._window_size, im_shape[0]))
        j_end = int(min(j_start+self._window_size, im_shape[1]))
        step = int(1 / self._sample_rate)

        # read 3D points in the window
        points = point_cloud_image[i_start:i_end:step, j_start:j_end:step]
        stacked_points = points.reshape(points.shape[0]*points.shape[1], -1)
        
        # check the distance from the center point
        contact_point = point_cloud_image[int(action.center[1]),
                                          int(action.center[0])]
        dists = np.linalg.norm(stacked_points - contact_point, axis=1)
        stacked_points = stacked_points[dists <= self._radius]

        # form the matrices for plane-fitting
        return stacked_points

class ComDiscCurvatureSuctionQualityFunction(DiscCurvatureSuctionQualityFunction):
    def __init__(self, config):
        """Create approach planarity suction metric. """
        self._curvature_pctile = config['curvature_pctile']

        DiscCurvatureSuctionQualityFunction.__init__(self, config)

    def quality(self, state, actions, params=None): 
        """Given a suction point, compute a score based on the Gaussian curvature.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            An RgbdImageState instance that encapsulates rgbd_im, camera_intr, segmask, full_observed.
        action: :obj:`SuctionPoint2D`
            A suction grasp in image space that encapsulates center, approach direction, depth, camera_intr.
        params: dict
            Stores params used in computing suction quality.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array of the quality for each grasp
        """
        # compute planarity
        curvature_q = DiscCurvatureSuctionQualityFunction.quality(self, state, actions, params=params)

        if params['vis']['hist']:
            plt.figure()
            utils.histogram(curvature_q, 100, (np.min(curvature_q), np.max(curvature_q)), normalized=False, plot=True)
            plt.show()

        # compute object centroid
        object_com = state.rgbd_im.center
        if state.segmask is not None:
            nonzero_px = state.segmask.nonzero_pixels()
            object_com = np.mean(nonzero_px, axis=0)

        # threshold
        curvature_q_thresh = abs(np.percentile(curvature_q, 100-self._curvature_pctile))
        qualities = []
        max_q = max(state.rgbd_im.height, state.rgbd_im.width)
        for k, action in enumerate(actions):
            q = max_q
            if curvature_q[k] > curvature_q_thresh:
                grasp_center = np.array([action.center[1], action.center[0]])
                q = np.linalg.norm(grasp_center - object_com)

            q = (np.exp(-q/max_q) - np.exp(-1)) / (1 - np.exp(-1))
            qualities.append(q)

        return np.array(qualities)

class GQCnnQualityFunction(GraspQualityFunction):
    def __init__(self, config):
        """Create a GQCNN suction quality function. """
        GraspQualityFunction.__init__(self)

        # store parameters
        self._config = config
        self._gqcnn_model_dir = config['gqcnn_model']
        self._crop_height = config['crop_height']
        self._crop_width = config['crop_width']
 
        # init GQ-CNN
        self._gqcnn = get_gqcnn_model().load(self._gqcnn_model_dir)

    def __del__(self):
        try:
            self._gqcnn.close_session()
        except:
            pass
        del self

    @property
    def gqcnn(self):
        """ Returns the GQ-CNN. """
        return self._gqcnn

    @property
    def gqcnn_recep_height(self):
        return self._gqcnn.im_height

    @property
    def gqcnn_recep_width(self):
        return self._gqcnn.im_width

    @property
    def gqcnn_stride(self):
        return self._gqcnn.stride

    @property
    def config(self):
        """ Returns the GQCNN quality function parameters. """
        return self._config

    def grasps_to_tensors(self, grasps, state):
        """Converts a list of grasps to an image and pose tensor
        for fast grasp quality evaluation.

        Attributes
        ----------
        grasps : :obj:`list` of :obj:`object`
            list of image grasps to convert
        state : :obj:`RgbdImageState`
            RGB-D image to plan grasps on

        Returns
        -------
        image_arr : :obj:`numpy.ndarray`
            4D numpy tensor of image to be predicted
        pose_arr : :obj:`numpy.ndarray`
            2D numpy tensor of depth values
        """
        # parse params
        gqcnn_im_height = self.gqcnn.im_height
        gqcnn_im_width = self.gqcnn.im_width
        gqcnn_num_channels = self.gqcnn.num_channels
        gqcnn_pose_dim = self.gqcnn.pose_dim
        gripper_mode = self.gqcnn.gripper_mode
        num_grasps = len(grasps)
        depth_im = state.rgbd_im.depth

        # allocate tensors
        tensor_start = time()
        image_tensor = np.zeros([num_grasps, gqcnn_im_height, gqcnn_im_width, gqcnn_num_channels])
        pose_tensor = np.zeros([num_grasps, gqcnn_pose_dim])
        scale = float(gqcnn_im_height) / self._crop_height
        depth_im_scaled = depth_im.rescale(scale)
        for i, grasp in enumerate(grasps):
            translation = scale * np.array([depth_im.center[0] - grasp.center[1],
                                            depth_im.center[1] - grasp.center[0]])
            im_tf = depth_im_scaled
            im_tf = depth_im_scaled.transform(translation, grasp.angle)
            im_tf = im_tf.crop(gqcnn_im_height, gqcnn_im_width)
            image_tensor[i,...] = im_tf.raw_data
            
            if gripper_mode == GripperMode.PARALLEL_JAW:
                pose_tensor[i] = grasp.depth
            elif gripper_mode == GripperMode.SUCTION:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.approach_angle])
            elif gripper_mode == GripperMode.MULTI_SUCTION:
                pose_tensor[i] = grasp.depth
            elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
                pose_tensor[i] = grasp.depth
            elif gripper_mode == GripperMode.LEGACY_SUCTION:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.approach_angle])
            else:
                raise ValueError('Gripper mode %s not supported' %(gripper_mode))
        self._logger.debug('Tensor conversion took %.3f sec' %(time()-tensor_start))
        return image_tensor, pose_tensor

    def quality(self, state, actions, params): 
        """ Evaluate the quality of a set of actions according to a GQ-CNN.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            state of the world described by an RGB-D image
        actions: :obj:`object`
            set of grasping actions to evaluate
        params: dict
            optional parameters for quality evaluation

        Returns
        -------
        :obj:`list` of float
            real-valued grasp quality predictions for each action, between 0 and 1
        """
        # open tensorflow session for gqcnn
        if self._gqcnn.sess is None:
            self._gqcnn.open_session()

        # form tensors
        tensor_start = time()
        image_tensor, pose_tensor = self.grasps_to_tensors(actions, state)
        self._logger.info('Image transformation took %.3f sec' %(time() - tensor_start))
        if params is not None and params['vis']['tf_images']:
            # read vis params
            k = params['vis']['k']
            d = utils.sqrt_ceil(k)

            # display grasp transformed images
            from ambicore import Visualizer2D as vis2d
            vis2d.figure(size=(FIGSIZE,FIGSIZE))
            for i, image_tf in enumerate(image_tensor[:k,...]):
                depth = pose_tensor[i][0]
                vis2d.subplot(d,d,i+1)
                vis2d.imshow(DepthImage(image_tf))
                vis2d.title('Image %d: d=%.3f' %(i, depth))
            vis2d.show()

        # predict grasps
        predict_start = time()
        output_arr = self.gqcnn.predict(image_tensor, pose_tensor)

        if self._gqcnn._angular_bins > 0:
            q_values = output_arr[:, 1] # first index since images were rotated into alignment
        else:
            q_values = output_arr[:,-1]

        self._logger.info('Inference took %.3f sec' %(time() - predict_start))
        return q_values.tolist()

class GQCnnPlanarityQualityFunction(GraspQualityFunction):
    def __init__(self, config):
        """Create a GQCNN suction quality function. """
        GraspQualityFunction.__init__(self)
        self._gqcnn_fn = GQCnnQualityFunction(config)
        self._planarity_fn = DiscApproachPlanaritySuctionQualityFunction(config)
        
        self._planarity_q_thresh = config['planarity_q_thresh']

    def quality(self, state, actions, params): 
        """ Evaluate the quality of a set of actions according to a GQ-CNN.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            state of the world described by an RGB-D image
        actions: :obj:`object`
            set of grasping actions to evaluate
        params: dict
            optional parameters for quality evaluation

        Returns
        -------
        :obj:`list` of float
            real-valued grasp quality predictions for each action, between 0 and 1
        """
        q_values = self._gqcnn_fn.quality(state, actions, params)
        planarity_q_values = self._planarity_fn(state, actions, params)
        for i in range(len(q_values)):
            planarity = planarity_q_values[i]
            if planarity < self._planarity_q_thresh:
                q_values[i] = 0
        return q_values

class GQCnnHeightQualityFunction(GraspQualityFunction):
    def __init__(self, config):
        """Create a GQCNN suction quality function. """
        GraspQualityFunction.__init__(self)
        self._gqcnn_fn = GQCnnQualityFunction(config)
        
        self._height_thresh = config['height_thresh']

    def quality(self, state, actions, params): 
        """ Evaluate the quality of a set of actions according to a GQ-CNN.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            state of the world described by an RGB-D image
        actions: :obj:`object`
            set of grasping actions to evaluate
        params: dict
            optional parameters for quality evaluation

        Returns
        -------
        :obj:`list` of float
            real-valued grasp quality predictions for each action, between 0 and 1
        """
        q_values = self._gqcnn_fn.quality(state, actions, params)
        for i in range(len(q_values)):
            T_grasp_camera = actions[i].pose()
            T_grasp_world = state.T_camera_world * T_grasp_camera
            if T_grasp_world.translation[2] < self._height_thresh:
                q_values[i] = 0
        return q_values
    
class NoMagicQualityFunction(GraspQualityFunction):
    def __init__(self, config):
        """Create a quality that uses nomagic_net as a quality function. """
        from nomagic_submission import ConvNetModel
        from tensorpack.predict.config import PredictConfig
        from tensorpack import SaverRestore
        from tensorpack.predict import OfflinePredictor
        import json
        import os
        
        GraspQualityFunction.__init(self)
        
        # store parameters
        self._model_path = config['gqcnn_model']
        self._batch_size = config['batch_size']
        self._crop_height = config['crop_height']
        self._crop_width = config['crop_width']
        self._im_height = config['im_height']
        self._im_width = config['im_width']
        self._num_channels = config['num_channels']
        self._pose_dim = config['pose_dim']
        self._gripper_mode = config['gripper_mode']
        self._data_mean = config['data_mean']
        self._data_std = config['data_std']

        # init config
        model = ConvNetModel()
        self._config = PredictConfig(
            model=model,
            session_init=SaverRestore(self._model_path),
            output_names=['prob'])
        self._predictor = OfflinePredictor(self._config)
        
    @property
    def gqcnn(self):
        """ Returns the GQ-CNN. """
        return self._predictor

    @property
    def config(self):
        """ Returns the GQCNN suction quality function parameters. """
        return self._config

    def grasps_to_tensors(self, grasps, state):
        """Converts a list of grasps to an image and pose tensor
        for fast grasp quality evaluation.

        Attributes
        ----------
        grasps : :obj:`list` of :obj:`object`
            list of image grasps to convert
        state : :obj:`RgbdImageState`
            RGB-D image to plan grasps on

        Returns
        -------
        image_arr : :obj:`numpy.ndarray`
            4D numpy tensor of image to be predicted
        pose_arr : :obj:`numpy.ndarray`
            2D numpy tensor of depth values
        """
        # parse params
        gqcnn_im_height = self._im_height
        gqcnn_im_width = self._im_width
        gqcnn_num_channels = self._num_channels
        gqcnn_pose_dim = self._pose_dim
        gripper_mode = self._gripper_mode
        num_grasps = len(grasps)
        depth_im = state.rgbd_im.depth

        # allocate tensors
        tensor_start = time()
        image_tensor = np.zeros([num_grasps, gqcnn_im_height,
                                 gqcnn_im_width, gqcnn_num_channels])
        pose_tensor = np.zeros([num_grasps, gqcnn_pose_dim])
        scale = float(gqcnn_im_height) / self._crop_height
        depth_im_scaled = depth_im.rescale(scale)
        for i, grasp in enumerate(grasps):
            translation = scale * np.array([depth_im.center[0] - grasp.center[1],
                                            depth_im.center[1] - grasp.center[0]])
            im_tf = depth_im_scaled
            im_tf = depth_im_scaled.transform(translation, grasp.angle)
            im_tf = im_tf.crop(gqcnn_im_height, gqcnn_im_width)

            im_encoded = cv2.imencode('.png', np.uint8(im_tf.raw_data*255))[1].tostring()
            im_decoded = cv2.imdecode(np.frombuffer(im_encoded, np.uint8), 0) / 255.0
            image_tensor[i,:,:,0] = ((im_decoded - self._data_mean) / self._data_std)
            
            if gripper_mode == GripperMode.PARALLEL_JAW:
                pose_tensor[i] = grasp.depth
            elif gripper_mode == GripperMode.SUCTION:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.approach_angle])
            elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
                pose_tensor[i] = grasp.depth
            elif gripper_mode == GripperMode.LEGACY_SUCTION:
                pose_tensor[i,...] = np.array([grasp.depth, grasp.approach_angle])
            else:
                raise ValueError('Gripper mode %s not supported' %(gripper_mode))
        self._logger.debug('Tensor conversion took %.3f sec' %(time()-tensor_start))
        return image_tensor, pose_tensor

    def quality(self, state, actions, params): 
        """ Evaluate the quality of a set of actions according to a GQ-CNN.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            state of the world described by an RGB-D image
        actions: :obj:`object`
            set of grasping actions to evaluate
        params: dict
            optional parameters for quality evaluation

        Returns
        -------
        :obj:`list` of float
            real-valued grasp quality predictions for each action, between 0 and 1
        """
        # form tensors
        image_tensor, pose_tensor = self.grasps_to_tensors(actions, state)
        if params is not None and params['vis']['tf_images']:
            # read vis params
            k = params['vis']['k']
            d = utils.sqrt_ceil(k)

            # display grasp transformed images
            from ambicore import Visualizer2D as vis2d
            vis2d.figure(size=(FIGSIZE,FIGSIZE))
            for i, image_tf in enumerate(image_tensor[:k,...]):
                depth = pose_tensor[i][0]
                vis2d.subplot(d,d,i+1)
                vis2d.imshow(DepthImage(image_tf))
                vis2d.title('Image %d: d=%.3f' %(i, depth))
            vis2d.show()

        # predict grasps
        num_actions = len(actions)
        null_arr = -1 * np.ones(self._batch_size)
        predict_start = time()
        output_arr = np.zeros([num_actions, 2])
        cur_i = 0
        end_i = cur_i + min(self._batch_size, num_actions - cur_i)
        while cur_i < num_actions:
            output_arr[cur_i:end_i,:] = self.gqcnn(image_tensor[cur_i:end_i,:,:,0], pose_tensor[cur_i:end_i,0], null_arr)[0]
            cur_i = end_i
            end_i = cur_i + min(self._batch_size, num_actions - cur_i)
        q_values = output_arr[:,-1]
        self._logger.debug('Prediction took %.3f sec' %(time()-predict_start))
        return q_values.tolist()

class FCGQCnnQualityFunction(GraspQualityFunction):
    def __init__(self, config):
        """ Grasp quality function using the fully-convolutional gqcnn """
        GraspQualityFunction.__init__(self)

        # store parameters
        self._config = config
        self._model_dir = config['gqcnn_model']
        self._backend = config['gqcnn_backend']
        self._fully_conv_config = config['fully_conv_gqcnn_config']

        # init fcgqcnn
        self._fcgqcnn = get_fc_gqcnn_model(backend=self._backend).load(self._model_dir, self._fully_conv_config)

    def __del__(self):
        try:
            self._fcgqcnn.close_session()
        except:
            pass
        del self

    @property
    def gqcnn(self):
        """ Returns the FC-GQCNN. """
        return self._fcgqcnn

    @property
    def config(self):
        """ Returns the FC-GQCNN quality function parameters. """
        return self._config

    def quality(self, images, depths, params=None): 
        # open tensorflow session for fcgqcnn
        if self._fcgqcnn.sess is None:
            self._fcgqcnn.open_session()

        return self._fcgqcnn.predict(images, depths)

class GraspQualityFunctionFactory(object):
    """Factory for grasp quality functions. """
    @staticmethod
    def quality_function(metric_type, config):
        if metric_type == 'zero':
            return ZeroGraspQualityFunction()
        elif metric_type == 'parallel_jaw_com_force_closure':
            return ComForceClosureParallelJawQualityFunction(config)
        elif metric_type == 'suction_best_fit_planarity':
            return BestFitPlanaritySuctionQualityFunction(config)
        elif metric_type == 'suction_approach_planarity':
            return ApproachPlanaritySuctionQualityFunction(config)
        elif metric_type == 'suction_com_approach_planarity':
            return ComApproachPlanaritySuctionQualityFunction(config)
        elif metric_type == 'suction_disc_approach_planarity':
            return DiscApproachPlanaritySuctionQualityFunction(config)
        elif metric_type == 'suction_com_disc_approach_planarity':
            return ComDiscApproachPlanaritySuctionQualityFunction(config)
        elif metric_type == 'suction_gaussian_curvature':
            return GaussianCurvatureSuctionQualityFunction(config)
        elif metric_type == 'suction_disc_curvature':
            return DiscCurvatureSuctionQualityFunction(config)
        elif metric_type == 'suction_com_disc_curvature':
            return ComDiscCurvatureSuctionQualityFunction(config)
        elif metric_type == 'gqcnn':
            return GQCnnQualityFunction(config)
        elif metric_type == 'gqcnn_planarity':
            return GQCnnPlanarityQualityFunction(config)
        elif metric_type == 'gqcnn_height':
            return GQCnnHeightQualityFunction(config)
        elif metric_type == 'nomagic':
            return NoMagicQualityFunction(config)
        elif metric_type == 'fcgqcnn':
            return FCGQCnnQualityFunction(config)
        else:
            raise ValueError('Grasp function type %s not supported!' %(metric_type))
