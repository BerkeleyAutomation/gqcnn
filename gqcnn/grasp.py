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
Classes to encapsulate parallel-jaw grasps in image space
Author: Jeff
"""
import numpy as np

from autolab_core import Point, RigidTransform
from perception import CameraIntrinsics

class Grasp2D(object):
    """
    Parallel-jaw grasp in image space.

    Attributes
    ----------
    center : :obj:`autolab_core.Point`
        point in image space
    angle : float
        grasp axis angle with the camera x-axis
    depth : float
        depth of the grasp center in 3D space
    width : float
        distance between the jaws in meters
    camera_intr : :obj:`perception.CameraIntrinsics`
        frame of reference for camera that the grasp corresponds to
    """
    def __init__(self, center, angle, depth, width=0.0, camera_intr=None):
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width = width
        # if camera_intr is none use default primesense camera intrinsics
        if not camera_intr:
            self.camera_intr = CameraIntrinsics('primesense_overhead', fx=525, fy=525, cx=319.5, cy=239.5, width=640, height=480)
        else: 
            self.camera_intr = camera_intr

    @property
    def axis(self):
        """ Returns the grasp axis. """
        return np.array([np.cos(self.angle), np.sin(self.angle)])        

    @property
    def approach_angle(self):
        """ The angle between the grasp approach axis and camera optical axis. """
        return 0.0
    
    @property
    def frame(self):
        """ The name of the frame of reference for the grasp. """
        if self.camera_intr is None:
            raise ValueError('Must specify camera intrinsics')
        return self.camera_intr.frame

    @property
    def width_px(self):
        """ Returns the width in pixels. """
        if self.camera_intr is None:
            raise ValueError('Must specify camera intrinsics to compute gripper width in 3D space')
        # form the jaw locations in 3D space at the given depth
        p1 = Point(np.array([0, 0, self.depth]), frame=self.frame)
        p2 = Point(np.array([self.width, 0, self.depth]), frame=self.frame)
        
        # project into pixel space
        u1 = self.camera_intr.project(p1)
        u2 = self.camera_intr.project(p2)
        return np.linalg.norm(u1.data - u2.data)

    @property
    def endpoints(self):
        """ Returns the grasp endpoints """
        p1 = self.center.data - (float(self.width_px) / 2) * self.axis
        p2 = self.center.data + (float(self.width_px) / 2) * self.axis
        return p1, p2

    @property
    def feature_vec(self):
        """ Returns the feature vector for the grasp.
        v = [p1, p2, depth]
        where p1 and p2 are the jaw locations in image space
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self.depth]

    @staticmethod
    def from_feature_vec(v, width=0.0, camera_intr=None):
        """ Creates a Grasp2D obj from a feature vector and additional parameters.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            feature vector, see Grasp2D.feature_vec
        width : float
            grasp opening width, in meters
        camera_intr : :obj:`perception.CameraIntrinsics`
            frame of reference for camera that the grasp corresponds to
        """
        # read feature vec
        p1 = v[:2]
        p2 = v[2:4]
        depth = v[4]

        # compute center and angle
        center_px = (p1 + p2) / 2
        center = Point(center_px, camera_intr.frame)
        axis = p2 - p1
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        if axis[1] > 0:
            angle = np.arccos(axis[0])
        else:
            angle = -np.arccos(axis[0])
        return Grasp2D(center, angle, depth, width=width, camera_intr=camera_intr)

    def pose(self, grasp_approach_dir=None):
        """ Computes the 3D pose of the grasp relative to the camera.
        If an approach direction is not specified then the camera
        optical axis is used.
     
        Parameters
        ----------
        grasp_approach_dir : :obj:`numpy.ndarray`
            approach direction for the grasp in camera basis (e.g. opposite to table normal)

        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            the transformation from the grasp to the camera frame of reference
        """
        # check intrinsics
        if self.camera_intr is None:
            raise ValueError('Must specify camera intrinsics to compute 3D grasp pose')

        # compute 3D grasp center in camera basis
        grasp_center_im = self.center.data
        center_px_im = Point(grasp_center_im, frame=self.camera_intr.frame)
        grasp_center_camera = self.camera_intr.deproject_pixel(self.depth, center_px_im)
        grasp_center_camera = grasp_center_camera.data

        # compute 3D grasp axis in camera basis
        grasp_axis_im = self.axis
        grasp_axis_im = grasp_axis_im / np.linalg.norm(grasp_axis_im)
        grasp_axis_camera = np.array([grasp_axis_im[0], grasp_axis_im[1], 0])
        grasp_axis_camera = grasp_axis_camera / np.linalg.norm(grasp_axis_camera)
        
        # convert to 3D pose
        grasp_rot_camera, _, _ = np.linalg.svd(grasp_axis_camera.reshape(3,1))
        grasp_x_camera = grasp_approach_dir
        if grasp_approach_dir is None:
            grasp_x_camera = np.array([0,0,1]) # aligned with camera Z axis
        grasp_y_camera = grasp_axis_camera
        grasp_z_camera = np.cross(grasp_x_camera, grasp_y_camera)
        grasp_z_camera = grasp_z_camera / np.linalg.norm(grasp_z_camera)
        grasp_y_camera = np.cross(grasp_z_camera, grasp_x_camera)
        grasp_rot_camera = np.array([grasp_x_camera, grasp_y_camera, grasp_z_camera]).T
        if np.linalg.det(grasp_rot_camera) < 0: # fix possible reflections due to SVD
            grasp_rot_camera[:,0] = -grasp_rot_camera[:,0]
        T_grasp_camera = RigidTransform(rotation=grasp_rot_camera,
                                        translation=grasp_center_camera,
                                        from_frame='grasp',
                                        to_frame=self.camera_intr.frame)
        return T_grasp_camera

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """ Computes the distance between grasps in image space.
        Euclidean distance with alpha weighting of angles

        Parameters
        ----------
        g1 : :obj:`Grasp2D`
            first grasp
        g2 : :obj:`Grasp2D`
            second grasp
        alpha : float
            weight of angle distance (rad to meters)

        Returns
        -------
        float
            distance between grasps
        """
        # point to point distances
        point_dist = np.linalg.norm(g1.center.data - g2.center.data)

        # axis distances
        dot = max(min(np.abs(g1.axis.dot(g2.axis)), 1.0), -1.0)
        axis_dist = np.arccos(dot)
        return point_dist + alpha * axis_dist

class SuctionPoint2D(object):
    """
    Suction grasp in image space.

    Attributes
    ----------
    center : :obj:`autolab_core.Point`
        point in image space
    axis : :obj:`numpy.ndarray`
        normalized 3-vector representing the direction of the suction tip
    depth : float
        depth of the suction point in 3D space
    camera_intr : :obj:`perception.CameraIntrinsics`
        frame of reference for camera that the suction point corresponds to
    """
    def __init__(self, center, axis, depth, camera_intr=None):
        self.center = center
        self.axis = axis
        if np.abs(np.linalg.norm(self.axis) - 1.0) > 1e-3:
            raise ValueError('Illegal axis. Must be norm 1.')

        self.depth = depth
        # if camera_intr is none use default primesense camera intrinsics
        if not camera_intr:
            self.camera_intr = CameraIntrinsics('primesense_overhead', fx=525, fy=525, cx=319.5, cy=239.5, width=640, height=480)
        else: 
            self.camera_intr = camera_intr

    @property
    def frame(self):
        """ The name of the frame of reference for the grasp. """
        if self.camera_intr is None:
            raise ValueError('Must specify camera intrinsics')
        return self.camera_intr.frame

    @property
    def angle(self):
        """ The angle that the grasp pivot axis makes in image space. """
        rotation_axis = np.cross(self.axis, np.array([0,0,1]))
        rotation_axis_image = np.array([rotation_axis[0], rotation_axis[1]])
        angle = 0
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis_image = rotation_axis_image / np.linalg.norm(rotation_axis_image)
            angle = np.arccos(rotation_axis_image[0])
        if rotation_axis[1] < 0:
            angle = -angle
        return angle

    @property
    def approach_angle(self):
        """ The angle between the grasp approach axis and camera optical axis. """
        dot = max(min(self.axis.dot(np.array([0,0,1])), 1.0), -1.0)
        return np.arccos(dot)

    @property
    def feature_vec(self):
        """ Returns the feature vector for the suction point.
        v = [center, axis, depth]
        """
        #return np.r_[self.center.data, self.axis, self.depth]
        #return np.r_[self.center.data, self.axis]
        return self.center.data
        
    @staticmethod
    def from_feature_vec(v, camera_intr=None, depth=None, axis=None):
        """ Creates a SuctionPoint2D obj from a feature vector and additional parameters.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            feature vector, see Grasp2D.feature_vec
        camera_intr : :obj:`perception.CameraIntrinsics`
            frame of reference for camera that the grasp corresponds to
        depth : float
            hard-set the depth for the suction grasp
        axis : :obj:`numpy.ndarray`
            normalized 3-vector specifying the approach direction
        """
        # read feature vec
        center_px = v[:2]

        grasp_axis = np.array([0,0,-1])
        if v.shape > 2 and axis is None:
            grasp_axis = v[2:5]
            grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
        elif axis is not None:
            grasp_axis = axis
            
        grasp_depth = 0.5    
        if v.shape[0] > 5 and depth is None:
            grasp_depth = v[5]
        elif depth is not None:
            grasp_depth = depth
            
        # compute center and angle
        center = Point(center_px, camera_intr.frame)
        return SuctionPoint2D(center,
                              grasp_axis,
                              grasp_depth,
                              camera_intr=camera_intr)

    def pose(self):
        """ Computes the 3D pose of the grasp relative to the camera.

        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            the transformation from the grasp to the camera frame of reference
        """
        # check intrinsics
        if self.camera_intr is None:
            raise ValueError('Must specify camera intrinsics to compute 3D grasp pose')

        # compute 3D grasp center in camera basis
        suction_center_im = self.center.data
        center_px_im = Point(suction_center_im, frame=self.camera_intr.frame)
        suction_center_camera = self.camera_intr.deproject_pixel(self.depth, center_px_im)
        suction_center_camera = suction_center_camera.data

        # compute 3D grasp axis in camera basis
        suction_axis_camera = self.axis
        
        # convert to 3D pose
        suction_x_camera = suction_axis_camera
        suction_z_camera = np.array([-suction_x_camera[1], suction_x_camera[0], 0])
        suction_z_camera = suction_z_camera / np.linalg.norm(suction_z_camera)
        suction_y_camera = np.cross(suction_z_camera, suction_x_camera)
        suction_rot_camera = np.c_[suction_x_camera, suction_y_camera, suction_z_camera]

        T_suction_camera = RigidTransform(rotation=suction_rot_camera,
                                          translation=suction_center_camera,
                                          from_frame='grasp',
                                          to_frame=self.camera_intr.frame)
        return T_suction_camera

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """ Computes the distance between grasps in image space.
        Euclidean distance with alpha weighting of angles

        Parameters
        ----------
        g1 : :obj:`SuctionPoint2D`
            first suction point
        g2 : :obj:`SuctionPoint2D`
            second suction point
        alpha : float
            weight of angle distance (rad to meters)

        Returns
        -------
        float
            distance between grasps
        """
        # point to point distances
        point_dist = np.linalg.norm(g1.center.data - g2.center.data)

        # axis distances
        dot = max(min(np.abs(g1.axis.dot(g2.axis)), 1.0), -1.0)
        axis_dist = np.arccos(dot)

        return point_dist + alpha * axis_dist
