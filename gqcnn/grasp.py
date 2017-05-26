"""
Classes to encapsulate parallel-jaw grasps in image space
Author: Jeff
"""
import IPython
import numpy as np

from core import Point

class Grasp2D(object):
    """
    Parallel-jaw grasp in image space.

    Attributes
    ----------
    center : :obj:`core.Point`
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
        self.camera_intr = camera_intr

    @property
    def axis(self):
        """ Returns the grasp axis. """
        return np.array([np.cos(self.angle), np.sin(self.angle)])        
        
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
        angle = np.arccos(axis[0])
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
        :obj:`core.RigidTransform`
            the transformation from the grasp to the camera frame of reference
        """
        # check intrinsics
        if self.camera_intr is None:
            raise ValueError('Must specify camera intrinsics to compute 3D grasp pose')

        # compute 3D grasp center in camera basis
        grasp_center_im = self.center.data
        center_px_im = Point(grasp_center_im, frame=ir_intrinsics.frame)
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
        grasp_x_camera = np.cross(grasp_z_camera, grasp_y_camera)
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
        axis_dist = np.arccos(np.abs(g1.axis.dot(g2.axis)))

        return point_dist + alpha * axis_dist
