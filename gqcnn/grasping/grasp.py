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

Classes to encapsulate parallel-jaw grasps in image space.

Author
------
Jeff Mahler
"""
import numpy as np

from autolab_core import Point, RigidTransform
from perception import CameraIntrinsics


class Grasp2D(object):
    """Parallel-jaw grasp in image space.

    Attributes
    ----------
    center : :obj:`autolab_core.Point`
        Point in image space.
    angle : float
        Grasp axis angle with the camera x-axis.
    depth : float
        Depth of the grasp center in 3D space.
    width : float
        Distance between the jaws in meters.
    camera_intr : :obj:`perception.CameraIntrinsics`
        Frame of reference for camera that the grasp corresponds to.
    contact_points : list of :obj:`numpy.ndarray`
        Pair of contact points in image space.
    contact_normals : list of :obj:`numpy.ndarray`
        Pair of contact normals in image space.
    """

    def __init__(self,
                 center,
                 angle=0.0,
                 depth=1.0,
                 width=0.0,
                 camera_intr=None,
                 contact_points=None,
                 contact_normals=None):
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width = width
        # If `camera_intr` is none use default primesense camera intrinsics.
        if not camera_intr:
            self.camera_intr = CameraIntrinsics("primesense_overhead",
                                                fx=525,
                                                fy=525,
                                                cx=319.5,
                                                cy=239.5,
                                                width=640,
                                                height=480)
        else:
            self.camera_intr = camera_intr
        self.contact_points = contact_points
        self.contact_normals = contact_normals

        frame = "image"
        if camera_intr is not None:
            frame = camera_intr.frame
        if isinstance(center, np.ndarray):
            self.center = Point(center, frame=frame)

    @property
    def axis(self):
        """Returns the grasp axis."""
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def approach_axis(self):
        return np.array([0, 0, 1])

    @property
    def approach_angle(self):
        """The angle between the grasp approach axis and camera optical axis.
        """
        return 0.0

    @property
    def frame(self):
        """The name of the frame of reference for the grasp."""
        if self.camera_intr is None:

            raise ValueError("Must specify camera intrinsics")
        return self.camera_intr.frame

    @property
    def width_px(self):
        """Returns the width in pixels."""
        if self.camera_intr is None:
            missing_camera_intr_msg = ("Must specify camera intrinsics to"
                                       " compute gripper width in 3D space.")
            raise ValueError(missing_camera_intr_msg)
        # Form the jaw locations in 3D space at the given depth.
        p1 = Point(np.array([0, 0, self.depth]), frame=self.frame)
        p2 = Point(np.array([self.width, 0, self.depth]), frame=self.frame)

        # Project into pixel space.
        u1 = self.camera_intr.project(p1)
        u2 = self.camera_intr.project(p2)
        return np.linalg.norm(u1.data - u2.data)

    @property
    def endpoints(self):
        """Returns the grasp endpoints."""
        p1 = self.center.data - (self.width_px / 2) * self.axis
        p2 = self.center.data + (self.width_px / 2) * self.axis
        return p1, p2

    @property
    def feature_vec(self):
        """Returns the feature vector for the grasp.

        `v = [p1, p2, depth]` where `p1` and `p2` are the jaw locations in
        image space.
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self.depth]

    @staticmethod
    def from_feature_vec(v, width=0.0, camera_intr=None):
        """Creates a `Grasp2D` instance from a feature vector and additional
        parameters.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            Feature vector, see `Grasp2D.feature_vec`.
        width : float
            Grasp opening width, in meters.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Frame of reference for camera that the grasp corresponds to.
        """
        # Read feature vec.
        p1 = v[:2]
        p2 = v[2:4]
        depth = v[4]

        # Compute center and angle.
        center_px = (p1 + p2) // 2
        center = Point(center_px, camera_intr.frame)
        axis = p2 - p1
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        if axis[1] > 0:
            angle = np.arccos(axis[0])
        else:
            angle = -np.arccos(axis[0])
        return Grasp2D(center,
                       angle,
                       depth,
                       width=width,
                       camera_intr=camera_intr)

    def pose(self, grasp_approach_dir=None):
        """Computes the 3D pose of the grasp relative to the camera.

        If an approach direction is not specified then the camera
        optical axis is used.

        Parameters
        ----------
        grasp_approach_dir : :obj:`numpy.ndarray`
            Approach direction for the grasp in camera basis (e.g. opposite to
            table normal).

        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            The transformation from the grasp to the camera frame of reference.
        """
        # Check intrinsics.
        if self.camera_intr is None:
            raise ValueError(
                "Must specify camera intrinsics to compute 3D grasp pose")

        # Compute 3D grasp center in camera basis.
        grasp_center_im = self.center.data
        center_px_im = Point(grasp_center_im, frame=self.camera_intr.frame)
        grasp_center_camera = self.camera_intr.deproject_pixel(
            self.depth, center_px_im)
        grasp_center_camera = grasp_center_camera.data

        # Compute 3D grasp axis in camera basis.
        grasp_axis_im = self.axis
        grasp_axis_im = grasp_axis_im / np.linalg.norm(grasp_axis_im)
        grasp_axis_camera = np.array([grasp_axis_im[0], grasp_axis_im[1], 0])
        grasp_axis_camera = grasp_axis_camera / np.linalg.norm(
            grasp_axis_camera)

        # Convert to 3D pose.
        grasp_rot_camera, _, _ = np.linalg.svd(grasp_axis_camera.reshape(3, 1))
        grasp_x_camera = grasp_approach_dir
        if grasp_approach_dir is None:
            grasp_x_camera = np.array([0, 0, 1])  # Align with camera Z axis.
        grasp_y_camera = grasp_axis_camera
        grasp_z_camera = np.cross(grasp_x_camera, grasp_y_camera)
        grasp_z_camera = grasp_z_camera / np.linalg.norm(grasp_z_camera)
        grasp_y_camera = np.cross(grasp_z_camera, grasp_x_camera)
        grasp_rot_camera = np.array(
            [grasp_x_camera, grasp_y_camera, grasp_z_camera]).T
        if np.linalg.det(grasp_rot_camera) < 0:  # Fix reflections due to SVD.
            grasp_rot_camera[:, 0] = -grasp_rot_camera[:, 0]
        T_grasp_camera = RigidTransform(rotation=grasp_rot_camera,
                                        translation=grasp_center_camera,
                                        from_frame="grasp",
                                        to_frame=self.camera_intr.frame)
        return T_grasp_camera

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """Computes the distance between grasps in image space.

        Uses Euclidean distance with alpha weighting of angles

        Parameters
        ----------
        g1 : :obj:`Grasp2D`
            First grasp.
        g2 : :obj:`Grasp2D`
            Second grasp.
        alpha : float
            Weight of angle distance (rad to meters).

        Returns
        -------
        float
            Distance between grasps.
        """
        # Point to point distances.
        point_dist = np.linalg.norm(g1.center.data - g2.center.data)

        # Axis distances.
        dot = max(min(np.abs(g1.axis.dot(g2.axis)), 1.0), -1.0)
        axis_dist = np.arccos(dot)
        return point_dist + alpha * axis_dist


class SuctionPoint2D(object):
    """Suction grasp in image space.

    Attributes
    ----------
    center : :obj:`autolab_core.Point`
        Point in image space.
    axis : :obj:`numpy.ndarray`
        Dormalized 3-vector representing the direction of the suction tip.
    depth : float
        Depth of the suction point in 3D space.
    camera_intr : :obj:`perception.CameraIntrinsics`
        Frame of reference for camera that the suction point corresponds to.
    """

    def __init__(self, center, axis=None, depth=1.0, camera_intr=None):
        if axis is None:
            axis = np.array([0, 0, 1])

        self.center = center
        self.axis = axis

        frame = "image"
        if camera_intr is not None:
            frame = camera_intr.frame
        if isinstance(center, np.ndarray):
            self.center = Point(center, frame=frame)
        if isinstance(axis, list):
            self.axis = np.array(axis)
        if np.abs(np.linalg.norm(self.axis) - 1.0) > 1e-3:
            raise ValueError("Illegal axis. Must be norm 1.")

        self.depth = depth
        # If `camera_intr` is `None` use default primesense camera intrinsics.
        if not camera_intr:
            self.camera_intr = CameraIntrinsics("primesense_overhead",
                                                fx=525,
                                                fy=525,
                                                cx=319.5,
                                                cy=239.5,
                                                width=640,
                                                height=480)
        else:
            self.camera_intr = camera_intr

    @property
    def frame(self):
        """The name of the frame of reference for the grasp."""
        if self.camera_intr is None:
            raise ValueError("Must specify camera intrinsics")
        return self.camera_intr.frame

    @property
    def angle(self):
        """The angle that the grasp pivot axis makes in image space."""
        rotation_axis = np.cross(self.axis, np.array([0, 0, 1]))
        rotation_axis_image = np.array([rotation_axis[0], rotation_axis[1]])
        angle = 0
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis_image = rotation_axis_image / np.linalg.norm(
                rotation_axis_image)
            angle = np.arccos(rotation_axis_image[0])
        if rotation_axis[1] < 0:
            angle = -angle
        return angle

    @property
    def approach_angle(self):
        """The angle between the grasp approach axis and camera optical axis.
        """
        dot = max(min(self.axis.dot(np.array([0, 0, 1])), 1.0), -1.0)
        return np.arccos(dot)

    @property
    def approach_axis(self):
        return self.axis

    @property
    def feature_vec(self):
        """Returns the feature vector for the suction point.

        Note
        ----
        `v = [center, axis, depth]`
        """
        return self.center.data

    @staticmethod
    def from_feature_vec(v, camera_intr=None, depth=None, axis=None):
        """Creates a `SuctionPoint2D` instance from a feature vector and
        additional parameters.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            Feature vector, see `Grasp2D.feature_vec`.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Frame of reference for camera that the grasp corresponds to.
        depth : float
            Hard-set the depth for the suction grasp.
        axis : :obj:`numpy.ndarray`
            Normalized 3-vector specifying the approach direction.
        """
        # Read feature vec.
        center_px = v[:2]

        grasp_axis = np.array([0, 0, -1])
        if v.shape[0] > 2 and axis is None:
            grasp_axis = v[2:5]
            grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
        elif axis is not None:
            grasp_axis = axis

        grasp_depth = 0.5
        if v.shape[0] > 5 and depth is None:
            grasp_depth = v[5]
        elif depth is not None:
            grasp_depth = depth

        # Compute center and angle.
        center = Point(center_px, camera_intr.frame)
        return SuctionPoint2D(center,
                              grasp_axis,
                              grasp_depth,
                              camera_intr=camera_intr)

    def pose(self):
        """Computes the 3D pose of the grasp relative to the camera.

        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            The transformation from the grasp to the camera frame of reference.
        """
        # Check intrinsics.
        if self.camera_intr is None:
            raise ValueError(
                "Must specify camera intrinsics to compute 3D grasp pose")

        # Compute 3D grasp center in camera basis.
        suction_center_im = self.center.data
        center_px_im = Point(suction_center_im, frame=self.camera_intr.frame)
        suction_center_camera = self.camera_intr.deproject_pixel(
            self.depth, center_px_im)
        suction_center_camera = suction_center_camera.data

        # Compute 3D grasp axis in camera basis.
        suction_axis_camera = self.axis

        # Convert to 3D pose.
        suction_x_camera = suction_axis_camera
        suction_z_camera = np.array(
            [-suction_x_camera[1], suction_x_camera[0], 0])
        if np.linalg.norm(suction_z_camera) < 1e-12:
            suction_z_camera = np.array([1.0, 0.0, 0.0])
        suction_z_camera = suction_z_camera / np.linalg.norm(suction_z_camera)
        suction_y_camera = np.cross(suction_z_camera, suction_x_camera)
        suction_rot_camera = np.c_[suction_x_camera, suction_y_camera,
                                   suction_z_camera]

        T_suction_camera = RigidTransform(rotation=suction_rot_camera,
                                          translation=suction_center_camera,
                                          from_frame="grasp",
                                          to_frame=self.camera_intr.frame)
        return T_suction_camera

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """Computes the distance between grasps in image space.

        Uses Euclidean distance with alpha weighting of angles.

        Parameters
        ----------
        g1 : :obj:`SuctionPoint2D`
            First suction point.
        g2 : :obj:`SuctionPoint2D`
            Second suction point.
        alpha : float
            Weight of angle distance (rad to meters).

        Returns
        -------
        float
            Distance between grasps.
        """
        # Point to point distances.
        point_dist = np.linalg.norm(g1.center.data - g2.center.data)

        # Axis distances.
        dot = max(min(np.abs(g1.axis.dot(g2.axis)), 1.0), -1.0)
        axis_dist = np.arccos(dot)

        return point_dist + alpha * axis_dist


class MultiSuctionPoint2D(object):
    """Multi-Cup Suction grasp in image space.

    Equivalent to projecting a 6D pose to image space.

    Attributes
    ----------
    pose : :obj:`autolab_core.RigidTransform`
        Pose in 3D camera space.
    camera_intr : :obj:`perception.CameraIntrinsics`
        Frame of reference for camera that the suction point corresponds to.
    """

    def __init__(self, pose, camera_intr=None):
        self._pose = pose

        # TODO(vsatish): Confirm that this is really not needed.
        #        frame = "image"
        #        if camera_intr is not None:
        #            frame = camera_intr.frame

        # If `camera_intr` is `None` use default primesense camera intrinsics.
        if not camera_intr:
            self.camera_intr = CameraIntrinsics("primesense_overhead",
                                                fx=525,
                                                fy=525,
                                                cx=319.5,
                                                cy=239.5,
                                                width=640,
                                                height=480)
        else:
            self.camera_intr = camera_intr

    def pose(self):
        return self._pose

    @property
    def frame(self):
        """The name of the frame of reference for the grasp."""
        if self.camera_intr is None:
            raise ValueError("Must specify camera intrinsics")
        return self.camera_intr.frame

    @property
    def center(self):
        center_camera = Point(self._pose.translation,
                              frame=self.camera_intr.frame)
        center_px = self.camera_intr.project(center_camera)
        return center_px

    @property
    def axis(self):
        return self._pose.x_axis

    @property
    def approach_axis(self):
        return self.axis

    @property
    def approach_angle(self):
        """The angle between the grasp approach axis and camera optical axis.
        """
        dot = max(min(self.axis.dot(np.array([0, 0, 1])), 1.0), -1.0)
        return np.arccos(dot)

    @property
    def angle(self):
        g_axis = self._pose.y_axis
        g_axis_im = np.array([g_axis[0], g_axis[1], 0])
        if np.linalg.norm(g_axis_im) == 0:
            return 0
        theta = np.arctan2(g_axis[1], g_axis[0])
        return theta

    @property
    def depth(self):
        return self._pose.translation[2]

    @property
    def orientation(self):
        x_axis = self.axis
        y_axis = np.array([x_axis[1], -x_axis[0], 0])
        if np.linalg.norm(y_axis) == 0:
            y_axis = np.array([1, 0, 0])
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        R = np.array([x_axis, y_axis, z_axis]).T
        delta_R = R.T.dot(self._pose.rotation)
        orientation = np.arccos(delta_R[1, 1])
        if delta_R[1, 2] > 0:
            orientation = 2 * np.pi - orientation
        return orientation

    @property
    def feature_vec(self):
        """Returns the feature vector for the suction point.

        Note
        ----
        `v = [center, axis, depth]`
        """
        return np.r_[self.center.data,
                     np.cos(self.orientation),
                     np.sin(self.orientation)]

    @staticmethod
    def from_feature_vec(v,
                         camera_intr=None,
                         angle=None,
                         depth=None,
                         axis=None):
        """Creates a `SuctionPoint2D` instance from a feature vector and
        additional parameters.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            Feature vector, see `Grasp2D.feature_vec`.
        camera_intr : :obj:`perception.CameraIntrinsics`
            Frame of reference for camera that the grasp corresponds to.
        depth : float
            Hard-set the depth for the suction grasp.
        axis : :obj:`numpy.ndarray`
            Normalized 3-vector specifying the approach direction.
        """
        # Read feature vec.
        center_px = v[:2]

        grasp_angle = 0
        if v.shape[0] > 2 and angle is None:
            # grasp_angle = v[2]
            grasp_vec = v[2:]
            grasp_vec = grasp_vec / np.linalg.norm(grasp_vec)
            grasp_angle = np.arctan2(grasp_vec[1], grasp_vec[0])
        elif angle is not None:
            grasp_angle = angle

        grasp_axis = np.array([1, 0, 0])
        if axis is not None:
            grasp_axis = axis

        grasp_depth = 0.5
        if depth is not None:
            grasp_depth = depth

        x_axis = grasp_axis
        y_axis = np.array([grasp_axis[1], -grasp_axis[0], 0])
        if np.linalg.norm(y_axis) == 0:
            y_axis = np.array([1, 0, 0])
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        R = np.array([x_axis, y_axis, z_axis]).T
        R = R.dot(RigidTransform.x_axis_rotation(grasp_angle))
        t = camera_intr.deproject_pixel(
            grasp_depth, Point(center_px, frame=camera_intr.frame)).data
        T = RigidTransform(rotation=R,
                           translation=t,
                           from_frame="grasp",
                           to_frame=camera_intr.frame)

        # Compute center and angle.
        return MultiSuctionPoint2D(T, camera_intr=camera_intr)

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """Computes the distance between grasps in image space.

        Uses Euclidean distance with alpha weighting of angles.

        Parameters
        ----------
        g1 : :obj:`SuctionPoint2D`
            First suction point.
        g2 : :obj:`SuctionPoint2D`
            Second suction point.
        alpha : float
            Weight of angle distance (rad to meters).

        Returns
        -------
        float
            Distance between grasps.
        """
        # Point to point distances.
        point_dist = np.linalg.norm(g1.center.data - g2.center.data)

        # Axis distances.
        dot = max(min(np.abs(g1.axis.dot(g2.axis)), 1.0), -1.0)
        axis_dist = np.arccos(dot)

        return point_dist + alpha * axis_dist
