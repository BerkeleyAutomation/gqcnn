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

Action classes for representing 3D grasp actions.

Author
------
Jeff Mahler
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from future.utils import with_metaclass
import numpy as np

from autolab_core import Point

from .grasp import Grasp2D, SuctionPoint2D, MultiSuctionPoint2D


class Action(with_metaclass(ABCMeta, object)):
    """Abstract action class.

    Attributes
    ----------
    q_value : float
        Grasp quality.
    id : int
        Integer identifier for the action.
    metadata : dict
        Key-value dict of extra data about the action.
    """

    def __init__(self, q_value=0.0, id=-1, metadata={}):
        self._q_value = q_value
        self._id = id
        self._metadata = metadata

    @property
    def q_value(self):
        return self._q_value

    @property
    def id(self):
        return self._id

    @property
    def metadata(self):
        return self._metadata


class NoAction(Action):
    """Proxy for taking no action when none can be found."""
    pass


class GraspAction3D(with_metaclass(ABCMeta, Action)):
    """Abstract grasp class with grasp specified as an end-effector pose.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        Pose of the grasp w.r.t. world coordinate frame.
    """

    def __init__(self, T_grasp_world, q_value=0.0, id=-1, metadata={}):
        self.T_grasp_world = T_grasp_world
        Action.__init__(self, q_value, id, metadata)

    @abstractmethod
    def project(self, camera_intr, T_camera_world):
        pass


class ParallelJawGrasp3D(GraspAction3D):
    """Grasping with a parallel-jaw gripper.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        Pose of the grasp wrt world coordinate frame.
    """

    def project(self, camera_intr, T_camera_world, gripper_width=0.05):
        # Compute pose of grasp in camera frame.
        T_grasp_camera = T_camera_world.inverse() * self.T_grasp_world
        y_axis_camera = T_grasp_camera.y_axis[:2]
        if np.linalg.norm(y_axis_camera) > 0:
            y_axis_camera = y_axis_camera / np.linalg.norm(y_axis_camera)

        # Compute grasp axis rotation in image space.
        rot_grasp_camera = np.arccos(y_axis_camera[0])
        if y_axis_camera[1] < 0:
            rot_grasp_camera = -rot_grasp_camera
        while rot_grasp_camera < 0:
            rot_grasp_camera += 2 * np.pi
        while rot_grasp_camera > 2 * np.pi:
            rot_grasp_camera -= 2 * np.pi

        # Compute grasp center in image space.
        t_grasp_camera = T_grasp_camera.translation
        p_grasp_camera = Point(t_grasp_camera, frame=camera_intr.frame)
        u_grasp_camera = camera_intr.project(p_grasp_camera)
        d_grasp_camera = t_grasp_camera[2]
        return Grasp2D(u_grasp_camera,
                       rot_grasp_camera,
                       d_grasp_camera,
                       width=gripper_width,
                       camera_intr=camera_intr)


class SuctionGrasp3D(GraspAction3D):
    """Grasping with a suction-based gripper.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        Pose of the grasp wrt world coordinate frame.
    """

    def project(self, camera_intr, T_camera_world):
        # Compute pose of grasp in camera frame.
        T_grasp_camera = T_camera_world.inverse() * self.T_grasp_world
        x_axis_camera = T_grasp_camera.x_axis

        # Compute grasp center in image space.
        t_grasp_camera = T_grasp_camera.translation
        p_grasp_camera = Point(t_grasp_camera, frame=camera_intr.frame)
        u_grasp_camera = camera_intr.project(p_grasp_camera)
        d_grasp_camera = t_grasp_camera[2]
        return SuctionPoint2D(u_grasp_camera,
                              x_axis_camera,
                              d_grasp_camera,
                              camera_intr=camera_intr)


class MultiSuctionGrasp3D(GraspAction3D):
    """Grasping with a multi-cup suction-based gripper.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        Pose of the grasp wrt world coordinate frame.
    """

    def project(self, camera_intr, T_camera_world):
        # Compute pose of grasp in camera frame.
        T_grasp_camera = T_camera_world.inverse() * self.T_grasp_world
        return MultiSuctionPoint2D(T_grasp_camera, camera_intr=camera_intr)
