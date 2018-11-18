"""
Action classes for representing 3D grasp actions
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from autolab_core import Point, RigidTransform

from gqcnn.grasping import Grasp2D, SuctionPoint2D, MultiSuctionPoint2D

class Action(object):
    """ Abstract action class.

    Attributes
    ----------
    q_value : float
        q_value of the grasp
    id : int
        integer identifier for the action
    metadata : dict
        key-value dict of extra data about the action
    """
    def __init__(self, q_value=0.0,
                 id=-1,
                 metadata={}):
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
    """ Proxy for taking no action when none can be found! """
    pass
    
class GraspAction3D(Action):
    """ Generic grasping with grasps specified as an end-effector pose

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        pose of the grasp wrt world coordinate frame
    """
    def __init__(self, T_grasp_world,
                 q_value=0.0,
                 id=-1,
                 metadata={}):
        self.T_grasp_world = T_grasp_world
        Action.__init__(self, q_value, id, metadata)

class ParallelJawGrasp3D(GraspAction3D):
    """ Grasping with a parallel-jaw gripper.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        pose of the grasp wrt world coordinate frame
    """        
    def project(self, camera_intr,
                T_camera_world,
                gripper_width=0.05):
        # compute pose of grasp in camera frame
        T_grasp_camera = T_camera_world.inverse() * self.T_grasp_world
        y_axis_camera = T_grasp_camera.y_axis[:2]
        if np.linalg.norm(y_axis_camera) > 0:
            y_axis_camera = y_axis_camera / np.linalg.norm(y_axis_camera)

        # compute grasp axis rotation in image space
        rot_grasp_camera = np.arccos(y_axis_camera[0])
        if y_axis_camera[1] < 0:
            rot_grasp_camera = -rot_grasp_camera
        while rot_grasp_camera < 0:
            rot_grasp_camera += 2 * np.pi
        while rot_grasp_camera > 2 * np.pi:
            rot_grasp_camera -= 2 * np.pi

        # compute grasp center in image space
        t_grasp_camera = T_grasp_camera.translation
        p_grasp_camera = Point(t_grasp_camera,
                               frame=camera_intr.frame)
        u_grasp_camera = camera_intr.project(p_grasp_camera)
        d_grasp_camera = t_grasp_camera[2]
        return Grasp2D(u_grasp_camera,
                       rot_grasp_camera,
                       d_grasp_camera,
                       width=gripper_width,
                       camera_intr=camera_intr)
    
class SuctionGrasp3D(GraspAction3D):
    """ Grasping with a suction-based gripper.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        pose of the grasp wrt world coordinate frame
    """
    def project(self, camera_intr,
                T_camera_world):
        # compute pose of grasp in camera frame
        T_grasp_camera = T_camera_world.inverse() * self.T_grasp_world
        x_axis_camera = T_grasp_camera.x_axis

        # compute grasp center in image space
        t_grasp_camera = T_grasp_camera.translation
        p_grasp_camera = Point(t_grasp_camera, frame=camera_intr.frame)
        u_grasp_camera = camera_intr.project(p_grasp_camera)
        d_grasp_camera = t_grasp_camera[2]
        return SuctionPoint2D(u_grasp_camera,
                              x_axis_camera,
                              d_grasp_camera,
                              camera_intr=camera_intr)

class MultiSuctionGrasp3D(GraspAction3D):
    """ Grasping with a multi-cup suction-based gripper.

    Attributes
    ----------
    T_grasp_world : :obj:`RigidTransform`
        pose of the grasp wrt world coordinate frame
    """
    def project(self, camera_intr,
                T_camera_world):
        # compute pose of grasp in camera frame
        T_grasp_camera = T_camera_world.inverse() * self.T_grasp_world
        return MultiSuctionPoint2D(T_grasp_camera,
                                   camera_intr=camera_intr)    
