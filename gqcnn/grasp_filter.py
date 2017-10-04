"""
Classes for filtering out grasps, such as grasps that are unreachable for the 
robot arm
Author: Chris Correa
"""
from abc import ABCMeta, abstractmethod

from autolab_core import RigidTransform

DEFAULT_T_CAMERA_WORLD = '/home/autolab/Public/alan/calib/primesense_overhead/primesense_overhead_to_world.tf'

class GraspFilter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def filter(grasps):
        """
        Takes in a set of grasps, and returns a new list of grasps, each of 
        which passes through the filter.  An example of one such filter is the
        reachability filter, which filters out grasps that are outside the 
        robot workspace.

        Parameters
        ----------
        grasps : :obj:'list' of :obj:'Grasp2D'
            list of 2D grasp candidates

        Returns
        -------
        :obj:'list' of :obj:'Grasp2D'
            list of 2D grasp candidates that pass through the filter
        """
        pass

class ReachabilityFilter(GraspFilter):
    def __init__(self, T_camera_world=None,
                 arm_name='left'):
        if T_camera_world is None:
            T_camera_world = RigidTransform.load(DEFAULT_T_CAMERA_WORLD)
        self.T_camera_world = T_camera_world

        from yumipy import YuMiArm
        self.arm = YuMiArm(arm_name)

    def filter(self, grasps):
        """
        filters out grasps that are outside the robot workspace.

        Parameters
        ----------
        grasps : :obj:'list' of :obj:'Grasp2D'
            list of 2D grasp candidates

        Returns
        -------
        :obj:'list' of :obj:'Grasp2D'
            list of 2D grasp candidates that pass through the filter
        """
        def filter_grasp(grasp):
            T_grasp_camera = grasp.pose()
            return self.arm.is_pose_reachable(T_grasp_camera * self.T_camera_world)
        return filter(self.arm.is_pose_reachable, grasps)
