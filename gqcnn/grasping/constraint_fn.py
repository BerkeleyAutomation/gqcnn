"""
Constraint functions for grasp sampling
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import numpy as np

class GraspConstraintFn(object):
    """
    Abstract constraint functions for grasp sampling.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config):
        # set params
        self._config = config
    
    def __call__(self, grasp):
        """
        Evaluates whether or not a grasp is valid.

        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            grasp to evaluate

        Returns
        -------
        bool
            True if the grasp satisfies constraints, False otherwise
        """
        return self.satisfies_constraints(grasp)

    @abstractmethod    
    def satisfies_constraints(self, grasp):
        """
        Evaluates whether or not a grasp is valid.

        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            grasp to evaluate

        Returns
        -------
        bool
            True if the grasp satisfies constraints, False otherwise
        """
        pass

class DiscreteApproachGraspConstraintFn(GraspConstraintFn):
    """
    Constrains the grasp approach direction into a discrete set of
    angles from the world z direction.
    """
    def __init__(self, config):
        # init superclass
        GraspConstraintFn.__init__(self, config)
        
        self._max_approach_angle = self._config['max_approach_angle']
        self._angular_tolerance = self._config['angular_tolerance']
        self._angular_step = self._config['angular_step']
        self._T_camera_world = self._config['camera_pose']
        
    def satisfies_constraints(self, grasp):
        """
        Evaluates whether or not a grasp is valid by evaluating the
        angle between the approach axis and the world z direction.

        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            grasp to evaluate

        Returns
        -------
        bool
            True if the grasp satisfies constraints, False otherwise
        """
        # find grasp angle in world coordinates
        axis_world = self._T_camera_world.rotation.dot(grasp.approach_axis)
        angle = np.arccos(-axis_world[2])

        # check closest available angle
        available_angles = np.array([0.0])
        if self._angular_step > 0:
            available_angles = np.arange(start=0.0,
                                         stop=self._max_approach_angle,
                                         step=self._angular_step)
        diff = np.abs(available_angles - angle)
        angle_index = np.argmin(diff)
        closest_angle = available_angles[angle_index]
        if diff[angle_index] < self._angular_tolerance:
            return True
        return False

class GraspConstraintFnFactory(object):
    @staticmethod
    def constraint_fn(fn_type, config):
        if fn_type == 'none':
            return None
        elif fn_type == 'discrete_approach_angle':
            return DiscreteApproachGraspConstraintFn(config)
        else:
            raise ValueError('Grasp constraint function type %s not supported!' %(fn_type))
