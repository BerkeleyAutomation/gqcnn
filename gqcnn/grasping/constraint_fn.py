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

Constraint functions for grasp sampling.

Author
------
Jeff Mahler
"""
from abc import ABC, abstractmethod

import numpy as np


class GraspConstraintFn(ABC):
    """Abstract constraint functions for grasp sampling."""

    def __init__(self, config):
        # Set params.
        self._config = config

    def __call__(self, grasp):
        """Evaluates whether or not a grasp is valid.

        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            Grasp to evaluate.

        Returns
        -------
        bool
            True if the grasp satisfies constraints, False otherwise.
        """
        return self.satisfies_constraints(grasp)

    @abstractmethod
    def satisfies_constraints(self, grasp):
        """Evaluates whether or not a grasp is valid.

        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            Grasp to evaluate.

        Returns
        -------
        bool
            True if the grasp satisfies constraints, False otherwise.
        """
        pass


class DiscreteApproachGraspConstraintFn(GraspConstraintFn):
    """Constrains the grasp approach direction into a discrete set of
    angles from the world z direction."""

    def __init__(self, config):
        # Init superclass.
        GraspConstraintFn.__init__(self, config)

        self._max_approach_angle = self._config["max_approach_angle"]
        self._angular_tolerance = self._config["angular_tolerance"]
        self._angular_step = self._config["angular_step"]
        self._T_camera_world = self._config["camera_pose"]

    def satisfies_constraints(self, grasp):
        """Evaluates whether or not a grasp is valid by evaluating the
        angle between the approach axis and the world z direction.

        Parameters
        ----------
        grasp : :obj:`Grasp2D`
            Grasp to evaluate.

        Returns
        -------
        bool
            True if the grasp satisfies constraints, False otherwise.
        """
        # Find grasp angle in world coordinates.
        axis_world = self._T_camera_world.rotation.dot(grasp.approach_axis)
        angle = np.arccos(-axis_world[2])

        # Check closest available angle.
        available_angles = np.array([0.0])
        if self._angular_step > 0:
            available_angles = np.arange(start=0.0,
                                         stop=self._max_approach_angle,
                                         step=self._angular_step)
        diff = np.abs(available_angles - angle)
        angle_index = np.argmin(diff)
        if diff[angle_index] < self._angular_tolerance:
            return True
        return False


class GraspConstraintFnFactory(object):

    @staticmethod
    def constraint_fn(fn_type, config):
        if fn_type == "none":
            return None
        elif fn_type == "discrete_approach_angle":
            return DiscreteApproachGraspConstraintFn(config)
        else:
            raise ValueError(
                "Grasp constraint function type {} not supported!".format(
                    fn_type))
