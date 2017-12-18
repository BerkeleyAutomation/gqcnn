"""
Exceptions that can be thrown by sub-classes of GraspingPolicy.
Author: Vishal Satish
"""


class NoValidGraspsException(Exception):
    """ Exception for when antipodal point pairs can be found from the depth image but none are valid grasps that can be executed """
    pass


class NoAntipodalPairsFoundException(Exception):
    """ Exception for when no antipodal point pairs can be found from the depth image """
    pass
