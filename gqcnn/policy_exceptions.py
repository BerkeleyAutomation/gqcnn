"""
Exceptions that can be thrown by sub-classes of GraspingPolicy.
Author: Vishal Satish 
"""
class NoValidGraspsException(Exception):
	""" Exception for when antipodal point pairs can be found from the depth image but none are valid grasps that can be executed """
        def __init__(self, in_collision=True, not_confident=False, *args, **kwargs):
                self.in_collision = in_collision
                self.not_confident = not_confident
                Exception.__init__(self, *args, **kwargs)

class NoAntipodalPairsFoundException(Exception):
	""" Exception for when no antipodal point pairs can be found from the depth image """
	pass
