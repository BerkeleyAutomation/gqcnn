Grasping
========

Grasp2D
~~~~~~~
Wrapper for parallel jaw grasps in image space.

.. autoclass:: gqcnn.Grasp2D

RobotGripper
~~~~~~~~~~~~
Wrapper for robot grippers. Used for collision checking and encapsulation of various grasp parameters.

.. autoclass:: gqcnn.RobotGripper

ImageGraspSampler
~~~~~~~~~~~~~~~~~
Abstract class from Image Grasp Samplers.

.. autoclass:: gqcnn.ImageGraspSampler

AntipodalDepthImageGraspSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Image Grasp Sampler that samples antipodal point pairs from the gradients in provided depth image.

.. autoclass:: gqcnn.AntipodalDepthImageGraspSampler

ImageGraspSamplerFactory
~~~~~~~~~~~~~~~~~~~~~~~~
Factory for generating Image Grasp Samplers.

.. autoclass:: gqcnn.ImageGraspSamplerFactory
