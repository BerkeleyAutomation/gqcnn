Policy
======

Policy
~~~~~~
Abstract class for policies.

.. autoclass:: gqcnn.Policy

GraspingPolicy
~~~~~~~~~~~~~~
A policy for robust grasping with GQCNN's

.. autoclass:: gqcnn.GraspingPolicy

AntipodalGraspingPolicy
~~~~~~~~~~~~~~~~~~~~~~~
A policy for robust grasping with GQCNN's using sampling of
antipodal grasp candidates from an image space and ranking using a 
GQCNN instance. 

.. autoclass:: gqcnn.AntipodalGraspingPolicy

CrossEntropyAntipodalGraspingPolicy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A policy for robust grasping with GQCNN's using antipodal point sampling and 
a cross-entropy method.

.. autoclass:: gqcnn.CrossEntropyAntipodalGraspingPolicy

RgbdImageState
~~~~~~~~~~~~~~
A state wrapper for RGBD images.

.. autoclass:: gqcnn.RgbdImageState

ParallelJawGrasp
~~~~~~~~~~~~~~~~
An action wrapper for parallel jaw grasps.

.. autoclass:: gqcnn.ParallelJawGrasp

NoValidGraspsException
~~~~~~~~~~~~~~~~~~~~~~
Exception class for handling when the policy can sample antipodal grasps but cannot
determine any valid grasps in them.

.. autoclass:: gqcnn.NoValidGraspsException

NoAntipodalPairsFoundException
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exception class for handling when the policy cannot sample any antipodal point pairs from the input
image data.

.. autoclass:: gqcnn.NoAntipodalPairsFoundException 


