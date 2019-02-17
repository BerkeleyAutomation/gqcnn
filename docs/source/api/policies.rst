Policies
========

All GQ-CNN grasping policies are child classes of the base :ref:`GraspingPolicy` class that implements `__call__()`, which operates on :ref:`RgbdImageStates <RgbdImageState>` and returns a :ref:`GraspAction`. :: 

    $ from gqcnn import RgbdImageState, CrossEntropyRobustGraspingPolicy
    $
    $ im = RgbdImageState.load(<saved rgbd image dir>)
    $ my_policy = CrossEntropyRobustGraspingPolicy(<policy initializer args>)
    $
    $ my_grasp_action = my_policy(im)

Primary Policies
~~~~~~~~~~~~~~~~

CrossEntropyRobustGraspingPolicy
--------------------------------
An implementation of the `Cross Entropy Method (CEM)`_ used in `Dex-Net 2.0`_, `Dex-Net 2.1`_, `Dex-Net 3.0`_, and `Dex-Net 4.0`_ to iteratively locate the best grasp.

.. autoclass:: gqcnn.CrossEntropyRobustGraspingPolicy

FullyConvolutionalGraspingPolicyParallelJaw
-------------------------------------------
An implementation of the `FC-GQ-CNN`_ parallel jaw policy that uses dense, parallelized fully convolutional networks.

.. autoclass:: gqcnn.FullyConvolutionalGraspingPolicyParallelJaw 

FullyConvolutionalGraspingPolicySuction
---------------------------------------
An implementation of the `FC-GQ-CNN`_ suction policy that uses dense, parallelized fully convolutional networks.

.. autoclass:: gqcnn.FullyConvolutionalGraspingPolicySuction

Grasps and Image Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _RgbdImageState:

RgbdImageState
--------------
A wrapper for states containing an RGBD (RGB + Depth) image, camera intrinisics, and segmentation masks.

.. autoclass:: gqcnn.RgbdImageState

.. _GraspAction:

GraspAction
-----------
A wrapper for 2D grasp actions such as :ref:`Grasp2D` or :ref:`SuctionPoint2D`.

.. autoclass:: gqcnn.grasping.policy.policy.GraspAction

.. _Grasp2D:

Grasp2D
-------
A wrapper for 2D parallel jaw grasps.

.. autoclass:: gqcnn.grasping.grasp.Grasp2D

.. _SuctionPoint2D:

SuctionPoint2D
--------------
A wrapper for 2D suction grasps.

.. autoclass:: gqcnn.grasping.grasp.SuctionPoint2D

Miscellaneous and Parent Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Policy
------

.. autoclass:: gqcnn.grasping.policy.policy.Policy


.. _GraspingPolicy:

GraspingPolicy
--------------

.. autoclass:: gqcnn.grasping.policy.policy.GraspingPolicy

FullyConvolutionalGraspingPolicy
--------------------------------
.. autoclass:: gqcnn.grasping.policy.fc_policy.FullyConvolutionalGraspingPolicy

RobustGraspingPolicy
--------------------

.. autoclass:: gqcnn.RobustGraspingPolicy

UniformRandomGraspingPolicy
---------------------------

.. autoclass:: gqcnn.UniformRandomGraspingPolicy

.. _Cross Entropy Method (CEM): https://en.wikipedia.org/wiki/Cross-entropy_method
.. _Dex-Net 2.0: https://berkeleyautomation.github.io/dex-net/#dexnet_2
.. _Dex-Net 2.1: https://berkeleyautomation.github.io/dex-net/#dexnet_21
.. _Dex-Net 3.0: https://berkeleyautomation.github.io/dex-net/#dexnet_3
.. _Dex-Net 4.0: https://berkeleyautomation.github.io/dex-net/#dexnet_4
.. _FC-GQ-CNN: https://berkeleyautomation.github.io/fcgqcnn

