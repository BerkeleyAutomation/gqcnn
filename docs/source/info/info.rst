What are GQ-CNNs?
-----------------
GQ-CNNs are neural network architectures that take as input a depth image and grasp, and output the predicted probability that the grasp will successfully hold the object while lifting, transporting, and shaking the object.

.. image:: ../images/gqcnn.png
   :height: 800px
   :width: 800 px
   :scale: 100 %
   :align: center

The GQ-CNN weights are trained on datasets of synthetic point clouds, parallel-jaw grasps, and grasp metrics generated from physics-based models with domain randomization for sim-to-real transfer. See the ongoing `Dexterity Network (Dex-Net)`_ project for more information.

.. _Dexterity Network (Dex-Net): https://berkeleyautomation.github.io/dex-net

