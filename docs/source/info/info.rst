What are GQ-CNNs?
-----------------
GQ-CNNs are neural network architectures that take as input a depth image and grasp, and output the predicted probability that the grasp will successfully hold the object while lifting, transporting, and shaking the object.

.. figure:: ../images/gqcnn.png
   :width: 100%
   :align: center

   Original GQ-CNN architecture from `Dex-Net 2.0`_.

.. figure:: ../images/fcgqcnn_arch_diagram.png
   :width: 100%
   :align: center

   Alternate faster GQ-CNN architecture from `FC-GQ-CNN`_.


The GQ-CNN weights are trained on datasets of synthetic point clouds, parallel jaw grasps, and grasp metrics generated from physics-based models with domain randomization for sim-to-real transfer. See the ongoing `Dexterity Network (Dex-Net)`_ project for more information.

.. _Dexterity Network (Dex-Net): https://berkeleyautomation.github.io/dex-net
.. _Dex-Net 2.0: https://berkeleyautomation.github.io/dex-net/#dexnet_2
.. _FC-GQ-CNN: https://berkeleyautomation.github.io/fcgqcnn
