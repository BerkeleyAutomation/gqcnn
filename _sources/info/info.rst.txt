What are GQ-CNNs?
-----------------
GQ-CNNs are neural network architectures that take as input a depth image and grasp and output the predicted probability that the grasp will successfully hold the object while lifting, transporting, and shaking the object.

.. image:: ../images/gqcnn.png
   :height: 800px
   :width: 800 px
   :scale: 100 %
   :align: center

The GQ-CNN weights are trained on datasets of synthetic point clouds, parallel-jaw grasps, and grasp metrics generated from physics-based models with the `Dexterity Network (Dex-Net)`_.
See the `Dex-Net 2.0 paper`_ for more info.

.. _Dexterity Network (Dex-Net): https://berkeleyautomation.github.io/dex-net

.. _Dex-Net 2.0 paper: https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf

