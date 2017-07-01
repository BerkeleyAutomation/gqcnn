.. GQCNN documentation master file, created by
   sphinx-quickstart on Thu May  4 16:09:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Berkeley AUTOLAB's GQCNN Package
================================

Overview
--------
The `gqcnn` package is a Python API for training and deploying `Grasp Quality Convolutional Neural Networks (GQ-CNNs)`_ for grasp planning using training datasets from the `Dexterity Network (Dex-Net)`_, developed by the `Berkeley AUTOLAB`_ and introduced in the `Dex-Net 2.0 paper`_.

Links
-----
* `Source Code`_
* `Datasets`_
* `Pretrained Models`_
* `Dex-Net Website`_
* `UC Berkeley AUTOLAB`_

.. _Source Code: https://github.com/BerkeleyAutomation/gqcnn
.. _Datasets: http://bit.ly/2rIM7Jk
.. _Pretrained Models: http://bit.ly/2tAFMko
.. _Dex-Net Website: https://berkeleyautomation.gitub.io/dex-net
.. _UC Berkeley AUTOLAB: http://autolab.berkeley.edu

.. image:: images/gqcnn.png
   :height: 800px
   :width: 800 px
   :scale: 100 %
   :align: center

Project Goals
-------------
The goals of this project are to facilitate:

1) **Replicability** of GQ-CNN training from the `Dex-Net 2.0 paper`_.
2) **Research extensions** on novel GQ-CNN architectures that have higher performance on Dex-Net 2.0 training datasets.

Our longer-term goal is to encourage development of GQ-CNNs that can be used to plan grasps on different hardware setups with different robots and cameras.

Disclaimer
----------
The `gqcnn` package currently supports only training of GQ-CNN on Dex-Net 2.0 datasets.
We are working toward a grasp planning ROS service based on GQ-CNNs to work toward GQ-CNNs that work on other robot hardware setups.

Please note that **performance on current datasets is not indicative of performance on other hardware setups** because our datasets are specific to:

1) An ABB YuMi parallel-jaw gripper due to collision geometry.
2) A Primense Carmine 1.08 due to camera parameters.
3) A camera positioned between 50-70cm above a table looking down due to image rendering parameters.

We are currently researching how to generate datasets that can generalize across robots, cameras, and viewpoints.

Development
-----------
The package is currently under active development. Installation has been tested on Ubuntu 12.04, 14.04, and 16.04.

Please raise all bugs, feature requests, and other issues under the `Github Issues`_.
For other questions or concerns, please contact Jeff Mahler (jmahler@berkeley.edu) with the subject line starting with "gqcnn development: "

Academic Use
------------
If you use the code, datasets, or models in a publication, please `cite the Dex-Net 2.0 paper`_.

.. _Grasp Quality Convolutional Neural Networks (GQ-CNNs): info/info.html

.. _Dexterity Network (Dex-Net): https://berkeleyautomation.github.io/dex-net

.. _Berkeley AUTOLAB: http://autolab.berkeley.edu/

.. _Dex-Net 2.0 paper: https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf

.. _Github Issues: https://github.com/BerkeleyAutomation/gqcnn/issues

.. _cite the Dex-Net 2.0 paper: https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017.bib

.. toctree::
   :maxdepth: 2
   :caption: Background

   info/info.rst

.. toctree::
   :maxdepth: 2
   :caption: Installation Guide

   install/install.rst

.. toctree::
   :maxdepth: 2
   :caption: Data

   data/data.rst

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   /benchmarks/benchmarks.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorials/tutorial.rst

.. toctree::
   :maxdepth: 2
   :caption: Documentation for Scripts
   :glob:

   scripts/*

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :glob:

   api/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
