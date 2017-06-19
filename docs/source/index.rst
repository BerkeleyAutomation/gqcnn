.. GQCNN documentation master file, created by
   sphinx-quickstart on Thu May  4 16:09:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Berkeley AUTOLAB's GQCNN Package
================================
The `gqcnn` package facilitates training Grasp Quality Neural Networks (GQ-CNNs) on datasets from the `Dexterity Network`_ (Dex-Net) and using GQ-CNNs to plan parallel-jaw grasps from point clouds on a physical robot.
Installation has been tested on Ubuntu 12.04, 14.04, and 16.04.

Please raise all bugs, feature requests, and other issues under the `Github Issues`_.
For other questions or concerns, please contact Jeff Mahler (jmahler@berkeley.edu).

.. _Github Issues: https://github.com/BerkeleyAutomation/gqcnn/issues

.. _Dexterity Network: https://berkeleyautomation.github.io/dex-net

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
   :caption: Tutorial

   tutorials/tutorial.rst

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :glob:

   api/*

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   /benchmarks/benchmarks.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
