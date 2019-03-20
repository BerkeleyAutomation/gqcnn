.. GQCNN documentation master file, created by
   sphinx-quickstart on Thu May  4 16:09:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Berkeley AUTOLAB's GQCNN Package
================================

Overview
--------
The `gqcnn` package is a Python API for training and deploying `Grasp Quality Convolutional Neural Networks (GQ-CNNs)`_ for grasp planning using training datasets from the `Dexterity Network (Dex-Net)`_, developed by the `Berkeley AUTOLAB`_ and introduced in the `Dex-Net 2.0 paper`_.

.. note:: We're excited to announce **version 1.0**, which brings the GQ-CNN package up to date with recent research in `Dex-Net`_.
   Version 1.0 introduces support for:

   #. **Dex-Net 4.0:** Composite policies that decide whether to use a suction cup or parallel-jaw gripper.
   #. **Fully Convolutional GQ-CNNs:** Fully convolutional architectures that efficiently evaluate millions of grasps faster than prior GQ-CNNs.

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
.. _Dex-Net Website: https://berkeleyautomation.github.io/dex-net
.. _UC Berkeley AUTOLAB: http://autolab.berkeley.edu

.. image:: images/gqcnn.png
   :width: 100 %

Project Goals
-------------
The goals of this project are to facilitate:

#. **Replicability** of GQ-CNN training and deployment from:
    #. The latest `Dex-Net 4.0`_ results.
    #. Older results such as `Dex-Net 2.0`_, `Dex-Net 2.1`_ and `Dex-Net 3.0`_.
    #. Experimental results such as `FC-GQ-CNN`_.
#. **Research extensions** on novel GQ-CNN architectures that have higher performance on `Dex-Net 4.0`_ training datasets.

.. _Dex-Net 2.0: https://berkeleyautomation.github.io/dex-net/#dexnet_2
.. _Dex-Net 2.1: https://berkeleyautomation.github.io/dex-net/#dexnet_21
.. _Dex-Net 3.0: https://berkeleyautomation.github.io/dex-net/#dexnet_3
.. _Dex-Net 4.0: https://berkeleyautomation.github.io/dex-net/#dexnet_4
.. _FC-GQ-CNN: https://berkeleyautomation.github.io/fcgqcnn

Our longer-term goal is to encourage development of robust GQ-CNNs that can be used to plan grasps on different hardware setups with different robots and cameras.

Disclaimer
----------
GQ-CNN models are sensitive to the following parameters used during dataset generation:
   #. The robot gripper
   #. The depth camera
   #. The distance between the camera and workspace.

As a result, we cannot guarantee performance of our pre-trained models on other physical setups. If you have a specific use-case in mind, please reach out to us. It might be possible to generate a custom dataset for you particular setup. We are actively researching how to generate more robust datasets that can generalize across robots, cameras, and viewpoints.

Development
-----------
The package is currently under active development. Please raise all bugs, feature requests, and other issues under the `Github Issues`_.
For other questions or concerns, please contact Jeff Mahler (jmahler@berkeley.edu) or Vishal Satish (vsatish@berkeley.edu) with the subject line starting with "GQ-CNN Development".

Academic Use
------------
If you use the code, datasets, or models in a publication, please cite the appropriate paper:

#. **Dex-Net 2.0** `(bibtex) <https://raw.githubusercontent.com/BerkeleyAutomation/dex-net/gh-pages/docs/dexnet_rss2017.bib>`__
#. **Dex-Net 2.1** `(bibtex) <https://raw.githubusercontent.com/BerkeleyAutomation/dex-net/gh-pages/docs/dexnet_corl2017.bib>`__
#. **Dex-Net 3.0** `(bibtex) <https://raw.githubusercontent.com/BerkeleyAutomation/dex-net/gh-pages/docs/dexnet_icra2018.bib>`__
#. **Dex-Net 4.0** `(bibtex) <https://raw.githubusercontent.com/BerkeleyAutomation/dex-net/gh-pages/docs/dexnet_sciencerobotics2019.bib>`__
#. **FC-GQ-CNN** `(bibtex) <https://raw.githubusercontent.com/BerkeleyAutomation/fcgqcnn/gh-pages/docs/fcgqcnn_ral2019.bib>`__

.. _Grasp Quality Convolutional Neural Networks (GQ-CNNs): info/info.html

.. _Dexterity Network (Dex-Net): https://berkeleyautomation.github.io/dex-net

.. _Dex-Net: https://berkeleyautomation.github.io/dex-net

.. _Berkeley AUTOLAB: http://autolab.berkeley.edu/

.. _Dex-Net 2.0 paper: https://github.com/BerkeleyAutomation/dex-net/raw/gh-pages/docs/dexnet_rss2017_final.pdf

.. _Github Issues: https://github.com/BerkeleyAutomation/gqcnn/issues

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
   :caption: Getting Started

   tutorials/tutorial.rst

.. toctree::
   :maxdepth: 2
   :caption: Replication

   replication/replication.rst

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   benchmarks/benchmarks.rst

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   :glob:

   api/gqcnn.rst
   api/training.rst
   api/analysis.rst
   api/policies.rst

.. toctree::
   :maxdepth: 2
   :caption: License

   license/license.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
