Overview
~~~~~~~~
There are two main use cases of the `gqcnn` package:

#. :ref:`training` a `Dex-Net 4.0`_ GQ-CNN model on an offline `Dex-Net`_ dataset of point clouds, grasps, and grasp success metrics, and then grasp planning on RGBD images.
#. :ref:`grasp planning` on RGBD images using a pre-trained `Dex-Net 4.0`_ GQ-CNN model.

.. _Dex-Net 4.0: https://berkeleyautomation.github.io/dex-net/#dexnet_4
.. _Dex-Net: https://berkeleyautomation.github.io/dex-net/

Click on the links or scroll down to get started!

Prerequisites
-------------
Before running the tutorials please download the example models and datasets: ::

    $ cd /path/to/your/gqcnn
    $ ./scripts/downloads/download_example_data.sh
    $ ./scripts/downloads/models/download_models.sh


Running Python Scripts
----------------------
All `gqcnn` Python scripts are designed to be run from the top-level directory of your `gqcnn` repo by default. This is because every script takes in a YAML file specifying parameters for the script, and this YAML file is stored relative to the repository root directory.

We recommend that you run all scripts using this paradigm: ::

  cd /path/to/your/gqcnn
  python /path/to/script.py

.. _training:
.. include:: training.rst

.. _analysis:
.. include:: analysis.rst

.. _grasp planning:
.. include:: planning.rst

