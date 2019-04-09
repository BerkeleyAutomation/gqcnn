Prerequisites
~~~~~~~~~~~~~

Python
""""""

The `gqcnn` package has only been tested with `Python 2.7`.

Ubuntu
""""""

The `gqcnn` package has only been tested with `Ubuntu 12.04`, `Ubuntu 14.04` and `Ubuntu 16.04`.

Virtualenv
""""""""""

We highly recommend using a Python environment management system, in particular `Virtualenv`. **Note: Several users have encountered problems with dependencies when using Conda.**

Pip Installation
~~~~~~~~~~~~~~~~

The pip installation is intended for users who are **only interested in 1) Training GQ-CNNs or 2) Grasp planning on saved RGBD images**, not
interfacing with a physical robot.
If you have intentions of using GQ-CNNs for grasp planning on a physical robot, we suggest you `install as a ROS package`_.

.. _install as a ROS package: https://berkeleyautomation.github.io/gqcnn/install/install.html#ros-installation

1. Clone the repository
"""""""""""""""""""""""
Clone or download the `project`_ from Github. ::

    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

.. _project: https://github.com/BerkeleyAutomation/gqcnn

2. Run pip installation
""""""""""""""""""""""""""
Change directories into the `gqcnn` repository and run the pip installation. ::

    $ pip install .

This will install `gqcnn` in your current virtual environment and automatically download the example models and datasets.

.. _ros-install:

ROS Installation
~~~~~~~~~~~~~~~~

Installation as a ROS package is intended for users who wish to use GQ-CNNs to plan grasps on a physical robot.

1. Clone the repository
"""""""""""""""""""""""
Clone or download the `project`_ from Github. ::

    $ cd <PATH_TO_YOUR_CATKIN_WORKSPACE>/src
    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

2. Build the catkin package
"""""""""""""""""""""""""""
Build the catkin package. ::

    $ cd <PATH_TO_YOUR_CATKIN_WORKSPACE>
    $ catkin_make

Then re-source `devel/setup.bash` for the package to be available through Python.

