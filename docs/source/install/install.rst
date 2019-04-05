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

Dependencies
~~~~~~~~~~~~

PyPI Packages
"""""""""""""
The `gqcnn` package  depends on `numpy`_, `scipy`_, `matplotlib`_, `tensorflow`_, `cv2`_, `sklearn`_, `gputil`_, and `psutil`_, which should be installed automatically.
You can also install these manually if necessary: ::

    $ pip install numpy>=1.14.0 scipy matplotlib>=1.5.0 tensorflow>=1.10.0 opencv-python scikit-learn psutil gputil

.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy/org/
.. _matplotlib: http://www.matplotlib.org/
.. _tensorflow: https://www.tensorflow.org/
.. _cv2: http://opencv.org/
.. _sklearn: http://scikit-image.org/
.. _psutil: https://github.com/giampaolo/psutil
.. _gputil: https://github.com/anderskm/gputil

If you wish to use a GPU, then substitute `tensorflow-gpu` for `tensorflow`.

BerkeleyAutomation Packages
"""""""""""""""""""""""""""
The `gqcnn` package also depends on `BerkeleyAutomation's`_ `autolab_core`_, `perception`_, and `visualization`_ packages. These will also be installed automatically through PyPI.

To install these dependencies manually, follow the `installation instructions for autolab_core`_, the `installation instructions for perception`_, and the `installation instructions for visualization`_.
If you are installing `gqcnn` as a ROS package, we suggest installing both `autolab_core`_ and `perception`_ as ROS packages by checking out the repos into your catkin workspace and running `catkin_make`.

.. _BerkeleyAutomation's: https://github.com/BerkeleyAutomation
.. _autolab_core: https://github.com/BerkeleyAutomation/autolab_core
.. _perception: https://github.com/BerkeleyAutomation/perception
.. _visualization: https://github.com/BerkeleyAutomation/visualization
.. _installation instructions for autolab_core: https://BerkeleyAutomation.github.io/autolab_core/install/install.html
.. _installation instructions for perception: https://berkeleyautomation.github.io/perception/install/install.html
.. _installation instructions for visualization: https://berkeleyautomation.github.io/visualization/install/install.html

Python Installation
~~~~~~~~~~~~~~~~~~~

The Python-only installation is intended for users who are **only interested in 1) Training GQ-CNNs or 2) Grasp planning on saved RGBD images**, not
interfacing with a physical robot.
If you have intentions of using GQ-CNNs for grasp planning on a physical robot, we suggest you `install as a ROS package`_.

.. _install as a ROS package: https://berkeleyautomation.github.io/gqcnn/install/install.html#ros-installation

1. Clone the repository
"""""""""""""""""""""""
Clone or download the `project`_ from Github. ::

    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

.. _project: https://github.com/BerkeleyAutomation/gqcnn

2. Run installation script
""""""""""""""""""""""""""
Change directories into the `gqcnn` repository and run the setup script. ::

    $ python setup.py install

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

