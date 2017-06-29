Dependencies
~~~~~~~~~~~~

PyPI Packages
"""""""""""""
The `gqcnn` package  depends on `numpy`_, `scipy`_, `matplotlib`_, `tensorflow`_, `cv2`_, `skimage`_, `sklearn`_, and `pillow`_ which should be installed automatically when using pip.
You can also install these manually if necessary ::

    $ pip install numpy scipy matplotlib tensorflow-gpu opencv-python scikit-image scikit-learn pillow

.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy/org/
.. _matplotlib: http://www.matplotlib.org/
.. _tensorflow: https://www.tensorflow.org/
.. _cv2: http://opencv.org/
.. _pillow: https://python-pillow.org/
.. _skimage: http://scikit-learn.org/stable/
.. _sklearn: http://scikit-image.org/

If you do not have a GPU, then substitute `tensorflow` for `tensorflow-gpu` in the installation command.
Note that `TensorFlow installation`_ with GPU support requires CUDA 8.0.

.. _TensorFlow installation: https://www.tensorflow.org/install

BerkeleyAutomation Packages
"""""""""""""""""""""""""""
The `gqcnn` package also depends on `BerkeleyAutomation's`_ `autolab_core`_ and `perception`_ packages.
To install these dependencies, follow the `installation instructions for autolab_core`_ and the `installation instructions for perception`_.
If you are installing gqcnn as a ROS package, we suggest installing both `autolab_core`_ and `perception`_ as ROS packages by checking out the repos into your catkin workspace and running catkin_make.

.. _BerkeleyAutomation's: https://github.com/BerkeleyAutomation
.. _autolab_core: https://github.com/BerkeleyAutomation/autolab_core
.. _perception: https://github.com/BerkeleyAutomation/perception
.. _installation instructions for autolab_core: https://BerkeleyAutomation.github.io/autolab_core/install/install.html
.. _installation instructions for perception: https://berkeleyautomation.github.io/perception/install/install.html


Python Installation
~~~~~~~~~~~~~~~~~~~

Python-only installation is intended for users who are **only interested in training GQ-CNNs**, not
using them on a physical robot.
If you have intentions of using GQ-CNNs for grasp planning on a physical robot, we suggest you `install as a ROS package`_.

The `gqcnn` package is known to work for Python 2.7 and has not been tested for Python 3.

.. _install as a ROS package: https://berkeleyautomation.github.io/gqcnn/install/install.html#ros-installation

1. Clone the repository
"""""""""""""""""""""""
Clone or download the project from `Github`_. ::

    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

.. _Github: https://github.com/BerkeleyAutomation/gqcnn

2. Run installation script
""""""""""""""""""""""""""
Change directories into the `gqcnn` repository and run ::

    $ python setup.py install

or ::

    $ pip install -r requirements.txt

Alternatively, you can run ::

    $ pip install /path/to/gqcnn

to install `gqcnn` from anywhere.
This will install `gqcnn` in your current Python environment.

ROS Installation
~~~~~~~~~~~~~~~~

Installation as a ROS package is intended for users who wish to use GQ-CNNs to plan grasps on a physical robot.

1. Clone the repository
"""""""""""""""""""""""
Clone or download our source code from `Github`_. ::

    $ cd {PATH_TO_YOUR_CATKIN_WORKSPACE}/src
    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

2. Build the catkin package
"""""""""""""""""""""""""""
Build the catkin pacakge by running ::

    $ cd {PATH_TO_YOUR_CATKIN_WORKSPACE}
    $ catkin_make

Then re-source devel/setup.bash for the package to be available through Python.

Quick Start Guide
~~~~~~~~~~~~~~~~~
Once gqcnn is installed, see `our tutorial page`_ to get started!

.. _our tutorial page: https://berkeleyautomation.github.io/gqcnn/tutorials/tutorial.html

Documentation
~~~~~~~~~~~~~

Building
""""""""
The API documentation is available on the `gqcnn website`_.

.. _gqcnn website: https://berkeleyautomation.github.io/gqcnn

You can re-build `gqcnn`'s documentation from scratch with a few extra dependencies --
specifically, `sphinx`_ and a few plugins.
This is important for developers only.

.. _sphinx: http://www.sphinx-doc.org/en/1.4.8/

To install the dependencies required, simply run ::

    $ pip install -r docs_requirements.txt

Then, go to the `docs` directory and run ``make`` with the appropriate target.
For example, ::

    $ cd docs/
    $ make html

will generate a set of web pages. Any documentation files
generated in this manner can be found in `docs/build`.

Deploying
"""""""""
To deploy documentation to the Github Pages site for the repository,
simply push any changes to the documentation source to master
and then run ::

    $ . gh_deploy.sh

from the `docs` folder. This script will automatically checkout the
``gh-pages`` branch, build the documentation from source, and push it
to Github.


