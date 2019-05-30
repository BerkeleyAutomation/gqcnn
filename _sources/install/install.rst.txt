Prerequisites
~~~~~~~~~~~~~

Python
""""""

The `gqcnn` package has only been tested with `Python 2.7`, `Python 3.5`, `Python 3.6`, and `Python 3.7`.

Ubuntu
""""""

The `gqcnn` package has only been tested with `Ubuntu 12.04`, `Ubuntu 14.04` and `Ubuntu 16.04`.

Virtualenv
""""""""""

We highly recommend using a Python environment management system, in particular `Virtualenv`, with the Pip and ROS installations. **Note: Several users have encountered problems with dependencies when using Conda.**

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
"""""""""""""""""""""""
Change directories into the `gqcnn` repository and run the pip installation. ::

    $ pip install .

This will install `gqcnn` in your current virtual environment.

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

Docker Installation
~~~~~~~~~~~~~~~~~~~

We currently do not provide pre-built Docker images, but you can build them yourself. This will require you to have installed `Docker`_ or `Nvidia-Docker`_ if you plan on using GPUs. Note that our provided build for GPUs uses CUDA 10.0 and cuDNN 7.0, so make sure that this is compatible with your GPU hardware. If you wish to use a different CUDA/cuDNN version, change the base image in `docker/gpu/Dockerfile` to the desired `CUDA/cuDNN image distribution`_. **Note that other images have not yet been tested.**

.. _Docker: https://www.docker.com/
.. _Nvidia-Docker: https://github.com/NVIDIA/nvidia-docker
.. _CUDA/cuDNN image distribution: https://hub.docker.com/r/nvidia/cuda/

1. Clone the repository
"""""""""""""""""""""""
Clone or download the `project`_ from Github. ::

    $ git clone https://github.com/BerkeleyAutomation/gqcnn.git

.. _project: https://github.com/BerkeleyAutomation/gqcnn

2. Build Docker images
""""""""""""""""""""""
Change directories into the `gqcnn` repository and run the build script. ::

    $ ./scripts/docker/build-docker.sh

This will build the images `gqcnn/cpu` and `gqcnn/gpu`.

3. Run Docker image
""""""""""""""""""""
To run `gqcnn/cpu`: ::

    $ docker run --rm -it gqcnn/cpu

To run `gqcnn/gpu`: ::
    
    $ nvidia-docker run --rm -it gqcnn/gpu

Note the use of `nvidia-docker` in the latter to enable the Nvidia runtime.

You will then see an interactive shell like this: ::

    $ root@a96488604093:~/Workspace/gqcnn#

Now you can proceed to run the examples and tutorial!

