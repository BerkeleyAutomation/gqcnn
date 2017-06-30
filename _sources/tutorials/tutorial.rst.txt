Overview
~~~~~~~~
The tutorial covers the two main use cases of the `gqcnn` package:

1) :ref:`training` GQ-CNNs on offline datasets of point clouds, grasps, and grasp success metrics.
2) :ref:`grasp planning` on RGBD images using trained GQ-CNNs.

Click on the links or scroll down to get started!

Running Python Scripts
----------------------
All `gqcnn` Python scripts are designed to be run from the top-level directory of your gqcnn repo by default.
This is because every script takes in a YAML file specifying parameters for the algorithm, and this YAML file is stored relative to the repository root directory.

We recommend that you run all scripts using this paradigm::

  cd /path/to/your/gqcnn
  python examples/policy.py

If you see an error like::
 
  Traceback (most recent call last):
    File "policy.py", line 30, in <module>
      config = YamlConfig(config_filename)
    File "/home/autolab/Workspace/jeff_working/catkin_ws/src/core/./autolab_core/yaml_config.py", line 28, in __init__
      self._load_config(filename)
    File "/home/autolab/Workspace/jeff_working/catkin_ws/src/core/./autolab_core/yaml_config.py", line 69, in _load_config
      fh = open(filename, 'r')
  IOError: [Errno 2] No such file or directory: 'cfg/examples/policy.yaml'

then you should specify the `--config_filename` flag to point to the appropriate configuration file, like this::

  python policy.py --config_filename /path/to/your/gqcnn/cfg/examples/policy.yaml

.. _training:
.. include:: training.rst

.. _grasp planning:
.. include:: planning.rst

