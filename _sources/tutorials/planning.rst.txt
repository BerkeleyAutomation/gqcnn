Grasp Planning
~~~~~~~~~~~~~~
Grasp planning involves searching for the grasp with the highest predicted probability of success given a point cloud.
In the `gqcnn` package this is implemented as policies that map an RGBD image to a 6-DOF grasping pose by maximizing the output of a GQ-CNN. The maximization can be implemented with iterative methods such as the `Cross Entropy Method (CEM)`_, which is used in `Dex-Net 2.0`_, `Dex-Net 2.1`_, `Dex-Net 3.0`_, `Dex-Net 4.0`_, or much faster `fully convolutional networks`, which are used in the `FC-GQ-CNN`_. 

.. _Cross Entropy Method (CEM): https://en.wikipedia.org/wiki/Cross-entropy_method
.. _Dex-Net 2.0: https://berkeleyautomation.github.io/dex-net/#dexnet_2
.. _Dex-Net 2.1: https://berkeleyautomation.github.io/dex-net/#dexnet_21
.. _Dex-Net 3.0: https://berkeleyautomation.github.io/dex-net/#dexnet_3
.. _Dex-Net 4.0: https://berkeleyautomation.github.io/dex-net/#dexnet_4
.. _FC-GQ-CNN: https://berkeleyautomation.github.io/fcgqcnn

We provide example policies in `examples/`. In particular, we provide both an example Python policy and an example ROS policy. **Note that the ROS policy requires the ROS gqcnn installation**, which can be found :ref:`here <ros-install>`. We highly recommend using the Python policy unless you need to plan grasps on a physical robot using ROS.

.. _sample-inputs:

Sample Inputs
-------------
Sample inputs from our experimental setup are provided with the repo:

#. **data/examples/clutter/phoxi/dex-net_4.0**: Set of example images from a PhotoNeo PhoXi S containing objects used in `Dex-Net 4.0`_ experiments arranged in heaps. 
#. **data/examples/clutter/phoxi/fcgqcnn**: Set of example images from a PhotoNeo PhoXi S containing objects in `FC-GQ-CNN`_ experiments arranged in heaps.
#. **data/examples/single_object/primesense/**: Set of example images from a Primesense Carmine containing objects used in `Dex-Net 2.0`_ experiments in singulation. 
#. **data/examples/clutter/primesense/**: Set of example images from a Primesense Carmine containing objects used in `Dex-Net 2.1`_ experiments arranged in heaps.

**\*\*Note that when trying these sample inputs, you must make sure that the GQ-CNN model you are using was trained for the corresponding camera and input type (singulation/clutter). See the following section for more details.\*\***

.. _pre-trained-models:

Pre-trained Models
------------------
Pre-trained parallel jaw and suction models for `Dex-Net 4.0`_ are automatically downloaded with the `gqcnn` package installation. If you do wish to try out models for older results (or for our experimental `FC-GQ-CNN`_), all pre-trained models can be downloaded with: ::

    $ ./scripts/downloads/models/download_models.sh

The models are: 

#. **GQCNN-2.0**: For `Dex-Net 2.0`_, trained on images of objects in singulation with parameters for a Primesense Carmine.
#. **GQCNN-2.1**: For `Dex-Net 2.1`_, a `Dex-Net 2.0`_ model fine-tuned on images of objects in clutter with parameters for a Primesense Carmine.
#. **GQCNN-3.0**: For `Dex-Net 3.0`_, trained on images of objects in clutter with parameters for a Primesense Carmine.
#. **GQCNN-4.0-PJ**: For `Dex-Net 4.0`_, trained on images of objects in clutter with parameters for a PhotoNeo PhoXi S.
#. **GQCNN-4.0-SUCTION**: For `Dex-Net 4.0`_, trained on images of objects in clutter with parameters for a PhotoNeo PhoXi S.
#. **FC-GQCNN-4.0-PJ**: For `FC-GQ-CNN`_, trained on images of objects in clutter with parameters for a PhotoNeo PhoXi S.
#. **FC-GQCNN-4.0-SUCTION**: For `FC-GQ-CNN`_, trained on images of objects in clutter with parameters for a PhotoNeo PhoXi S.  

**\*\*Note that GQ-CNN models are sensitive to the parameters used during dataset generation, specifically 1) Gripper geometry, an ABB YuMi Parallel Jaw Gripper for all our pre-trained models 2) Camera intrinsics, either a Primesense Carmine or PhotoNeo Phoxi S for all our pre-trained models (see above for which one) 3) Distance between camera and workspace during rendering, 50-70cm for all our pre-trained models. Thus we cannot guarantee performance of our pre-trained models on other physical setups. If you have a specific use-case in mind, please reach out to us.\*\*** We are actively researching how to generate more robust datasets that can generalize across robots, cameras, and viewpoints!

Python Policy
-------------
The example Python policy can be queried on saved images using: ::

    $ python examples/policy.py <model_name> --depth_image <depth_image_filename> --segmask <segmask_filename> --camera_intr <camera_intr_filename>

The args are:

#. **model_name**: Name of the GQ-CNN model to use.
#. **depth_image_filename**: Path to a depth image (float array in .npy format).
#. **segmask_filename**: Path to an object segmentation mask (binary image in .png format). 
#. **camera_intr_filename**: Path to a camera intrinsics file (.intr file generated with `BerkeleyAutomation's`_ `perception`_ package).

.. _BerkeleyAutomation's: https://github.com/BerkeleyAutomation
.. _perception: https://github.com/BerkeleyAutomation/perception

To evaluate the pre-trained `Dex-Net 4.0`_ **parallel jaw** network on sample images of objects in heaps run: ::

    $ python examples/policy.py GQCNN-4.0-PJ --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr

To evaluate the pre-trained `Dex-Net 4.0`_ **suction** network on sample images of objects in heaps run: ::

    $ python examples/policy.py GQCNN-4.0-SUCTION --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr

.. _ros-policy:

ROS Policy
----------
First start the grasp planning service: ::

    $ roslaunch gqcnn grasp_planning_service.launch model_name:=<model_name>

The args are:

#. **model_name**: Name of the GQ-CNN model to use. Default is **GQCNN-4.0-PJ**.
#. **model_dir**: Path to the directory where the GQ-CNN models are located. Default is **models/**. If you are using the provided **download_models.sh** script, you shouldn't have to modify this.

To start the grasp planning service with the pre-trained `Dex-Net 4.0`_ **parallel jaw** network run: ::

    $ roslaunch gqcnn grasp_planning_service.launch model_name:=GQCNN-4.0-PJ

To start the grasp planning service with the pre-trained `Dex-Net 4.0`_ **suction** network run: ::

    $ roslaunch gqcnn grasp_planning_service.launch model_name:=GQCNN-4.0-SUCTION

The example ROS policy can then be queried on saved images using: ::

    $ python examples/policy_ros.py --depth_image <depth_image_filename> --segmask <segmask_filename> --camera_intr <camera_intr_filename>

The args are:

#. **depth_image_filename**: Path to a depth image (float array in .npy format).
#. **segmask_filename**: Path to an object segmentation mask (binary image in .png format).
#. **camera_intr_filename**: Path to a camera intrinsics file (.intr file generated with `BerkeleyAutomation's`_ `perception`_ package).

To query the policy on sample images of objects in heaps run: ::

    $ python examples/policy_ros.py --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr

Usage on a Physical Robot with ROS
----------------------------------
To run the GQ-CNN on a physical robot with ROS, you will want to implement your own ROS node to query the grasp planning service similar to what `examples/policy_ros.py` does. If you are interested in replicating this functionality on your own robot, please contact Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in GQ-CNN ROS Service".

FC-GQ-CNN Policy
----------------
Our most recent research result, the `FC-GQ-CNN`_, combines novel fully convolutional network architectures with our prior work on GQ-CNNs to increase policy rate and reliability. Instead of relying on the `Cross Entropy Method (CEM)`_ to iteratively search over the policy action space for the best grasp, the FC-GQ-CNN instead densely and efficiently evaluates the entire action space in parallel. It is thus able to consider 5000x more grasps in 0.625s, resulting in a MPPH (Mean Picks Per Hour) of 296, compared to the prior 250 MPPH of `Dex-Net 4.0`_.

.. figure:: ../images/fcgqcnn_arch_diagram.png
    :width: 100 % 
    :align: center

    FC-GQ-CNN architecture.

You can download the pre-trained `FC-GQ-CNN`_ parallel jaw and suction models along with the other pre-trained models: ::
    
    $ ./scripts/downloads/models/download_models.sh

Then run the Python policy with the `\\--fully_conv` flag.

To evaluate the pre-trained `FC-GQ-CNN`_ **parallel jaw** network on sample images of objects in heaps run: ::

    $ python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_0.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr

To evaluate the pre-trained `FC-GQ-CNN`_ **suction** network on sample images of objects in heaps run: ::

    $ python examples/policy.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_0.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr

With ROS
^^^^^^^^

Review the section on using the normal ROS policy first, which can be found :ref:`here <ros-policy>`.

Add the additional arg **fully_conv:=True** when launching the grasp planning service and provide the corresponding network (**FC-GQCNN-4.0-PJ** for **parallel jaw** and **FC-GQCNN-4.0-SUCTION** for **suction**).

If you wish to test on inputs other than those provided in `data/examples/clutter/phoxi/fcgqcnn/`, you will need to edit the input height and width configuration in the appropriate `cfg/examples/<fc_gqcnn_pj.yaml or fc_gqcnn_suction.yaml>` under `["policy"]["metric"]["fully_conv_gqcnn_config"]`.
