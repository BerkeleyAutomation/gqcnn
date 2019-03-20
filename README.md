# Berkeley AUTOLAB's GQCNN Package

Documentation: https://berkeleyautomation.github.io/gqcnn

## Version 1.0
We're excited to announce the version 1.0 release, which brings the GQ-CNN package up to speed with recent research in the [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project.
Version 1.0 introduces support for:
**[Dex-Net 4.0](https://goldberg.berkeley.edu/pubs/Ambidextrous-Grasping-Science-Robotics-Jan-2019.pdf):** Extends Dex-Net to multiple grippers working together in composite policies.
**[Fully Convolutional GQ-CNN](https://goldberg.berkeley.edu/pubs/Fully-Convolutional-Dex-Net-RA-Letters-Journal-Feb-2019.pdf):** Extends Dex-Net to fully convolutional architectures that efficiently evaluate millions of grasps faster than prior GQ-CNNs.

Version 1.0 also provide a more robust ROS grasp planning service that includes built-in pre-processing.

**What’s new**
* Support for training GQ-CNNs on Dex-Net 4.0 parallel jaw and suction datasets
* Support for faster Fully Convolutional GQ-CNNs (FC-GQ-CNNs)
* More robust ROS policy with integrated pre-processing
* Improved interface for training GQ-CNNs and evaluating policies
* Faster training due to improved parallelism in data prefetch/pre-processing
* Easy-to-use shell scripts for replication of published results from Dex-Net 2.0/2.1/3.0/4.0/FC-GQ-CNN

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs).

This package is part of the Dexterity Network (Dex-Net) project: https://berkeleyautomation.github.io/dex-net

Created and maintained by the AUTOLAB at UC Berkeley: https://autolab.berkeley.edu

## Installation
See the website at [https://berkeleyautomation.github.io/gqcnn](https://berkeleyautomation.github.io/gqcnn) for installation instructions and API Documentation.

## Usage
As of Feb. 1, 2018, the code is licensed according to the UC Berkeley Copyright and Disclaimer Notice.
The code is available for educational, research, and not-for-profit purposes (for full details, see LICENSE).
If you use this code in a publication, please cite:

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics." Robotics: Science and Systems (2017). Boston, MA.

## Datasets
Our GQ-CNN training datasets and trained models can be downloaded from [this link](https://berkeley.box.com/s/p85ov4dx7vbq6y1l02gzrnsexg6yyayb).

## ROS Service
We developed a [ROS service for grasp planning with GQ-CNNs](https://github.com/BerkeleyAutomation/gqcnn/blob/master/ros_nodes/grasp_planner_node.py).
The service takes as input a color image, depth image, camera info topic, and bounding box for the object in image space, and returns a parallel-jaw gripper pose relative to the camer along with a predicted probability of success.
This has been tested on our setup with ROS Jade on Ubuntu 14.04

To illustrate using our ROS service, we've shared [the ROS node that we use to plan grasps for and control an ABB YuMi on our local setup](https://github.com/BerkeleyAutomation/gqcnn/blob/master/ros_nodes/yumi_control_node.py).
This file should be considered READ-ONLY as it uses parameters specific to our setup.
If you have interest in replicating this functionality on your own robot, please contact Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in GQ-CNN ROS Service".

