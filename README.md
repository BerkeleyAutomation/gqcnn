# Berkeley AUTOLAB's GQCNN Package

Documentation: https://berkeleyautomation.github.io/gqcnn

## Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs).

This package is part of the Dexterity Network (Dex-Net) project: https://berkeleyautomation.github.io/dex-net

Created and maintained by the AUTOLAB at UC Berkeley: https://autolab.berkeley.edu

## Installation
See the website at [https://berkeleyautomation.github.io/gqcnn](https://berkeleyautomation.github.io/gqcnn) for installation instructions and API Documentation.

## Datasets
Our GQ-CNN training datasets and trained models can be downloaded from [this link](https://berkeley.box.com/s/p85ov4dx7vbq6y1l02gzrnsexg6yyayb).

## ROS Service
We developed a [ROS service for grasp planning with GQ-CNNs](https://github.com/BerkeleyAutomation/gqcnn/blob/master/ros_nodes/grasp_planner_node.py).
The service takes as input a color image, depth image, camera info topic, and bounding box for the object in image space, and returns a parallel-jaw gripper pose relative to the camer along with a predicted probability of success.
This has been tested on our setup with ROS Jade on Ubuntu 14.04

To illustrate using our ROS service, we've shared [the ROS node that we use to plan grasps for and control an ABB YuMi on our local setup](https://github.com/BerkeleyAutomation/gqcnn/blob/master/ros_nodes/yumi_control_node.py).
This file should be considered READ-ONLY as it uses parameters specific to our setup.
If you have interest in replicating this functionality on your own robot, please contact Jeff Mahler (jmahler@berkeley.edu) with the subject line: "Interested in GQ-CNN ROS Service".

