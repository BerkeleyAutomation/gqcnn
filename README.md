# Berkeley AUTOLAB's GQCNN Package

Documentation: https://berkeleyautomation.github.io/gqcnn.

## Version 1.0 Release
We're excited to announce version 1.0, which brings the GQ-CNN package up to date with recent research in the [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project.
Version 1.0 introduces support for:

* **[Dex-Net 4.0](https://berkeleyautomation.github.io/dex-net/#dexnet_4):** Composite policies that decide whether to use a suction cup or parallel-jaw gripper.
* **[Fully Convolutional GQ-CNNs](https://berkeleyautomation.github.io/fcgqcnn):** Fully convolutional architectures that efficiently evaluate millions of grasps faster than prior GQ-CNNs.

Version 1.0 also provide a more robust ROS grasp planning service that includes built-in pre-processing.

### New Features
* Support for training GQ-CNNs on Dex-Net 4.0 parallel jaw and suction datasets.
* Support for faster Fully Convolutional GQ-CNNs (FC-GQ-CNNs).
* More robust ROS policy with integrated pre-processing.
* Improved interface for training GQ-CNNs and evaluating policies.
* Faster training due to improved parallelism in data prefetch/pre-processing.
* Easy-to-use shell scripts for replication of published results from Dex-Net {[2.0](https://berkeleyautomation.github.io/dex-net/#dexnet_2),[2.1](https://berkeleyautomation.github.io/dex-net/#dexnet_21),[3.0](https://berkeleyautomation.github.io/dex-net/#dexnet_3),[4.0](https://berkeleyautomation.github.io/dex-net/#dexnet_4)} and [FC-GQ-CNN](https://berkeleyautomation.github.io/fcgqcnn).

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs).

This package is part of the Dexterity Network (Dex-Net) project: https://berkeleyautomation.github.io/dex-net

Created and maintained by the AUTOLAB at UC Berkeley: https://autolab.berkeley.edu

## Installation
See the website at [https://berkeleyautomation.github.io/gqcnn](https://berkeleyautomation.github.io/gqcnn) for installation instructions and API Documentation.

## Datasets
Our GQ-CNN training datasets and trained models can be downloaded from [this link](https://berkeley.box.com/s/p85ov4dx7vbq6y1l02gzrnsexg6yyayb).

## Usage
As of Feb. 1, 2018, the code is licensed according to the UC Berkeley Copyright and Disclaimer Notice.
The code is available for educational, research, and not-for-profit purposes (for full details, see [LICENSE](https://github.com/BerkeleyAutomation/gqcnn/blob/release-prep/LICENSE)).
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).
