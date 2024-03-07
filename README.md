## Note: Python 2.x support has officially been dropped.
## Note: Unofficial Python 3.11 support is at the end of this README.

# Berkeley AUTOLAB's GQCNN Package
<p>
   <a href="https://travis-ci.org/BerkeleyAutomation/gqcnn/">
       <img alt="Build Status" src="https://travis-ci.org/BerkeleyAutomation/gqcnn.svg?branch=master">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/releases/latest">
       <img alt="Release" src="https://img.shields.io/github/release/BerkeleyAutomation/gqcnn.svg?style=flat">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/blob/master/LICENSE">
       <img alt="Software License" src="https://img.shields.io/badge/license-REGENTS-brightgreen.svg">
   </a>
   <a>
       <img alt="Python 3 Versions" src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-yellow.svg">
   </a>
</p>

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs). It is part of the ongoing [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project created and maintained by the [AUTOLAB](https://autolab.berkeley.edu) at UC Berkeley.

## Installation and Usage
Please see the [docs](https://berkeleyautomation.github.io/gqcnn/) for installation and usage instructions.

## Citation
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).

*** 

# Updated installation of GQCNN

This repo has updated the codebase of GQCNN to newer versions and tested on Ubuntu20.04:

python 3.11

CUDA 12.2

tensorflow 2.15.0.post1

to set up the environment, follow the README in [docker/updated_gpu](docker/updated_gpu/README.md)

## Download the pretrained models for Dexnet:
The link provided in the official website is obsolete, use the link below:

https://drive.google.com/file/d/1fbC0sGtVEUmAy7WPT_J-50IuIInMR9oO/view

or you can download it from huggingface using:
```bash
wget --content-disposition https://huggingface.co/WoodenHeart0214/gqcnn/resolve/main/model_zoo.zip?download=true && \
    unzip model_zoo.zip && \
    rm model_zoo.zip && \
    mv model_zoo models
```

## Test Command
Check the Makefile

`make test_single_object`

