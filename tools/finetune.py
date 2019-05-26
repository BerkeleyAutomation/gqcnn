# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Script for fine-tuning a Grasp Quality Convolutional Neural Network (GQ-CNN).

Author
------
Vishal Satish & Jeff Mahler
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

from autolab_core import YamlConfig, Logger
import autolab_core.utils as utils
from gqcnn import get_gqcnn_model, get_gqcnn_trainer
from gqcnn import utils as gqcnn_utils

# Setup logger.
logger = Logger.get_logger("tools/finetune.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description=(
        "Fine-Tune a pre-trained Grasp Quality Convolutional Neural Network"
        " with TensorFlow"))
    parser.add_argument(
        "dataset_dir",
        type=str,
        default=None,
        help="path to the dataset to use for training and validation")
    parser.add_argument("base_model_name",
                        type=str,
                        default=None,
                        help="name of the pre-trained model to fine-tune")
    parser.add_argument("--split_name",
                        type=str,
                        default="image_wise",
                        help="name of the split to train on")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="path to store the model")
    parser.add_argument("--tensorboard_port",
                        type=int,
                        default=None,
                        help="port to launch tensorboard on")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="random seed for training")
    parser.add_argument("--config_filename",
                        type=str,
                        default=None,
                        help="path to the configuration file to use")
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="path to the pre-trained model to fine-tune")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="name for the trained model")
    parser.add_argument(
        "--save_datetime",
        type=bool,
        default=False,
        help=("whether or not to save a model with the date and time of"
              " training"))
    parser.add_argument("--backend",
                        type=str,
                        default="tf",
                        help="the deep learning framework to use")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    base_model_name = args.base_model_name
    split_name = args.split_name
    output_dir = args.output_dir
    tensorboard_port = args.tensorboard_port
    seed = args.seed
    config_filename = args.config_filename
    model_dir = args.model_dir
    name = args.name
    save_datetime = args.save_datetime
    backend = args.backend

    # Set default output dir.
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../models")

    # Set default config filename.
    if config_filename is None:
        config_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "cfg/finetune.yaml")

    # Set default model dir.
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")

    # Turn relative paths absolute.
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.getcwd(), dataset_dir)
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(os.getcwd(), model_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # Create full path to the pre-trained model.
    model_dir = os.path.join(model_dir, base_model_name)

    # Create output dir if necessary.
    utils.mkdir_safe(output_dir)

    # Open train config.
    train_config = YamlConfig(config_filename)
    if seed is not None:
        train_config["seed"] = seed
        train_config["gqcnn"]["seed"] = seed
    if tensorboard_port is not None:
        train_config["tensorboard_port"] = tensorboard_port
    gqcnn_params = train_config["gqcnn"]

    # Create a unique output folder based on the date and time.
    if save_datetime:
        # Create output dir.
        unique_name = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(output_dir, unique_name)
        utils.mkdir_safe(output_dir)

    # Set visible devices.
    if "gpu_list" in train_config:
        gqcnn_utils.set_cuda_visible_devices(train_config["gpu_list"])

    # Fine-tune the network.
    start_time = time.time()
    gqcnn = get_gqcnn_model(backend)(gqcnn_params)
    trainer = get_gqcnn_trainer(backend)(gqcnn,
                                         dataset_dir,
                                         split_name,
                                         output_dir,
                                         train_config,
                                         name=name)
    trainer.finetune(model_dir)
    logger.info("Total Fine-tuning Time: " +
                str(utils.get_elapsed_time(time.time() - start_time)))
