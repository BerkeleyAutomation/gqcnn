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

Script to run saved policy output from user.

Author
------
Jeff Mahler
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time

import numpy as np

from autolab_core import YamlConfig, Logger
from visualization import Visualizer2D as vis2d
from gqcnn import RgbdImageState, ParallelJawGrasp
from gqcnn import CrossEntropyRobustGraspingPolicy

# Set up logger.
logger = Logger.get_logger("tools/run_policy.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(
        description=("Run a saved test case through a GQ-CNN policy. For"
                     " debugging purposes only."))
    parser.add_argument("test_case_path",
                        type=str,
                        default=None,
                        help="path to test case")
    parser.add_argument("--config_filename",
                        type=str,
                        default="cfg/tools/run_policy.yaml",
                        help="path to configuration file to use")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="directory to store output")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    test_case_path = args.test_case_path
    config_filename = args.config_filename
    output_dir = args.output_dir
    seed = args.seed

    # Make output dir.
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Make relative paths absolute.
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", config_filename)

    # Set random seed.
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Read config.
    config = YamlConfig(config_filename)
    policy_config = config["policy"]

    # Load test case.
    state_path = os.path.join(test_case_path, "state")
    action_path = os.path.join(test_case_path, "action")
    state = RgbdImageState.load(state_path)
    original_action = ParallelJawGrasp.load(action_path)

    # Init policy.
    policy = CrossEntropyRobustGraspingPolicy(policy_config)

    if policy_config["vis"]["input_images"]:
        vis2d.figure()
        if state.segmask is None:
            vis2d.subplot(1, 2, 1)
            vis2d.imshow(state.rgbd_im.color)
            vis2d.title("COLOR")
            vis2d.subplot(1, 2, 2)
            vis2d.imshow(state.rgbd_im.depth,
                         vmin=policy_config["vis"]["vmin"],
                         vmax=policy_config["vis"]["vmax"])
            vis2d.title("DEPTH")
        else:
            vis2d.subplot(1, 3, 1)
            vis2d.imshow(state.rgbd_im.color)
            vis2d.title("COLOR")
            vis2d.subplot(1, 3, 2)
            vis2d.imshow(state.rgbd_im.depth,
                         vmin=policy_config["vis"]["vmin"],
                         vmax=policy_config["vis"]["vmax"])
            vis2d.title("DEPTH")
            vis2d.subplot(1, 3, 3)
            vis2d.imshow(state.segmask)
            vis2d.title("SEGMASK")
        filename = None
        if output_dir is not None:
            filename = os.path.join(output_dir, "input_images.png")
        vis2d.show(filename)

    # Query policy.
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))

    # Vis final grasp.
    if policy_config["vis"]["final_grasp"]:
        vis2d.figure(size=(10, 10))
        vis2d.subplot(1, 2, 1)
        vis2d.imshow(state.rgbd_im.depth,
                     vmin=policy_config["vis"]["vmin"],
                     vmax=policy_config["vis"]["vmax"])
        vis2d.grasp(original_action.grasp,
                    scale=policy_config["vis"]["grasp_scale"],
                    show_center=False,
                    show_axis=True,
                    color="r")
        vis2d.title("Original (Q=%.3f) (Z=%.3f)" %
                    (original_action.q_value, original_action.grasp.depth))
        vis2d.subplot(1, 2, 2)
        vis2d.imshow(state.rgbd_im.depth,
                     vmin=policy_config["vis"]["vmin"],
                     vmax=policy_config["vis"]["vmax"])
        vis2d.grasp(action.grasp,
                    scale=policy_config["vis"]["grasp_scale"],
                    show_center=False,
                    show_axis=True,
                    color="r")
        vis2d.title("New (Q=%.3f) (Z=%.3f)" %
                    (action.q_value, action.grasp.depth))
        filename = None
        if output_dir is not None:
            filename = os.path.join(output_dir, "planned_grasp.png")
        vis2d.show(filename)
