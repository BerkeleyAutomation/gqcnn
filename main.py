# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np

from autolab_core import YamlConfig, Logger
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode

# Set up logger.
logger = Logger.get_logger("examples/policy.py")

if __name__ == "__main__":
    model_name = "GQCNN-4.0-PJ"
    depth_im_filename = "data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy"
    camera_intr_filename = "data/calib/phoxi/phoxi.intr"
    config_filename = "cfg/examples/gqcnn_pj.yaml"
    model_path = f"models/{model_name}"
    model_config = f"{model_path}/config.json"

    with open(model_config, "r") as f:
        model_config = json.load(f)

    gqcnn_config = model_config["gqcnn"]
    gripper_mode = gqcnn_config["gripper_mode"]

    # Read config.
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = policy_config["metric"]["gqcnn_model"]

    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    depth_data = np.load(depth_im_filename)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8), frame=camera_intr.frame)

    # Optionally read a segmask.
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    segmask = valid_px_mask

    # Inpaint.
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

    # Create state.
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    policy = CrossEntropyRobustGraspingPolicy(policy_config)

    # Query policy.
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))
    print(action.grasp.depth)
    print(action.q_value)

    # Vis final grasp.
    if policy_config["vis"]["final_grasp"]:
        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config["vis"]["vmin"],
                   vmax=policy_config["vis"]["vmax"])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
            action.grasp.depth, action.q_value))
        vis.show()
