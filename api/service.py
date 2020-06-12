import json
import os
import time
import uuid

import numpy as np
from autolab_core import Logger, YamlConfig
from flask import request
from gqcnn.grasping import CrossEntropyRobustGraspingPolicy, RgbdImageState
from perception import CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis

from api.exceptions import FileInputException, FileNotFoundException

# Set up logger.
logger = Logger.get_logger("examples/policy.py")


def heathcheck_service():
    return {"version": f"GQCNN App v0.1-{str(uuid.uuid4())}"}


def grasp_planning_service(is_vis=False):
    model_name = "GQCNN-4.0-PJ"
    # depth_im_filename = "data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy"
    camera_intr_filename = "data/calib/phoxi/phoxi.intr"
    config_filename = "cfg/examples/gqcnn_pj.yaml"
    model_path = f"models/{model_name}"
    model_config = f"{model_path}/config.json"

    if "file" in request.files:
        try:
            depth_data = np.load(request.files["file"])
        except:
            raise FileInputException()
    else:
        raise FileNotFoundException()

    with open(model_config, "r") as f:
        model_config = json.load(f)

    # Read config.
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = policy_config["metric"][
                "gqcnn_model"
            ]

    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    # depth_data = np.load(depth_im_filename)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(
        np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
        frame=camera_intr.frame,
    )

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
    time_lapse = time.time() - policy_start
    logger.info("Planning took %.3f sec" % (time_lapse))

    if is_vis:
        # Vis final grasp.
        if policy_config["vis"]["final_grasp"]:
            vis.figure(size=(10, 10))
            vis.imshow(
                rgbd_im.depth,
                vmin=policy_config["vis"]["vmin"],
                vmax=policy_config["vis"]["vmax"],
            )
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title(
                "Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                    action.grasp.depth, action.q_value
                )
            )
            vis.show()

    # https://berkeleyautomation.github.io/gqcnn/api/policies.html#grasp2d
    print(action.grasp.center)
    return {
        "time_lapse": time_lapse,
        "q_value": action.q_value,
        "angle": action.grasp.angle,
        "depth": action.grasp.depth,
        "width": action.grasp.width,
        "approach_angle": action.grasp.approach_angle,
    }
    return {"status": "ok"}
