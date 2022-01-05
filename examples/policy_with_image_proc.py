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

Displays robust grasps planned using a GQ-CNN-based policy on a set of saved
RGB-D images. The default configuration is cfg/examples/policy.yaml.

Author
------
Jeff Mahler
"""
import argparse
import os
import time

import numpy as np
import pcl
import skimage

from autolab_core import (PointCloud, RigidTransform, YamlConfig,
                          Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage,
                          SegmentationImage)
from visualization import Visualizer2D as vis

from gqcnn import (RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy,
                   RgbdImageState)

CLUSTER_TOL = 0.0015
MIN_CLUSTER_SIZE = 100
MAX_CLUSTER_SIZE = 1000000

# Set up logger.
logger = Logger.get_logger("tools/policy_with_image_proc.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Run a grasping policy on an example image")
    parser.add_argument(
        "--depth_image",
        type=str,
        default=None,
        help="path to a test depth image stored as a .npy file")
    parser.add_argument("--segmask",
                        type=str,
                        default=None,
                        help="path to an optional segmask to use")
    parser.add_argument("--camera_intrinsics",
                        type=str,
                        default=None,
                        help="path to the camera intrinsics")
    parser.add_argument("--camera_pose",
                        type=str,
                        default=None,
                        help="path to the camera pose")
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="path to a trained model to run")
    parser.add_argument("--config_filename",
                        type=str,
                        default=None,
                        help="path to configuration file to use")
    args = parser.parse_args()
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intrinsics
    camera_pose_filename = args.camera_pose
    model_dir = args.model_dir
    config_filename = args.config_filename

    if depth_im_filename is None:
        depth_im_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/examples/single_object/depth_0.npy")
    if camera_intr_filename is None:
        camera_intr_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/calib/primesense.intr")
    if camera_pose_filename is None:
        camera_pose_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "data/calib/primesense.tf")
    if config_filename is None:
        config_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            "cfg/examples/replication/dex-net_2.0.yaml")

    # Read config.
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if model_dir is not None:
        policy_config["metric"]["gqcnn_model"] = model_dir
    if "gqcnn_model" in policy_config["metric"] and not os.path.isabs(
            policy_config["metric"]["gqcnn_model"]):
        policy_config["metric"]["gqcnn_model"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..",
            policy_config["metric"]["gqcnn_model"])

    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
    T_camera_world = RigidTransform.load(camera_pose_filename)

    # Read images.
    depth_data = np.load(depth_im_filename)
    depth_data = depth_data.astype(np.float32) / 1000.0
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Optionally read a segmask.
    mask = np.zeros((camera_intr.height, camera_intr.width, 1), dtype=np.uint8)
    c = np.array([165, 460, 500, 135])
    r = np.array([165, 165, 370, 370])
    rr, cc = skimage.draw.polygon(r, c, shape=mask.shape)
    mask[rr, cc, 0] = 255
    segmask = BinaryImage(mask)
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename)
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)

    # Create new cloud.
    point_cloud = camera_intr.deproject(depth_im)
    point_cloud.remove_zero_points()
    pcl_cloud = pcl.PointCloud(point_cloud.data.T.astype(np.float32))
    tree = pcl_cloud.make_kdtree()

    # Find large clusters (likely to be real objects instead of noise).
    ec = pcl_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(CLUSTER_TOL)
    ec.set_MinClusterSize(MIN_CLUSTER_SIZE)
    ec.set_MaxClusterSize(MAX_CLUSTER_SIZE)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    num_clusters = len(cluster_indices)

    obj_segmask_data = np.zeros(depth_im.shape)

    # Read out all points in large clusters.
    cur_i = 0
    for j, indices in enumerate(cluster_indices):
        num_points = len(indices)
        points = np.zeros([3, num_points])

        for i, index in enumerate(indices):
            points[0, i] = pcl_cloud[index][0]
            points[1, i] = pcl_cloud[index][1]
            points[2, i] = pcl_cloud[index][2]

        segment = PointCloud(points, frame=camera_intr.frame)
        depth_segment = camera_intr.project_to_image(segment)
        obj_segmask_data[depth_segment.data > 0] = j + 1
    obj_segmask = SegmentationImage(obj_segmask_data.astype(np.uint8))
    obj_segmask = obj_segmask.mask_binary(segmask)

    # Inpaint.
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

    if "input_images" in policy_config["vis"] and policy_config["vis"][
            "input_images"]:
        vis.figure(size=(10, 10))
        num_plot = 3
        vis.subplot(1, num_plot, 1)
        vis.imshow(depth_im)
        vis.subplot(1, num_plot, 2)
        vis.imshow(segmask)
        vis.subplot(1, num_plot, 3)
        vis.imshow(obj_segmask)
        vis.show()

        from visualization import Visualizer3D as vis3d
        point_cloud = camera_intr.deproject(depth_im)
        vis3d.figure()
        vis3d.points(point_cloud,
                     subsample=3,
                     random=True,
                     color=(0, 0, 1),
                     scale=0.001)
        vis3d.pose(RigidTransform())
        vis3d.pose(T_camera_world.inverse())
        vis3d.show()

    # Create state.
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # Init policy.
    policy_type = "cem"
    if "type" in policy_config:
        policy_type = policy_config["type"]
    if policy_type == "ranking":
        policy = RobustGraspingPolicy(policy_config)
    else:
        policy = CrossEntropyRobustGraspingPolicy(policy_config)
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))

    # Vis final grasp.
    if policy_config["vis"]["final_grasp"]:
        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config["vis"]["vmin"],
                   vmax=policy_config["vis"]["vmax"])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp on depth (Q=%.3f)" % (action.q_value))
        vis.show()

    # Get grasp pose.
    T_grasp_camera = action.grasp.pose(
        grasp_approach_dir=-T_camera_world.inverse().z_axis)
    grasp_pose_msg = T_grasp_camera.pose_msg

    # TODO: Control to reach the grasp pose.
