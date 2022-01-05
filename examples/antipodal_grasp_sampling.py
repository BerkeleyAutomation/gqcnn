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

Demonstrates image-based antipodal grasp candidate sampling, which is used in
the Cross Entropy Method (CEM)-based GQ-CNN grasping policy. Samples images
from a BerkeleyAutomation/perception `RgbdSensor`.

Author
------
Jeff Mahler
"""
import argparse
import os

from autolab_core import RigidTransform, YamlConfig, Logger, RgbdImage
from perception import RgbdSensorFactory
from visualization import Visualizer2D as vis

from gqcnn.grasping import AntipodalDepthImageGraspSampler

# Set up logger.
logger = Logger.get_logger("examples/antipodal_grasp_sampling.py")

if __name__ == "__main__":
    # Parse args.
    parser = argparse.ArgumentParser(description=(
        "Sample antipodal grasps on a depth image from an RgbdSensor"))
    parser.add_argument("--config_filename",
                        type=str,
                        default="cfg/examples/antipodal_grasp_sampling.yaml",
                        help="path to configuration file to use")
    args = parser.parse_args()
    config_filename = args.config_filename

    # Read config.
    config = YamlConfig(config_filename)
    sensor_type = config["sensor"]["type"]
    sensor_frame = config["sensor"]["frame"]
    num_grasp_samples = config["num_grasp_samples"]
    gripper_width = config["gripper_width"]
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    visualize_sampling = config["visualize_sampling"]
    sample_config = config["sampling"]

    # Read camera calib.
    tf_filename = "%s_to_world.tf" % (sensor_frame)
    T_camera_world = RigidTransform.load(
        os.path.join(config["calib_dir"], sensor_frame, tf_filename))

    # Setup sensor.
    sensor = RgbdSensorFactory.sensor(sensor_type, config["sensor"])
    sensor.start()
    camera_intr = sensor.ir_intrinsics

    # Read images.
    color_im, depth_im, _ = sensor.frames()
    color_im = color_im.inpaint(rescale_factor=inpaint_rescale_factor)
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)

    # Sample grasps.
    grasp_sampler = AntipodalDepthImageGraspSampler(sample_config,
                                                    gripper_width)
    grasps = grasp_sampler.sample(rgbd_im,
                                  camera_intr,
                                  num_grasp_samples,
                                  segmask=None,
                                  seed=100,
                                  visualize=visualize_sampling)

    # Visualize.
    vis.figure()
    vis.imshow(depth_im)
    for grasp in grasps:
        vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
    vis.title("Sampled grasps")
    vis.show()
