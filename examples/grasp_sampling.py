"""
Demonstrates image-based grasp candidate sampling, which is used in the
GQ-CNN-based grasping policy
Author: Jeff Mahler
"""
import argparse
import logging
import IPython
import numpy as np
import os
import sys

from core import RigidTransform, YamlConfig
from perception import RgbdImage, RgbdSensorFactory

from gqcnn.image_grasp_sampler import AntipodalDepthImageGraspSampler
from gqcnn import Visualizer as vis

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.DEBUG)

    # parse args
    parser = argparse.ArgumentParser(description='Capture a set of test images from the Kinect2')
    parser.add_argument('--config_filename', type=str, default='cfg/examples/grasp_sampling.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename

    # read config
    config = YamlConfig(config_filename)
    sensor_type = config['sensor']['type']
    sensor_frame = config['sensor']['frame']
    num_grasp_samples = config['num_grasp_samples']
    gripper_width = config['gripper_width']
    inpaint_rescale_factor = config['inpaint_rescale_factor']
    visualize_sampling = config['visualize_sampling']
    sample_config = config['sampling']

    # read camera calib
    tf_filename = '%s_to_world.tf' %(sensor_frame)
    T_camera_world = RigidTransform.load(os.path.join(config['calib_dir'], sensor_frame, tf_filename))

    # setup sensor
    sensor = RgbdSensorFactory.sensor(sensor_type, config['sensor'])
    sensor.start()
    camera_intr = sensor.ir_intrinsics

    # read images
    color_im, depth_im, _ = sensor.frames()
    color_im = color_im.inpaint(rescale_factor=inpaint_rescale_factor)
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)

    # sample grasps
    grasp_sampler = AntipodalDepthImageGraspSampler(sample_config, gripper_width)
    grasps = grasp_sampler.sample(rgbd_im, camera_intr, num_grasp_samples, segmask=None,
                                  seed=100, visualize=visualize_sampling)

    # visualize
    vis.figure()
    vis.imshow(depth_im)
    for grasp in grasps:
        vis.grasp(grasp, scale=1.5, show_center=False, show_axis=True)
    vis.title('Sampled grasps')
    vis.show()
