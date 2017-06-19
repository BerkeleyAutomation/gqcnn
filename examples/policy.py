"""
Demonstrates robust grasping policies with GQ-CNNS
Author: Jeff Mahler
"""
import argparse
import logging
import IPython
import numpy as np
import os
import sys
import time

from autolab_core import RigidTransform, YamlConfig
from perception import RgbdImage, RgbdSensorFactory

from gqcnn import CrossEntropyAntipodalGraspingPolicy, RgbdImageState
from gqcnn import Visualizer as vis

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.DEBUG)

    # parse args
    parser = argparse.ArgumentParser(description='Capture a set of test images from the Kinect2')
    parser.add_argument('config_filename', type=str, default='cfg/examples/policy.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    config_filename = args.config_filename

    # read config
    config = YamlConfig(config_filename)
    sensor_type = config['sensor']['type']
    sensor_frame = config['sensor']['frame']
    inpaint_rescale_factor = config['inpaint_rescale_factor']
    policy_config = config['policy']

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
    state = RgbdImageState(rgbd_im, camera_intr)

    # init policy
    policy = CrossEntropyAntipodalGraspingPolicy(policy_config)
    policy_start = time.time()
    action = policy(state)
    logging.info('Planning took %.3f sec' %(time.time() - policy_start))

    # vis final grasp
    if policy_config['vis']['final_grasp']:
        vis.figure(size=(10,10))
        vis.subplot(1,2,1)
        vis.imshow(rgbd_im.color)
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on color (Q=%.3f)' %(action.q_value))
        vis.subplot(1,2,2)
        vis.imshow(rgbd_im.depth)
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
        vis.show()

