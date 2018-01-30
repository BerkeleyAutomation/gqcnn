"""
Displays robust grasps planned using a GQ-CNN-based policy on a set of saved RGB-D images.
The default configuration is cfg/examples/policy.yaml.

Author
------
Jeff Mahler
"""
import argparse
import cPickle as pkl
import logging
import IPython
import numpy as np
import os
import sys
import time

from autolab_core import RigidTransform, YamlConfig
from perception import RgbdImage, BinaryImage, ColorImage, DepthImage, CameraIntrinsics

from gqcnn import UniformRandomGraspingPolicy, RgbdImageState
from gqcnn import Visualizer as vis

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.DEBUG)

    # parse args
    parser = argparse.ArgumentParser(description='Run the GQ-CNN policy on an example')
    parser.add_argument('depth_im_filename', type=str, default=None, help='path to the depth image to run')
    parser.add_argument('camera_intr_filename', type=str, default=None, help='path to example camera intrinsics')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/sample_antipodal_grasp.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    depth_im_filename = args.depth_im_filename
    camera_intr_filename = args.camera_intr_filename
    config_filename = args.config_filename

    # read config
    config = YamlConfig(config_filename)
    policy_config = config['policy']

    # read image
    depth_im_arr = pkl.load(open(depth_im_filename, 'r'))
    depth_im = DepthImage(depth_im_arr)
    color_im = ColorImage(np.zeros([depth_im.height,
                                    depth_im.width,
                                    3]).astype(np.uint8))
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    segmask_arr = 255 * (depth_im_arr < 1.0)
    segmask = BinaryImage(segmask_arr.astype(np.uint8))
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
    
    # init policy
    policy = UniformRandomGraspingPolicy(policy_config)
    policy_start = time.time()
    action = policy(state)
    logging.info('Planning took %.3f sec' %(time.time() - policy_start))

    # vis final grasp
    if policy_config['vis']['final_grasp']:
        vis.figure(size=(10,10))
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config['vis']['vmin'],
                   vmax=policy_config['vis']['vmax'])
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Antipodal grasp')
        vis.show()

