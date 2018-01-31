"""
Script to run saved policy output from a user run
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

from gqcnn import RgbdImageState, ParallelJawGrasp
from gqcnn import CrossEntropyRobustGraspingPolicy
from gqcnn import Visualizer as vis

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.DEBUG)

    # parse args
    parser = argparse.ArgumentParser(description='Run a saved test case through a GQ-CNN policy. For debugging purposes only.')
    parser.add_argument('test_case_path', type=str, default=None, help='path to test case')
    parser.add_argument('--config_filename', type=str, default='cfg/tools/run_policy.yaml', help='path to configuration file to use')
    args = parser.parse_args()
    test_case_path = args.test_case_path
    config_filename = args.config_filename

    # make relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       config_filename)

    # read config
    config = YamlConfig(config_filename)
    policy_config = config['policy']
        
    # load test case
    state_path = os.path.join(test_case_path, 'state')
    action_path = os.path.join(test_case_path, 'action')
    state = RgbdImageState.load(state_path)
    action = ParallelJawGrasp.load(action_path)

    # init policy
    policy = CrossEntropyRobustGraspingPolicy(policy_config)

    if policy_config['vis']['input_images']:
        vis.figure()
        if state.segmask is None:
            vis.subplot(1,2,1)
            vis.imshow(state.rgbd_im.color)
            vis.title('COLOR')
            vis.subplot(1,2,2)
            vis.imshow(state.rgbd_im.depth)
            vis.title('DEPTH')
        else:
            vis.subplot(1,3,1)
            vis.imshow(state.rgbd_im.color)
            vis.title('COLOR')
            vis.subplot(1,3,2)
            vis.imshow(state.rgbd_im.depth)            
            vis.title('DEPTH')
            vis.subplot(1,3,3)
            vis.imshow(state.segmask)            
            vis.title('SEGMASK')
        filename = None
        if policy._logging_dir is not None:
            filename = os.path.join(policy._logging_dir, 'input_images.png')
        vis.show(filename)    

    # query policy
    policy_start = time.time()
    action = policy(state)
    logging.info('Planning took %.3f sec' %(time.time() - policy_start))

    # vis final grasp
    if policy_config['vis']['final_grasp']:
        vis.figure(size=(10,10))
        vis.subplot(1,2,1)
        vis.imshow(state.rgbd_im.color)
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on color (Q=%.3f)' %(action.q_value))
        vis.subplot(1,2,2)
        vis.imshow(state.rgbd_im.depth)
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
        filename = None
        if policy._logging_dir is not None:
            filename = os.path.join(policy._logging_dir, 'planned_grasp.png')
        vis.show(filename)
    
