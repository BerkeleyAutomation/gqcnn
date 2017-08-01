"""
Example file for loading a policy test case.
Author: Jeff Mahler
"""
import argparse
import cPickle as pkl
import logging
import numpy as np
import os
import sys

from perception import RgbdImage, CameraIntrinsics
from visualization import Visualizer2D as vis
from gqcnn import RgbdImageState, Grasp2D, SuctionPoint2D

if __name__ == '__main__':
    # setup logger
    logging.getLogger().setLevel(logging.INFO)
    
    # parse args
    parser = argparse.ArgumentParser(description='Load a test case to facilitate debugging of new image-based grasp metrics')
    parser.add_argument('test_case_dir', type=str, default=None, help='path to the test case directory')
    args = parser.parse_args()
    test_case_dir = args.test_case_dir
    
    # load the test case
    if not os.path.exists(test_case_dir):
        raise ValueError('Test case %s does not exist!')

    candidate_actions_filename = os.path.join(test_case_dir, 'actions.pkl')
    candidates = pkl.load(open(candidate_actions_filename, 'rb'))
    image_state_filename = os.path.join(test_case_dir, 'state.pkl')
    state = pkl.load(open(image_state_filename, 'rb'))

    # visualize first image for an example
    seg_depth = state.rgbd_im.depth.mask_binary(state.segmask)
    vis.figure()
    vis.subplot(1,4,1)
    vis.imshow(state.rgbd_im.depth)
    vis.title('Depth')
    vis.subplot(1,4,2)
    vis.imshow(state.segmask)
    vis.title('Segmask')
    vis.subplot(1,4,3)
    vis.imshow(state.rgbd_im.color)
    vis.title('Segmented color')
    vis.subplot(1,4,4)
    vis.imshow(seg_depth)
    vis.title('Segmented depth')
    vis.show()

    # TODO: implement "quality_function"

    # call quality function for each candidate
    for candidate in candidates:
        quality = quality_function(candidate, state)

    # TODO: visualize grasps by quality 
    
