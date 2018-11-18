# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Displays robust grasps planned using a GQ-CNN-based policy on a set of saved RGB-D images.
The default configuration is cfg/examples/policy.yaml.

Author
------
Jeff Mahler
"""
import argparse
import cPickle as pkl
import os
import time

import numpy as np

from autolab_core import RigidTransform, YamlConfig, Logger
from perception import RgbdImage, BinaryImage, ColorImage, DepthImage, CameraIntrinsics
from visualization import Visualizer2D as vis
from gqcnn import UniformRandomGraspingPolicy, RgbdImageState

# set up logger
logger = Logger.get_logger('tools/sample_antipodal_grasp.py')

if __name__ == '__main__':
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
    _, ext = os.path.splitext(depth_im_filename)
    if ext.lower() == '.pkl':
        depth_im_arr = pkl.load(open(depth_im_filename, 'r'))
    else:
        depth_im_arr = np.load(depth_im_filename)
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
    logger.info('Planning took %.3f sec' %(time.time() - policy_start))

    # vis final grasp
    if policy_config['vis']['final_grasp']:
        vis.figure(size=(10,10))
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config['vis']['vmin'],
                   vmax=policy_config['vis']['vmax'])
        vis.grasp(action.grasp, scale=1.5, show_center=False, show_axis=True)
        vis.title('Antipodal grasp')
        vis.show()

