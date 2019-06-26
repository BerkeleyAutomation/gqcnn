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
The default configuration for the standard GQ-CNN policy is cfg/examples/policy.yaml. The default configuration for the Fully-Convolutional GQ-CNN policy is cfg/examples/fc_policy.yaml.

Author
------
Jeff Mahler and Vishal Satish
"""
import argparse
import json
import os
import time
import glob

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from autolab_core import RigidTransform, YamlConfig, Logger
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis

from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction
from gqcnn.utils import GripperMode, NoValidGraspsException

# set up logger
logger = Logger.get_logger('examples/policy.py')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run a grasping policy on an example image')
    parser.add_argument('--depth_image', type=str, default=None, help='path to a test depth image stored as a .npy file')
    parser.add_argument('--segmask', type=str, default=None, help='path to an optional segmask to use')
    parser.add_argument('--camera_intr', type=str, default=None, help='path to the camera intrinsics')
    parser.add_argument('--model_dir', type=str, default=None, help='path to the folder in which the model is stored')
    parser.add_argument('--config_filename', type=str, default=None, help='path to configuration file to use')
    parser.add_argument('--fully_conv', action='store_true', help='run Fully-Convolutional GQ-CNN policy instead of standard GQ-CNN policy')
    args = parser.parse_args()
    #depth_im_filename = args.depth_image
    print("starting")
    for iter in xrange(17):
        depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            '..',
            'data/examples/mobile/depth_' + str(iter) + '.npy')
        rgb_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
            '..',
            'data/examples/mobile/rgb_' + str(iter) + '.png')
        segmask_filename = args.segmask
        camera_intr_filename = args.camera_intr
        model_dir = args.model_dir
        config_filename = args.config_filename
        fully_conv = args.fully_conv

        assert not (fully_conv and depth_im_filename is not None and segmask_filename is None), 'Fully-Convolutional policy expects a segmask.'

        if depth_im_filename is None:
            if fully_conv:
                depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 '..',
                                                 'data/examples/clutter/primesense/depth_0.npy')
            else:
                depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 '..',
                                                 'data/examples/single_object/primesense/depth_0.npy')
        if fully_conv and segmask_filename is None:
            segmask_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '..',
                                            'data/examples/clutter/primesense/segmask_0.png')
        if camera_intr_filename is None:
            camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '..',
                                                'data/calib/primesense/primesense.intr')    
        if config_filename is None:
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/fc_policy.yaml')
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/policy.yaml')

        # read config
        config = YamlConfig(config_filename)
        inpaint_rescale_factor = config['inpaint_rescale_factor']
        policy_config = config['policy']

        # set model if provided and make relative paths absolute
        if model_dir is not None:
            policy_config['metric']['gqcnn_model'] = model_dir
        if 'gqcnn_model' in policy_config['metric'].keys() and not os.path.isabs(policy_config['metric']['gqcnn_model']):
            policy_config['metric']['gqcnn_model'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                  '..',
                                                                  policy_config['metric']['gqcnn_model'])

        # setup sensor
        camera_intr = CameraIntrinsics.load(camera_intr_filename)
            
        # read images
        depth_data = np.load(depth_im_filename)
        # crop image to make gqcnn focus on target zone of HSR
        # desired cropped image: depth_data[190:330, 60:170]
        depth_data[:, :245] = 0
        depth_data[:, 420:] = 0
        depth_data[:150, :] = 0
        depth_data[310:, :] = 0
        depth_im = DepthImage(depth_data, frame=camera_intr.frame)
        color_data = Image.open(rgb_im_filename)
        color_im = ColorImage(np.array(color_data),
                                frame=camera_intr.frame)

        
        # optionally read a segmask
        segmask = None
        if segmask_filename is not None:
            segmask = BinaryImage.open(segmask_filename)
        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        if segmask is None:
            segmask = valid_px_mask
        else:
            segmask = segmask.mask_binary(valid_px_mask)
        
        # inpaint
        depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
            
        if 'input_images' in policy_config['vis'].keys() and policy_config['vis']['input_images']:
            vis.figure(size=(10,10))
            num_plot = 1
            if segmask is not None:
                num_plot = 2
            vis.subplot(1,num_plot,1)
            vis.imshow(depth_im)
            if segmask is not None:
                vis.subplot(1,num_plot,2)
                vis.imshow(segmask)
            vis.show()
            
        # create state
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

        # set input sizes for fully-convolutional policy
        if fully_conv:
            policy_config['metric']['fully_conv_gqcnn_config']['im_height'] = depth_im.shape[0]
            policy_config['metric']['fully_conv_gqcnn_config']['im_width'] = depth_im.shape[1]

        # init policy
        if fully_conv:
            #TODO: @Vishal we should really be doing this in some factory policy
            if policy_config['type'] == 'fully_conv_suction':
                policy = FullyConvolutionalGraspingPolicySuction(policy_config)
            elif policy_config['type'] == 'fully_conv_pj':
                policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
            else:
                raise ValueError('Invalid fully-convolutional policy type: {}'.format(policy_config['type']))
        else:
            policy_type = 'cem'
            if 'type' in policy_config.keys():
                policy_type = policy_config['type']
            if policy_type == 'ranking':
                policy = RobustGraspingPolicy(policy_config)
            elif policy_type == 'cem':
                policy = CrossEntropyRobustGraspingPolicy(policy_config)
            else:
                raise ValueError('Invalid policy type: {}'.format(policy_type))

        # query policy
        policy_start = time.time()
        import ipdb; ipdb.set_trace()
        action = policy(state)
        logger.info('Planning took %.3f sec' %(time.time() - policy_start))

        # vis final grasp
        if policy_config['vis']['final_grasp']:
            #vis.figure(size=(40,40))
            #vis.subplot(1,2,1)
            #vis.imshow(rgbd_im.depth,
            #           vmin=policy_config['vis']['vmin'],
            #           vmax=policy_config['vis']['vmax'])
            #vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            #vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            #vis.show()
            vis.figure(size=(10,10))
            #vis.subplot(1,2,2)
            vis.imshow(rgbd_im.color)
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
            vis.savefig(os.getcwd()+ '/data/examples/mobile/result_depth_' + str(iter) + '.png')
            vis.show()
