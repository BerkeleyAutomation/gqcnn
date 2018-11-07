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
import os
import sys
import time

import numpy as np
import rosgraph.roslogging as rl
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

from autolab_core import Point, RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis

from gqcnn import CrossEntropyRobustGraspingPolicy, RgbdImageState, Grasp2D, SuctionPoint2D
from gqcnn.grasping.policy import GraspAction
from gqcnn.msg import GQCNNGrasp, BoundingBox
from gqcnn.srv import GQCNNGraspPlanner

# set up logger
logger = Logger.get_logger('examples/grasping_client.py')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run a grasping policy on an example image')
    parser.add_argument('--color_image', type=str, default=None, help='path to a test color image stored as a .png file')
    parser.add_argument('--depth_image', type=str, default=None, help='path to a test depth image stored as a .npy file')
    parser.add_argument('--segmask', type=str, default=None, help='path to an optional segmask to use')
    parser.add_argument('--camera_intrinsics', type=str, default=None, help='path to the camera intrinsics')
    parser.add_argument('--gripper_width', type=float, default=0.05, help='width of the gripper to plan for')
    parser.add_argument('--config_filename', type=str, default=None, help='path to configuration file to use')
    args = parser.parse_args()
    color_im_filename = args.color_image
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intrinsics
    gripper_width = args.gripper_width
    config_filename = args.config_filename

    # initialize the ROS node
    rospy.init_node('grasp_planning_example')
    logger.addHandler(rl.RosStreamHandler())

    # setup filenames
    if color_im_filename is None:
        color_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         '..',
                                         'data/examples/single_object/primesense/color_0.png')
    if depth_im_filename is None:
        depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         '..',
                                         'data/examples/single_object/primesense/depth_0.npy')
    if segmask_filename is None:
        segmask_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        '..',
                                        'data/examples/single_object/primesense/segmask_0.png')
    if camera_intr_filename is None:
        camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '..',
                                            'data/calib/primesense/primesense.intr')    
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/examples/grasping_client.yaml')
    
    # read config
    config = YamlConfig(config_filename)

    # wait for Grasp Planning Service and create Service Proxy
    rospy.wait_for_service('grasping_policy')
    plan_grasp = rospy.ServiceProxy('grasping_policy', GQCNNGraspPlanner)
    cv_bridge = CvBridge()    

    # setup sensor
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
        
    # read images
    color_im = ColorImage.open(color_im_filename, frame=camera_intr.frame)
    depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
    
    # optionally read a segmask
    segmask = BinaryImage(255 * np.ones(depth_im.shape).astype(np.uint8),
                          frame=camera_intr.frame)
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename)

    # optionally set a bounding box
    bounding_box = BoundingBox()
    bounding_box.minY = 0
    bounding_box.minX = 0
    bounding_box.maxY = color_im.height
    bounding_box.maxX = color_im.width
    
    # plan grasp
    grasp_resp = plan_grasp(color_im.rosmsg,
                            depth_im.rosmsg,
                            camera_intr.rosmsg,
                            bounding_box,
                            segmask.rosmsg)
    grasp = grasp_resp.grasp
    
    # convert to a grasp action
    grasp_type = grasp.grasp_type
    if grasp_type == GQCNNGrasp.PARALLEL_JAW:
        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                       frame=camera_intr.frame)
        grasp_2d = Grasp2D(center,
                           grasp.angle,
                           grasp.depth,
                           width=gripper_width,
                           camera_intr=camera_intr)
    elif grasp_type == GQCNNGrasp.SUCTION:
        center = Point(np.array([grasp.center_px[0], grasp.center_px[1]]),
                       frame=camera_intr.frame)
        grasp_2d = SuctionPoint2D(center,
                                  np.array([0,0,1]),
                                  grasp.depth,
                                  camera_intr=camera_intr)        
    else:
        raise ValueError('Grasp type %d not recognized!' %(grasp_type))
    try:
        thumbnail = DepthImage(cv_bridge.imgmsg_to_cv2(grasp.thumbnail,
                                                       desired_encoding="passthrough"),
                               frame=camera_intr.frame)
    except CVBridgeError as e:
        logger.error(e)
        logger.error('Failed to convert image')
        sys.exit(1)
    action = GraspAction(grasp_2d, grasp.q_value, thumbnail)
    
    # vis final grasp
    if config['vis']['final_grasp']:
        vis.figure(size=(10,10))
        vis.imshow(depth_im, vmin=0.6, vmax=0.9)
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
        vis.show()
