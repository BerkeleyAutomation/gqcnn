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
import logging
import IPython
import numpy as np
import os
import rosgraph.roslogging as rl
import rospy
import sys
import time

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

from autolab_core import Point, RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis

from gqcnn.grasping import CrossEntropyRobustGraspingPolicy, RgbdImageState, Grasp2D, SuctionPoint2D, GraspAction
from gqcnn.msg import GQCNNGrasp, BoundingBox
from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerBoundingBox, GQCNNGraspPlannerSegmask

if __name__ == '__main__':
    # set up logger
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Run a grasping policy on an example image')
    parser.add_argument('--depth_image', type=str, default=None, help='path to a test depth image stored as a .npy file')
    parser.add_argument('--segmask', type=str, default=None, help='path to an optional segmask to use')
    parser.add_argument('--camera_intr', type=str, default=None, help='path to the camera intrinsics')
    parser.add_argument('--gripper_width', type=float, default=0.05, help='width of the gripper to plan for')
    parser.add_argument('--namespace', type=str, default='gqcnn', help='namespace of the ROS grasp planning service')
    parser.add_argument('--vis_grasp', type=bool, default=True, help='whether or not to visualize the grasp')
    args = parser.parse_args()
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intr
    gripper_width = args.gripper_width
    namespace = args.namespace
    vis_grasp = args.vis_grasp

    # initialize the ROS node
    rospy.init_node('grasp_planning_example')
    logging.getLogger().addHandler(rl.RosStreamHandler())

    # setup filenames
    if depth_im_filename is None:
        depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         '..',
                                         'data/examples/single_object/primesense/depth_0.npy')
    if camera_intr_filename is None:
        camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '..',
                                            'data/calib/primesense/primesense.intr')    

    # wait for Grasp Planning Service and create Service Proxy
    rospy.wait_for_service('%s/grasp_planner' %(namespace))
    rospy.wait_for_service('%s/grasp_planner_segmask' %(namespace))
    plan_grasp = rospy.ServiceProxy('%s/grasp_planner' %(namespace), GQCNNGraspPlanner)
    plan_grasp_segmask = rospy.ServiceProxy('%s/grasp_planner_segmask' %(namespace), GQCNNGraspPlannerSegmask)
    cv_bridge = CvBridge()    

    # setup sensor
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
        
    # read images
    depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # read segmask
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename, frame=camera_intr.frame)
        grasp_resp = plan_grasp_segmask(color_im.rosmsg,
                                        depth_im.rosmsg,
                                        camera_intr.rosmsg,
                                        segmask.rosmsg)
    else:
        grasp_resp = plan_grasp(color_im.rosmsg,
                                depth_im.rosmsg,
                                camera_intr.rosmsg)
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
        logging.error(e)
        logging.error('Failed to convert image')
        sys.exit(1)
    action = GraspAction(grasp_2d, grasp.q_value, thumbnail)
    
    # vis final grasp
    if vis_grasp:
        vis.figure(size=(10,10))
        vis.imshow(depth_im, vmin=0.6, vmax=0.9)
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title('Planned grasp on depth (Q=%.3f)' %(action.q_value))
        vis.show()
