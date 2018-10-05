#!/usr/bin/env python
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
ROS server for planning GQCNN grasps 
Author: Vishal Satish
"""
import math
import time

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from autolab_core import YamlConfig
from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
from visualization import Visualizer2D as vis

from gqcnn import CrossEntropyRobustGraspingPolicy, RgbdImageState, Grasp2D, SuctionPoint2D
from gqcnn import NoValidGraspsException
from gqcnn.srv import GQCNNGraspPlanner
from gqcnn.msg import GQCNNGrasp

class GraspPlanner(object):
    def __init__(self, cfg, cv_bridge, grasping_policy, grasp_pose_publisher):
        """
        Parameters
        ----------
        cfg : dict
            dictionary of configuration parameters
        cv_bridge: :obj:`CvBridge`
            ROS CvBridge
        grasping_policy: :obj:`GraspingPolicy`
            grasping policy to use
        grasp_pose_publisher: :obj:`Publisher`
            ROS Publisher to publish planned grasp's ROS Pose only for visualization
        """
        self.cfg = cfg
        self.cv_bridge = cv_bridge
        self.grasping_policy = grasping_policy
        self.grasp_pose_publisher = grasp_pose_publisher

    def plan_grasp(self, req):
        """ Grasp planner request handler .
        
        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service
        """
        rospy.loginfo('Planning Grasp')

        # set min dimensions
        pad = max(
            math.ceil(np.sqrt(2) * (float(self.cfg['policy']['metric']['crop_width']) / 2)),
            math.ceil(np.sqrt(2) * (float(self.cfg['policy']['metric']['crop_height']) / 2))
        )        
        min_width = 2 * pad + self.cfg['policy']['metric']['crop_width']
        min_height = 2 * pad + self.cfg['policy']['metric']['crop_height']

        # get the raw depth and color images as ROS Image objects
        raw_color = req.color_image
        raw_depth = req.depth_image
        segmask = None
        raw_segmask = req.segmask

        # get the raw camera info as ROS CameraInfo object
        raw_camera_info = req.camera_info
        
        # get the bounding box as a custom ROS BoundingBox msg 
        bounding_box = req.bounding_box

        # wrap the camera info in a perception CameraIntrinsics object
        camera_intrinsics = CameraIntrinsics(raw_camera_info.header.frame_id, raw_camera_info.K[0], raw_camera_info.K[4], raw_camera_info.K[2], raw_camera_info.K[5], raw_camera_info.K[1], raw_camera_info.height, raw_camera_info.width)

        ### create wrapped Perception RGB and Depth Images by unpacking the ROS Images using CVBridge ###
        try:
            color_image = ColorImage(self.cv_bridge.imgmsg_to_cv2(raw_color, "rgb8"), frame=camera_intrinsics.frame)
            depth_image = DepthImage(self.cv_bridge.imgmsg_to_cv2(raw_depth, desired_encoding = "passthrough"), frame=camera_intrinsics.frame)
            segmask = BinaryImage(self.cv_bridge.imgmsg_to_cv2(raw_segmask, desired_encoding = "passthrough"), frame=camera_intrinsics.frame)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

        # check image sizes
        if color_image.height != depth_image.height or \
           color_image.width != depth_image.width:
            rospy.logerr('Color image and depth image must be the same shape! Color is %d x %d but depth is %d x %d' %(color_image.height, color_image.width, depth_image.height, depth_image.width))
            raise rospy.ServiceException('Color image and depth image must be the same shape! Color is %d x %d but depth is %d x %d' %(color_image.height, color_image.width, depth_image.height, depth_image.width))            

        if color_image.height < min_height or color_image.width < min_width:
            rospy.logerr('Color image is too small! Must be at least %d x %d resolution but the requested image is only %d x %d' %(min_height, min_width, color_image.height, color_image.width))
            raise rospy.ServiceException('Color image is too small! Must be at least %d x %d resolution but the requested image is only %d x %d' %(min_height, min_width, color_image.height, color_image.width))

        # inpaint images
        color_image = color_image.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        depth_image = depth_image.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        
        # visualize
        if self.cfg['vis']['color_image']:
            vis.imshow(color_image)
            vis.show()
        if self.cfg['vis']['depth_image']:
            vis.imshow(depth_image)
            vis.show()
        if self.cfg['vis']['segmask'] and segmask is not None:
            vis.imshow(segmask)
            vis.show()

        # aggregate color and depth images into a single perception rgbdimage
        rgbd_image = RgbdImage.from_color_and_depth(color_image, depth_image)
        
        # calc crop parameters
        minX = bounding_box.minX - pad
        minY = bounding_box.minY - pad
        maxX = bounding_box.maxX + pad
        maxY = bounding_box.maxY + pad

        # contain box to image->don't let it exceed image height/width bounds
        if minX < 0:
            minX = 0
        if minY < 0:
            minY = 0
        if maxX > rgbd_image.width:
            maxX = rgbd_image.width
        if maxY > rgbd_image.height:
            maxY = rgbd_image.height

        centroidX = (maxX + minX) / 2
        centroidY = (maxY + minY) / 2

        # compute width and height
        width = maxX - minX
        height = maxY - minY
  
        # crop camera intrinsics and rgbd image
        cropped_camera_intrinsics = camera_intrinsics.crop(height, width, centroidY, centroidX)
        cropped_rgbd_image = rgbd_image.crop(height, width, centroidY, centroidX)
        cropped_segmask = None
        if segmask is not None:
            cropped_segmask = segmask.crop(height, width, centroidY, centroidX)
        
        # visualize  
        if self.cfg['vis']['cropped_rgbd_image']:
            vis.imshow(cropped_rgbd_image)
            vis.show()

        # create an RGBDImageState with the cropped RGBDImage and CameraIntrinsics
        image_state = RgbdImageState(cropped_rgbd_image,
                                     cropped_camera_intrinsics,
                                     segmask=cropped_segmask)
  
        # execute policy
        try:
            return self.execute_policy(image_state, self.grasping_policy, self.grasp_pose_publisher, cropped_camera_intrinsics.frame)
        except NoValidGraspsException:
            rospy.logerr('While executing policy found no valid grasps from sampled antipodal point pairs. Aborting Policy!')
            raise rospy.ServiceException('While executing policy found no valid grasps from sampled antipodal point pairs. Aborting Policy!')

    def execute_policy(self, rgbd_image_state, grasping_policy, grasp_pose_publisher, pose_frame):
        """ Executes a grasping policy on an RgbdImageState
        
        Parameters
        ----------
        rgbd_image_state: :obj:`RgbdImageState`
            RgbdImageState from perception module to encapsulate depth and color image along with camera intrinsics
        grasping_policy: :obj:`GraspingPolicy`
            grasping policy to use
        grasp_pose_publisher: :obj:`Publisher`
            ROS Publisher to publish planned grasp's ROS Pose only for visualization
        pose_frame: :obj:`str`
            frame of reference to publish pose alone in
        """
        # execute the policy's action
        rospy.loginfo('Planning Grasp')
        grasp_planning_start_time = time.time()
        grasp = grasping_policy(rgbd_image_state)
  
        # create GQCNNGrasp return msg and populate it
        gqcnn_grasp = GQCNNGrasp()
        gqcnn_grasp.q_value = grasp.q_value
        gqcnn_grasp.pose = grasp.grasp.pose().pose_msg
        if isinstance(grasp.grasp, Grasp2D):
            gqcnn_grasp.grasp_type = GQCNNGrasp.PARALLEL_JAW
        elif isinstance(grasp.grasp, SuctionPoint2D):
            gqcnn_grasp.grasp_type = GQCNNGrasp.SUCTION
        else:
            rospy.logerr('Grasp type not supported!')
            raise rospy.ServiceException('Grasp type not supported!')

        # store grasp representation in image space
        gqcnn_grasp.center_px[0] = grasp.grasp.center[0]
        gqcnn_grasp.center_px[1] = grasp.grasp.center[1]
        gqcnn_grasp.angle = grasp.grasp.angle
        gqcnn_grasp.depth = grasp.grasp.depth
        gqcnn_grasp.thumbnail = grasp.image.rosmsg
        
        # create and publish the pose alone for visualization ease of grasp pose in rviz
        pose_stamped = PoseStamped()
        pose_stamped.pose = grasp.grasp.pose().pose_msg
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = pose_frame
        pose_stamped.header = header
        grasp_pose_publisher.publish(pose_stamped)

        # return GQCNNGrasp msg
        rospy.loginfo('Total grasp planning time: ' + str(time.time() - grasp_planning_start_time) + ' secs.')

        return gqcnn_grasp

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Grasp_Sampler_Server')

    # initialize cv_bridge
    cv_bridge = CvBridge()

    # get configs
    cfg = YamlConfig(rospy.get_param('~config'))
    model_dir = rospy.get_param('~model_dir')
    policy_cfg = cfg['policy']
    policy_cfg['metric']['gqcnn_model'] = model_dir

    # create publisher to publish pose only of final grasp
    grasp_pose_publisher = rospy.Publisher('/gqcnn_grasp/pose', PoseStamped, queue_size=10)

    # create a policy 
    rospy.loginfo('Creating Grasp Policy')
    grasping_policy = CrossEntropyRobustGraspingPolicy(policy_cfg)

    # create a grasp planner object
    grasp_planner = GraspPlanner(cfg, cv_bridge, grasping_policy, grasp_pose_publisher)

    # initialize the service        
    service = rospy.Service('grasping_policy', GQCNNGraspPlanner, grasp_planner.plan_grasp)
    rospy.loginfo('Grasping Policy Initialized')

    # spin
    rospy.spin()
