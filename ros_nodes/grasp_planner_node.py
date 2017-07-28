#!/usr/bin/env python
""" 
ROS Server for planning GQCNN grasps 
Author: Vishal Satish
"""
import rospy
import time
import perception

from cv_bridge import CvBridge, CvBridgeError

from autolab_core import YamlConfig
from gqcnn import CrossEntropyAntipodalGraspingPolicy, RgbdImageState
from gqcnn import NoValidGraspsException, NoAntipodalPairsFoundException
from gqcnn import Visualizer as vis

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
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
        """ Grasp planner request handler 
        
        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service
        """
        rospy.loginfo('Planning Grasp')
        
        # get the raw depth and color images as ROS Image objects
        raw_color = req.color_image
        raw_depth = req.depth_image

        # get the raw camera info as ROS CameraInfo object
        raw_camera_info = req.camera_info
        
        # get the bounding box as a custom ROS BoundingBox msg 
        bounding_box = req.bounding_box

        # wrap the camera info in a perception CameraIntrinsics object
        camera_intrinsics = perception.CameraIntrinsics(raw_camera_info.header.frame_id, raw_camera_info.K[0], raw_camera_info.K[4], raw_camera_info.K[2], raw_camera_info.K[5], raw_camera_info.K[1], raw_camera_info.height, raw_camera_info.width)

        ### Create wrapped Perception RGB and Depth Images by unpacking the ROS Images using CVBridge ###
        try:
            color_image = perception.ColorImage(self.cv_bridge.imgmsg_to_cv2(raw_color, "rgb8"), frame=camera_intrinsics.frame)
            depth_image = perception.DepthImage(self.cv_bridge.imgmsg_to_cv2(raw_depth, desired_encoding = "passthrough"), frame=camera_intrinsics.frame)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

        # visualize
        if self.cfg['vis']['vis_uncropped_color_image']:
            vis.imshow(color_image)
            vis.show()
        if self.cfg['vis']['vis_uncropped_depth_image']:
            vis.imshow(depth_image)
            vis.show()

        # aggregate color and depth images into a single perception rgbdimage
        rgbd_image = perception.RgbdImage.from_color_and_depth(color_image, depth_image)
        
        # calc crop parameters
        minX = bounding_box.minX
        minY = bounding_box.minY
        maxX = bounding_box.maxX
        maxY = bounding_box.maxY

        # contain box to image->don't let it exceed image height/width bounds
        no_pad = False
        if minX < 0:
            minX = 0
            no_pad = True
        if minY < 0:
            minY = 0
            no_pad = True
        if maxX > rgbd_image.width:
            maxX = rgbd_image.width
            no_pad = True
        if maxY > rgbd_image.height:
            maxY = rgbd_image.height
            no_pad = True

        centroidX = (maxX + minX) / 2
        centroidY = (maxY + minY) / 2

        # add some padding to bounding box to prevent empty pixel regions when the image is rotated during grasp planning
        if not no_pad:
            width = (maxX - minX) + self.cfg['width_pad']
            height = (maxY - minY) + self.cfg['height_pad']
        else:
            width = (maxX - minX)
            height = (maxY - minY)
  
        # crop camera intrinsics and rgbd image
        cropped_camera_intrinsics = camera_intrinsics.crop(height, width, centroidY, centroidX)
        cropped_rgbd_image = rgbd_image.crop(height, width, centroidY, centroidX)
        
        # visualize  
        if self.cfg['vis']['vis_cropped_rgbd_image']:
            vis.imshow(cropped_rgbd_image)
            vis.show()

        # create an RGBDImageState with the cropped RGBDImage and CameraIntrinsics
        image_state = RgbdImageState(cropped_rgbd_image, cropped_camera_intrinsics)
  
        # execute policy
        try:
            return self.execute_policy(image_state, self.grasping_policy, self.grasp_pose_publisher, cropped_camera_intrinsics.frame)
        except NoValidGraspsException:
            rospy.logerr('While executing policy found no valid grasps from sampled antipodal point pairs. Aborting Policy!')
            raise rospy.ServiceException('While executing policy found no valid grasps from sampled antipodal point pairs. Aborting Policy!')
        except NoAntipodalPairsFoundException:
            rospy.logerr('While executing policy could not sample any antipodal point pairs from input image. Aborting Policy! Please check if there is an object in the workspace or if the output of the object detector is reasonable.')
            raise rospy.ServiceException('While executing policy could not sample any antipodal point pairs from input image. Aborting Policy! Please check if there is an object in the workspace or if the output of the object detector is reasonable.')

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
        gqcnn_grasp.grasp_success_prob = grasp.q_value
        gqcnn_grasp.pose = grasp.grasp.pose().pose_msg

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
    policy_cfg = cfg['policy']

    # create publisher to publish pose only of final grasp
    grasp_pose_publisher = rospy.Publisher('/gqcnn_grasp/pose', PoseStamped, queue_size=10)

    # create a policy 
    rospy.loginfo('Creating Grasp Policy')
    grasping_policy = CrossEntropyAntipodalGraspingPolicy(policy_cfg)

    # create a grasp planner object
    grasp_planner = GraspPlanner(cfg, cv_bridge, grasping_policy, grasp_pose_publisher)

    # initialize the service        
    service = rospy.Service('plan_gqcnn_grasp', GQCNNGraspPlanner, grasp_planner.plan_grasp)
    rospy.loginfo('Grasp Sampler Server Initialized')

    # spin
    rospy.spin()
