#!/usr/bin/env python
""" 
ROS Node for planning GQCNN grasps 
Author: Vishal Satish
"""
import rospy
import time
import perception as percep
from core import YamlConfig
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from gqcnn.msg import GQCNNGrasp, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
from visualization import Visualizer2D as vis
from visualization import Visualizer3D as vis3d
from gqcnn import CrossEntropyAntipodalGraspingPolicy, RgbdImageState
from gqcnn import NoValidGraspsException, NoAntipodalPairsFoundException
import IPython

camera_intrinsics = None
rgb_image = None
depth_image = None
cv_bridge = None
cfg = None

def camera_intrinsics_callback(data):
    """ Callback for Camera Intrinsics. Wrap the incoming intrinsic data in perception CameraIntrinsics object """    
    global camera_intrinsics
    camera_intrinsics = percep.CameraIntrinsics(data.header.frame_id, data.K[0], data.K[4], data.K[2], data.K[5], data.K[1], data.height, data.width)

def bounding_box_callback(data):
    """ Callback for Object Detector Bounding Boxes """

    # make sure camera intrinsics, rgb image and depth image are not None, then generate RGBD image and proceed
    if camera_intrinsics is not None and rgb_image is not None and depth_image is not None:

        # inpaint to remove holes
        inpainted_rgb_image = rgb_image.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
        inpainted_depth_image = depth_image.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])

        rgbd_image = percep.RgbdImage.from_color_and_depth(inpainted_rgb_image, inpainted_depth_image)
    
        # find crop parameters
        minX = data.minX
        minY = data.minY
        maxX = data.maxX
        maxY = data.maxY
        centroidX = (maxX + minX) / 2
        centroidY = (maxY + minY) / 2
        width = (maxX - minX) + cfg['width_pad']
        height = (maxY - minY) + cfg['height_pad']
    
        # crop camera intrinsics and rgbd image
        cropped_camera_intrinsics = camera_intrinsics.crop(height, width, centroidX, centroidY)
        cropped_rgbd_image = rgbd_image.crop(height, width, centroidX, centroidY)
    
        # visualize
        # rospy.loginfo('Unregistering from nodes')
        # camera_intrinsics_sub.unregister()
        # bounding_boxes_sub.unregister()
        # depth_image_sub.unregister()
        # rgb_image_sub.unregister()
        
        if cfg['vis_cropped_images']:
            vis.imshow(per_color_image)
            vis.show()
            vis.imshow(per_depth_image)
            vis.show()
            vis.imshow(rgbd_image)
            vis.show()

        # execute policy
        try:
            grasp = execute_policy(RgbdImageState(cropped_rgbd_image, cropped_camera_intrinsics))
        except NoValidGraspsException:
            rospy.logerr('While executing policy found no valid grasps from sampled antipodal point pairs. Aborting Policy!')
        except NoAntipodalPairsFoundException:
            rospy.logerr('While executing policy could not sample any antipodal point pairs from input image. Aborting Policy! Please check if there is an object in the workspace or if the output of the object detector is reasonable.')

def execute_policy(rgbd_image_state):
    # execute the policy's action
    rospy.loginfo('Planning Grasp')
    grasp_planning_start_time = time.time()
    grasp = grasping_policy(rgbd_image_state)

    # create GQCNNGrasp return msg and populate it
    gqcnn_grasp = GQCNNGrasp()
    gqcnn_grasp.grasp_success_prob = grasp.p_success[0]
    gqcnn_grasp.pose = grasp.grasp.pose().pose_msg

    pose_stamped = PoseStamped()
    pose_stamped.pose = grasp.grasp.pose().pose_msg
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = camera_intrinsics.frame
    pose_stamped.header = header
    grasp_pose_publisher.publish(pose_stamped)

    # publish GQCNNGrasp
    rospy.loginfo('Publishing Grasp, Total planning time: ' + str(time.time() - grasp_planning_start_time) + ' secs.')
    grasp_publisher.publish(gqcnn_grasp)

    return grasp

def rgb_im_callback(data):
    """ Callback for Object Detector RGB Image. Get the raw ROS image, convert it to a cv image, then wrap it with the perception ColorImage. """
    global rgb_image
    try:
        rgb_image = percep.ColorImage(cv_bridge.imgmsg_to_cv2(data, "rgb8"), frame=camera_intrinsics.frame)
    except CvBridgeError as cv_bridge_exception:
        rospy.logerr(cv_bridge_exception)

def depth_im_callback(data):
    """ Callback for Object Detector Depth Image. Get the raw ROS image, convert it to a cv image, the wrap it with the perception DepthImage. """
    global depth_image
    try:
        depth_image = percep.DepthImage(cv_bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough"), frame=camera_intrinsics.frame)
    except CvBridgeError as cv_bridge_exception:
        rospy.logerr(cv_bridge_exception)

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('grasp_sampler_node')
    rospy.loginfo('Grasp Sampler Node initialized')

    # initialize cv_bridge
    cv_bridge = CvBridge()

    # get configs
    cfg = YamlConfig('cfg/ros_nodes/grasp_planner_node.yaml')
    topics = cfg['ros_topics']
    policy_cfg = cfg['policy']

    # create a subscriber to get camera intrinsics 
    camera_intrinsics_sub = rospy.Subscriber(topics['camera_intrinsics'], CameraInfo, camera_intrinsics_callback)

    # create a subscriber to get bounding boxes from object_detector
    bounding_boxes_sub = rospy.Subscriber(topics['object_detector']['bounding_boxes'], BoundingBox, bounding_box_callback)

    # create a subscriber to get depth image from object detector
    depth_image_sub = rospy.Subscriber(topics['object_detector']['depth_image'], Image, depth_im_callback)

    # create a subscriber to get rgb image from object detector
    rgb_image_sub = rospy.Subscriber(topics['object_detector']['rgb_image'], Image, rgb_im_callback)

    # create publisher to publish final grasp and confidence
    grasp_publisher = rospy.Publisher('/gqcnn_grasp', GQCNNGrasp, queue_size=10)

    # create publisher to publish pose of final grasp only
    grasp_pose_publisher = rospy.Publisher('/gqcnn_grasp/pose', PoseStamped, queue_size=10)

    # create a policy 
    rospy.loginfo('Creating Grasp Policy')
    grasping_policy = CrossEntropyAntipodalGraspingPolicy(policy_cfg)

    rospy.spin()