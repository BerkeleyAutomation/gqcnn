#!/usr/bin/env python
import rospy
from core import YamlConfig
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Quaternion
from gqcnn.msg import GQCNNGrasp
import perception as per
from cv_bridge import CvBridge, CvBridgeError
from visualization import Visualizer2D as vis
from gqcnn import CrossEntropyAntipodalGraspingPolicy, RgbdImageState

grasp_publisher = None
camera_intrinsics = None
rgb_image = None
depth_image = None
cv_bridge = None
rgbd_image = None
grasping_policy = None

def camera_intrinsics_callback(data):
    """ Callback for Camera Intrinsics """    
    global camera_intrinsics
    camera_intrinsics = per.CameraIntrinsics(data.header.frame_id, data.K[0], data.K[4], data.K[2], data.K[5], data.K[1], data.height, data.width)

def bounding_box_callback(data):
    """ Callback for Object Detector Bounding Boxes """
    global camera_intrinsics
    global rgbd_image

    # make sure camera intrinsics, rgb image and depth image are not None, then generate RGBD image
    if camera_intrinsics is not None and rgb_image is not None and depth_image is not None:
        rgbd_image = per.RgbdImage.from_color_and_depth(per.ColorImage(rgb_image), per.DepthImage(depth_image))
    
    # find crop parameters
    minX = data.x
    minY = data.y
    maxX = data.z
    maxY = data.w
    centroidX = (maxX + minX) / 2
    centroidY = (maxY + minY) / 2
    width = (maxX - minX)
    height = (maxY - minY)

    if rgbd_image is not None:
        # crop camera intrinsics and rgbd image
        camera_intrinsics = camera_intrinsics.crop(height, width, centroidX, centroidY)
        rospy.loginfo('Crop height: ' + str(height) + 'Crop Width: ' + str(width) + 'CenterX: ' + str(centroidX) + 'CenterY: ' + str(centroidY))
        rgbd_image = rgbd_image.crop(height, width, centroidX, centroidY)

    # visualize
    vis.imshow(per.ColorImage(rgb_image))
    vis.show()
    vis.imshow(per.DepthImage(depth_image))
    vis.show()
    vis.imshow(rgbd_image)
    vis.show()

    # execute policy
    execute_policy()

def execute_policy():
    # execute the policy's action
    rospy.loginfo('Planning Grasp')
    grasp = grasping_policy(RgbdImageState(rgbd_image, camera_intrinsics))

    # create GQCNNGrasp return msg and populate it
    gqcnn_grasp = GQCNNGrasp()
    gqcnn_grasp.grasp_success_prob = grasp.p_success
    gqcnn_grasp.pose = grasp.pose().pose_msg

    # publish GQCNNGrasp
    rospy.loginfo('Publishing Grasp')
    grasp_publisher.publish(gqcnn_grasp)

def rgb_im_callback(data):
    """ Callback for Object Detector RGB Image """
    global rgb_image
    try:
        rgb_image = cv_bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as cv_bridge_exception:
        rospy.logerr(cv_bridge_exception)

def depth_im_callback(data):
    """ Callback for Object Detector Depth Image """
    global depth_image
    try:
        depth_image = cv_bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
    except CvBridgeError as cv_bridge_exception:
        rospy.logerr(cv_bridge_exception)

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('grasp_sampler_node')
    rospy.loginfo('Grasp Sampler Node initialized')

    # initialize cv_bridge
    cv_bridge = CvBridge()

    # get configs
    cfg = YamlConfig('/home/autolab/Workspace/vishal_working/catkin_ws/src/gqcnn/cfg/ros_nodes/grasp_sampler_node.yaml')
    topics = cfg['ros_topics']
    policy_cfg = cfg['policy_cfg']['policy']

    # create a subscriber to get camera intrinsics 
    rospy.Subscriber(topics['camera_intrinsics'], CameraInfo, camera_intrinsics_callback)

    # create a subscriber to get bounding boxes from object_detector
    rospy.Subscriber(topics['object_detector']['bounding_boxes'], Quaternion, bounding_box_callback)

    # create a subscriber to get depth image from object detector
    rospy.Subscriber(topics['object_detector']['depth_image'], Image, depth_im_callback)

    # create a subscriber to get rgb image from object detector
    rospy.Subscriber(topics['object_detector']['rgb_image'], Image, rgb_im_callback)

    # create publisher to publish final grasp and confidence
    grasp_publisher = rospy.Publisher('/gqcnn_grasp', GQCNNGrasp, queue_size=10)

    # create a policy 
    rospy.loginfo('Creating Grasp Policy')
    grasping_policy = CrossEntropyAntipodalGraspingPolicy(policy_cfg)
    # grasping_policy.gqcnn.open_session()

    rospy.spin()