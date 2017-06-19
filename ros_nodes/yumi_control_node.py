#!/usr/bin/env python
import rospy
from gqcnn.msg import GQCNNGrasp
from sensor_msgs.msg import Image, CameraInfo
from autolab_core import RigidTransform
from gqcnn import RobotGripper
from autolab_core import YamlConfig
from yumipy import YuMiRobot, YuMiCommException, YuMiControlException, YuMiSubscriber
from yumipy import YuMiConstants as YMC
import logging
import numpy as np
from visualization import Visualizer3D as vis3d
import perception as per
from cv_bridge import CvBridge, CvBridgeError

robot = None
arm = None
subscriber = None
config = None
home_pose = None
gripper = None
T_camera_world = None
depth_im = None
ir_intrinsics = None
cv_bridge = None
in_grasp = False

def grasp_callback(data):
    rospy.loginfo('Received grasp from GQCNN')

    rotation_quaternion = np.asarray([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z]) 
    translation = np.asarray([data.pose.position.x, data.pose.position.y, data.pose.position.z])
    T_grasp_world = RigidTransform(rotation_quaternion, translation, 'grasp', T_camera_world.from_frame)
    
    T_gripper_world = T_camera_world * T_grasp_world * gripper.T_grasp_gripper
    
    global in_grasp
    if not in_grasp:
        rospy.loginfo('Executing Grasp!')
        in_grasp = True
        execute_grasp(T_gripper_world, robot, arm, subscriber, config)

        # bring arm back to home pose 
        arm.goto_pose(home_pose)
        in_grasp = False


def execute_grasp(T_gripper_world, robot, arm, subscriber, config):
    """ Executes a single grasp for the hand pose T_gripper_world up to the point of lifting the object """
    # snap gripper to valid depth
    if T_gripper_world.translation[2] < config['grasping']['min_gripper_depth']:
        T_gripper_world.translation[2] = config['grasping']['min_gripper_depth']

    # get cur pose
    T_cur_world = arm.get_pose()

    # compute approach pose
    t_approach_target = np.array([0,0,config['grasping']['approach_dist']])
    T_gripper_approach = RigidTransform(translation=t_approach_target,
                                        from_frame='gripper',
                                        to_frame='gripper')
    T_approach_world = T_gripper_world * T_gripper_approach.inverse()
    t_lift_target = np.array([0,0,config['grasping']['lift_height']])
    T_gripper_lift = RigidTransform(translation=t_lift_target,
                                    from_frame='gripper',
                                    to_frame='gripper')
    T_lift_world = T_gripper_world * T_gripper_lift.inverse()

    # compute lift pose
    t_delta_approach = T_approach_world.translation - T_cur_world.translation

    # perform grasp on the robot, up until the point of lifting
    for _ in range(10):
        _, torques = subscriber.left.get_torque()        
    resting_torque = torques[3]
    arm.open_gripper(wait_for_res=True)
    robot.set_z(config['control']['approach_zoning'])
    arm.goto_pose(YMC.L_KINEMATIC_AVOIDANCE_POSE)
    arm.goto_pose(T_approach_world)

    # grasp
    robot.set_v(config['control']['approach_velocity'])
    robot.set_z(config['control']['standard_zoning'])
    if config['control']['test_collision']:
        robot.set_z('z200')
        T_gripper_world.translation[2] = 0.0
        arm.goto_pose(T_gripper_world, wait_for_res=True)
        _, T_cur_gripper_world = subscriber.left.get_pose()
        dist_from_goal = np.linalg.norm(T_cur_gripper_world.translation - T_gripper_world.translation)
        collision = False
        for i in range(10):
            _, torques = subscriber.left.get_torque()        
        while dist_from_goal > 1e-3:
            _, T_cur_gripper_world = subscriber.left.get_pose()
            dist_from_goal = np.linalg.norm(T_cur_gripper_world.translation - T_gripper_world.translation)
            _, torques = subscriber.left.get_torque()
            print torques
            if torques[1] > 0.001:
                logging.info('Detected collision!!!!!!')
                robot.set_z('fine')
                arm.goto_pose(T_approach_world, wait_for_res=True)
                logging.info('Commanded!!!!!!')
                collision = True
                break
            arm.goto_pose(T_gripper_world, wait_for_res=False)
    else:
        arm.goto_pose(T_gripper_world, wait_for_res=True)
    
    # pick up object
    arm.close_gripper(force=config['control']['gripper_close_force'], wait_for_res=True)
    pickup_gripper_width = arm.get_gripper_width()
    robot.set_v(config['control']['standard_velocity'])
    robot.set_z(config['control']['standard_zoning'])
    arm.goto_pose(T_lift_world)
    arm.goto_pose(YMC.L_KINEMATIC_AVOIDANCE_POSE, wait_for_res=True)
    arm.goto_pose(YMC.L_PREGRASP_POSE, wait_for_res=True)

    # shake test
    if config['control']['shake_test']:

        # compute shake poses
        radius = config['control']['shake_radius']
        angle = config['control']['shake_angle'] * np.pi
        delta_T = RigidTransform(translation=[0,0,radius], from_frame='gripper', to_frame='gripper')
        R_shake = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
        delta_T_up = RigidTransform(rotation=R_shake, translation=[0,0,-radius], from_frame='gripper', to_frame='gripper')
        delta_T_down = RigidTransform(rotation=R_shake.T, translation=[0,0,-radius], from_frame='gripper', to_frame='gripper')
        T_shake_up = YMC.L_PREGRASP_POSE.as_frames('gripper', 'world') * delta_T_up * delta_T
        T_shake_down = YMC.L_PREGRASP_POSE.as_frames('gripper', 'world') * delta_T_down * delta_T

        robot.set_v(config['control']['shake_velocity'])
        robot.set_z(config['control']['shake_zoning'])
        for i in range(config['control']['num_shakes']):
            arm.goto_pose(T_shake_up, wait_for_res=False)
            arm.goto_pose(YMC.L_PREGRASP_POSE, wait_for_res=False)
            arm.goto_pose(T_shake_down, wait_for_res=False)
            arm.goto_pose(YMC.L_PREGRASP_POSE, wait_for_res=False)
        robot.set_v(config['control']['standard_velocity'])
        robot.set_z(config['control']['standard_zoning'])

    # check gripper width
    for _ in range(10):
        _, torques = subscriber.left.get_torque()        
        lift_torque = torques[3]
        lift_gripper_width = arm.get_gripper_width()

    # check drops
    lifted_object = False
    if np.abs(lift_gripper_width) > config['grasping']['pickup_min_width']:
        lifted_object = True

    # check lifts
    # table_clear = False
    # for i in range(3):
    #     _, depth_im, _ = sensor.frames()

    return lifted_object, lift_gripper_width, lift_torque

def init_robot(config):
    """ Initializes a robot """
    robot = None
    subscriber = None
    initialized = False
    while not initialized:
        try:
            robot = YuMiRobot(debug=config['robot_off'])
            robot.set_v(config['control']['standard_velocity'])
            robot.set_z(config['control']['standard_zoning'])
            if config['control']['use_left']:
                arm = robot.left
                arm.goto_state(YMC.L_HOME_STATE)
                home_pose = YMC.L_PREGRASP_POSE
            else:
                arm = robot.right
                arm.goto_state(YMC.R_HOME_STATE)
                home_pose = YMC.R_AWAY_STATE

            subscriber = YuMiSubscriber()
            subscriber.start()

            initialized = True
        except YuMiCommException as ymc:
            if robot is not None:
                robot.stop()
            if subscriber is not None and subscriber._started:
                subscriber.stop()
            logging.error(str(ymc))
            logging.error('Failed to initialize YuMi. Check the FlexPendant and connection to the YuMi.')
            human_input = raw_input('Hit [ENTER] when YuMi is ready')
    return robot, subscriber, arm, home_pose

# def depth_im_callback(data):
#     global depth_im
#     # rospy.loginfo('Received Depth Image')
#     try:
#         depth_image = cv_bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
#     except CvBridgeError as cv_bridge_exception:
#         rospy.logerr(cv_bridge_exception)
#     depth_im = per.DepthImage(depth_image, 'primesense_overhead')

# def camera_intrinsics_callback(data):
#     """ Callback for Camera Intrinsics """    
#     global ir_intrinsics
#     ir_intrinsics = per.CameraIntrinsics('primesense_overhead', data.K[0], data.K[4], data.K[2], data.K[5], data.K[1], data.height, data.width)

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Yumi_Control_Node')

    rospy.loginfo('Initializing YuMi')
    config = YamlConfig('/home/autolab/Workspace/vishal_working/catkin_ws/src/gqcnn/cfg/ros_nodes/yumi_control_node.yaml')
    robot, subscriber, arm, home_pose = init_robot(config)

    rospy.loginfo('Loading Gripper')
    gripper = RobotGripper.load('yumi_metal_spline')

    rospy.loginfo('Loading T_camera_world')
    T_camera_world = RigidTransform.load('/home/autolab/Public/alan/calib/primesense_overhead/primesense_overhead_to_world.tf')

    # create a subscriber to get GQCNN Grasp 
    rospy.loginfo('Subscribing to GQCNN Grasp Topic')
    rospy.Subscriber('/gqcnn_grasp', GQCNNGrasp, grasp_callback)

    rospy.spin()