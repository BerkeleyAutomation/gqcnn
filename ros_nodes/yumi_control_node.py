#!/usr/bin/env python
"""
Example node for planning grasps from point clouds using the gqcnn module and
executing the grasps with an ABB YuMi.
Additionally depends on the dex-net, meshpy, and yumipy modules.

This file is intended as an example, not as code that will run with standard installation.

Author: Vishal Satish
"""
import rospy
import logging
import numpy as np
import signal
import time

from autolab_core import RigidTransform
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from yumipy import YuMiRobot, YuMiCommException, YuMiControlException, YuMiSubscriber
from yumipy import YuMiConstants as YMC
from visualization import Visualizer2D as vis
import perception as perception
from perception import RgbdDetectorFactory, RgbdSensorFactory
from gqcnn import Visualizer as vis

from gqcnn.msg import GQCNNGrasp, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from gqcnn.srv import GQCNNGraspPlanner

from cv_bridge import CvBridge, CvBridgeError

from gqcnn import GraspIsolatedObjectExperimentLogger

def process_GQCNNGrasp(grasp, robot, left_arm, right_arm, subscriber, home_pose, config):
    """ Processes a ROS GQCNNGrasp message and executes the resulting grasp on the ABB Yumi """
    
    grasp = grasp.grasp
    rospy.loginfo('Processing Grasp')

    rotation_quaternion = np.asarray([grasp.pose.orientation.w, grasp.pose.orientation.x, grasp.pose.orientation.y, grasp.pose.orientation.z]) 
    translation = np.asarray([grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z])
    T_grasp_world = RigidTransform(rotation_quaternion, translation, 'grasp', T_camera_world.from_frame)
    
    T_gripper_world = T_camera_world * T_grasp_world * gripper.T_grasp_gripper
    
    if not config['robot_off']:
        rospy.loginfo('Executing Grasp!')
        lifted_object, lift_gripper_width, lift_torque = execute_grasp(T_gripper_world, robot, left_arm, right_arm, subscriber, config)
    
        # bring arm back to home pose 
        left_arm.goto_pose(home_pose)
        right_arm.open_gripper(wait_for_res=True)

        return lift_gripper_width, T_gripper_world

def execute_grasp(T_gripper_world, robot, left_arm, right_arm, subscriber, config):
    """ Executes a single grasp for the hand pose T_gripper_world up to the point of lifting the object """
    # snap gripper to valid depth
    if T_gripper_world.translation[2] < config['grasping']['min_gripper_depth']:
        T_gripper_world.translation[2] = config['grasping']['min_gripper_depth']

    # get cur pose
    T_cur_world = left_arm.get_pose()

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
    right_arm.open_gripper(wait_for_res=True)
    robot.set_z(config['control']['approach_zoning'])
    left_arm.goto_pose(YMC.L_KINEMATIC_AVOIDANCE_POSE)
    left_arm.goto_pose(T_approach_world)

    # grasp
    robot.set_v(config['control']['approach_velocity'])
    robot.set_z(config['control']['standard_zoning'])
    if config['control']['test_collision']:
        robot.set_z('z200')
        T_gripper_world.translation[2] = 0.0
        left_arm.goto_pose(T_gripper_world, wait_for_res=True)
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
                left_arm.goto_pose(T_approach_world, wait_for_res=True)
                logging.info('Commanded!!!!!!')
                collision = True
                break
            left_arm.goto_pose(T_gripper_world, wait_for_res=False)
    else:
        left_arm.goto_pose(T_gripper_world, wait_for_res=True)
    
    # pick up object
    right_arm.close_gripper(force=config['control']['gripper_close_force'], wait_for_res=True)
    pickup_gripper_width = right_arm.get_gripper_width()
    robot.set_v(config['control']['standard_velocity'])
    robot.set_z(config['control']['standard_zoning'])
    left_arm.goto_pose(T_lift_world)
    left_arm.goto_pose(YMC.L_KINEMATIC_AVOIDANCE_POSE, wait_for_res=True)
    left_arm.goto_pose(YMC.L_PREGRASP_POSE, wait_for_res=True)

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
            left_arm.goto_pose(T_shake_up, wait_for_res=False)
            left_arm.goto_pose(YMC.L_PREGRASP_POSE, wait_for_res=False)
            left_arm.goto_pose(T_shake_down, wait_for_res=False)
            left_arm.goto_pose(YMC.L_PREGRASP_POSE, wait_for_res=False)
        robot.set_v(config['control']['standard_velocity'])
        robot.set_z(config['control']['standard_zoning'])

    # check gripper width
    for _ in range(10):
        _, torques = subscriber.left.get_torque()        
        lift_torque = torques[3]
        lift_gripper_width = right_arm.get_gripper_width()

    # check drops
    lifted_object = False
    if np.abs(lift_gripper_width) > config['grasping']['pickup_min_width']:
        lifted_object = True

    return lifted_object, lift_gripper_width, lift_torque

def init_robot(config):
    """ Initializes the robot """
    robot = None
    subscriber = None
    initialized = False
    while not initialized:
        try:
            robot = YuMiRobot(debug=config['robot_off'])
            robot.set_v(config['control']['standard_velocity'])
            robot.set_z(config['control']['standard_zoning'])
            left_arm = robot.left
            left_arm.goto_state(YMC.L_HOME_STATE)
            home_pose = YMC.L_PREGRASP_POSE

            right_arm = robot.right
            right_arm.goto_state(YMC.R_HOME_STATE)

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
    return robot, subscriber, left_arm, right_arm, home_pose

def run_experiment():
    """ Run the experiment """

    if not config['robot_off']:
        rospy.loginfo('Initializing YuMi')
        robot, subscriber, left_arm, right_arm, home_pose = init_robot(config)
    
    # create ROS CVBridge
    cv_bridge = CvBridge()
    
    # wait for Grasp Planning Service and create Service Proxy
    rospy.wait_for_service('plan_gqcnn_grasp')
    plan_grasp = rospy.ServiceProxy('plan_gqcnn_grasp', GQCNNGraspPlanner)

    # get camera intrinsics
    camera_intrinsics = sensor.ir_intrinsics

    # setup experiment logger
    experiment_logger = GraspIsolatedObjectExperimentLogger(config['experiment_dir'],
                                                            config['supervisor'],
                                                            camera_intrinsics,
                                                            T_camera_world,
                                                            '/home/autolab/Workspace/vishal_working/catkin_ws/src/gqcnn/cfg/ros_nodes/yumi_control_node.yaml',
                                                            planner_type=config['planner_type'])

    logging.info('Saving experiment to %s' %(experiment_logger.experiment_path))
    object_keys = config['test_object_keys']
    trial_number = 1
    re_try = False

    logging.info('Beginning experiment')

    while True:
        if not re_try:
            experiment_logger.start_trial()
            obj = np.random.choice(object_keys, size=1)[0]
        else:
            re_try = False
        
        rospy.loginfo('Please place object: ' + obj + ' on the workspace.')
        raw_input("Press ENTER when ready ...")
        # start the next trial
        rospy.loginfo('Trial %d' % (trial_number))

        # get the images from the sensor
        color_image, depth_image, _ = sensor.frames()
        
        # log some trial info        
        experiment_logger.update_trial_attribute('trial_num', trial_number)
        experiment_logger.update_trial_attribute('color_im', color_image)
        experiment_logger.update_trial_attribute('depth_im', depth_image)

        # inpaint to remove holes
        inpainted_color_image = color_image.inpaint(rescale_factor=config['inpaint_rescale_factor'])
        inpainted_depth_image = depth_image.inpaint(rescale_factor=config['inpaint_rescale_factor'])

        detector = RgbdDetectorFactory.detector('point_cloud_box')
        detection = detector.detect(inpainted_color_image, inpainted_depth_image, detector_cfg, camera_intrinsics, T_camera_world, vis_foreground=False, vis_segmentation=False
            )[0]

        if config['vis']['vis_detector_output']:
            vis.figure()
            vis.subplot(1,2,1)
            vis.imshow(detection.color_thumbnail)
            vis.subplot(1,2,2)
            vis.imshow(detection.depth_thumbnail)
            vis.show()

        boundingBox = BoundingBox()
        boundingBox.minY = detection.bounding_box.min_pt[0]
        boundingBox.minX = detection.bounding_box.min_pt[1]
        boundingBox.maxY = detection.bounding_box.max_pt[0]
        boundingBox.maxX = detection.bounding_box.max_pt[1]

        try:
            start_time = time.time()
            planned_grasp_data = plan_grasp(inpainted_color_image.rosmsg, inpainted_depth_image.rosmsg, camera_intrinsics.rosmsg, boundingBox)
            grasp_plan_time = time.time() - start_time

            lift_gripper_width, T_gripper_world = process_GQCNNGrasp(planned_grasp_data, robot, left_arm, right_arm, subscriber, home_pose, config)

            # get human label
            human_input = raw_input('Grasp success, or grasp failure? [s/f] ')
            while human_input.lower() != 's' and human_input.lower() != 'f':
                logging.info('Did not understand input. Please answer \'s\' or \'f\'')
                human_input = raw_input('Grasp success, or grasp failure? [s/f] ')
            if human_input.lower() == 's':
                experiment_logger.update_trial_attribute('human_label', 1)
            else:
                experiment_logger.update_trial_attribute('human_label', 0)
            
            # log result
            experiment_logger.update_trial_attribute('gripper_pose', T_gripper_world)
            experiment_logger.update_trial_attribute('planning_time', grasp_plan_time)
            experiment_logger.update_trial_attribute('gripper_width', lift_gripper_width)
            experiment_logger.update_trial_attribute('found_grasp', 1)
            experiment_logger.update_trial_attribute('completed', True)
            experiment_logger.update_trial_attribute('object_key', obj)
            trial_number += 1

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: \n %s" % e)  
            experiment_logger.update_trial_attribute('found_grasp', 0)
            experiment_logger.update_trial_attribute('completed', True)

            experiment_logger.update_trial_attribute('object_key', obj)     
            trial_number += 1
        except (YuMiCommException, YuMiControlException) as yce:
            rospy.logerr(str(yce))
            if sensor is not None:
                sensor.stop()
            if robot is not None:
                robot.stop()
            if subscriber is not None and subscriber._started:
                subscriber.stop()        
            rospy.loginfo("Re-trying")
            re_try = True
            robot, subscriber, left_arm, right_arm, home_pose = init_robot(config)

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Yumi_Control_Node')

    config = YamlConfig('/home/autolab/Workspace/vishal_working/catkin_ws/src/gqcnn/cfg/ros_nodes/yumi_control_node.yaml')

    rospy.loginfo('Loading Gripper')
    gripper = RobotGripper.load('yumi_metal_spline')

    rospy.loginfo('Loading T_camera_world')
    T_camera_world = RigidTransform.load('/home/autolab/Public/alan/calib/primesense_overhead/primesense_overhead_to_world.tf')

    detector_cfg = config['detector']

    # create rgbd sensor
    rospy.loginfo('Creating RGBD Sensor')
    sensor_cfg = config['sensor_cfg']
    sensor_type = sensor_cfg['type']
    sensor = RgbdSensorFactory.sensor(sensor_type, sensor_cfg)
    sensor.start()
    rospy.loginfo('Sensor Running')

    # setup safe termination
    def handler(signum, frame):
        logging.info('caught CTRL+C, exiting...')        
        if sensor is not None:
            sensor.stop()
        if robot is not None:
            robot.stop()
        if subscriber is not None and subscriber._started:
            subscriber.stop()            
        exit(0)
    signal.signal(signal.SIGINT, handler)
    
    # run experiment
    run_experiment()

    rospy.spin()
