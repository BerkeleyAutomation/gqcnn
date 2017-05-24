#!/usr/bin/env python
import rospy

publisher = None

def subscriber_callback(data):
    """ Callback for detector output """
    
    rospy.loginfo('Received: ' + data)
    
    # process data

    rospy.loginfo('Publishing: ' + output)
    publisher.publish(output)    
    
if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('gqcnn_node')
    rospy.loginfo('GQCNN node initialized')

    #create a subsriber to get detector output that subscribes to the topic 'detector'
    rospy.Subscriber('detector', msg_type, subscriber_callback)

    publisher = rospy.Publisher('gqcnn', msg_type, queue_size=10)

    rospy.spin()