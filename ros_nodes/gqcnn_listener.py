
""" Provides a service to get planned grasps using the GQCNN """

import rospy

try:
    from gqcnn.srv import *
except ImportError:
    raise RuntimeError("gqcnn_listener service unavailable outside of catkin package")

if __name__ == '__main__':
    rospy.init_node('gqcnn_listener')
    
    def gqcnn_request_handler(req):
        trans = tfBuffer.lookup_transform(req.from_frame, req.to_frame, rospy.Time())
        return RigidTransformListenerResponse(trans.transform.translation.x,
                                              trans.transform.translation.y,
                                              trans.transform.translation.z,
                                              trans.transform.rotation.w,
                                              trans.transform.rotation.x,
                                              trans.transform.rotation.y,
                                              trans.transform.rotation.z)
    s = rospy.Service('gqcnn_listener', GQCNNListener, gqcnn_request_handler)
    
    rospy.spin()