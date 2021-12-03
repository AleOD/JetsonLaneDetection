#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32

while True:
    pub_throttle = rospy.Publisher('throttle', Float32, queue_size=8)
    pub_steering = rospy.Publisher('steering', Float32, queue_size=8)
    rospy.init_node('teleop', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    val = input()
    pub_throttle.publish(-0.2)
    pub_steering.publish(float(val))

    rate.sleep()