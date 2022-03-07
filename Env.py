import rospy, tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from sensor_msgs.msg import Image
import rospkg

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import DeleteModel, SpawnModel

import time
import math
import random
import numpy as np
from math import log10,floor,sqrt
import torch

import cv2
from cv_bridge import CvBridge, CvBridgeError


class Environment():
	def __init__(self):
		self.X = [2,-2,-2,2,0,-2,-1.5,1,2,2,-1,1]
		self.Y = [2,-2,2,-2,-2,-1,1,0.5,0.5,1,0,-1.5]
		self.targetIndex = 0
		self.laserData = []
		self.State = []
		self.pos = []
		self.orientation = []
		self.Pub = rospy.Publisher("cmd_vel",Twist,queue_size=10)
		self.startTime = 0
		self.maxTime = 100
		self.Found = [0,0,0,0]
		self.prevAngle = 180
		self.prevTime = 999999
		self.stuck = False
		self.actionsTaken = 0
		self.resetTarget()
		self.resetRobot()


	def addTarget(self,index):
		rospy.wait_for_service("gazebo/spawn_sdf_model")
		spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
		targetFile = open("/home/jamalahmed2001/model_editor_models/TargetGreen/model.sdf", "r")
		targetXML = targetFile.read()
		targetFile.close()
		t = tf.transformations.quaternion_from_euler(0,0,0)
		orient = Quaternion(t[0],t[1],t[2],t[3])
		item_name = "Target"+str(index)
		print("Spawning model:%s", item_name)
		item_pose   =   Pose(Point(x=self.X[index], y=self.Y[index],    z=0),   orient)
		spawn_model(item_name, targetXML, "", item_pose, "world")

	def removeTarget(self,index):
		rospy.wait_for_service("gazebo/delete_model")
		delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
		item_name = "Target"
		print("Deleting model:%s", item_name)
		delete_model(item_name+str(index))
