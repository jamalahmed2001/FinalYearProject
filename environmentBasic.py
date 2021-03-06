#!/usr/bin/env python3

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
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from math import log10,floor,sqrt
import torch

class Environment():
	def __init__(self):
		self.X = 0
		self.Y = 0
		self.State = []
		self.pos = []
		self.orientation = []
		self.laserData = []
		self.Pub = rospy.Publisher("cmd_vel",Twist,queue_size=10)
		self.startTime = 0
		self.maxTime = 60
		self.Found = [0,0,0,0]
		self.numActions = 5
		self.prevAngle = 180
		self.prevTime = 999999
		self.stuck = False
		self.actionsTaken = 0
		self.inSight = False
		self.resetRobot()
		self.resetTarget()

	def addTarget(self):
		rospy.wait_for_service("gazebo/spawn_sdf_model")
		spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
		targetFile = open("/home/jamalahmed2001/model_editor_models/TargetGreen/model.sdf", "r")
		targetXML = targetFile.read()
		targetFile.close()
		t = tf.transformations.quaternion_from_euler(0,0,0)
		orient = Quaternion(t[0],t[1],t[2],t[3])
		item_name = "Target"
		print("Spawning model:%s", item_name)
		item_pose   =   Pose(Point(x=self.X, y=self.Y,    z=0),   orient)
		spawn_model(item_name, targetXML, "", item_pose, "world")

	def removeTarget(self):
		rospy.wait_for_service("gazebo/delete_model")
		delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
		item_name = "Target"
		print("Deleting model:%s", item_name)
		delete_model(item_name)

	def resetRobot(self):
		state_msg = ModelState()
		state_msg.model_name = 'turtlebot3_waffle_pi'
		state_msg.pose.position.x = 0
		state_msg.pose.position.y = 0
		state_msg.pose.position.z = 0.1
		state_msg.pose.orientation.x = 0
		state_msg.pose.orientation.y = 0
		state_msg.pose.orientation.z = random.randint(0,1)
		state_msg.pose.orientation.w = random.randint(0,1)
		rospy.wait_for_service('/gazebo/set_model_state')
		try:
			set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			resp = set_state( state_msg )
		except:
			print("Service call failed: %s")
		self.State = []

	def resetTarget(self):
		self.removeTarget()
		self.Y = random.uniform(-1.5,1.5)
		self.X = random.uniform(-1.5,1.5)
		while -0.5<self.X<0.5:
			self.X = random.uniform(-1.5,1.5)
		while -0.5<self.Y<0.5:
			self.Y = random.uniform(-1.5,1.5)
		# self.X = random.uniform(-0.75,0.75)
		# self.Y = random.uniform(-0.75,0.75)
		time.sleep(0.75)
		self.addTarget()
		self.startTime = time.time()


	def setState(self,image):
		try:
		  cv_image = bridge.imgmsg_to_cv2(image, "passthrough")
		except CvBridgeError:
		  rospy.logerr("CvBridge Error")
		self.State.append(cv2.resize(np.uint8(cv2.flip(cv_image,1)), (84, 84)))#resize and flip image


	def setPos(self,position):
		self.prevPos = self.pos
		self.pos = position.pose.pose.position
		self.pos.x = roundSig(self.pos.x,3)
		self.pos.y = roundSig(self.pos.y,3)
		self.pos.z = roundSig(self.pos.z,3)
		self.orientation = position.pose.pose.orientation

	def setLasers(self,laserData):
		self.laserData = laserData.ranges
		# left = self.laserData[-60:]
		# right = self.laserData[:60]
		# self.laserData = left + right

	def getAngleToTarget(self):
		try:
			a = tf.transformations.euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
			optimal = a[2]*(180 / math.pi )
			angle = math.atan2(self.Y-self.pos.y, self.X-self.pos.x)  *(180 / math.pi )
			#optimal is bset orientation to ponit
			#angle is actual orientation to point
			diff = optimal - angle
			if diff < -180:
				diff = 360+diff
			else:
				diff = abs(diff)
			return diff
		except:
			return 180
	def takeAction(self,action):
		move = Twist()
		rate = rospy.Rate(1)

		action = ((action == 1).nonzero(as_tuple=True)[0])
		move.linear.x = 0.15
		if action==torch.Tensor([0]):
			move.angular.z = -0.45
			# move.linear.x = 0
		elif action==torch.Tensor([1]):
			move.angular.z = -0.25
		elif action==torch.Tensor([2]):
			move.angular.z = 0
			# move.linear.x = 0.25
		elif action==torch.Tensor([3]):
			move.angular.z = 0.25
		elif action==torch.Tensor([4]):
			move.angular.z = 0.45
			# move.linear.x = 0
		# move.linear.x = 0.1
		self.Pub.publish(move)
		rate.sleep()

	def getReward(self,action):
		elapsed = time.time()-self.startTime
		print("T.S.L - " + str(elapsed))
		print("Prev Time - ", str(self.prevTime))
		if self.actionsTaken>=60:
			self.actionsTaken  =0
		# # if elapsed>=self.maxTime:
			self.resetRobot()
			# self.resetTarget()
			self.startTime = time.time()
			self.State = []
			self.Found[-1]-=1
			return np.zeros((480, 640, 3),dtype=np.float32),0,True
		self.actionsTaken += 1
		self.takeAction(action)
		action = torch.argmax(action)
		self.Found[1] = self.actionsTaken
		print(self.Found)

		difference = self.State[-1]
		self.State = []

		angle = self.getAngleToTarget()
		if (self.X-0.175 <= self.pos.x <= self.X+0.175)and (self.Y-0.175 <= self.pos.y <= self.Y+0.175):
			self.stuck=False
			self.Found[0]+=1
			self.actionsTaken  =0
			self.prevTime = elapsed
			print()
			print("FOUND ITT")
			print()
			self.resetTarget()
			# self.resetRobot()
			return difference,100,True
		for item in self.laserData:
			if item<=0.175:
				self.Found[-2]-=1
				self.actionsTaken  =0
				self.startTime = time.time()
				# self.resetTarget()
				self.resetRobot()
				self.State = []
				return difference,-100,True

		image = difference
		self.inSight = False
		for x in range(0,len(image)-1):
			for y in range(0,len(image[x])):
				if image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
					self.inSight = True
		if self.inSight:
			angle = convertRange(angle,0,180,0,-1) #angle mapped to -1(180 deg) to 1(0deg)
			distance = convertRange(((((self.X- self.pos.x )**2) + ((self.Y-self.pos.y)**2) )**0.5),0,4,0,-1)
			Reward = (0.5*angle)+(0.5*distance)
			self.prevAngle = abs(angle)
			if angle<=self.prevAngle:
				Reward/=2
			else:
				Reward*=2
			return difference,Reward,False
		else:
			return difference,-1,False
			# Reward=-1
			# self.prevAngle = abs(angle)
			# return difference,Reward,False

		# if self.prevPos==self.pos:
		# 	if action==torch.Tensor([2])or action==torch.Tensor([3]) or action==torch.Tensor([1]):
		# 		if self.stuck==True:
		# 			self.stuck=False
		# 			self.Found[-2]-=1
		# 			self.startTime = time.time()
		# 			# self.resetTarget()
		# 			self.resetRobot()
		# 			self.State = []
		# 			return difference,-999, True
		#
		# 		else:
		# 			self.stuck=True
		# 	return difference,Reward,False
		# else:
		# 	self.stuck=False
			# return difference,Reward,False

def roundSig(num,sig):
	return round(num, sig-int(floor(log10(abs(num))))-1)

def convertRange(value, leftMin, leftMax, rightMin, rightMax):
	# Figure out how 'wide' each range is
	leftSpan = leftMax - leftMin
	rightSpan = rightMax - rightMin

	# Convert the left range into a 0-1 range (float)
	if leftSpan>0:
		valueScaled = float(value - leftMin) / float(leftSpan)
		# Convert the 0-1 range into a value in the right range.
		return rightMin + (valueScaled * rightSpan)
	else:
		return 0.5

bridge = CvBridge()
