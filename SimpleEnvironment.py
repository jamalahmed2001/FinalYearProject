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
	def __init__(self,type):
		# self.X = [2,-2,-2,2,0,-2,-1.5,1,2,2,-1,1]
		# self.Y = [2,-2,2,-2,-2,-1,1,0.5,0.5,1,0,-1.5]
		if type=="8x8":
			self.targetPositions = [[random.uniform(-2,2),random.uniform(-2,2)]]
			for i in range(0,20):
				self.targetPositions.append([random.uniform(-2,2),random.uniform(-2,2)])
		elif type=="10x10":
			self.targetPositions = [ [1.5,2], [-1,0],[1.5,-1], [0,3] ,[4,3],[4,-3],[-3,4],[-2,-4],[-4,4]] #10x10
		# self.targetPositions = [ [1,1], [-1,0],[2,-2], [2,2] ,[-2,-2],[-2,2],[-2,-1],[-2,1],[0.5,1.5]] #plaza
		# self.targetPositions = [ [-6.5,-2], [-6.5,3],[-1,4], [-4,1] ,[1,3],[4,1],[6,1],[6.5,-1.5],[2,0.5]] #flat
		self.targetIndex = 0
		self.RobotIndex = 0
		self.laserData = []
		self.State = []
		self.pos = []
		self.orientation = []
		self.Pub = rospy.Publisher("cmd_vel",Twist,queue_size=10)
		self.startTime = 0
		self.maxTime = 100
		self.Found = [0,0,0,0]
		self.actions = 5
		self.prevAngle = 180
		self.prevTime = 999999
		self.stuck = 0
		self.actionsTaken = 0
		self.resetTarget()
		self.resetRobot(True)


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
		item_pose   =   Pose(Point(x=self.targetPositions[index][0], y=self.targetPositions[index][1],    z=0),   orient)
		spawn_model(item_name, targetXML, "", item_pose, "world")

	def removeTarget(self,index):
		rospy.wait_for_service("gazebo/delete_model")
		delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
		item_name = "Target"
		print("Deleting model:%s", item_name)
		delete_model(item_name+str(index))

	def resetRobot(self,new=False):
		state_msg = ModelState()
		state_msg.model_name = 'turtlebot3_waffle_pi'
		if new:
			self.RobotIndex = random.randint(0,len(self.targetPositions)-1)
			while self.RobotIndex == self.targetIndex:
				self.RobotIndex  = random.randint(0,len(self.targetPositions)-1)
		state_msg.pose.position.x = self.targetPositions[self.RobotIndex ][0]
		state_msg.pose.position.y = self.targetPositions[self.RobotIndex ][1]
		state_msg.pose.position.z = 0.1
		state_msg.pose.orientation.x = 0
		state_msg.pose.orientation.y =0
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
		if self.targetIndex==0:
			for i in range(0,len(self.targetPositions)):
				self.removeTarget(i)
		else:
			self.removeTarget(self.targetIndex)
		index = random.randint(0,len(self.targetPositions)-1)
		time.sleep(0.5)
		self.addTarget(index)
		self.targetIndex = index
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
		self.pos.x = roundSig(self.pos.x,4)
		self.pos.y = roundSig(self.pos.y,4)
		self.pos.z = roundSig(self.pos.z,1)
		self.orientation = position.pose.pose.orientation

	def setLasers(self,laserData):
		mid = len(laserData.ranges)//2
		self.laserData  = (list(laserData.ranges)[mid:])
		self.laserData.extend(list(laserData.ranges)[:mid])

	def getAngleToTarget(self):
		try:
			a = tf.transformations.euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
			optimal = a[2]*(180 / math.pi )
			angle = math.atan2(self.targetPositions[self.targetIndex][1]-self.pos.y, self.targetPositions[self.targetIndex][0]-self.pos.x)  *(180 / math.pi )
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
		move.linear.x = 0.3
		if action==torch.Tensor([0]):
			move.angular.z = -0.6
			# move.linear.x = 0
		elif action==torch.Tensor([1]):
			move.angular.z = -0.3
		elif action==torch.Tensor([2]):
			move.angular.z = 0
			# move.linear.x = 0.3
		elif action==torch.Tensor([3]):
			move.angular.z = 0.3
		elif action==torch.Tensor([4]):
			move.angular.z = 0.6
			# move.linear.x = 0
		self.Pub.publish(move)
		rate.sleep()

	def getReward(self,action):
		elapsed = time.time()-self.startTime
		print("T.S.L - " + str(elapsed))
		print("Prev Time - ", str(self.prevTime))
		if self.actionsTaken>=100:
			self.actionsTaken  =0
		# # # if elapsed>=self.maxTime:
			self.resetRobot(True)
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

		if (self.targetPositions[self.targetIndex][0]-0.175 <= self.pos.x <= self.targetPositions[self.targetIndex][0]+0.175)and (self.targetPositions[self.targetIndex][1]-0.175 <= self.pos.y <= self.targetPositions[self.targetIndex][1]+0.175):
			self.stuck=False
			self.Found[0]+=1
			self.actionsTaken  =0
			self.prevTime = elapsed
			print()
			print("FOUND ITT")
			print()
			self.resetTarget()
			self.resetRobot(True)
			return difference,100,True
		for item in self.laserData[len(self.laserData)//2-15:len(self.laserData)//2+15]:
			if item<=0.175:
				self.Found[-2]-=1
				offset = self.actionsTaken
				self.actionsTaken  =0
				self.startTime = time.time()
				# self.resetTarget()
				self.resetRobot(True)
				return difference,-100,True
				# return difference,-200+offset,True

		image = difference
		self.inSight = False
		for x in range(0,len(image)-1):
			for y in range(0,len(image[x])):
				if 100<image[x][y][1]<110 and image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
					self.inSight = True
		maximisingDistance = True
		for l in self.laserData:
			if l>self.laserData[len(self.laserData)//2]:
				maximisingDistance = False

		distance = ((((self.targetPositions[self.targetIndex][0]- self.pos.x )**2) + ((self.targetPositions[self.targetIndex][1]-self.pos.y)**2) )**0.5)
		if self.inSight:
			angle = self.getAngleToTarget()
			angleReward = convertRange(angle,0,90,0,-1) #angle mapped to -1(90 deg) to 1(0deg)
			distanceReward = convertRange(((((self.targetPositions[self.targetIndex][0]- self.pos.x )**2) + ((self.targetPositions[self.targetIndex][1]-self.pos.y)**2) )**0.5),0,6,0,-1)
			Reward = (0.5*angleReward)+(0.5*distanceReward)
			if angle<=self.prevAngle:
				Reward/=2
			# else:
			# 	Reward*=2
			self.prevAngle = abs(angle)
			return difference,Reward,False
		elif maximisingDistance:
			self.prevAngle = 180
			return difference,-0.75,False
		else:
			self.prevAngle = 180
			return difference,-1,False

def roundSig(num,sig):
	return round(num, sig-int(floor(log10(abs(num))))-1)

def convertRange(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

bridge = CvBridge()
