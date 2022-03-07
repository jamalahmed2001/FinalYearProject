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
		self.X = [2,-2,-2,2,0,-2,-1.5,1,2,2,-1,1]
		self.Y = [2,-2,2,-2,-2,-1,1,0.5,0.5,1,0,-1.5]
		# self.X = [1, -6.5, -6.5, 6.5, 7,4, 1,-3,-4 ]
		# self.Y = [2, -3 , 3, -4, 4, 4, 4, 4, 1]
		self.targetIndex = 0
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

	def resetRobot(self):
		state_msg = ModelState()
		state_msg.model_name = 'turtlebot3_waffle_pi'
		index = random.randint(0,len(self.X)-1)
		while index== self.targetIndex:
			index = random.randint(0,len(self.X)-1)
		state_msg.pose.position.x = self.X[index]
		state_msg.pose.position.y = self.Y[index]
		state_msg.pose.position.z = 0.1
		state_msg.pose.orientation.x = 0
		state_msg.pose.orientation.y = 0
		state_msg.pose.orientation.z = 0
		state_msg.pose.orientation.w = 0
		rospy.wait_for_service('/gazebo/set_model_state')
		try:
			set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			resp = set_state( state_msg )
		except:
			print("Service call failed: %s")
		self.State = []

	def resetTarget(self):
		if self.targetIndex==0:
			for i in range(0,len(self.X)):
				self.removeTarget(i)
		else:
			self.removeTarget(self.targetIndex)
		index = random.randint(0,len(self.X)-1)
		time.sleep(0.5)
		self.addTarget(index)
		self.targetIndex = index
		self.startTime = time.time()


	def setState(self,image):
		try:
		  cv_image = bridge.imgmsg_to_cv2(image, "passthrough")
		except CvBridgeError:
		  rospy.logerr("CvBridge Error")
		self.State.append(np.uint8(cv2.flip(cv_image,1)))

	def setPos(self,position):
		self.prevPos = self.pos
		self.pos = position.pose.pose.position
		self.pos.x = roundSig(self.pos.x,3)
		self.pos.y = roundSig(self.pos.y,3)
		self.pos.z = roundSig(self.pos.z,3)
		self.orientation = position.pose.pose.orientation

	def setLasers(self,laserData):
		mid = len(laserData.ranges)//2
		self.laserData  = (list(laserData.ranges)[mid:])
		self.laserData.extend(list(laserData.ranges)[:mid])

	def getAngleToTarget(self):
		try:
			a = tf.transformations.euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
			optimal = a[2]*(180 / math.pi )
			angle = math.atan2(self.Y[self.targetIndex]-self.pos.y, self.X[self.targetIndex]-self.pos.x)  *(180 / math.pi )
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
		move.linear.x = 0.25
		if action==torch.Tensor([0]):
			move.angular.z = -0.35
			move.linear.x = 0
		elif action==torch.Tensor([1]):
			move.angular.z = -0.25
		elif action==torch.Tensor([2]):
			move.angular.z = 0
			# move.linear.x = 0.25
		elif action==torch.Tensor([3]):
			move.angular.z = 0.25
		elif action==torch.Tensor([4]):
			move.angular.z = 0.35
			move.linear.x = 0
		# move.linear.x = 0.1
		self.Pub.publish(move)
		rate.sleep()


	def getReward(self,action):
		elapsed = time.time()-self.startTime
		print("T.S.L - " + str(elapsed))
		print("Prev Time - ", str(self.prevTime))
		self.actionsTaken += 1
		self.takeAction(action)
		action = torch.argmax(action)
		self.Found[1] = self.actionsTaken
		print(self.Found)

		difference = self.State[-1]
		self.State = []

		if (self.X[self.targetIndex]-0.175 <= self.pos.x <= self.X[self.targetIndex]+0.175) and (self.Y[self.targetIndex]-0.175 <= self.pos.y <= self.Y[self.targetIndex]+0.175):
			self.stuck=False
			self.Found[0]+=1
			self.prevTime = elapsed
			print()
			print("FOUND ITT")
			print()
			self.resetTarget()
			self.actionsTaken  =0
			# self.resetRobot()
			return difference,1000, True

			#make a reward function based on:
			#distance to target
			#orientation to target
			#laser distance around object?

		angle = self.getAngleToTarget()
		Reward = 2-2**(angle/20)#reward for orientation
		Reward/=1000
		# print("AngleReward - ",Reward)
		distance = ((((self.X[self.targetIndex] - self.pos.x )**2) + ((self.Y[self.targetIndex]-self.pos.y)**2) )**0.5)
		Reward+= (2-(2**distance))/1000
		# print("DistAngleReward - ",Reward)

		laserTotal = 0
		for i in range(0,len(self.laserData)):
			if self.laserData[i]==np.inf:
				laserTotal+=10
			else:
				laserTotal+=self.laserData[i]
		laserTotal/=1000
		# print(laserTotal)
		Reward+=laserTotal

		# if angle<15:
			# distance = ((((self.X[self.targetIndex] - self.pos.x )**2) + ((self.Y[self.targetIndex]-self.pos.y)**2) )**0.5)
			# Reward = -abs((distance-2)**2)
		# Reward=-1
		inSight = False
		image = cv2.resize(difference, (84, 84))
		for x in range(0,len(image)-1):
			for y in range(0,len(image[x])):
				if image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
					inSight = True


		self.prevAngle = abs(angle)
		#checks for collision
		for i in range(0,len(self.laserData)):
			#front collision
			if i > 30 and i<50:
				if self.laserData[i]<0.175:
					if action==torch.Tensor([2]) or action==torch.Tensor([3]) or action==torch.Tensor([1]):
						self.Found[-2]-=1
						self.startTime = time.time()
						# self.resetTarget()
						self.resetRobot()
						self.State = []
						self.actionsTaken  =0
						return difference,-1000,True
					# return difference,Reward,False

		if self.prevPos==self.pos:
			self.stuck+=1;
			if self.stuck>=3 and (action==torch.Tensor([1]) or action==torch.Tensor([2]) or action==torch.Tensor([3])):
				return difference,-900,True
			else:
				if inSight:
					return difference,Reward-self.stuck,False
				else:
					return difference,-1-self.stuck,False
		else:
			self.stuck=0
			if inSight:
				return difference,Reward-self.stuck,False
			else:
				return difference,-1-self.stuck,False
def roundSig(num,sig):
	return round(num, sig-int(floor(log10(abs(num))))-1)

bridge = CvBridge()
