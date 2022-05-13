#!/usr/bin/env python3
import os,glob
import random
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import cv2
from SimpleEnvironment import Environment
import rospy, tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from sensor_msgs.msg import Image,LaserScan

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.number_of_actions =5
		self.gamma = 0.975
		self.final_epsilon = 0.05 # 0.0001
		self.initial_epsilon = 0.5 # 0.1
		self.number_of_iterations = 50#1000
		self.replay_memory_size = 10000
		self.minibatch_size = 32
		#4 frames, 32 chann out,, 8x8kernel, stride 4
		self.conv1 = nn.Conv2d(4 , 16, 8, 4)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(16, 32, 4, 2)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(32, 64, 3, 1)
		self.relu3 = nn.ReLU(inplace=True)
		self.fc4 = nn.Linear(3136, 512)
		self.relu4 = nn.ReLU(inplace=True)
		self.fc5 = nn.Linear(512, self.number_of_actions)

	def forward(self, x):
		out = self.conv1(x)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.relu2(out)
		out = self.conv3(out)
		out = self.relu3(out)
		out = out.view(out.size()[0], -1)
		out = self.fc4(out)
		out = self.relu4(out)
		out = self.fc5(out)
		return out

def InitialiseWeights(m):
	if type(m) == nn.Conv2d or type(m) == nn.Linear:
		torch.nn.init.uniform(m.weight, -0.01, 0.01)
		m.bias.data.fill_(0.01)

def InputToTensor(image):
	image_tensor = image.transpose(2, 0, 1)
	image_tensor = image_tensor.astype(np.float32)
	image_tensor = torch.from_numpy(image_tensor)
	if torch.cuda.is_available():  # put on GPU if CUDA is available
		image_tensor = image_tensor.cuda()
	return image_tensor

def FormulateInput(image,env):
	#check for green and makes white so in conversion it has max contrast
	for x in range(0,len(image)-1):
		for y in range(0,len(image[x])):
			#Image is BGR not RGB. Green was used just as it would prevent any conversion issues
			if 100<image[x][y][1]<110 and image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
				image[x][y][0]=255
				image[x][y][1]=255
				image[x][y][2]=255
	image = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)#converts to greyscale
	#normalises input for cv2 format
	normalisedImage = np.zeros((85,84))
	for x in range(0,len(image)-1):
		for y in range(0,len(image[x])):
			normalisedImage[x][y] = image[x][y]/255
	rawLaser = list(env.laserData)
	for i in range(0,len(rawLaser)):
		if rawLaser[i]==np.inf:
			rawLaser[i] = 10
		rawLaser[i] = round(rawLaser[i],1)

	normalisedImage[-1] = rawLaser
	normalisedImage = np.reshape(normalisedImage, (85, 84, 1))
	# cv2.imshow("Camera view diff", normalisedImage)#
	# cv2.waitKey(3)
	return normalisedImage


def train(model, start,envmode):
	# define Adam optimizer
	learningrate = 0.00000015625#0.369#0.000005
	optimizer = optim.Adam(model.parameters(), lr=learningrate)#0.0025)
	# initialize mean squared error loss
	criterion = nn.MSELoss()
	env = Environment(envmode)
	rate = rospy.Rate(1)
	image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, env.setState)
	odom_sub = rospy.Subscriber('/odom', Odometry, env.setPos)
	laser_sub = rospy.Subscriber('/scan', LaserScan, env.setLasers)
	rate.sleep()

	# initialize replay memory
	ReplayMemory = []

	# get action
	action = torch.zeros([model.number_of_actions], dtype=torch.float32)
	action[0] = 1
	returnedstate, reward, terminal = env.getReward(action)
	returnedstate = FormulateInput(returnedstate,env)
	returnedstate = InputToTensor(returnedstate)
	state = torch.cat((returnedstate, returnedstate, returnedstate, returnedstate)).unsqueeze(0)

	# initialize training value
	epsilon = model.initial_epsilon
	episode = 0
	epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
	taken = [0,0]
	PrevEpisodeReward =[-1000]
	prevcoll = 0
	CurrentEpisodeReward = 0
	EPSDECAY = 0.99
	# main loop
	while episode < model.number_of_iterations:
		optimizer = optim.Adam(model.parameters(), lr=learningrate)#0.0025)
		output = model(state)[0]
		action = torch.zeros([model.number_of_actions], dtype=torch.float32)
		# epsilon greedy exploration
		# print(round(epsilon,1))
		Random = random.random() <= epsilon#ound(epsilon,1)
		if Random:
			print("Performed random action!")
			AIndex = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
		else:
			AIndex = torch.argmax(output)
		action[AIndex] = 1

		# get next state and reward
		nextReturnedState, reward, terminal = env.getReward(action)
		nextReturnedState = FormulateInput(nextReturnedState,env)
		nextReturnedState = InputToTensor(nextReturnedState)
		nextState = torch.cat((state.squeeze(0)[1:, :, :], nextReturnedState)).unsqueeze(0)

		action = action.unsqueeze(0)
		reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
		ReplayMemory.append((state, action, reward, nextState, terminal))
		# if replay memory is full, remove the oldest transition
		if len(ReplayMemory) > model.replay_memory_size:
			ReplayMemory.pop(0)

		#action based decay
		epsilon*=(EPSDECAY)
		if epsilon<=model.final_epsilon:
			epsilon = epsilon_decrements[episode]

		# #reward based
		# # EPSRatio = abs(CurrentEpisodeReward/(fails*10*PrevEpisodeReward[episode]))
		# EPSRatio = abs(CurrentEpisodeReward/(PrevEpisodeReward[-1]))
		# #eps ratio goes negative when prev is positive
		# # EPSRatio = abs(CurrentEpisodeReward-abs(fails*PrevEpisodeReward[episode])/PrevEpisodeReward[episode]))
		# # if EPSRatio>=1:
		# # 	fails+=1
		# # 	# EPSRatio = abs(CurrentEpisodeReward-abs(fails*PrevEpisodeReward[episode])/PrevEpisodeReward[episode]))
		# # 	EPSRatio = abs(CurrentEpisodeReward/(fails*10*PrevEpisodeReward[episode]))
		# epsilon= convertRange(EPSRatio,0,1,epsilon_decrements[episode],model.final_epsilon)
		# print("EPSRATIO - ", EPSRatio)


		CurrentEpisodeReward+=float(reward)
		print("REWARDS")
		print(PrevEpisodeReward,float(CurrentEpisodeReward))
		# sample random minibatch
		minibatch = random.sample(ReplayMemory, min(len(ReplayMemory), model.minibatch_size))
		# unpack minibatch
		stateBatch = torch.cat(tuple(d[0] for d in minibatch))
		actionBatch = torch.cat(tuple(d[1] for d in minibatch))
		rewardBatch = torch.cat(tuple(d[2] for d in minibatch))
		nextStateBatch = torch.cat(tuple(d[3] for d in minibatch))
		# get output for the next state
		outputBatch = model(nextStateBatch)
		#DQN Update
		# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
		y_batch = torch.cat(tuple(rewardBatch[i] if minibatch[i][4]
						  else rewardBatch[i] + model.gamma * torch.max(outputBatch[i])
						  for i in range(len(minibatch))))
		# get Q-value
		q_value = torch.sum(model(stateBatch) * actionBatch, dim=1)

		#not needed as reinit of optimiser donee every action
		# # PyTorch accumulates gradients by default so reset in each pass
		# optimizer.zero_grad()
		# returns a new Tensor, detached from the current graph, the result will never require gradient
		y_batch = y_batch.detach()
		# calculate loss
		loss = criterion(q_value, y_batch)
		# backward pass
		loss.backward()
		optimizer.step()
		# set state to be nextState
		state = nextState

		if terminal:
			#adaptive learning
			#commenting 214 and 215 turns this off
			ageSuccess = (abs(env.Found[-2])-prevcoll)/(env.Found[0]+1)
			EpisodeSuccess =  (abs(env.Found[-2])-prevcoll)#finds number of collisons in a given episode
			if EpisodeSuccess>10:
				EpisodeSuccess = int(str(EpisodeSuccess)[-1])

			EPSDECAY = convertRange(EpisodeSuccess,0,10,0.99,0.999)
			learningrate = abs(convertRange(ageSuccess,0,10,0.00000015625,0.000000015625))# random action taking so generalise about good data and reinforce negative
			print()
			print("LR - ",learningrate)
			print("ED - ",EPSDECAY)
			print()

			PrevEpisodeReward.append(CurrentEpisodeReward)
			CurrentEpisodeReward = 0
			if reward>=10:
				prevcoll = abs(env.Found[-2])
				episode += 1
				#episodic decay
				epsilon = epsilon_decrements[episode]
				taken.append(0)
		#increment actions counter
		taken[0]+=1
		taken[episode+1]+=1

		if episode % 5 == 0:
			name = str(model.gamma)+"_"+str(model.final_epsilon)+"_"+str(model.initial_epsilon)+"_"+str(model.number_of_iterations)+"_"+str(learningrate)+"_"+str(model.minibatch_size)+"_"
			torch.save(model, "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/"+name + str(episode) + ".pth")
		elif PrevEpisodeReward[episode-1]>0:
			name = str(model.gamma)+"_"+str(model.final_epsilon)+"_"+str(model.initial_epsilon)+"_"+str(model.number_of_iterations)+"_"+str(learningrate)+"_"+str(model.minibatch_size)+"_"
			torch.save(model, "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/"+name + str(episode) + ".pth")
		print()
		print("\nEpisode:", episode,"\nactionsPerEpisode:",taken, "\nTraining Time:", time.time() - start, "\nEpsilon:", epsilon,"\nAction:",
			  AIndex.cpu().detach().numpy(), "\nreward:", reward.numpy()[0][0], "\nQ max:",
			  np.max(output.cpu().detach().numpy()))


def convertRange(value, leftMin, leftMax, rightMin, rightMax):
	# Figure out how breadth of ranges
	leftSpan = leftMax - leftMin
	rightSpan = rightMax - rightMin
	# Convert the left range into a 0-1 range (float)
	valueScaled = float(value - leftMin) / float(leftSpan)
	# Convert the 0-1 range into a value in the right range.
	return rightMin + (valueScaled * rightSpan)

def test(model,envmode):
	env = Environment(envmode)
	rate = rospy.Rate(1)
	image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, env.setState)
	odom_sub = rospy.Subscriber('/odom', Odometry, env.setPos)
	laser_sub = rospy.Subscriber('/scan', LaserScan, env.setLasers)
	rate.sleep()

	action = torch.zeros([model.number_of_actions], dtype=torch.float32)
	action[0] = 1
	returnedstate, reward, terminal = env.getReward(action)
	returnedstate = FormulateInput(returnedstate,env)
	returnedstate = InputToTensor(returnedstate)
	state = torch.cat((returnedstate, returnedstate, returnedstate, returnedstate)).unsqueeze(0)

	while True:
		# get output from the neural network
		output = model(state)[0]
		action = torch.zeros([model.number_of_actions], dtype=torch.float32)
		# get action
		AIndex = torch.argmax(output)
		action[AIndex] = 1
		# get next state
		nextReturnedState, reward, terminal = env.getReward(action)
		nextReturnedState = FormulateInput(nextReturnedState,env)
		nextReturnedState = InputToTensor(nextReturnedState)
		nextState = torch.cat((state.squeeze(0)[1:, :, :], nextReturnedState)).unsqueeze(0)
		# set state to be nextState
		state = nextState

def main(mode,envmode):
	cuda_is_available = torch.cuda.is_available()
	if mode == 'test':
		models = glob.glob("/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/*.pth")
		latest = max(models,key=os.path.getctime)
		latest = "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/#10x10FinalModel.pth"
		print(latest)
		model = torch.load(
			latest,
			map_location='cpu' if not cuda_is_available else None
		).eval()
		if cuda_is_available:  # put on GPU if CUDA is available
			model = model.cuda()
		test(model,envmode)
	elif mode == 'train':
		if not os.path.exists('./pretrained_model/'):
			os.mkdir('pretrained_model/')
		model = NeuralNetwork()
		if cuda_is_available:  # put on GPU if CUDA is available
			model = model.cuda()
		model.apply(InitialiseWeights)
		start = time.time()
		train(model, start,envmode)
	elif mode == "load":
		models = glob.glob("/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/*.pth")
		latest = max(models,key=os.path.getctime)
		latest = "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/#10x10FinalModel.pth"
		print(latest)
		model = torch.load(
			latest,
			map_location='cpu' if not cuda_is_available else None
		)
		# model.number_of_actions =5
		# model.gamma = 0.933
		model.initial_epsilon = 0.1# 0.1
		model.final_epsilon = 0.05 # 0.0001
		model.number_of_iterations = 100 #10000
		model.replay_memory_size = 100000
		model.minibatch_size = 64
		start = time.time()
		train(model, start,envmode)

if __name__ == "__main__":
	rospy.init_node("Model_Interaction",disable_signals=True)
	main(sys.argv[1],sys.argv[2])
