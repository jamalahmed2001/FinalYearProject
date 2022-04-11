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
		self.gamma = 0.95
		self.final_epsilon = 0.05 # 0.0001
		self.initial_epsilon = 0.5 # 0.1
		self.number_of_iterations = 100#10000
		self.replay_memory_size = 100000000
		self.minibatch_size = 64
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

def ImageToTensor(image):
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
			if image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
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

		# if rawLaser[i]>0.5:
		#     rawLaser[i] = 1
		# else:
		#     rawLaser[i] = 0

	normalisedImage[-1] = rawLaser
	# print(rawLaser)
	normalisedImage = np.reshape(normalisedImage, (85, 84, 1))
	# cv2.imshow("Camera view diff", normalisedImage)#
	# cv2.waitKey(3)
	return normalisedImage


def train(model, start):
	# define Adam optimizer
	learningrate = 0.000025
	optimizer = optim.Adam(model.parameters(), lr=learningrate)#0.0025)
	# initialize mean squared error loss
	criterion = nn.MSELoss()
	env = Environment()
	rate = rospy.Rate(1)
	image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, env.setState)
	odom_sub = rospy.Subscriber('/odom', Odometry, env.setPos)
	laser_sub = rospy.Subscriber('/scan', LaserScan, env.setLasers)
	rate.sleep()

	# initialize replay memory
	replay_memory = []

	# initial action is do nothing
	action = torch.zeros([model.number_of_actions], dtype=torch.float32)
	action[0] = 1
	returnedstate, reward, terminal = env.getReward(action)
	returnedstate = FormulateInput(returnedstate,env)
	returnedstate = ImageToTensor(returnedstate)
	state = torch.cat((returnedstate, returnedstate, returnedstate, returnedstate)).unsqueeze(0)

	# initialize epsilon value
	epsilon = model.initial_epsilon
	episode = 0

	epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
	taken = [0,0]
	PrevEpisodeReward =[-1000]
	CurrentEpisodeReward = 0
	fails=1
	# main infinite loop
	while episode < model.number_of_iterations:
		# get output from the neural network
		output = model(state)[0]

		# initialize action
		action = torch.zeros([model.number_of_actions], dtype=torch.float32)
		# epsilon greedy exploration
		random_action = random.random() <= epsilon
		if random_action:
			print("Performed random action!")
			action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
		else:
			action_index = torch.argmax(output)

		action[action_index] = 1

		# get next state and reward
		image_data_1, reward, terminal = env.getReward(action)
		image_data_1 = FormulateInput(image_data_1,env)
		image_data_1 = ImageToTensor(image_data_1)
		state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

		action = action.unsqueeze(0)
		reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
		replay_memory.append((state, action, reward, state_1, terminal))
		# if replay memory is full, remove the oldest transition
		if len(replay_memory) > model.replay_memory_size:
			replay_memory.pop(0)



		# epsiodic decay
		# epsilon = epsilon_decrements[episode]

		#action based decay
		epsilon*=(0.99)
		if epsilon<=model.final_epsilon:
			epsilon = epsilon_decrements[episode]

		#reward based
		# EPSRatio = abs(CurrentEpisodeReward/(fails*10*PrevEpisodeReward[episode]))
		# #eps ratio goes negative when prev is positive
		# # EPSRatio = abs(CurrentEpisodeReward-abs(fails*PrevEpisodeReward[episode])/PrevEpisodeReward[episode]))
		# if EPSRatio>=1:
		# 	fails+=1
		# 	# EPSRatio = abs(CurrentEpisodeReward-abs(fails*PrevEpisodeReward[episode])/PrevEpisodeReward[episode]))
		# 	EPSRatio = abs(CurrentEpisodeReward/(fails*10*PrevEpisodeReward[episode]))
		# epsilon= convertRange(EPSRatio,0,1,epsilon_decrements[episode],model.final_epsilon)
		# print("EPSRATIO - ", EPSRatio)



		CurrentEpisodeReward+=float(reward)
		print("REWARDS")
		print(PrevEpisodeReward,float(CurrentEpisodeReward))
		# sample random minibatch
		minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
		# unpack minibatch
		state_batch = torch.cat(tuple(d[0] for d in minibatch))
		action_batch = torch.cat(tuple(d[1] for d in minibatch))
		reward_batch = torch.cat(tuple(d[2] for d in minibatch))
		state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
		# get output for the next state
		output_1_batch = model(state_1_batch)
		# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
		# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
		y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
						  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
						  for i in range(len(minibatch))))
		# extract Q-value
		q_value = torch.sum(model(state_batch) * action_batch, dim=1)
		# PyTorch accumulates gradients by default so reset in each pass
		optimizer.zero_grad()
		# returns a new Tensor, detached from the current graph, the result will never require gradient
		y_batch = y_batch.detach()
		# calculate loss
		loss = criterion(q_value, y_batch)
		# backward pass
		loss.backward()
		optimizer.step()
		# set state to be state_1
		state = state_1

		if terminal:
			# episodelogfile = open("EpisodeLogFile.txt","a")
			# episodelogfile.write(str(CurrentEpisodeReward)+"\n")
			# episodelogfile.close()


			PrevEpisodeReward.append(CurrentEpisodeReward)
			CurrentEpisodeReward = 0
			if reward>=10:
				episode += 1
				fails = 1
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
			  action_index.cpu().detach().numpy(), "\nreward:", reward.numpy()[0][0], "\nQ max:",
			  np.max(output.cpu().detach().numpy()))


def convertRange(value, leftMin, leftMax, rightMin, rightMax):
	# Figure out how 'wide' each range is
	leftSpan = leftMax - leftMin
	rightSpan = rightMax - rightMin

	# Convert the left range into a 0-1 range (float)
	valueScaled = float(value - leftMin) / float(leftSpan)

	# Convert the 0-1 range into a value in the right range.
	return rightMin + (valueScaled * rightSpan)
def test(model):
	env = Environment()
	rate = rospy.Rate(1)
	image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, env.setState)
	odom_sub = rospy.Subscriber('/odom', Odometry, env.setPos)
	laser_sub = rospy.Subscriber('/scan', LaserScan, env.setLasers)
	rate.sleep()

	# initial action is do nothing
	action = torch.zeros([model.number_of_actions], dtype=torch.float32)
	action[0] = 1
	image_data, reward, terminal = env.getReward(action)
	image_data = FormulateInput(image_data,env)
	image_data = ImageToTensor(image_data)
	state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

	while True:
		# get output from the neural network
		output = model(state)[0]

		action = torch.zeros([model.number_of_actions], dtype=torch.float32)
		if torch.cuda.is_available():  # put on GPU if CUDA is available
			action = action.cuda()

		# get action
		action_index = torch.argmax(output)
		if torch.cuda.is_available():  # put on GPU if CUDA is available
			action_index = action_index.cuda()
		action[action_index] = 1

		# get next state
		image_data_1, reward, terminal = env.getReward(action)
		image_data_1 = FormulateInput(image_data_1,env)
		image_data_1 = ImageToTensor(image_data_1)
		state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

		# set state to be state_1
		state = state_1

def main(mode):
	cuda_is_available = torch.cuda.is_available()
	if mode == 'test':
		models = glob.glob("/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/*.pth")
		latest = max(models,key=os.path.getctime)
		print(latest)
		model = torch.load(
			latest,
			map_location='cpu' if not cuda_is_available else None
		).eval()
		if cuda_is_available:  # put on GPU if CUDA is available
			model = model.cuda()
		test(model)
	elif mode == 'train':
		if not os.path.exists('./pretrained_model/'):
			os.mkdir('pretrained_model/')
		model = NeuralNetwork()
		if cuda_is_available:  # put on GPU if CUDA is available
			model = model.cuda()
		model.apply(InitialiseWeights)
		start = time.time()
		train(model, start)
	elif mode == "load":
		models = glob.glob("/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/*.pth")
		latest = max(models,key=os.path.getctime)
		# latest = "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/!10x10using4x4modelretrain0.85_0.05_0.5_200_2.5e-05_32_195.pth"
		print(latest)
		model = torch.load(
			latest,
			map_location='cpu' if not cuda_is_available else None
		)
		# model.number_of_actions =5
		model.gamma = 0.95

		model.initial_epsilon = 0.3# 0.1
		model.final_epsilon = 0.05 # 0.0001

		model.number_of_iterations = 300 #10000
		# model.replay_memory_size = 10000000
		# model.minibatch_size = 64

		start = time.time()
		train(model, start)

if __name__ == "__main__":
	rospy.init_node("Model_Interaction",disable_signals=True)
	main(sys.argv[1])
