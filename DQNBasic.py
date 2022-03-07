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
from environment import Environment
import rospy, tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from sensor_msgs.msg import Image,LaserScan
import matplotlib.pyplot as plt

plt.ion()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions =5
        self.gamma = 0.4
        self.final_epsilon = 0.05 # 0.0001
        self.initial_epsilon = 1# 0.1
        self.number_of_iterations = 250#10000
        self.replay_memory_size = 1000000
        self.minibatch_size = 32
        #4 frames, 32 chann out,, 8x8kernel, stride 4
        self.conv1 = nn.Conv2d(4 , 30, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(30, 15, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(15, 64, 3, 1)
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

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor

##no laser image resize
# def resize_and_bgr2gray(image,game_state):
#     # image = image[0:288, 0:404]
#     # cv2.imshow("Camera view diff", image)#
#     # cv2.waitKey(3)
#     image = cv2.resize(image, (84, 84))
#     newimage = np.zeros((84,84))
#     # BGR form
#     for x in range(0,len(image)):
#         for y in range(0,len(image[x])):
#             if image[x][y][2]>image[x][y][0] and image[x][y][2]>image[x][y][1]: #red
#                 newimage[x][y] = 0.1#*(image[x][y][2]/(image[x][y][2]+image[x][y][1]+image[x][y][0]))
#                 # print("r")
#             elif image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
#                 newimage[x][y]=255
#                 # print("g")
#                 # print("g")
#             elif image[x][y][0]>image[x][y][1] and image[x][y][0]>image[x][y][2]:#blue
#                 newimage[x][y]=0.5
#                 # print("b")
#             else:
#                 newimage[x][y] = 0
#
#     image_data = newimage
#     cv2.imshow("Camera view diff", newimage)#
#     cv2.waitKey(3)
#     #if red value =255 make 84x84 matrix
#     # image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
#     # image_data = cv2.resize(image, (84, 84))
#     # image_data[image_data > 0] = 255
#     image_data = np.reshape(image_data, (84, 84, 1))
#     return image_data

def resize_and_bgr2gray(image,game_state):
    # image = image[0:288, 0:404]
    # cv2.imshow("Camera view diff", image)#
    # cv2.waitKey(3)
    image = cv2.resize(image, (84, 84))


    newimage = np.zeros((85,84))
    # BGR form
    for x in range(0,len(image)-1):
        for y in range(0,len(image[x])):
            # if image[x][y][0]>=255:#if blue
            #     newimage[x][y] = 10
            # elif image[x][y][1]>=255:#green
            #     newimage[x][y] = 10

            if image[x][y][2]>image[x][y][0] and image[x][y][2]>image[x][y][1]: #red
                newimage[x][y] = 0.1#*(image[x][y][2]/(image[x][y][2]+image[x][y][1]+image[x][y][0]))
                # print("r")
            elif image[x][y][1]>image[x][y][0] and image[x][y][1]>image[x][y][2]:#green
                newimage[x][y]=255
                # print("g")
                # print("g")
            elif image[x][y][0]>image[x][y][1] and image[x][y][0]>image[x][y][2]:#blue
                newimage[x][y]=0.5
                # print("b")
            else:
                newimage[x][y] = 0

            # elif image[x][y][0]<=50 and image[x][y][1]<=50 and image[x][y][2]<=50: #make black
            #     newimage[x][y] = 10
    image_data = newimage
    rawLaser = list(game_state.laserData)
    for i in range(0,len(rawLaser)):
        if rawLaser[i]==np.inf:
            rawLaser[i] = 10
        rawLaser[i] = (round(rawLaser[i],1)/10)

        # if rawLaser[i]>0.5:
        #     rawLaser[i] = 1
        # else:
        #     rawLaser[i] = 0

    image_data[-1] = rawLaser
    # print(rawLaser)
    image_data = np.reshape(image_data, (85, 84, 1))
    cv2.imshow("Camera view diff", newimage)#
    cv2.waitKey(3)
    return image_data


def train(model, start):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00015)#0.0025)
    # initialize mean squared error loss
    criterion = nn.MSELoss()
    # instantiate game
    game_state = Environment()
    rate = rospy.Rate(1)
    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, game_state.setState)
    odom_sub = rospy.Subscriber('/odom', Odometry, game_state.setPos)
    laser_sub = rospy.Subscriber('/scan', LaserScan, game_state.setLasers)
    rate.sleep()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.getReward(action)
    image_data = resize_and_bgr2gray(image_data,game_state)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    taken = [0,0]
    fails = 1
    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.getReward(action)
        image_data_1 = resize_and_bgr2gray(image_data_1,game_state)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        replay_memory.append((state, action, reward, state_1, terminal))
        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)
        # epsilon annealing
        # epsilon = epsilon_decrements[iteration]

        epsilon*=(0.99)
        # print(iteration+1,taken[iteration+1]+1,taken[0]+1,fails)
        # epsilon*=(0.975**(0.05*(  ((iteration+1)/((taken[iteration+1]+1)/(taken[0]+1))) /fails)))
        #
        # if taken[iteration+1]%10==0:
        #     if (iteration+(taken[iteration+1]/10)) <len(epsilon_decrements):
        #         epsilon = epsilon_decrements[int(iteration+(taken[iteration+1]/10)) ]
        #     if taken[iteration+1]%100 == 0:
        #         # game_state.resetTarget()
        #         if epsilon<=model.final_epsilon:
        #                 #policy cant find in 100 steps. so try again with less initial randomness
        #             if (iteration+(taken[iteration+1]/100) )<len(epsilon_decrements):
        #                 epsilon = epsilon_decrements[int(iteration+(taken[iteration+1]/100)) ]
        #                 game_state.resetRobot()
        #             else:
        #                 epsilon = epsilon_decrements[iteration-1]
        #                 game_state.resetTarget()
        #                 game_state.resetRobot()
        #         else:
        #                 epsilon = epsilon_decrements[iteration-1]
        #                 game_state.resetTarget()
        #                 # game_state.resetRobot()


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
        # if reward>=50:
        #     iteration += 1
        #     taken.append(0)
        #     fails = 1
        #     epsilon = epsilon_decrements[iteration]
        #     game_state.actionsTaken = 0

        if terminal:
            if reward<0:
                fails+=1
            elif reward>=10:
                iteration += 1
                taken.append(0)
                fails = 1
                epsilon = epsilon_decrements[iteration]
                game_state.actionsTaken = 0

        taken[0]+=1
        taken[iteration+1]+=1

        if iteration % 5 == 0:
            name = str(model.gamma)+"_"+str(model.final_epsilon)+"_"+str(model.initial_epsilon)+"_"+str(model.number_of_iterations)+"_"+str(model.replay_memory_size)+"_"+str(model.minibatch_size)+"_"
            torch.save(model, "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/"+name + str(iteration) + ".pth")
        print()
        print("\nIteration:", iteration,"\nfailed:",fails,"\nactionsPerIteration:",taken, "\nTraining Time:", time.time() - start, "\nEpsilon:", epsilon, "\nEPSDecay:",str((0.99**(0.05*(  ((iteration+1)/((taken[iteration+1]+1)/(taken[0]+1))) /fails)))),"\nAction:",
              action_index.cpu().detach().numpy(), "\nreward:", reward.numpy()[0][0], "\nQ max:",
              np.max(output.cpu().detach().numpy()))


def test(model):
    game_state = Environment()
    rate = rospy.Rate(1)
    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, game_state.setState)
    odom_sub = rospy.Subscriber('/odom', Odometry, game_state.setPos)
    laser_sub = rospy.Subscriber('/scan', LaserScan, game_state.setLasers)
    rate.sleep()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.getReward(action)
    image_data = resize_and_bgr2gray(image_data,game_state)
    image_data = image_to_tensor(image_data)
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
        image_data_1, reward, terminal = game_state.getReward(action)
        image_data_1 = resize_and_bgr2gray(image_data_1,game_state)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

def main(mode):
    cuda_is_available = torch.cuda.is_available()
    if mode == 'test':
        models = glob.glob("/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/*.pth")
        latest = max(models,key=os.path.getctime)
        # latest = "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/best-nolaser-0.5_0.01_0.1_500_1000000_16_310.pth"
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
        model.apply(init_weights)
        start = time.time()
        train(model, start)
    elif mode == "load":
        models = glob.glob("/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/*.pth")
        latest = max(models,key=os.path.getctime)
        # latest = "/home/jamalahmed2001/catkin_ws/src/simulated_homing/src/pretrained_model/best-nolaser-0.5_0.01_0.1_500_1000000_16_310.pth"
        print(latest)
        model = torch.load(
            latest,
            map_location='cpu' if not cuda_is_available else None
        )
        model.number_of_actions =5
        model.gamma = 0.5
        model.final_epsilon = 0.01 # 0.0001
        model.initial_epsilon = 0.5# 0.1
        model.number_of_iterations = 100#10000
        model.replay_memory_size = 1000000
        model.minibatch_size = 64

        start = time.time()
        train(model, start)

if __name__ == "__main__":
    rospy.init_node("Model_Interaction",disable_signals=True)
    main(sys.argv[1])
