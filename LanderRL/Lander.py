import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import random

GymEnvironment = 'LunarLander-v2'

env = gym.make(GymEnvironment)
env.seed(random.randint(0, 100000))
torch.manual_seed(random.randint(0, 100000))

#Hyperparameters
learning_rate = 0.01
gamma = 0.99

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]

        self.action_space = env.action_space.n
        
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        #self.l2 = nn.Linear(128, 128, bias=False)
        self.l3 = nn.Linear(128, self.action_space, bias=False)
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            # self.l2,
            # nn.Dropout(p=0.6),
            # nn.ReLU(),
            self.l3,
            nn.Softmax(dim=-1)
        )
        return model(x)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action


def update_policy():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []


total_rewards = []

def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        done = False       

        total_reward = 0

        for frame in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data[0])
            total_reward += reward
            # if episode % 50 == 0: 
            #     env.render()
            #     time.sleep(0.020)

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                print(total_reward, running_reward)
                total_rewards.append(total_reward)
                break
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (frame * 0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, frame, running_reward))

        # Stop when the last 5 runs have averaged a good score.
        if len(total_rewards) > 5:
            if np.mean(total_rewards[-5:]) > 0:
                print("average of the last 5 episodes = ", np.mean(total_rewards[-5:]))
                break


episodes = 10000
main(episodes)

plt.plot(total_rewards)
plt.show()

while True:

    state = env.reset() # Reset environment and record the starting state
    done = False       

    total_reward = 0

    for frame in range(1000):
        action = select_action(state)
        # Step through environment using chosen action
        state, reward, done, _ = env.step(action.data[0])
        total_reward += reward
        env.render()
        time.sleep(0.020)

        if done:
            print(total_reward)
            break
