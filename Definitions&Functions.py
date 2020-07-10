# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:46:16 2020

@author: schep
"""

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


from IPython.display import clear_output
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")




##### Create Environment ####

from multiprocessing_env import SubprocVecEnv


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk



####  Create Neural Network ####
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        # NN is design for continious action tasks
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        # sample actions from gaussian distribution
        dist  = Normal(mu, std)
        return dist, value
    


#### Create Plots ####


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(model, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward



#### Compute GAE ####

def compute_gae(next_value, rewards, masks, values, gammas, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gammas * values[step + 1] * masks[step] - values[step]
        gae = delta + gammas * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


#### Define PPO
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        # erzeuge array mit 100 werten, mit randomwerten die zwischen 0 und 500 liegen (für beispiel
        # mit 500 batchsize und 100 minibatchsize; diesen spaß macht er 5 mal)
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_params, model, optimizer):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_params, 1.0 + clip_params) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            # run backpropagation as usual
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


## define configurations
def configurations(gammas, clip_params, lr, hidden_size, tau):
    
    configurations = []
    
    for i in range(len(gammas)):
        parameters = np.array([gammas[i], clip_params[i], lr[i], hidden_size[i], tau[i]])
        configurations.append(parameters)
    # ich muss in python funktionen immer "return" beenden!!!
    return configurations