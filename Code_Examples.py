# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:41:04 2020

@author: schep
"""


## Create Environments ##
num_envs = 4
env_name = "Pendulum-v0"

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)




## Example 1: Check tau parameter (smoothing parameter of PPO) ##

# generate configurations: 
#  - the length of each hyperparameter tuple must be the same (2 in this example)
#  - the length of the hyperparameter tuple determines the number of configurations (2 in this example)
#  - each configuration is build according to the positions of the hyperparameters
#  - in this example the first configuration has the following hyperparameters (according to the first
#    hyperparameter elements): 
#       gamma 0.9, clip_params 0.2, lr 3e-4, hidden_size 256, tau 0.95
#  - in this example the second configuration has the following hyperparameters (according to the second 
#    hyperparameter elements):
#       gamma 0.9, clip_params 0.2, lr 3e-4, hidden_size 256, tau 0.90


gammas = 0.9,0.9
clip_params = 0.2,0.2
lr = 3e-4, 3e-4
hidden_size = 256, 256
tau = 0.95, 0.9

configurations = configurations(gammas, clip_params, lr, hidden_size, tau)

# define names of benchmark curves
curve_names = "tau_0.95","tau_0.9"
    
# run benchmark
run_PPO(configurations, curve_names, envs, env, num_steps=20, ppo_epochs=4, mini_batch_size = 5, max_frames = 1500, modulo = 100)




# Example 2: Check clip parameter (clipping of PPO)
gammas = 0.9,0.9
clip_params = 0.1,0.4
lr = 3e-4, 3e-4
hidden_size = 256, 256
tau = 0.95, 0.95


configurations = configurations(gammas, clip_params, lr, hidden_size, tau)

curve_names = "clip_0.1","tau_0.4"
    

run_PPO(configurations, curve_names, envs, env, num_steps=20, ppo_epochs=4, mini_batch_size = 5, max_frames = 1500, modulo = 100)
