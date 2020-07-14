# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:39:21 2020

@author: schep
"""


def run_PPO(configurations, curve_names, envs, env, num_steps, ppo_epochs, mini_batch_size, max_frames, modulo):
    
    num_configs = len(configurations)
    
    plot_rewards = []
    
    
    for e in range(num_configs):
        
        num_inputs  = envs.observation_space.shape[0]
        num_outputs = envs.action_space.shape[0]
    
        hidden_size = int(configurations[e][3])
        model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=configurations[e][2])
    
        frame_idx  = 0

        test_rewards = []
    
        state = envs.reset()
        early_stop = False
    
    
        while frame_idx < max_frames and not early_stop:
            
            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0
            
            for _ in range(num_steps):
                
                state = torch.FloatTensor(state).to(device)
                dist, value = model(state)
                
                action = dist.sample()
                next_state, reward, done, _ = envs.step(action.cpu().numpy())
                
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state
                
                frame_idx += 1

                if frame_idx % modulo == 0:
                    test_reward = np.mean([test_env(model) for _ in range(10)])
                    test_rewards.append(test_reward)
                    plot(frame_idx, test_rewards)
                    #if test_reward > threshold_reward: early_stop = True
                    
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = model(next_state)
                    
            returns = compute_gae(next_value, rewards, masks, values, configurations[e][0], configurations[e][4])
        
            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
        
            ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, configurations[e][1], model, optimizer)
        
     
        plot_rewards.append(test_rewards)
    
    x = np.arange(len(plot_rewards[0]))


    for s in range(len(plot_rewards)):
        #plt.figure()
        plt.plot(x,plot_rewards[s], label = curve_names[s]) 
        #plt.plot(x,plot_rewards[s]) 
    plt.legend(loc="upper left")
    plt.savefig("plot", dpi= 300)
    plt.show()
