# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:39:04 2020

@author: schep
"""


import numpy as np
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    Die ganzen Funktionen, wie step_async, step_wait etc. werden bei SubprocVecEnv aufgerufen
    und auch erst dort richtig definiert
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    
class CloudpickleWrapper(object):
    """
    Wird bei SubprocVecEnv aufgerufen
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    CloudpickleWrapper saves env_fn function in x
    __getstate__ function returns the serialized representation of self.x
    if you save the thing returned from cloudpickle.dumps(self.x) to a file, you can later 
    load it by calling def __setstate__ to retrieve all the attributes
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
    
        # Pipe() ensures, that the two connectes objects can communicate to each other
        # self.remotes correspond to the list of parent nodes and 
        # self.work_remotes correspond to the list of the corresponding child nodes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        # starts process, with target (here workers); args are simply the arguments applied to
        # target (workers); with start() below, ps is run
        # CloudpickleWrapper (siehe oben wie definiert) saves env_fn function in x
        #  __getstate__ function returns the serialized representation of self.x
        # if you save the thing returned from cloudpickle.dumps(self.x) to a file, you can later 
        # load it by calling def __setstate__ to retrieve all the attributes
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        # VecEnv’s init function is called 
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    # for each step, the step_async function is run and the step_wait function is returned
    # the remote sends the string “step” and the action to its child and set the variable self.waiting to True    
    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    # We see that remote receive back a message from its child which is the states 
    # of the environment! We see that it returns obs, observations, rews, rewards, dones,
    # whether it’s over or not, and infos
    # Then, after some processing, it returns them! Now, let’s look at the worker, 
    # the function that actually runs everything(the target of the processes)
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs