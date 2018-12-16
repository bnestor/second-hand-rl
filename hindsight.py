"""
hindsight.py

hindsight experience replay
"""
import numpy as np
import random


class hindsight():
    """
    """
    def __init__(self, batch_size=16):
        #list of all objects
        self.objects=[]
        self.batch_size=batch_size
    def add(self, states, actions, rewards):
        completed=True
        if rewards [-1]<1:
            completed=False
        self.objects.append((states, actions, rewards, completed))
        if len(self.objects) > self.batch_size*3:
            self.objects=random.sample(self.objects, self.batch_size*10)
    def sample(self, with_hindsight=True):
         # find the index we last sampled from
        samples=random.sample(self.objects, self.batch_size)

        if with_hindsight:
            for i, sample in enumerate(samples):
                states, actions, rewards, completed=sample
                if completed:
                    continue
                #make end state a reward state
                if np.random.rand()>0.5:
                    # select a random index
                    ind=np.random.randint(0, len(rewards)-1)
                    states=states[:ind]
                    actions=actions[:ind]
                    rewards=rewards[:ind]
                    #set the end point as a reward
                    rewards[-1]=1
                    #set the end state as the correct state throughout the network.
                    new_states=[]
                    for state in states:
                        if len(state.shape)==3:
                            state[:,:,2:3]=states[-1][-1,-1,2:3]
                        if len(state.shape)==2:
                            state[:,2:3]=states[-1][-1,2:3]
                        if len(state.shape)==1:
                            state[2:3]=states[-1][2:3]
                        new_states.append(state)
                    samples[i]=(new_states, actions, rewards, True)


        return samples
