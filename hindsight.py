"""
hindsight.py

hindsight experience replay
"""
import numpy as np
import random
import copy


class hindsight():
    """
    """
    def __init__(self, batch_size=16):
        #list of all objects
        self.objects=[]
        self.batch_size=batch_size
        self.buffer_size=3
    def add(self, states, actions, rewards):
        completed=True
        if len(rewards)<3:
            return
        if rewards [-1]<1:
            completed=False
        assert len(states)==len(actions)

        values=copy.deepcopy((states, actions, rewards, completed))
        self.objects.append(values)
        if len(self.objects) > self.batch_size*self.buffer_size:
            # self.objects=random.sample(self.objects, self.batch_size*self.buffer_size)
            idxs=np.random.randint(0, high=len(self.objects), size=self.batch_size*self.buffer_size)
            self.objects=[self.objects[i] for i in idxs]
    def sample(self, with_hindsight=True):
         # find the index we last sampled from
        # samples=random.choices(self.objects, k=self.batch_size)
        # samples=random.sample(self.objects, self.batch_size)

        idxs=np.random.randint(0, high=len(self.objects), size=self.batch_size)
        samples=[self.objects[i] for i in idxs]

        # for i in range(len(self.objects)):
        #     states, actions, rewards, completed=self.objects[i]
        #     print(len(states), len(actions), len(rewards))

        if with_hindsight:
            for i, sample in enumerate(samples):
                states, actions, rewards, completed=sample
                assert len(states)==len(actions)
                if len(rewards)<=12:
                    continue
                if completed:
                    continue
                #make end state a reward state
                if np.random.rand()>0.5:
                    # select a random index
                    ind=np.random.randint(10, len(rewards)-1)
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
