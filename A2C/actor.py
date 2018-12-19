import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from .agent import Agent

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(self.out_dim, activation='relu')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.01 * entropy - K.sum(eligibility)
        with tf.device('/cpu:0'):
            updates = self.adam.get_updates(self.model.trainable_weights, [], loss)
            result=K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)
        return result
