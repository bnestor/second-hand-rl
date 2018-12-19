import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Activation, Dense, Flatten
from keras.optimizers import Adam
from .agent import Agent

class Critic(Agent):
    """ Critic for the A3C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.discounted_r = K.placeholder(shape=(None,))
        # Pre-compile for threading
        self.model._make_predict_function()

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(1, activation='relu')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        with tf.device('/cpu:0'):
            updates = self.adam.get_updates(self.model.trainable_weights, [], critic_loss)
            output=K.function([self.model.input, self.discounted_r], [], updates=updates)
        return output
