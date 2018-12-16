import numpy as np
from keras.optimizers import RMSprop

class Agent:
    """ Agent Generic Class
    """

    def __init__(self, inp_dim, out_dim, lr):
        self.inp_dim = inp_dim
        print(inp_dim)
        self.out_dim = out_dim
        self.rms_optimizer =  RMSprop(lr=lr, epsilon=0.1, rho=0.99)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Critic Value Prediction
        """
        # print('network input')
        # print(inp)
        # print('reshaped input')
        # print(self.reshape(inp))
        return self.model.predict(self.reshape(inp))

    def reshape(self, x):
        if len(x.shape) < 3:
            x=np.expand_dims(x,axis=0)
            x=self.reshape(x)
            return x
        else: return x
