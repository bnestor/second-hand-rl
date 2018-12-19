import random
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
import keras.backend as K

from .critic import Critic
from .actor import Actor
from utils.networks import tfSummary
from utils.stats import gather_stats
from .agent import Agent
from hindsight import hindsight

def softmax(z):
    assert len(z.shape) == 2
    z=z.astype(np.float32)
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


class random_network_distillation(Agent):
    def __init__(self, inp_dim, out_dim, network_rand, network_pred, lr):
        Agent.__init__(self, inp_dim, inp_dim, lr)
        self.inp_dim=np.asarray(inp_dim)
        self.random_model = self.addHead(network_rand)
        self.learned_model=self.addHead(network_pred)

    def addHead(self, network):
        """ Assemble Critic network to predict value of each state
        """
        x = Dense(128, activation='relu')(network.output)
        out = Dense(self.inp_dim[0], activation='relu')(x)
        return Model(network.input, out)

    def optimizer(self):
        """ Critic Optimization: Mean Squared Error over discounted rewards
        """
        exploration_loss = K.mean(K.square(self.random_model.output - self.learned_model.output))
        updates = self.rms_optimizer.get_updates(self.learned_model.trainable_weights, [], exploration_loss)
        return K.function([self.random_model.input, self.learned_model.input], [exploration_loss], updates=updates)


class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 1e-6):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = (k,) + env_dim
        # print(self.env_dim)
        # print(self.act_dim)
        # input()
        self.gamma = gamma
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        # self.rand_shallow = self.buildNetwork()
        # self.predict_shallow = self.buildNetwork()
        # build random network
        # self.random_network=random_network_distillation(self.env_dim, act_dim*k, self.rand_shallow,self.predict_shallow, lr*10)
        # print("init_actor")
        #act_dim=4 [accelx stopx accely stopy]
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        # print("init_critic")
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()
        # self.rnd_opt=self.random_network.optimizer()

        self.her=hindsight(batch_size=24)

        # self.pretrain_random()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        x = Flatten()(inp)
        x = Dense(64, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, s, e=10e9):
        """ Use the actor to predict the next action to take, using the policy
        """
        #modified for multidimensional agent
        # predictions=self.actor.predict(s).ravel().reshape((-1,2))
        # # print(predictions)
        # # predictions=predictions+10000.0/(e+100)


        # a=[]
        # for item, reason in enumerate(['move_x', 'move_y', 'stay_still_x', 'stay_still_y']):
        #     # or self.act_dim/2
        #     p=softmax(predictions[item, :].reshape(-1,2)).ravel() #apply a softmax layer
        #     p=(np.abs(predictions[item,:])/np.sum(np.abs(predictions[item,:]))).ravel()
        #     a.append(np.random.choice(np.arange(2), 1, p=p))
        #     # print(predictions[item, :].reshape(-1,2))
        #     # print(p)
        # # return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]

        # attempt 2

        predictions=self.actor.predict(s).ravel()
        # print(len(predictions))
        predictions=(predictions)-0.5


        # input()
        return predictions

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        state_values = self.critic.predict(np.array(states))
        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # print(np.asarray(states).shape)
        # print(np.asarray(actions).shape)
        # print(np.asarray(discounted_rewards).shape)
        # print(actions.reshape((-1, self.act_dim*4)).shape)
        # print(actions)
        actions[np.isnan(actions)]=0

        if np.any(np.isnan(states)):
            print("nan states")
        if np.any(np.isnan(actions)):
            actions[np.isnan(actions)]=0
            print("nan actions")
        if np.any(np.isnan(advantages)):
            print("nan advantages")
        if np.any(np.isnan(rewards)):
            print("nanrewards")
        # Networks optimization
        self.a_opt([states, actions.reshape((-1, self.act_dim)), advantages])
        self.c_opt([states, discounted_rewards])

    def pretrain_random(self, env, args, summary_writer, train_steps=200, env_steps=100):
        """
        Generate a somewhat random output so that the agent explores.
        """
        results = []

        # Main Loop
        tqdm_e = tqdm(range(train_steps), desc='pretrain', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []
            old_a=np.asarray(np.zeros_like(self.policy_action(old_state, e)))

            while not done:
                # if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state, e)
                # if np.random.rand()<0.1:
                #     print(a)
                # print(a)
                #feedforward
                # Retrieve new state, reward, and whether the state is terminal
                new_state, _, done, _ = env.step(a)

                r=np.random.choice(((np.asarray(a).reshape(-1)-old_a.reshape(-1))**2)[:2])
                old_a=np.asarray(a)
                #self.c_opt([states, discounted_rewards])


                # Memorize (s, a, r) for training
                actions.append(to_categorical(a, self.act_dim))
                rewards.append(r)
                states.append(old_state)

                # compute the novelty
                last_state=states[-1].reshape((1, 4, 4))
                novelty=self.rnd_opt([last_state,last_state])


                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Train using discounted rewards ie. compute updates
            try:
                self.train_models(states, np.asarray(actions), rewards, done)
            except:
                print('error training critic')
                self.train_models(states, np.asarray(actions), rewards, done)

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score:{}, Nov.: {}".format(str(cumul_reward), novelty))
            tqdm_e.refresh()
        # tqdm_e = tqdm(range(train_steps), desc='Pretrain', leave=True, unit=" episodes")
        # for step in tqdm_e:
        #     #generate random states
        #     #[self.p.x, self.p.y, self.user_x,self.user_y]
        #     states=[[np.random.randint(0, high=255, size=(4)),np.random.randint(0, high=255, size=(4)),np.random.randint(0, high=255, size=(4)),np.random.randint(0, high=255, size=(4))] for i in range(env_steps)]
        #     #generate random actions
        #     actions=[to_categorical([np.random.randint(0,1),np.random.randint(0,1)], self.act_dim) for i in range(env_steps)]
        #     actions=np.squeeze(actions)
        #     #generate random  rewards
        #     rewards=[1 for i in range(env_steps)]
        #     self.train_models(np.asarray(states), np.asarray(actions), np.asarray(rewards), True)

    def train(self, env, args, summary_writer):
        """ Main A2C Training Algorithm
        """
        # self.pretrain_random(env, args, summary_writer)
        results = []
        possible_states=[np.asarray(0), np.asarray(1)]

        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []

            while not done:
                if (e%64==1)&(e>30):
                    if args.render: env.render()
                # Actor picks an action (following the policy)
                
                if e<30:
                    a=[random.choice(possible_states),random.choice(possible_states),random.choice(possible_states),random.choice(possible_states)]
                elif np.random.rand()<0.5:
                    a=[random.choice(possible_states),random.choice(possible_states),random.choice(possible_states),random.choice(possible_states)]
                else:
                    a = self.policy_action(old_state, e)
                #feedforward
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)

                #self.c_opt([states, discounted_rewards])

                # print(novelty)

                # Memorize (s, a, r) for training
                # actions.append(to_categorical(a, self.act_dim))
                actions.append(a)
                states.append(old_state)

                # compute the novelty
                # print(self.env_dim[1])
                last_state=states[-1].reshape((1, 4, self.env_dim[1]))
                novelty=0
                # novelty=self.rnd_opt([last_state,last_state])[0]
                rewards.append(r+0.0001*novelty)
                # Update current state
                old_state = new_state
                cumul_reward += r+0.0001*novelty
                time += 1



            # Train using discounted rewards ie. compute updates
            self.her.add(states, np.asarray(actions), rewards)
            # print(np.asarray(states).shape, np.asarray(actions).shape, np.asarray(rewards).shape)
            # only update every 10 episodes?
            if e>24:
                for item in self.her.sample():
                    states, actions, rewards, completed=item
                    # print(np.asarray(states).shape, np.asarray(actions).shape, np.asarray(rewards).shape)
                    states=np.asarray(states)[-min(1000, len(rewards)):]
                    actions=np.asarray(actions)[-min(1000, len(rewards)):]
                    rewards=np.asarray(rewards)[-min(1000, len(rewards)):]
                    self.train_models(states, actions, rewards, completed)
                # try:
                #     for item in self.her.sample():
                #         states, actions, rewards, completed=item
                #         print(np.asarray(states).shape, np.asarray(actions).shape, np.asarray(rewards).shape)
                #         states=np.asarray(states)[-max(1000, len(rewards)):]
                #         actions=np.asarray(actions)[-max(1000, len(rewards)):]
                #         rewards=np.asarray(rewards)[-max(1000, len(rewards)):]
                #         self.train_models(states, actions, rewards, completed)
                # except:
                #     print('error training critic')
                #     for item in self.her.sample():
                #         states, actions, rewards, completed=item
                #         self.train_models(states, np.asarray(actions), rewards, done)
                #     # self.train_models(states, np.asarray(actions), rewards, done)

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results
