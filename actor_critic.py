"""
actor_critic.py
"""
import tensorflow as tf
import numpy as np
from single_cell_env import opticalTweezers
from utils.continuous_environments import Environment
from hindsight import hindsight

# This function selects the probability distribution over actions
# from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import cat_entropy, mse
from baselines.common import explained_variance
from stable_baselines.common.distributions import make_proba_dist_type
from baselines.common.distributions import make_pdtype
from gym import spaces


import scipy.stats

from tqdm import tqdm
import sys
import os
import csv


def obscurity(state):
	"""
	random obscurity of x an y
	"""
	x=state[2]
	y=state[3]
	x2=np.sign(x-512/2)*0.9*(x-512/2)+512/2
	y2=0.95*y
	state2=state
	state2[2]=x2
	state2[3]=y2
	return state2

def find_trainable_variables(key):
	with tf.variable_scope(key):
		return tf.trainable_variables()

class Actor():
	def __init__(self, states, actions, advantages, rewards, Entropy_coefficient, max_grad_norm, vf_coef=0.5, lr=0.5*1e-3):
		self.states=states
		self.actions=actions
		self.advantages=advantages
		self.rewards=rewards
		self.Entropy_coefficient=Entropy_coefficient
		self.vf_coef=vf_coef
		self.lr=lr
		self.pdtype = make_proba_dist_type(spaces.Discrete(4))
		# self.pdtype = make_pdtype(spaces.Discrete(4))
		self.build_model(max_grad_norm)

	def build_model(self, max_grad_norm, reuse=tf.AUTO_REUSE):
		#reuse  is true id loading
		#buld a 4 layer fc net for actor
		# X=tf.placeholder([-1,6,4])
		states_in=tf.layers.flatten(self.states)
		with tf.variable_scope("model", reuse = reuse):
			a1=tf.layers.dropout(inputs=tf.layers.dense(inputs=states_in, units=64, activation=tf.nn.relu), rate=0.3)
			self.a2=tf.layers.dropout(tf.layers.dense(inputs=a1, units=128, activation=tf.nn.relu), rate=0.2)
			self.a3=tf.layers.dropout(tf.layers.dense(inputs=self.a2, units=128, activation=tf.nn.relu), rate=0.1)
			self.out=tf.layers.dense(inputs=self.a3, units=4, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
			self.value=tf.layers.dense(inputs=self.a3, units=1, activation=None)

			#
			# self.pd, self.pi = self.pdtype.pdfromlatent(self.out, init_scale=0.01) # with baselines from openai
			self.pd, self.pi, _ = self.pdtype.proba_distribution_from_latent(self.out, self.value, init_scale=0.01) # with stable_baselines see https://stable-baselines.readthedocs.io/en/master/common/distributions.html?highlight=vf%20latent%20vector
			# self.pd, self.pi = self.pdtype.pdfromlatent(self.out, init_scale=0.01)
			self.a0=self.pd.sample()

		#calculate the loss function
		neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pi, labels=self.actions)

		# 1/n * sum A(si,ai) * -logpi(ai|si)
		pg_loss = tf.reduce_mean(self.advantages * neglogpac)

		# Value loss 1/2 SUM [R - V(s)]^2
		vf_loss = tf.reduce_mean(mse(tf.squeeze(self.value),self.rewards))

		# Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
		entropy = tf.reduce_mean(self.pd.entropy())


		self.loss = pg_loss - entropy * self.Entropy_coefficient + vf_loss * self.vf_coef

		# Update parameters using loss
		# 1. Get the model parameters
		params = find_trainable_variables("model")

		# 2. Calculate the gradients
		grads = tf.gradients(self.loss, params)
		if max_grad_norm is not None:
			# Clip the gradients (normalize)
			grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
		grads = list(zip(grads, params))
		# zip aggregate each gradient with parameters associated
		# For instance zip(ABCD, xyza) => Ax, By, Cz, Da

		# 3. Build our trainer
		trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, epsilon=1e-5)

		# 4. Backpropagation
		self.train_op = trainer.apply_gradients(grads)

		# optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		# self.train_op = optimizer.minimize(loss=self.loss,global_step=tf.train.get_global_step())

	def train(self):
		return self.loss, self.train_op
	def predict(self):
		return self.a0, self.value


class Network():
	def __init__(self, states, other_pred, scale_factor, lr=1e-4):
		self.states=states
		self.other_pred=other_pred
		self.lr=lr
		self.scale_factor=scale_factor
		self.build_model()

	def build_model(self):
		#buld a 4 layer fc net for actor
		# X=tf.placeholder([-1,6,4])
		states_in=tf.layers.flatten(self.states)
		a1=tf.layers.dense(inputs=states_in, units=64, activation=tf.nn.relu)
		self.a2=tf.layers.dense(inputs=a1, units=128, activation=tf.nn.relu)
		a3=tf.layers.dense(inputs=self.a2, units=128, activation=tf.nn.relu)
		self.out=tf.layers.dense(inputs=a3, units=2, activation=tf.nn.relu)
		#calculate the loss function
		self.loss=tf.reduce_mean(tf.scalar_mul(self.scale_factor, tf.reduce_mean(tf.square(tf.subtract(self.out, self.other_pred)))))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		self.train_op = optimizer.minimize(loss=self.loss,global_step=tf.train.get_global_step())
	def train(self):
		return self.train_op
	def predict(self):
		return self.out


class Network2():
	def __init__(self, states, other_pred, scale_factor, lr=1e-4):
		self.states=states
		self.other_pred=other_pred
		self.lr=lr
		self.scale_factor=scale_factor
		self.build_model()

	def build_model(self):
		#buld a 4 layer fc net for actor
		# X=tf.placeholder([-1,6,4])
		states_in=tf.layers.flatten(self.states)
		a1=tf.layers.dense(inputs=states_in, units=64, activation=tf.nn.relu)
		self.a2=tf.layers.dense(inputs=a1, units=128, activation=tf.nn.relu)
		a3=tf.layers.dense(inputs=self.a2, units=128, activation=tf.nn.relu)
		self.out=tf.layers.dense(inputs=a3, units=2, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
		#calculate the loss function
		loss=tf.scalar_mul(self.scale_factor, tf.reduce_mean(tf.square(tf.subtract(self.out, self.other_pred))))
		l2_loss=tf.losses.get_regularization_loss()
		# range_loss=tf.reduce_mean(tf.maximum(tf.minimum(tf.subtract(self.out, tf.constant([1, 1], dtype=tf.float32)), tf.constant([0,0], dtype=tf.float32)), tf.constant([1,1], dtype=tf.float32)))
		one_loss=tf.reduce_mean(tf.to_float(tf.greater_equal(self.out, tf.constant([1, 1], dtype=tf.float32))))
		zero_loss=tf.reduce_mean(tf.to_float(tf.less_equal(self.out, tf.constant([0, 0], dtype=tf.float32))))
		self.loss=tf.reduce_mean(loss+l2_loss+10*one_loss+10*zero_loss)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		self.train_op = optimizer.minimize(loss=self.loss,global_step=tf.train.get_global_step())
	def train(self):
		return self.train_op
	def predict(self):
		return self.out



class Critic():
	def __init__(self, discounted_rewards, Actor, lr=1e-4):
		self.rewards=discounted_rewards
		self.lr=lr
		self.Actor=Actor
		self.build_model()
	def build_model(self):
		#buld a 4 layer fc net for actor
		# c3=tf.layers.dense(inputs=self.Actor.a3, units=128, activation=tf.nn.relu)
		self.c_out=tf.layers.dense(inputs=self.Actor.a3, units=1, activation=None)

		# self.critic_loss=tf.reduce_mean(self.c_out, self.rewards)
		self.critic_loss=tf.losses.mean_squared_error(self.rewards, self.c_out)
		self.optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op=self.optimizer.minimize(loss=self.critic_loss, global_step=tf.train.get_global_step())
	def train(self):
		return self.critic_loss, self.train_op
	def predict(self):
		return self.c_out






# # Convolution layer
# def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
# 	return tf.layers.conv2d(inputs=inputs,filters=filters,kernel_size=kernel_size,strides=(strides, strides),activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer(gain=gain))


# # Fully connected layer
# def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
# 	return tf.layers.dense(inputs=inputs,units=units,activation=activation_fn,kernel_initializer=tf.orthogonal_initializer(gain))


# """
# This object creates the A2C Network architecture
# """
# class A2CPolicy(object):
# 	def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse = False):
# 		# This will use to initialize our kernels
# 		gain = np.sqrt(2)

# 		# Based on the action space, will select what probability distribution type
# 		# we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType
# 		# aka Diagonal Gaussian, 3D normal distribution
# 		self.pdtype = make_pdtype(action_space)

# 		height, weight, channel = ob_space.shape
# 		ob_shape = (height, weight, channel)

# 		# Create the input placeholder
# 		inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

# 		# Normalize the images
# 		scaled_images = tf.cast(inputs_, tf.float32) / 255.

# 		"""
# 		Build the model
# 		3 CNN for spatial dependencies
# 		Temporal dependencies is handle by stacking frames
# 		(Something funny nobody use LSTM in OpenAI Retro contest)
# 		1 common FC
# 		1 FC for policy
# 		1 FC for value
# 		"""
# 		with tf.variable_scope("model", reuse = reuse):
# 			conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
# 			conv2 = conv_layer(conv1, 64, 4, 2, gain)
# 			conv3 = conv_layer(conv2, 64, 3, 1, gain)
# 			flatten1 = tf.layers.flatten(conv3)
# 			fc_common = fc_layer(flatten1, 512, gain=gain)

# 			# This build a fc connected layer that returns a probability distribution
# 			# over actions (self.pd) and our pi logits (self.pi).
# 			self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)

# 			# Calculate the v(s)
# 			vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]

# 		self.initial_state = None

# 		# Take an action in the action distribution (remember we are in a situation
# 		# of stochastic policy so we don't always take the action with the highest probability
# 		# for instance if we have 2 actions 0.7 and 0.3 we have 30% chance to take the second)
# 		a0 = self.pd.sample()

# 	# Function use to take a step returns action to take and V(s)
# 	def step(state_in, *_args, **_kwargs):
# 		action, value = sess.run([a0, vf], {inputs_: state_in})

# 		#print("step", action)

# 		return action, value

# 	# Function that calculates only the V(s)
# 	def value(state_in, *_args, **_kwargs):
# 		return sess.run(vf, {inputs_: state_in})

# 	# Function that output only the action to take
# 	def select_action(state_in, *_args, **_kwargs):
# 		return sess.run(a0, {inputs_: state_in})

# 	self.inputs_ = inputs_
# 	self.vf = vf
# 	self.step = step
# 	self.value = value
# 	self.select_action = select_action


# class Model(object):
#     """
#     We use this object to :
#     __init__:
#     - Creates the step_model
#     - Creates the train_model
#     train():
#     - Make the training part (feedforward and retropropagation of gradients)
#     save/load():
#     - Save load the model
#     """
#     def __init__(self, policy,ob_space,action_space,nenvs,nsteps,ent_coef,vf_coef,max_grad_norm):
# 	    sess = tf.get_default_session()

#         # Here we create the placeholders
#         actions_ = tf.placeholder(tf.int32, [None], name="actions_")
#         advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
#         rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
#         lr_ = tf.placeholder(tf.float32, name="learning_rate_")

#         # Here we create our two models:
#         # Step_model that is used for sampling
#         step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)

#         # Train model for training
#         train_model = policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True)

#         """
#         Calculate the loss
#         Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
#         """
#         # Policy loss
#         # Output -log(pi)
#         neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)

#         # 1/n * sum A(si,ai) * -logpi(ai|si)
#         pg_loss = tf.reduce_mean(advantages_ * neglogpac)

#         # Value loss 1/2 SUM [R - V(s)]^2
#         vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf),rewards_))

#         # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
#         entropy = tf.reduce_mean(train_model.pd.entropy())


#         loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

#         # Update parameters using loss
#         # 1. Get the model parameters
#         params = find_trainable_variables("model")

#         # 2. Calculate the gradients
#         grads = tf.gradients(loss, params)
#         if max_grad_norm is not None:
#             # Clip the gradients (normalize)
#             grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
#         grads = list(zip(grads, params))
#         # zip aggregate each gradient with parameters associated
#         # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

#         # 3. Build our trainer
#         trainer = tf.train.RMSPropOptimizer(learning_rate=lr_, decay=0.99, epsilon=1e-5)

#         # 4. Backpropagation
#         _train = trainer.apply_gradients(grads)

#     def train(states_in, actions, returns, values, lr):
#         # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
#         # Returns = R + yV(s')
#         advantages = returns - values

#         # We create the feed dictionary
#         td_map = {train_model.inputs_: states_in,actions_: actions,advantages_: advantages, rewards_: returns, lr_: lr}

#         policy_loss, value_loss, policy_entropy, _= sess.run([pg_loss, vf_loss, entropy, _train], td_map)
        
#         return policy_loss, value_loss, policy_entropy


#     def save(save_path):
#         """
#         Save the model
#         """
#         saver = tf.train.Saver()
#         saver.save(sess, save_path)

#     def load(load_path):
#         """
#         Load the model
#         """
#         saver = tf.train.Saver()
#         print('Loading ' + load_path)
#         saver.restore(sess, load_path)

#     self.train = train
#     self.train_model = train_model
#     self.step_model = step_model
#     self.step = step_model.step
#     self.value = step_model.value
#     self.initial_state = step_model.initial_state
#     self.save = save
#     self.load = load
# 	tf.global_variables_initializer().run(session=sess)



def discount(r, gamma=0.95):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * gamma
            discounted_r[t] = cumul_r
        return np.asarray(discounted_r).ravel()


def train_model(env, actor, critic, States, Actions, Rewards, Advantages, k=5000):
	""" Main A2C Training Algorithm
	"""
	# self.pretrain_random(env, args, summary_writer)
	results = []
	her=hindsight(batch_size=16)

	with tf.Session() as sess:

		sess.run(tf.initializers.global_variables())
		sess.run(tf.initializers.local_variables())

		# Main Loop
		tqdm_e = tqdm(range(k), desc='Score', leave=True, unit=" episodes")
		for e in tqdm_e:

			# Reset episode
			time, cumul_reward, done = 0, 0, False
			old_state = env.reset()
			actions, states, rewards = [], [], []

			do_random=False
			if np.random.rand()>200/(e+200):
				do_random=True

			while not done:
				# if (e%64==1)&(e>30):
				# 	env.render()
				env.render()
				# Actor picks an action (following the policy)

				if e<128:
					a=np.random.normal(0, 2, size=(2))
				elif np.random.rand()<0.1:
					a=np.random.normal(0, 2, size=(2))
				else:
					a, v=sess.run(actor.predict(), feed_dict={States: np.asarray(old_state).reshape(-1,4,6)}).squeeze()
					if np.any(np.isnan(a)):
						print(a)
					# print(a)
				#feedforward
				# Retrieve new state, reward, and whether the state is terminal
				new_state, r, done, _ = env.step(a)

				# Memorize (s, a, r) for training
				# actions.append(to_categorical(a, self.act_dim))
				actions.append(a)
				states.append(old_state)

				# compute the novelty
				# print(self.env_dim[1])

				rewards.append(r)
				# Update current state
				old_state = new_state
				cumul_reward += r
				time += 1



				# Train using discounted rewards ie. compute updates
				# print(np.asarray(states).shape)
			assert len(states)==len(actions)
			her.add(states, np.asarray(actions), rewards)
				# print(np.asarray(states).shape, np.asarray(actions).shape, np.asarray(rewards).shape)
				# only update every 10 episodes?
			if e>24:
				for item in her.sample():
					states2, actions2, rewards2, completed=item
					# print(len(states2), len(actions2), len(rewards2))
					# print(np.asarray(states).shape, np.asarray(actions).shape, np.asarray(rewards).shape)
					states2=np.asarray(states2)[-min(1000, len(rewards2)):]
					actions2=np.asarray(actions2)[-min(1000, len(rewards2)):]
					rewards2=np.asarray(rewards2)[-min(1000, len(rewards2)):]


					# discount the rewards
					discounted_r=discount(rewards2).reshape((-1,1))

					# compute the advantages
					state_values = sess.run([critic.predict()], feed_dict={States:states2})

					# print(discounted_r.shape)
					# print(np.squeeze(state_values).shape)
					advantages = (np.squeeze(discounted_r) - np.squeeze(state_values)).reshape(-1,1)
					# print(len(advantages[np.where(advantages<0)]))
					#advantage between -1 and 1
					advantages=1+np.abs(advantages)

					# print(len(states2))
					# print(actions2.shape)
					# print(len(advantages))


					# run the training operations
					actor_loss,_=sess.run([actor.loss, actor.train_op], feed_dict={States:states2, Actions:actions2, Advantages:advantages})
					# print(actor_loss)
					critic_loss, _=sess.run([critic.critic_loss, critic.train_op], feed_dict={States:states2, Rewards:discounted_r})

			# Gather stats every episode for plotting
			results.append([e, np.mean(rewards), np.std(rewards)])

			# Display score
			tqdm_e.set_description("Score: " + str(cumul_reward))
			tqdm_e.refresh()


class ant():
    def __init__(self, x=0, y=0, theta=0, max_speed=3, dt=0.1, frame=(512,512)):
        self.x=x
        self.y=y
        self.theta=theta
        self.max_speed=max_speed
        self.dt=dt
        self.colour=(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        self.frame=frame


    def update(self):
        self.theta=self.theta+np.random.choice([-0.2, 0, 0.2], p=[0.45,0.1,0.45])
        r=self.max_speed*self.dt
        self.x+=r*np.cos(self.theta)
        self.y+=r*np.sin(self.theta)
        self.x=min(max(0, self.x), self.frame[0])
        self.y=min(max(0, self.y), self.frame[1])


def train_model2(env, actor, critic, States, Actions, Rewards, Advantages, Entropy_coefficient, k=10000):
	""" Main A2C Training Algorithm
	todo: add more intuitive sampling method.
	todo:train agent to approximate
	"""
	# self.pretrain_random(env, args, summary_writer)
	results = []
	hindsight_batch=48
	her=hindsight(batch_size=hindsight_batch)



	#state action placeholders
	# States=tf.placeholder(tf.float32, shape=[None,2], name='States')
	# Actions=tf.placeholder(tf.float32, shape=[None,2], name='Actions')
	# Rewards=tf.placeholder(tf.float32, shape=[None,1], name='Rewards')
	# Advantages=tf.placeholder(tf.float32, shape=[None,1], name='Advantages')
	other=tf.placeholder(tf.float32, shape=[None,2], name='other')
	scale_factor=tf.placeholder(tf.float32, shape=[], name='scale_factor')

	sparse_select=tf.placeholder(tf.float32, shape=[None, 4], name='sparse_select')


	#we need three networks:
	# random_net=Network(States, other, scale_factor)
	# certainty_net=Network(States, other, scale_factor)
	# approx_net=Network2(States, other, scale_factor)


	with tf.Session() as sess:

		sess.run(tf.initializers.global_variables())
		sess.run(tf.initializers.local_variables())

		# #pretrain approx_net
		# data=[]
		# for i in tqdm(range(10000), desc='pretrain'):
		# 	state=np.random.randint(0, 512, 2).reshape(-1,2)
		# 	loss, _=sess.run([approx_net.loss, approx_net.train_op], feed_dict={States:state, other:(state/512), scale_factor:np.asarray(1)})
		# 	data.append([str(i),str(loss)])

		# with open("output.csv", "w") as f:
		#     writer = csv.writer(f)
		#     writer.writerows(data)




		# Main Loop
		data=[]
		tqdm_e = tqdm(range(k), desc='Score', leave=True, unit=" episodes")
		for e in tqdm_e:

			# Reset episode
			time, done = 0, False
			old_state = env.reset()
			actions, states, rewards = [], [], []

			x, y=env.get_user_xy()
			target=ant(x=x, y=y, max_speed=3, dt=0.1, theta=0)

			do_random=False
			if np.random.rand()>500/(e+500):
				do_random=True

			err_x_old=[0,0]
			err_y_old=[0,0]

			#calculate the errors
			state_orig_1=old_state[-1]
			#add obscurity
			state=state_orig_1
			# state=obscurity(state_orig_1)
			#calculate distillation network uncertainty
			# pred=sess.run(random_net.out, feed_dict={States:np.asarray(state)[2:4].reshape((-1,2))})
			# loss, _ =sess.run([certainty_net.loss, certainty_net.train_op], feed_dict={States:np.asarray(state)[2:4].reshape((-1,2)), other:pred.reshape((-1,2)), scale_factor:np.asarray(1)})

			# state_estimate=sess.run(approx_net.out, feed_dict={States:np.asarray(state)[2:4].reshape((-1,2))})
			# state_estimate=np.squeeze(state_estimate)
			# state_estimate[np.where(state_estimate>1)]=1
			# state_estimate=(state_estimate)*512


			num_steps=0
			cumul_reward=0
			do_a2c=np.random.randint(2, size=1)
			if len(her.objects)<hindsight_batch:
				do_a2c=0
			# while not done:
			while True:
				state_orig=old_state[-1]
				state=state_orig
				# state=obscurity(state_orig)
				# state[2]=state_estimate[0]+bias_x
				# state[3]=state_estimate[1]+bias_y

				#noisy terminal state
				# if np.random.rand()<0.1:
				# 	bias_x=np.random.normal(state_orig[2], 7)
				# 	bias_y=np.random.normal(state_orig[3], 7)
				# state[2]=state_estimate[0]+bias_x
				# state[3]=state_estimate[1]+bias_y

				if (e%50==0)|(e>5000):
					do_a2c=True
					if e==0:
						do_a2c=0
					if do_a2c:
						env.render(pt=[[state[2], state[3]]], text="a2c {}".format(num_steps), episode=e)
					else:
						env.render(pt=[[state[2], state[3]]], text="ctrl {}".format(num_steps), episode=e)
				# env.render(pt=[[state[2], state[3]]])

				if do_a2c:
					# env.render(pt=[[state[2], state[3]]], text="a2c {}".format(num_steps))
					a, v=sess.run(actor.predict(), feed_dict={States: np.asarray(old_state).reshape(-1,4,6)})
					# print(a[0])
					b=np.zeros((4))
					b[a]=1
					a=b

				else:
					err_x=[state[2]-state[4], state[0]-state[4]]
					err_y=[state[3]-state[5], state[1]-state[5]]

					#pd controller
					kp=1
					ki=0
					kd=10

					Fx=[err_x[0]*kp+ki*(err_x[0]+err_x_old[0])/2+kd*(err_x[0]-err_x_old[0]), err_x[1]*kp+ki*(err_x[1]+err_x_old[1])/2+kd*(err_x[1]-err_x_old[1])]
					Fy=[err_y[0]*kp+ki*(err_y[0]+err_y_old[0])/2+kd*(err_y[0]-err_y_old[0]), err_y[1]*kp+ki*(err_y[1]+err_y_old[1])/2+kd*(err_y[1]-err_y_old[1])]


					blend_x=1-scipy.stats.norm(0, 5).cdf(np.abs(state[0]-state[4]))
					blend_y=1-scipy.stats.norm(0, 5).cdf(np.abs(state[1]-state[5]))

					# blend_x=0.5+0.5*np.erf(np.log(np.abs(state[0]-state[4]))/(np.sqrt(2)*2))
					# blend_y=0.5+0.5*np.erf(np.log(np.abs(state[1]-state[5]))/(np.sqrt(2)*2))

					err_x_old=err_x
					err_y_old=err_y

					a=[Fx[0]*blend_x+Fx[1]*(1-blend_x), Fy[0]*blend_y+Fy[1]*(1-blend_y)]
					# print(a)
					a=[a[0]+np.random.normal(0,abs(err_x[0])*0.5+0.1), a[1]+np.random.normal(0,abs(err_y[0])*0.5+0.1)]

					#reshape a to an appropriate format
					#a=[pos_x, neg_x, pos_y, neg_y]
					new_a=np.zeros((4)).reshape(-1)
					# new_a[[max(0, np.sign(a[0])), max(0, np.sign(a[1]))]]=[1,1]
					# if abs(a[0])>0.5:
					if abs(a[0])>abs(a[1]):
						new_a[int(1-max(0, np.sign(a[0])))]=1
					else:
						new_a[int(1-max(0, np.sign(a[1]))+2)]=1
					# print(np.sign(a[0]), np.sign(a[1]))
					# print(new_a)
					a=new_a


				# Retrieve new state, reward, and whether the state is terminal
				new_state, r, done, _ = env.step(a)
				target.update()
				#set env user points
				env.set_user(user_x=target.x, user_y=target.y)
				num_steps+=1


				# Memorize (s, a, r) for training
				# actions.append(to_categorical(a, self.act_dim))
				actions.append(a)
				states.append(old_state)

				# compute the novelty
				# print(self.env_dim[1])

				rewards.append(r)
				# Update current state
				old_state = new_state
				cumul_reward += r
				time += 1

				if done:
					break

				if num_steps==1500:
					break



				# Train using discounted rewards ie. compute updates
				# print(np.asarray(states).shape)

			tqdm_e.set_description("Score: " + str(cumul_reward))
			tqdm_e.refresh()
			data.append([str(e),str(do_a2c), str(cumul_reward), str(num_steps)])
			if e%100==33:
				with open("output.csv", "w") as f:
				    writer = csv.writer(f)
				    writer.writerows(data)
			# if do_a2c==False:
			# 	rewards[-1]=1
			if cumul_reward>=1:
				her.add(states, np.asarray(actions), rewards)

			mean_actor=0
			mean_critic=0
			denom=0

			if len(her.objects)>hindsight_batch:
				for item in her.sample():
					states2, actions2, rewards2, completed=item
					# print(len(states2), len(actions2), len(rewards2))
					# print(np.asarray(states).shape, np.asarray(actions).shape, np.asarray(rewards).shape)
					states2=np.asarray(states2)[-min(1500, len(rewards2)):]
					actions2=np.asarray(actions2)[-min(1500, len(rewards2)):]
					rewards2=np.asarray(rewards2)[-min(1500, len(rewards2)):]


					# actions2=actions2[:,:4]

					actions2=np.argmax(actions2, axis=1)
					# print(actions)

					# print(actions2.shape)
					# discount the rewards
					discounted_r=discount(rewards2).reshape((-1,1))

					# compute the advantages
					state_values = sess.run([critic.predict()], feed_dict={States:states2})
					state_values=sess.run(actor.value, feed_dict={States:states2})

					# print(discounted_r.shape)
					# print(np.squeeze(state_values).shape)
					advantages = (np.squeeze(discounted_r) - np.squeeze(state_values)).reshape(-1,1)
					# print(len(advantages[np.where(advantages<0)]))
					#advantage between -1 and 1
					advantages=1+np.abs(advantages)

					ev=explained_variance(np.squeeze(state_values), np.squeeze(discounted_r))
					# ev = np.asarray([explained_variance(i,j) for i,j in zip(np.squeeze(state_values), np.squeeze(discounted_r))])
					# print(ev.shape)
					# print(len(states2))
					# print(actions2.shape)
					# print(len(advantages))

					# run the training operations
					actor_loss,_=sess.run([actor.loss, actor.train_op], feed_dict={States:states2, Actions:actions2, Advantages:advantages, Rewards:discounted_r, Entropy_coefficient:ev})
					# print(actor_loss)
					# critic_loss, _=sess.run([critic.critic_loss, critic.train_op], feed_dict={States:states2, Rewards:discounted_r})

					mean_actor+=actor_loss
				print(np.mean(mean_actor))
			# _=sess.run(approx_net.train_op, feed_dict={States:np.asarray(state)[2:4].reshape((-1,2)), other:np.asarray(old_state)[-1, 0:2].reshape((-1,2)), scale_factor:np.asarray(0.001+loss)})
			# print(num_steps)
			# Display score


	with open("output.csv", "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(data)





def main(training=True):
	#load an environment
	env=Environment(opticalTweezers(), 4)
	env.reset()

	#state action placeholders
	States=tf.placeholder(tf.float32, shape=[None,4,6], name='States')
	Actions=tf.placeholder(tf.int32, shape=[None], name='Actions')
	Rewards=tf.placeholder(tf.float32, shape=[None,1], name='Rewards')
	Advantages=tf.placeholder(tf.float32, shape=[None,1], name='Advantages')
	Entropy_coefficient=tf.placeholder(tf.float32, shape=(), name='Entropy_coefficient')

	#load a model or else init
	if os.path.isfile(os.path.join(os.getcwd(), 'model')):
		#load model
		pass
	else:
		max_grad_norm=0.5
		actor=Actor(States, Actions, Advantages, Rewards, Entropy_coefficient, max_grad_norm)
		critic=Critic(Rewards, actor)
	if training:
		train_model2(env, actor, critic, States, Actions, Rewards, Advantages, Entropy_coefficient)
	else:
		pass


if __name__=="__main__":
	if len(sys.argv)==1:
		main(training=True)
	else:
		main(training=True)
