"""
actor_critic.py
"""
import tensorflow as tf
import numpy as np
from single_cell_env import opticalTweezers
from utils.continuous_environments import Environment
from hindsight import hindsight

import scipy.stats

from tqdm import tqdm
import sys
import os



class Actor():
	def __init__(self, states, actions, advantages, lr=1e-4):
		self.states=states
		self.actions=actions
		self.advantages=advantages
		self.lr=lr
		self.build_model()

	def build_model(self):
		#buld a 4 layer fc net for actor
		# X=tf.placeholder([-1,6,4])
		states_in=tf.layers.flatten(self.states)
		a1=tf.layers.dense(inputs=states_in, units=64, activation=tf.nn.relu)
		self.a2=tf.layers.dense(inputs=a1, units=128, activation=tf.nn.relu)
		a3=tf.layers.dense(inputs=self.a2, units=128, activation=tf.nn.relu)
		self.out=tf.layers.dense(inputs=a3, units=2, activation=None)


		#calculate the loss function
		self.loss=tf.reduce_sum(tf.multiply(tf.reduce_mean(tf.square(tf.subtract(self.out, self.actions))),self.advantages))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		self.train_op = optimizer.minimize(loss=self.loss,global_step=tf.train.get_global_step())

	def train(self):
		return self.loss, self.train_op
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
		c3=tf.layers.dense(inputs=self.Actor.a2, units=128, activation=tf.nn.relu)
		self.c_out=tf.layers.dense(inputs=c3, units=1, activation=tf.nn.sigmoid)

		# self.critic_loss=tf.reduce_mean(self.c_out, self.rewards)
		self.critic_loss=tf.losses.mean_squared_error(self.rewards, self.c_out)
		self.optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op=self.optimizer.minimize(loss=self.critic_loss, global_step=tf.train.get_global_step())
	def train(self):
		return self.critic_loss, self.train_op
	def predict(self):
		return self.c_out


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
				if (e%64==1)&(e>30):
					env.render()
				# Actor picks an action (following the policy)

				if e<128:
					a=np.random.normal(0, 2, size=(2))
				elif np.random.rand()<0.1:
					a=np.random.normal(0, 2, size=(2))
				else:
					a=sess.run(actor.predict(), feed_dict={States: np.asarray(old_state).reshape(-1,4,6)}).squeeze()
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




def train_model2(env, actor, critic, States, Actions, Rewards, Advantages, k=5000):
	""" Main A2C Training Algorithm
	"""
	# self.pretrain_random(env, args, summary_writer)
	results = []
	her=hindsight(batch_size=24)

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
			if np.random.rand()>500/(e+500):
				do_random=True

			err_x_old=[0,0]
			err_y_old=[0,0]

			blend_x=blend_y=1

			while not done:
				if True:
					env.render()
				# Actor picks an action (following the policy)
				# np.asarray([self.p.x/512, self.p.y/512, user_x/512, user_y/512, self.ix/512, self.iy/512])

				#calculate the errors
				state=old_state[-1]
				err_x=[state[2]-state[4], state[0]-state[4]]
				err_y=[state[3]-state[5], state[1]-state[5]]

				#disturbance rejection_controller
				kp=2
				ki=0
				kd=0

				Fx=[err_x[0]*kp+ki*(err_x[0]+err_x_old[0])/2+kd*(err_x[0]-err_x_old[0]), err_x[1]*kp+ki*(err_x[1]+err_x_old[1])/2+kd*(err_x[1]-err_x_old[1])]
				Fy=[err_y[0]*kp+ki*(err_y[0]+err_y_old[0])/2+kd*(err_y[0]-err_y_old[0]), err_y[1]*kp+ki*(err_y[1]+err_y_old[1])/2+kd*(err_y[1]-err_y_old[1])]
				

				# blend_x=blend_y=1
				# if np.sqrt((state[0]-state[4])**2+(state[1]-state[5])**2)<30:
				# 	blend_x=0
				# 	blend_y=0

				# blend_x=np.round(scipy.stats.norm(0, 100).cdf(np.abs(state[0]-state[4])))
				# blend_y=np.round(scipy.stats.norm(0, 100).cdf(np.abs(state[1]-state[5])))

				# blend_x=1-0.5+0.5*np.erf(np.log(np.abs(state[0]-state[4]))/(np.sqrt(2)*2))
				# blend_y=1-0.5+0.5*np.erf(np.log(np.abs(state[1]-state[5]))/(np.sqrt(2)*2))

				# blend_x=1
				# blend_y=1

				if (np.abs(state[0]-state[4])<40)|(np.abs(state[1]-state[5])<40):
					blend_x=blend_y=0




				err_x_old=err_x
				err_y_old=err_y

				a=[Fx[0]*blend_x+Fx[1]*(1-blend_x), Fy[0]*blend_y+Fy[1]*(1-blend_y)]



				# if e<128:
				# 	a=np.random.normal(0, 2, size=(2))
				# elif np.random.rand()<0.1:
				# 	a=np.random.normal(0, 2, size=(2))
				# else:
				# 	a=sess.run(actor.predict(), feed_dict={States: np.asarray(old_state).reshape(-1,4,6)}).squeeze()
				# 	if np.any(np.isnan(a)):
				# 		print(a)
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




def main(training=True):
	#load an environment
	env=Environment(opticalTweezers(), 4)
	env.reset()

	#state action placeholders
	States=tf.placeholder(tf.float32, shape=[None,4,6], name='States')
	Actions=tf.placeholder(tf.float32, shape=[None,2], name='Actions')
	Rewards=tf.placeholder(tf.float32, shape=[None,1], name='Rewards')
	Advantages=tf.placeholder(tf.float32, shape=[None,1], name='Advantages')

	#load a model or else init
	if os.path.isfile(os.path.join(os.getcwd(), 'model')):
		#load model
		pass
	else:
		actor=Actor(States, Actions, Advantages)
		critic=Critic(Rewards, actor)
	if training:
		train_model2(env, actor, critic, States, Actions, Rewards, Advantages)
	else:
		pass


if __name__=="__main__":
	if len(sys.argv)==1:
		main(training=True)
	else:
		main(training=True)