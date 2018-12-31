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
		self.out=tf.layers.dense(inputs=a3, units=2, activation=tf.nn.relu)


		#calculate the loss function
		self.loss=tf.reduce_sum(tf.multiply(tf.reduce_mean(tf.square(tf.subtract(self.out, tf.maximum(self.actions+100, 0)))),self.advantages))

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op = optimizer.minimize(loss=self.loss,global_step=tf.train.get_global_step())

	def train(self):
		return self.loss, self.train_op
	def predict(self):
		return self.out


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
		c3=tf.layers.dense(inputs=self.Actor.a2, units=128, activation=tf.nn.relu)
		self.c_out=tf.layers.dense(inputs=c3, units=1, activation=tf.nn.relu)

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
				# if (e%64==1)&(e>30):
				# 	env.render()
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




def train_model2(env, actor, critic, States, Actions, Rewards, Advantages, k=10000):
	""" Main A2C Training Algorithm
	todo: add more intuitive sampling method.
	todo:train agent to approximate
	"""
	# self.pretrain_random(env, args, summary_writer)
	results = []
	her=hindsight(batch_size=24)

	#state action placeholders
	# States=tf.placeholder(tf.float32, shape=[None,2], name='States')
	# Actions=tf.placeholder(tf.float32, shape=[None,2], name='Actions')
	# Rewards=tf.placeholder(tf.float32, shape=[None,1], name='Rewards')
	# Advantages=tf.placeholder(tf.float32, shape=[None,1], name='Advantages')
	other=tf.placeholder(tf.float32, shape=[None,2], name='other')
	scale_factor=tf.placeholder(tf.float32, shape=[], name='other')


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


			bias_x=0
			bias_y=0
			num_steps=0
			cumul_reward=0
			do_a2c=np.random.randint(2, size=1)
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

				if (e%100==0)|(e>5000):
					env.render(pt=[[state[2], state[3]]])
				# env.render(pt=[[state[2], state[3]]])

				if do_a2c:
					#this
					if e>1000:
						print('do_a2c')
					a=sess.run(actor.predict(), feed_dict={States: np.asarray(old_state).reshape(-1,4,6)}).squeeze()
					a=a-100
				else:
					if e>1000:
						print('controller')
					err_x=[state[2]-state[4], state[0]-state[4]]
					err_y=[state[3]-state[5], state[1]-state[5]]

					#pd controller
					kp=1
					ki=0
					kd=10

					Fx=[err_x[0]*kp+ki*(err_x[0]+err_x_old[0])/2+kd*(err_x[0]-err_x_old[0]), err_x[1]*kp+ki*(err_x[1]+err_x_old[1])/2+kd*(err_x[1]-err_x_old[1])]
					Fy=[err_y[0]*kp+ki*(err_y[0]+err_y_old[0])/2+kd*(err_y[0]-err_y_old[0]), err_y[1]*kp+ki*(err_y[1]+err_y_old[1])/2+kd*(err_y[1]-err_y_old[1])]


					blend_x=1-scipy.stats.norm(0, 10).cdf(np.abs(state[0]-state[4]))
					blend_y=1-scipy.stats.norm(0, 10).cdf(np.abs(state[1]-state[5]))

					# blend_x=0.5+0.5*np.erf(np.log(np.abs(state[0]-state[4]))/(np.sqrt(2)*2))
					# blend_y=0.5+0.5*np.erf(np.log(np.abs(state[1]-state[5]))/(np.sqrt(2)*2))

					err_x_old=err_x
					err_y_old=err_y

					a=[Fx[0]*blend_x+Fx[1]*(1-blend_x), Fy[0]*blend_y+Fy[1]*(1-blend_y)]
					# print(a)
					a=[a[0]+np.random.normal(0,abs(err_x[0])*0.5+0.1), a[1]+np.random.normal(0,abs(err_y[0])*0.5+0.1)]


				# Retrieve new state, reward, and whether the state is terminal
				new_state, r, done, _ = env.step(a)
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

				if num_steps==1500:
					break



				# Train using discounted rewards ie. compute updates
				# print(np.asarray(states).shape)

			tqdm_e.set_description("Score: " + str(cumul_reward))
			tqdm_e.refresh()
			data.append([str(e),str(do_a2c), str(cumul_reward)])
			if e%100==33:
				with open("output.csv", "w") as f:
				    writer = csv.writer(f)
				    writer.writerows(data)
			if cumul_reward<1:
				continue
			her.add(states, np.asarray(actions), rewards)

			if len(her.objects)>24:
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
					# print(actor_loss, critic_loss)
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
