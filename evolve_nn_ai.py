import tensorflow as tf
import numpy as np
from utils import left, right, up, down

num_nets = 20
index = 0


class Net:
	num_layers = 2
	num_units = 20
	sess = None

	def __init__(self):
		if Net.sess is None:
			Net.sess = tf.Session()

		x = tf.placeholder(tf.float32, shape=[None, 5])

		W1 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[5,Net.num_units]))
		b1 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[1,Net.num_units]))
		W2 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[Net.num_units,3]))
		b2 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[1,3]))
		# returns (go straight, go right, go left)

		self.weights = [W1,W2,b1,b2]

		a1 = tf.sigmoid(tf.matmul(x, W1) + b1)
		self.y_hat = tf.nn.softmax(tf.matmul(a1, W2) + b2)

		Net.sess.run(tf.global_variables_initializer())

	def mutate_weights(self):
		for w in self.weights:
			mask = .5*(np.random.rand(w.get_shape()[0], w.get_shape()[1]) - .5)
			w.assign(Net.sess.run(w) + np.multiply(Net.sess.run(w), mask)).eval()

	@classmethod
	def average_weights(cls, to_av_nns, to_assign_nns):
		all_weights = np.array([Net.sess.run(n.weights) for n in nns]).reshape((len(to_av_nns), Net.num_layers*2))
		avg_weights = [np.mean(all_weights[:,i]) for i in range(all_weights.shape[1])]

		for n in to_assign_nns:
			for i in range(len(n.weights)):
				n.weights[i].assign(avg_weights[i]).eval()

	@classmethod
	def close_sess(cls):
		Net.sess.close()


nets = [Net() for _ in range(num_nets)]
score = [0 for _ in range(num_nets)]


def get_act(state, cur_dir):
	dir_ = np.amax(Net.sess.run(nets[index].y_hat, feed_dict=state))
	if dir_ == 0:
		return cur_dir
	elif (dir_ == 1 and cur_dir == up) or (dir_ == 2 and cur_dir == down):
		return right
	elif (dir_ == 1 and cur_dir == left) or (dir_ == 2 and cur_dir == right):
		return up
	elif (dir_ == 1 and cur_dir == right) or (dir_ == 2 and cur_dir == left):
		return down
	elif (dir_ == 1 and cur_dir == down) or (dir_ == 2 and cur_dir == up):
		return left


def get_top(num=4):
	global score
	to_av = []
	for _ in range(num):
		p_in = np.amax(score)
		to_av.append(nets[p_in])
		score[p_in] = -np.inf
	score = [0 for _ in range(num_nets)]
	return to_av


def update_score(cur_pos, old_pos, f_pos):
	o_dist = np.sqrt((old_pos[0] - f_pos[0])**2 + (old_pos[1] - f_pos[1])**2)
	n_dist = np.sqrt((cur_pos[0] - f_pos[0])**2 + (cur_pos[1] - f_pos[1])**2)

	if n_dist == 0:
		score[index] += 20
	elif n_dist > o_dist:
		score[index] -= 2
	elif n_dist <= o_dist:
		score[index] += 1


def update_index():
	global index
	if index == num_nets-1:
		to_av = get_top()
		Net.average_weights(to_av, nets)
		index = 0
	else:
		index += 1

