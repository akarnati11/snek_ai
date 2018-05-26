import tensorflow as tf
import numpy as np
from utils import adjacent, collision, get_rel_in_glo, dirs, left, right, up, down
import _pickle as pickle

LAYER_0_SIZE = 20
LAYER_1_SIZE = 20

MIN_EPS = 0.05
GAMMA = 0.99
FOOD_R = 10
LIVING_R = -1
DEATH_R = -30
TRAINING_EPIS = 5000
BATCH_SIZE = 30
MEM_LENGTH = 100


sess = tf.InteractiveSession()

experience = None
EXPERIENCES = None

q_true = tf.placeholder(tf.float32, shape=[None, 1])
x = tf.placeholder(tf.float32, shape=[None, 3])

W0 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[3, LAYER_0_SIZE]))
b0 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[1, LAYER_0_SIZE]))
W1 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[LAYER_0_SIZE, 1]))
b1 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[1, 1]))
# W2 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[LAYER_1_SIZE, 1]))
# b2 = tf.Variable(tf.random_uniform(minval=-.1,maxval=.1,shape=[1, 1]))

q_hat = tf.tanh(tf.matmul(tf.tanh(tf.matmul(x, W0) + b0), W1) + b1)

loss = 0.5*tf.reduce_sum(tf.square(q_hat - q_true))

opt = tf.train.AdamOptimizer(learning_rate=0.05)
trainer = opt.minimize(loss)


def get_state(sn, seg, d, f, ss):
	ret = [0,0,0]

	sides = get_rel_in_glo(d)
	blocked = collision(sn, d, ss, blocking=True)
	# print(blocked)
	for side in blocked:
		if side in sides:
			ret[sides.index(side)] = 1


	com = (sum([i.x for i in sn]) // len(sn), sum([i.y for i in sn]) // len(sn))

	for l in [com, (f.x,f.y)]:
		if (d == up and l[1] == seg.y and l[0] > seg.x) or (d == left and l[0] == seg.x and l[1] < seg.y) \
			or (d == down and l[1] == seg.y and l[0] < seg.x) or (d == right and l[0] == seg.x and l[1] > seg.y):
			ret.append(0)
		elif (d == up and l[0] == seg.x and l[1] < seg.y) or (d == down and l[0] == seg.x and l[1] > seg.y) \
			or (d == left and l[1] == seg.y and l[0] < seg.x) or (d == right and l[1] == seg.y and l[0] > seg.x):
			ret.append(2)
		elif (d == up and l[1] == seg.y and l[0] < seg.x) or (d == left and l[0] == seg.x and l[1] > seg.y) \
			or (d == down and l[1] == seg.y and l[0] > seg.x) or (d == right and l[0] == seg.x and l[1] < seg.y):
			ret.append(4)
		elif (d == up and l[0] == seg.x and l[1] > seg.y) or (d == left and l[1] == seg.y and l[0] > seg.x) \
			or (d == down and l[0] == seg.x and l[1] < seg.y) or (d == right and l[1] == seg.y and l[0] < seg.x):
			ret.append(6)
		elif (d == up and l[0] > seg.x and l[1] < seg.y) or (d == left and l[0] < seg.x and l[1] < seg.y) \
			or (d == down and l[0] < seg.x and l[1] > seg.y) or (d == right and l[0] > seg.x and l[1] > seg.y):
			ret.append(1)
		elif (d == up and l[0] < seg.x and l[1] < seg.y) or (d == left and l[0] < seg.x and l[1] > seg.y) \
			or (d == down and l[0] > seg.x and l[1] > seg.y) or (d == right and l[0] > seg.x and l[1] < seg.y):
			ret.append(3)
		elif (d == up and l[0] < seg.x and l[1] > seg.y) or (d == left and l[0] > seg.x and l[1] > seg.y) \
			or (d == down and l[0] > seg.x and l[1] < seg.y) or (d == right and l[0] < seg.x and l[1] < seg.y):
			ret.append(5)
		else:
			ret.append(7)
	print(tuple(ret))
	return tuple(ret)


def get_eps(epi_num):
	return max(MIN_EPS, min(1.0, np.e**(-epi_num/(TRAINING_EPIS/4))))


def get_act(sn, d, f, ss, eps):
	state = get_state(sn, sn[0], d, f, ss)
	p_acts = get_rel_in_glo(d)

	if np.random.random() < eps:
		return p_acts[np.random.randint(0,2)]
	rel_act = np.argmax(sess.run(q_hat, feed_dict={x: state}))

	return p_acts[rel_act]


def add_exp(sn, di, f, ss):
	old_state = get_state(sn, sn[1], di[1], f, ss)
	new_state = get_state(sn, sn[0], di[0], f, ss)

	if di[1] == di[0]:
		old_act = 0
	elif (di[1] == up and di[0] == left) or (di[1] == left and di[0] == down) \
		or (di[1] == down and di[0] == right) or (di[1] == right and di[0] == up):
		old_act = 1
	else:
		old_act = 2

	if adjacent(sn[0], f, ss, ss, dir_=di[0]):
		r = FOOD_R
	elif collision(sn, di[0], ss):
		r = DEATH_R
	else:
		r = LIVING_R

	global experience
	if experience is None:
		experience = np.array([[old_state, old_act, r, new_state]])
	else:
		experience = np.append(experience, [[old_state, old_act, r, new_state]], axis=0)
		if len(experience) > MEM_LENGTH:
			np.delete(experience, 0,0)


# def train(batch_size=BATCH_SIZE):


