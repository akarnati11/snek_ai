from utils import adjacent, collision, get_rel_in_glo, dirs, left, right, up, down
import _pickle as pickle
import numpy as np
from functools import reduce


# Mapping {(state, action): reward}, ((tuple, tuple): double)
# State: (straight clear, right relative clear, left relative clear, body quadrant, food quadrant)
Q = np.random.rand(2, 2, 2, 4, 3)
space = np.zeros(Q.shape)
MIN_ALPHA = .15
MIN_EPS = 0.05
GAMMA = 0.99
FOOD_R = 50
LIVING_R = -.1
DEATH_R = -10
TRAINING_EPIS = 1000


def get_state(sn, seg, d, f, ss):
	# ret = []
	ret = [0, 0, 0]

	sides = get_rel_in_glo(d)
	blocked = collision(sn, d, ss, blocking=True)

	for side in blocked:
		if side in sides:
			ret[sides.index(side)] = 1

	# com = (sum([i.x for i in sn]) // len(sn), sum([i.y for i in sn]) // len(sn))

	for l in [(f.x, f.y)]:
		if (d == up and l[1] == seg.y and l[0] > seg.x) or  (d == left and l[0] == seg.x and l[1] < seg.y) \
			or (d == down and l[1] == seg.y and l[0] < seg.x) or (d == right and l[0] == seg.x and l[1] > seg.y):
			ret.append(0)
		elif (d == up and l[0] == seg.x and l[1] < seg.y) or (d == down and l[0] == seg.x and l[1] > seg.y) \
			or (d == left and l[1] == seg.y and l[0] < seg.x) or (d == right and l[1] == seg.y and l[0] > seg.x):
			# ret.append(2)
			ret.append(1)
		elif (d == up and l[1] == seg.y and l[0] < seg.x) or (d == left and l[0] == seg.x and l[1] > seg.y) \
			or (d == down and l[1] == seg.y and l[0] > seg.x) or (d == right and l[0] == seg.x and l[1] < seg.y):
			# ret.append(4)
			ret.append(2)
		elif (d == up and l[0] == seg.x and l[1] > seg.y) or (d == left and l[1] == seg.y and l[0] > seg.x) \
			or (d == down and l[0] == seg.x and l[1] < seg.y) or (d == right and l[1] == seg.y and l[0] < seg.x):
			# ret.append(6)
			ret.append(3)
		elif (d == up and l[0] > seg.x and l[1] < seg.y) or (d == left and l[0] < seg.x and l[1] < seg.y) \
			or (d == down and l[0] < seg.x and l[1] > seg.y) or (d == right and l[0] > seg.x and l[1] > seg.y):
			# ret.append(1)
			ret.append(0)
		elif (d == up and l[0] < seg.x and l[1] < seg.y) or (d == left and l[0] < seg.x and l[1] > seg.y) \
			or (d == down and l[0] > seg.x and l[1] > seg.y) or (d == right and l[0] > seg.x and l[1] < seg.y):
			# ret.append(3)
			ret.append(1)
		elif (d == up and l[0] < seg.x and l[1] > seg.y) or (d == left and l[0] > seg.x and l[1] > seg.y) \
			or (d == down and l[0] > seg.x and l[1] < seg.y) or (d == right and l[0] < seg.x and l[1] < seg.y):
			# ret.append(5)
			ret.append(2)
		else:
			# ret.append(7)
			ret.append(3)
	print(tuple(ret))
	return tuple(ret)

def get_eps(epi_num):
	return max(MIN_EPS, min(1.0, np.e**(-epi_num/(TRAINING_EPIS/4))))


def get_alpha(epi_num):
	return max(MIN_ALPHA, min(.3, np.e**(-epi_num/(TRAINING_EPIS/4))))


def get_act(sn, d, f, ss, eps):
	state = get_state(sn, sn[0], d, f, ss)
	p_acts = get_rel_in_glo(d)

	if np.random.random() < eps:
		return p_acts[np.random.randint(0,2)]
	rel_act = np.argmax(Q[state])

	return p_acts[rel_act]


def update_Q(sn, di, f, ss, alpha):
	old_state = get_state(sn, sn[1], di[1], f, ss)
	new_state = get_state(sn, sn[0], di[0], f, ss)

	if di[1] == di[0]:
		old_act = 0
	elif (di[1] == up and di[0] == left) or (di[1] == left and di[0] == down) \
		or (di[1] == down and di[0] == right) or (di[1] == right and di[0] == up):
		old_act = 1
	else:
		old_act = 2

	# print(space.shape, old_state)
	space[old_state] = 1

	max_Q = np.max(Q[new_state])
	if adjacent(sn[0], f, ss, ss, dir_=di[0]):
		Q[old_state][old_act] = (1-alpha)*Q[old_state][old_act] + alpha*(FOOD_R + GAMMA*max_Q)
		return FOOD_R
	elif collision(sn, di[0], ss):
		Q[old_state][old_act] = (1 - alpha) * Q[old_state][old_act] + alpha * (DEATH_R + GAMMA * max_Q)
		return DEATH_R
	else:
		Q[old_state][old_act] = (1 - alpha) * Q[old_state][old_act] + alpha * (LIVING_R + GAMMA * max_Q)
		return LIVING_R


def reduce_Q(q, r):
	reduced = q*GAMMA
	for reward in r:
		reduced = GAMMA*(reward + reduced)
	return reduced


def update_Q_sequence(sn, di, f, ss, alpha, depth, past_rewards):
	old_state = get_state(sn, sn[depth-1], di[depth-1], f, ss)
	new_state = get_state(sn, sn[0], di[0], f, ss)

	if di[depth-1] == di[depth-2]:
		old_act = 0
	elif (di[depth-1] == up and di[depth-2] == left) or (di[depth-1] == left and di[depth-2] == down) \
		or (di[depth-1] == down and di[depth-2] == right) or (di[depth-1] == right and di[depth-2] == up):
		old_act = 1
	else:
		old_act = 2

	space[old_state][old_act] = 1

	reduced_Q = reduce_Q(np.max(Q[new_state]), past_rewards)
	if adjacent(sn[0], f, ss, ss, dir_=di[0]):
		Q[old_state][old_act] = (1-alpha)*Q[old_state][old_act] + alpha*(FOOD_R + GAMMA*reduced_Q)
		return FOOD_R
	elif collision(sn, di[0], ss):
		Q[old_state][old_act] = (1 - alpha) * Q[old_state][old_act] + alpha * (DEATH_R + GAMMA * reduced_Q)
		return DEATH_R
	else:
		Q[old_state][old_act] = (1 - alpha) * Q[old_state][old_act] + alpha * (LIVING_R + GAMMA * reduced_Q)
		return LIVING_R


def save_Q_list():
	with open("Q_2000.pickle", 'wb') as f:
		pickle.dump(Q, f)
	print("saving Q:", Q)
	print(Q)

def load_Q_list():
	global Q
	with open("Q_2000.pickle", 'rb') as f:
		Q = pickle.load(f)
	print("loaded Q:", Q)
