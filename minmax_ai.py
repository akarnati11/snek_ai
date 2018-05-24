import math
from utils import collision, left, right, up, down
from copy import deepcopy


def get_acts(snake, snake_dirs, ss, f, depth):
	act = None
	min_ = math.inf

	if snake_dirs[0] == left or snake_dirs[0] == right:
		dirs = [snake_dirs[0], up, down]
	else:
		dirs = [snake_dirs[0], right, left]

	for a in dirs:
		n_snake, n_dirs = update_snake(deepcopy(snake), deepcopy(snake_dirs), a, ss)
		coll = collision(n_snake, n_dirs[0], ss)

		if not coll:
			val = math.sqrt((n_snake[0].x - f.x) ** 2 + (n_snake[0].y - f.y) ** 2)
		else:
			val = math.inf
		if depth != 1 and val != 0 and not coll:
			val += get_acts(n_snake, n_dirs, ss, f, depth-1)[1]

		if val < min_:
			act = a
			min_ = val
	return act, min_


def update_snake(s, dirs, new_dir, ss):
	t1 = new_dir
	for i in range(len(dirs)):
		t2 = dirs[i]
		dirs[i] = t1
		t1 = t2

	for i in range(len(s)):
		s[i].move_ip(ss * dirs[i][0], ss * dirs[i][1])

	return s, dirs
