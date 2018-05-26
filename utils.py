import random
from functools import reduce

size = 720, 720

left = (-1, 0)
right = (1, 0)
up = (0, -1)
down = (0, 1)

dirs = [up, left, down, right]


def prod(l):
	return reduce(lambda x, y: x*y, l)


def get_p_acts(s, d, ss):
	if d == up or d == down:
		dirs_ = [left, right, d]
	else:
		dirs_ = [up, down, d]

	kept = []
	for dir_ in dirs_:
		n_pos = (s[0].x + dir_[0] * ss, s[0].y + dir_[1] * ss)
		if not collision(s, d, ss, other_pos=n_pos):
			kept.append(dir_)
	return kept


def gen_new_food(s, ss):
	mark = True
	r_x,r_y = 0,0
	while mark:
		r_x, r_y = ((((size[0]-2*ss) * random.random()) // ss)+1) * ss, ((((size[0]-2*ss) * random.random()) // ss)+1) * ss
		mark = False
		for b in s:
			t = b.copy()
			t.x, t.y = r_x, r_y
			if (b.x == r_x and b.y == r_y) or adjacent(b, t, ss, ss)[0]:
				mark = True
	return (r_x,r_y)


def adjacent(s_b, b, off, ss, dir_=None):
	if dir_ is None:
		sides = []
		if (s_b.x == b.x and s_b.y == b.y + ss) or b.y == 0:
			sides.append(0)
		if (s_b.x == b.x and s_b.y == b.y - ss) or b.y+ss == size[0]:
			sides.append(2)
		if (s_b.y == b.y and s_b.x == b.x - ss) or b.x+ss == size[0]:
			sides.append(3)
		if (s_b.y == b.y and s_b.x == b.x + ss) or b.x == 0:
			sides.append(1)
		return len(sides) != 0, sides
	if dir_ == right and s_b.y == b.y and s_b.x == b.x-off:
		# print("1")
		return True
	elif dir_ == left and s_b.y == b.y and s_b.x == b.x+off:
		# print("2")
		return True
	elif dir_ == down and s_b.x == b.x and s_b.y == b.y-off:
		# print("3")
		return True
	elif dir_ == up and s_b.x == b.x and s_b.y == b.y+off:
		# print("4")
		return True
	return False


def init_game(init_size, step_size, block_rect, r_start=False):
	if r_start:
		r_x = random.randint(init_size + 1, size[0]//step_size - init_size - 1)*step_size
		r_y = random.randint(init_size + 1, size[0]//step_size - init_size - 1)*step_size
		r_dir = random.randint(0,4)
		# print(r_x, r_y, r_dir)
	else:
		r_x = 0
		r_y = 0
		r_dir = 0
	block_rect.x, block_rect.y = r_x, r_y

	if r_dir == 0:
		snake = [block_rect.move(step_size * i, 0) for i in range(init_size-1, -1, -1)]
		snake_dirs = [right for _ in range(init_size)]
	elif r_dir == 1:
		snake = [block_rect.move(0, step_size * i) for i in range(init_size-1, -1, -1)]
		snake_dirs = [down for _ in range(init_size)]
	elif r_dir == 2:
		snake = [block_rect.move(0, -step_size * i) for i in range(init_size - 1, -1, -1)]
		snake_dirs = [up for _ in range(init_size)]
	else:
		snake = [block_rect.move(-step_size * i, 0) for i in range(init_size - 1, -1, -1)]
		snake_dirs = [left for _ in range(init_size)]


	# print(snake)
	# print(snake_dirs)

	f = gen_new_food(snake, step_size)

	return 0, right, snake, snake_dirs, 0, f


def get_rel_in_glo(d):
	if d == up:
		return [d, left, right]
	elif d == down:
		return [d, right, left]
	elif d == left:
		return [d, down, up]
	return [d, up, down]


def collision(s, dir_, ss, other_pos=None, self_collide=True, blocking=False):
	pos = s[0].copy()
	if other_pos is not None:
		pos.x, pos.y = other_pos[0], other_pos[1]
	# Against wall
	if not blocking:
		if dir_ == right and pos.x == size[0]:
			return True
		elif dir_ == left and pos.x+ss == 0:
			return True
		elif dir_ == down and pos.y == size[0]:
			return True
		elif dir_== up and pos.y+ss==0:
			return True

	if blocking:
		sides = list()
		for seg in s[1:]:
			_, si = adjacent(pos, seg, 0, ss)
			sides.extend(si)
		return [dirs[i] for i in set(sides)]
	elif self_collide:
		# Against itself
		return any([adjacent(pos,seg, 0, ss, dir_=dir_) for seg in s[1:]])
