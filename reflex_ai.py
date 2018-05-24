import math


def get_acts(p_acts, snake, ss, f):
	act = None
	min_ = math.inf
	for a in p_acts:
		new_pos = (snake[0].x + a[0] * ss, snake[0].y + a[1] * ss)
		val = math.sqrt((new_pos[0] - f.x) ** 2 + (new_pos[1] - f.y) ** 2)
		if val < min_:
			act = a
			min_ = val
	return act
