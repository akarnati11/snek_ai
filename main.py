import sys, pygame
import reflex_ai
import q_ai, dqn_ai, minmax_ai
from utils import adjacent, collision, init_game, gen_new_food, get_p_acts, left, right, up, down, size, prod
import evolve_nn_ai
import numpy as np


pygame.init()

# if len(sys.argv) > 2:
# 	load = bool(sys.argv[3])
# if len(sys.argv) > 1:
# 	control = sys.argv[1]
# 	game_per = int(sys.argv[2])
# else:
# 	control = "reflex_ai"
# 	game_per = 100
# 	load = False

CONTROL = "q_ai"
GAME_PER = 3000
LOAD = False
TESTING = False
GRAPHICS = True


block = pygame.image.load("block.png")
food = pygame.image.load("food.png")
screen = pygame.display.set_mode(size)
label_font = pygame.font.SysFont("monospace", 50)

block_rect = block.get_rect()
step_size = block_rect.width

t, new_dir, snake, snake_dirs, score, f_pos = init_game(5, step_size, block_rect)
food_rect = food.get_rect().move(f_pos[0], f_pos[1])
clock = pygame.time.Clock()

start_size = 5

if LOAD and CONTROL == "q_ai":
	q_ai.load_Q_list()

c = 0
epi_rew = 0
epis = 0
reward_buffer = []
while 1:
	if CONTROL == "q_ai" and epis == q_ai.TRAINING_EPIS:
		q_ai.save_Q_list()
		print("Episodes:", epis)
		sys.exit()
	# elif CONTROL == "dqn_ai" and epis == dqn_ai.TRAINING_EPIS:
	# 	dqn_ai.save_Q_net()
	# 	print("Episodes:", epis)
	# 	dqn_ai.sess.close()
	# 	sys.exit()
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			if CONTROL == "q_ai":
				q_ai.save_Q_list()
				print("Episodes:", epis)
			sys.exit()
		if CONTROL is "q_ai":
			# Key event logic
			pressed = pygame.key.get_pressed()
			if pressed[pygame.K_UP] and snake_dirs[0] != down:
				new_dir = up
				# print("up")
			elif pressed[pygame.K_RIGHT] and snake_dirs[0] != left:
				new_dir = right
				# print("left")
			elif pressed[pygame.K_DOWN] and snake_dirs[0] != up:
				new_dir = down
				# print("down")
			elif pressed[pygame.K_LEFT] and snake_dirs[0] != right:
				new_dir = left
				# print("right")

	alpha, eps = 0, 0
	if t > GAME_PER:
		c += 1
		if CONTROL == "reflex_ai":
			possible_acts = get_p_acts(snake, new_dir, step_size)
			new_dir = reflex_ai.get_acts(possible_acts, snake, step_size, food_rect)
		elif CONTROL == "minmax_ai":
			new_dir = minmax_ai.get_acts(snake, snake_dirs, step_size, food_rect, 3)[0]
		elif CONTROL == "q_ai":
			if TESTING:
				eps, alpha = 0, 0
			else:
				eps, alpha = q_ai.get_eps(epis), q_ai.get_alpha(epis)
			# cur_rew = q_ai.update_Q(snake, snake_dirs, food_rect, step_size, alpha)
			cur_rew = q_ai.update_Q_sequence(snake, snake_dirs, food_rect, step_size, alpha, start_size, reward_buffer)
			epi_rew += cur_rew
			reward_buffer.insert(0, cur_rew)
			if len(reward_buffer) > start_size:
				reward_buffer.pop()
			new_dir = q_ai.get_act(snake, snake_dirs[0], food_rect, step_size, eps)
		elif CONTROL == "dqn_ai":
			if TESTING:
				eps = 0
			else:
				eps = dqn_ai.get_eps(epis)
			epi_rew += dqn_ai.add_exp(snake, snake_dirs, food_rect, step_size)
			# dqn_ai.train()
		elif CONTROL == "evolve_nn_ai":
			print("ddd")
			# act = evolve_nn_ai.get_act()
		else:
			new_dir = left

		# print(snake_dirs)


		# If snake eats food
		if adjacent(snake[0], food_rect, step_size, step_size, dir_=new_dir):
			rx, ry = gen_new_food(snake, step_size)
			snake.insert(0, block_rect.copy())
			snake[0].x, snake[0].y = food_rect.x, food_rect.y
			snake_dirs.insert(0, new_dir)
			food_rect.x, food_rect.y = rx, ry
			score += 1

		# Update directions
		t1 = new_dir
		for i in range(len(snake_dirs)):
			t2 = snake_dirs[i]
			snake_dirs[i] = t1
			t1 = t2

		# Update positions
		for i in range(len(snake)):
			snake[i].move_ip(step_size * snake_dirs[i][0], step_size * snake_dirs[i][1])

		# print(snake_dirs)

		if CONTROL == "q_ai" and collision(snake, snake_dirs[0], step_size):
			if TESTING:
				eps, alpha = 0, 0
			else:
				eps, alpha = q_ai.get_eps(epis), q_ai.get_alpha(epis)
			# cur_rew = q_ai.update_Q(snake, snake_dirs, food_rect, step_size, alpha)
			cur_rew = q_ai.update_Q_sequence(snake, snake_dirs, food_rect, step_size, alpha, start_size, reward_buffer)
			epi_rew += cur_rew
			reward_buffer.insert(0, cur_rew)
			if len(reward_buffer) > start_size:
				reward_buffer.pop()

			t, new_dir, snake, snake_dirs, score, f_pos = init_game(start_size, step_size, block_rect, True)
			food_rect.x, food_rect.y = f_pos[0], f_pos[1]
			print("Episode {} terminated after {} cycles".format(epis, c))
			print("Episode {} reward: {}".format(epis, epi_rew))
			print("Episode {} alpha and eps: {}, {}".format(epis, alpha, eps))
			print("Q-space explored: {}".format(np.sum(q_ai.space)/prod(q_ai.space.shape)))
			c = 0
			epi_rew = 0
			epis += 1
		elif CONTROL != "q_ai" and collision(snake, snake_dirs[0], step_size):
			t, new_dir, snake, snake_dirs, score, f_pos = init_game(start_size, step_size, block_rect)
			food_rect.x, food_rect.y = f_pos[0], f_pos[1]


		#Paint to screen
		if GRAPHICS:
			screen.fill((0,0,0))
			screen.blit(food, food_rect)
			for b in snake:
				screen.blit(block, b)
			label = label_font.render(str(score), 1, (255, 255, 255))
			screen.blit(label, (size[0] - 50, 25))
			pygame.display.flip()


		# Reset timer
		t = 0

	t += clock.tick(45)
