

### Dependencies:
- pygame
- tensorflow (evolve_nn_ai.py)
- _ pickle (q_ai.py, evolve_nn_ai.py)

### user_op:
Allows user to play snake using the arrow keys. Second command line arg chooses game period. game period must be at lowest 100 otherwise rendering issues will occur when snake eats food.  
"python3 main.py user_op 300"

### reflex_ai:
Snake chooses actions that put it into a state with a closer Euclidean distance to food, from available actions (actions that do not kill snake through collisions). This AI can be seen as a depth 1 search agent or a reflex agent as it only considers immediate local actions, and risks being 'blocked in' by its own body at large lengths. This can be seen below, with the yellow square highlighting the head:

![alt text](caught.png)  

"python3 main.py dumb_ai 100"

### minmax_ai:
The snake performs a depth-3 search on possible actions in order to find the best action, which is the one with minimum Euclidean distance sum. Collisions produce an infinite distance too the food and actions that produce collisions are pruned from the search tree.

### q_ai
Snake uses a reinforcement learning scheme (Q-learning) to take actions. In this model the state consists of relative mapping of the food to the head of the snake, reduced to 4 quadrants, and the relative mapping of the center of mass of the snake to the head of the snake, also reduced to 4 quadrants. The idea behind the reduction is to dramatically reduce the size of the state space, as this is a common issue with classical Q-learning for an environment with thousands of states. I've also kept track of the CoM of the snake as a way for the snake to take into account collisions with the itself, although more testing has to be done to improve this. Additionally, since there is only one high reward state at any given moment (food location), a single step of the snake translates to a small negative reward, as a way to force the snake to reach the food quicker. The actual Q update is a Temporal Difference (TD) error.

Third command line arg chooses whether to load a previously trained Q list or to begin exploring with a blank list.  
"python3 main.py q_ai 100 True"
