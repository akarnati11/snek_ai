

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

"python3 main.py reflex_ai 100"

### minmax_ai:
The snake performs a depth-3 search on possible actions in order to find the best action, which is the one with minimum Euclidean distance sum. Collisions produce an infinite distance too the food and actions that produce collisions are pruned from the search tree.

### q_ai:
Using vanilla Q-learning, snake learns the best actions to take given certain states. The snake game is modeled as a deterministic Markov Decision Process with a reduced state space representation. This reduced state space approach is akin to general dimensionality reduction and allows this simplest type of Q-learning to converge fairly quickly. The features chosen were position of food and position of center of mass of snake relative to the head and binned as one of 8 regions. Epsilon-greedy exploration and temporal difference error approaches were used, both with adaptively shrinking parameters and both starting at 0.5. Reaching food was rewarded with 100, death with -25 and living with -1, as an attempt to drive the snake directly to food. Since the game is deterministic, the discount factor was set high (0.99) to help propogate the sparse positive rewards to past states. The pickle file contains a table trained over 2000 episodes.

Although I tried to account for avoiding self collisions, with indicator variables accounting for relative sides of the snake's head, the snake still tends to kill itself quickly. I chalk this up to the simplified nature of binned Q-learning and have not found an example online of a binned-agent who does better without forbidding self-killing actions.

Third command line arg chooses whether to load a previously trained Q list or to begin exploring with a blank list.  
"python3 main.py q_ai 100 True"

### dqn_ai:
