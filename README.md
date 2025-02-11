# Wumpus Quest - Markov Decision Process (MDP) Implementation

This project implements a Markov Decision Process (MDP) to solve the Wumpus Quest problem. The goal is to guide an agent through a cave to collect gold and exit safely, while avoiding obstacles like walls and pits.

---

## **Code Overview**

### **1. Imports and Constants**
```python
import random
import logging
import sys
from itertools import chain, combinations
from client import run  # Assuming this is provided by the server protocol

# Constants
GAMMA = 0.95  # Discount factor
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT"]
```
- **Imports**:
  - `random`: Used for introducing randomness in agent movement.
  - `logging`: Used for logging information during execution.
  - `sys`: Used for command-line arguments.
  - `itertools.chain` and `itertools.combinations`: Used to generate all possible subsets of gold locations.
  - `client.run`: Assumed to be provided by the server protocol for running the agent.

- **Constants**:
  - `GAMMA = 0.95`: The discount factor for future rewards in the MDP.
  - `EPSILON = 1e-6`: The convergence threshold for Value Iteration.
  - `ACTIONS`: The list of possible actions the agent can take (`NORTH`, `SOUTH`, `EAST`, `WEST`, `EXIT`).

---

### **2. Helper Functions**

#### **`powerset(iterable)`**
```python
def powerset(iterable):
  #...
```
- **Purpose**: Generates all possible subsets of the gold locations.
- **Example**: If `gold_locations = [(1, 2), (3, 4)]`, the powerset will be `[(), ((1, 2)), ((3, 4)), ((1, 2), (3, 4))]`.
- **Usage**: Used to represent all possible states where the agent has collected different combinations of gold.

#### **`parse_map(raw_map)`**
```python
def parse_map(raw_map):
  #...
```
- **Purpose**: Parses the map string into a 2D grid and extracts the positions of gold (`G`) and the starting position (`S`).
- **Output**:
  - `grid`: A 2D list representing the map.
  - `gold_locations`: A list of tuples representing the coordinates of gold.
  - `start_pos`: A tuple representing the starting position of the agent.

#### **`get_walkable_positions(grid)`**
```python
def get_walkable_positions(grid):
  #...
```
- **Purpose**: Returns a list of all walkable positions in the grid (i.e., positions that are not walls `X`).
- **Output**: A list of tuples representing walkable positions.

#### **`move_agent(position, action, grid)`**
```python
def move_agent(position, action, grid):
  #...
```
- **Purpose**: Moves the agent based on the chosen action, with a 10% chance of deviating left or right.
- **Logic**:
  - The agent has an 80% chance of moving in the intended direction.
  - A 10% chance of moving left.
  - A 10% chance of moving right.
- **Output**: The new position after attempting to move.

#### **`is_position_walkable(position, grid)`**
```python
def is_position_walkable(position, grid):
  #...
```
- **Purpose**: Checks if a position is within the grid bounds and not a wall (`X`).
- **Output**: `True` if the position is walkable, `False` otherwise.

#### **`get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)`**
```python
def get_reward(position, action, next_position, gold_collected, gold_locations, start_pos):
  #...
```
- **Purpose**: Computes the reward for a given transition.
- **Rewards**:
  - `-0.01`: Small penalty for each step (encourages the agent to exit quickly).
  - `-0.1`: Additional penalty for hitting a wall.
  - `+1`: Reward for collecting gold.
  - `+len(gold_collected)`: Reward for exiting the cave with collected gold.

#### **`value_iteration(grid, gold_locations, start_pos)`**
```python
def value_iteration(grid, gold_locations, start_pos):
  #...
```
- **Purpose**: Performs Value Iteration to compute the optimal policy.
- **Steps**:
  1. Initialize the value function `V` for all states.
  2. Iteratively update the value function until convergence.
  3. Extract the optimal policy by choosing the action that maximizes the expected reward for each state.

#### **`get_possible_next_positions(position, action, grid)`**
```python
def get_possible_next_positions(position, action, grid):
  #...
```
- **Purpose**: Returns all possible next positions for a given action, accounting for deviations.
- **Logic**:
  - For `EXIT`, the agent stays at the current position if it’s at the stairs.
  - For movement actions, the agent can move in the intended direction or deviate left/right with a 10% chance each.

#### **`get_transition_prob(position, action, next_position, grid)`**
```python
def get_transition_prob(position, action, next_position, grid):
  #...
```
- **Purpose**: Computes the transition probability for a given action and next position.
- **Logic**:
  - For `EXIT`, the probability is 1.0 if the agent is at the stairs and stays there.
  - For movement actions, the probability is 0.8 for the intended direction and 0.1 for left/right deviations.

#### **`agent_function(request_data, request_info)`**
```python
def agent_function(request_data, request_info):
  #...
```
- **Purpose**: The main function that processes the server’s request and decides the agent’s action.
- **Steps**:
  1. Parse the game state (map, gold locations, start position).
  2. Allocate skill points if needed.
  3. Extract the current position and collected gold from the history.
  4. Compute the optimal policy using Value Iteration.
  5. Choose the best action based on the current state.

---

### **3. Main Execution**
- **Purpose**: Runs the agent using the server protocol.
- **Steps**:
  1. Set up logging.
  2. Run the agent with the provided configuration file.

---

## **Key Values**
- **States**: Represented as `(position, frozenset(gold_collected))`.
- **Actions**: `NORTH`, `SOUTH`, `EAST`, `WEST`, `EXIT`.
- **Rewards**:
  - `-0.01`: Penalty for each step.
  - `-0.1`: Penalty for hitting a wall.
  - `+1`: Reward for collecting gold.
  - `+len(gold_collected)`: Reward for exiting the cave with collected gold.
- **Transition Probabilities**:
  - `0.8`: Intended direction.
  - `0.1`: Left/right deviation.

---

## **How to Run**
1. Ensure the required dependencies are installed.
2. Run the agent using the following command:
   ```bash
   python example.py <agent_config_file>
