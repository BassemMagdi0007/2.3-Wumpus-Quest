# Assignment 2.3: Wumpus Quest

This project implements an AI agent for Wumpus Quest, a Markov Decision Process (MDP)-based game where the agent navigates a cave, collects gold, and returns safely while avoiding hazards such as pits, Wumpuses, and bridges. The agent must learn an optimal policy for decision-making using policy iteration, a reinforcement learning technique.

The agent operates in an uncertain environment where movements may not always succeed as intended, and some actions (such as fighting or crossing bridges) depend on skill-based probability rolls.


## Table of Contents

- [Introduction](#introduction)
  - Key Features
- [Setup](#setup)
  - Repository content
  - How to run the code
  - Used libraries
- [Code Structure](#code-structure)
- [Self Evaluation and Design Decisions](#design-decision)
- [Output Format](#output-format)

## Introduction

### Key Features 
- **Markov Decision Process (MDP) Framework:** The agent models the game as an MDP and applies policy iteration to determine the best strategy.
- **State Representation:** The state consists of the agent’s position and gold collected in the cave.
- **Reward System:** The agent receives rewards for collecting gold and penalties for movement costs and hitting walls.
- **Skill Allocation:** The agent can allocate skill points for agility (crossing bridges) and fighting (defeating the Wumpus).
- **Deterministic Transition Model:** The movement follows deterministic rules, but skill-dependent actions use probability-based success/failure.
- **Grid-Based Navigation:** The cave map is represented as a 2D grid where different symbols (S, G, P, W, B, X) define terrain and obstacles.

## Setup
### This repository contains:
 1) **`example.py`**: Core implementation of navigation logic
 2) **`client.py`**: A Python implementation of the AISysProj server protocol
 3) **agent-configs/**: Configuration files for different game scenarios.

### How to run the code: 
1) **`example.py`**, **`client.py`** and **agent-configs/** folder must all be on the same folder
2) Run the **cmd** on the current path.
3) Run the following command **python example.py agent-configs/env-*.json**

### Used libraries:
**_random:_**
Used for random dice rolls in skill checks.
**_itertools:_**
Helps in generating state subsets for MDP state representation.
**_logging:_**
Tracks runtime events and debugging information during environment interactions.


## **Code Overview**

### **1. Imports and Constants**
```python
import random
import logging
from itertools import chain, combinations

# Constants
GAMMA = 0.95  # Discount factor
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT"]
```
- **Imports**:
  - `random`: Used for random choices (e.g., initial policy).
  - `logging`: For logging information during execution.
  - `sys`: For system-related operations (e.g., command-line arguments).
  - `itertools`: Provides utility functions like `chain` and `combinations` for generating subsets.
  - `client.run`: Assumed to be a function provided by the server to run the agent.
- **Constants**:
  - `GAMMA`: Discount factor for future rewards.
  - `EPSILON`: Threshold for convergence in Policy Iteration.
  - `ACTIONS`: List of possible actions the agent can take.

---

### **2. Helper Functions**
#### **`powerset(iterable)`**
```python
def powerset(iterable):
  #...
```
- Generates all possible subsets of an iterable (e.g., gold locations).
- Used to represent all possible states of collected gold.

#### **`parse_map(raw_map)`**
```python
def parse_map(raw_map):
  #...
```
- Parses the raw map string into a 2D grid.
- Extracts the positions of gold (`G`) and the starting position (`S`).

#### **`get_walkable_positions(grid)`**
```python
get_walkable_positions(grid):
  #...
```
- Returns a list of all walkable positions (non-wall cells) in the grid.

#### **`is_position_walkable(position, grid)`**
```python
is_position_walkable(position, grid):
  #...
```
- Checks if a position is within the grid bounds and not a wall.

#### **`get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)`**
```python
get_reward(position, action, next_position, gold_collected, gold_locations, start_pos):
  #...
```
- Computes the reward for a given transition:
  - Small penalty for each step (-0.01).
  - Additional penalty for hitting a wall (-0.1).
  - Reward for collecting gold (+1).
  - Reward for exiting the cave with collected gold (+len(gold_collected)).

#### **`get_possible_next_positions(position, action, grid)`**
```python
get_possible_next_positions(position, action, grid):
  #...
```
- Returns all possible next positions for a given action.
- For `EXIT`, the agent can only stay at the current position if it's at the stairs (`S`).

#### **`get_transition_prob(position, action, next_position, grid)`**
```python
get_transition_prob(position, action, next_position, grid):
  #...
```
- Computes the transition probability for a given action and next position.
- Since the environment is deterministic, the probability is either `1.0` or `0.0`.

---

### **3. Policy Iteration**
- **Policy Iteration**:
```python
policy_iteration(grid, gold_locations, start_pos):
  #...
```
  - Initializes a random policy and value function.
  - Alternates between **Policy Evaluation** (updating the value function) and **Policy Improvement** (updating the policy).
  - Stops when the policy stabilizes (no further changes).

---

### **4. Agent Function**
- **Agent Function**:
  - Parses the game state (map, history, etc.).
  - Allocates skill points if available.
  - Computes the optimal policy using Policy Iteration.
  - Follows the policy to move the agent and collect gold.
  - Returns the chosen action.

---

### **5. Main Function**
- Runs the agent using the `client.run` function.
- Sets up logging and runs the agent for a maximum of 1000 iterations.

---

## **How It Works**
1. The agent receives the game state (map, history, etc.) from the server.
2. It parses the map to identify walkable positions, gold locations, and the starting position.
3. Using **Policy Iteration**, the agent computes the optimal policy for maximizing rewards.
4. The agent follows the policy to navigate the grid, collect gold, and exit the cave.
5. The agent returns the chosen action to the server.

---

## **Usage**
1. Ensure the required dependencies are installed (`random`, `logging`, `sys`, `itertools`).
2. Run the agent using the provided `client.run` function:
   ```bash
   python main.py agent_config.json


# Current Output 

```python

HISTORY:
Action: {'agility': 6, 'fighting': 0}, Outcome: {'agility': 6, 'fighting': 0}
Action: NORTH, Outcome: {'position': [7, 8]}
Action: NORTH, Outcome: {'position': [7, 7]}
Action: NORTH, Outcome: {'position': [7, 6]}
Action: NORTH, Outcome: {'position': [7, 5]}
Action: NORTH, Outcome: {'position': [7, 4]}
Action: NORTH, Outcome: {'position': [7, 3]}
Action: NORTH, Outcome: {'position': [7, 2]}
Action: EAST, Outcome: {'position': [7, 3]}
Action: NORTH, Outcome: {'position': [7, 2]}
Action: EAST, Outcome: {'position': [8, 2]}


GAME MAP:
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX

CURRENT POSITION: (8, 2)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX A XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((8, 2), frozenset()): EAST
Agent moved to: (9, 2)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX  AXXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((9, 2), frozenset()): SOUTH
Agent moved to: (9, 3)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XAXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((9, 3), frozenset({(9, 3)})): SOUTH
Agent moved to: (9, 4)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX XAXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((9, 4), frozenset({(9, 3)})): SOUTH
Agent moved to: (9, 5)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX XAXXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((9, 5), frozenset({(9, 3)})): SOUTH
Agent moved to: (9, 6)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX XAXXXX
XX G      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((9, 6), frozenset({(9, 3)})): SOUTH
Agent moved to: (9, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G     AXXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((9, 7), frozenset({(9, 3)})): WEST
Agent moved to: (8, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G    A XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((8, 7), frozenset({(9, 3)})): WEST
Agent moved to: (7, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G   A  XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((7, 7), frozenset({(9, 3)})): WEST
Agent moved to: (6, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G  A   XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((6, 7), frozenset({(9, 3)})): WEST
Agent moved to: (5, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G A    XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((5, 7), frozenset({(9, 3)})): WEST
Agent moved to: (4, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX GA     XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((4, 7), frozenset({(9, 3)})): WEST
Agent moved to: (3, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX A      XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((3, 7), frozenset({(3, 7), (9, 3)})): EAST
Agent moved to: (4, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX GA     XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((4, 7), frozenset({(3, 7), (9, 3)})): EAST
Agent moved to: (5, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G A    XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((5, 7), frozenset({(3, 7), (9, 3)})): EAST
Agent moved to: (6, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G  A   XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((6, 7), frozenset({(3, 7), (9, 3)})): EAST
Agent moved to: (7, 7)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G   A  XXXX
XX XXXX XXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((7, 7), frozenset({(3, 7), (9, 3)})): SOUTH
Agent moved to: (7, 8)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXXAXXXXXX
XX     SXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((7, 8), frozenset({(3, 7), (9, 3)})): SOUTH
Agent moved to: (7, 9)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     AXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX


Chosen action for state ((7, 9), frozenset({(3, 7), (9, 3)})): EXIT
Agent moved to: (7, 9)
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX
XXXXXXX   XXXX
XXXXXXX XGXXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XXXXXXX X XXXX
XX G      XXXX
XX XXXX XXXXXX
XX     AXXXXXX
XXXXXXXXXXXXXX
XXXXXXXXXXXXXX

CURRENT POSITION: (7, 9)

NEW HISTORY:
Action: EAST, Outcome: {'position': [9, 2], 'collected-gold-at': None}
Action: SOUTH, Outcome: {'position': [9, 3], 'collected-gold-at': [9, 3]}
Action: SOUTH, Outcome: {'position': [9, 4], 'collected-gold-at': None}
Action: SOUTH, Outcome: {'position': [9, 5], 'collected-gold-at': None}
Action: SOUTH, Outcome: {'position': [9, 6], 'collected-gold-at': None}
Action: SOUTH, Outcome: {'position': [9, 7], 'collected-gold-at': None}
Action: WEST, Outcome: {'position': [8, 7], 'collected-gold-at': None}
Action: WEST, Outcome: {'position': [7, 7], 'collected-gold-at': None}
Action: WEST, Outcome: {'position': [6, 7], 'collected-gold-at': None}
Action: WEST, Outcome: {'position': [5, 7], 'collected-gold-at': None}
Action: WEST, Outcome: {'position': [4, 7], 'collected-gold-at': None}
Action: WEST, Outcome: {'position': [3, 7], 'collected-gold-at': [3, 7]}
Action: EAST, Outcome: {'position': [4, 7], 'collected-gold-at': None}
Action: EAST, Outcome: {'position': [5, 7], 'collected-gold-at': None}
Action: EAST, Outcome: {'position': [6, 7], 'collected-gold-at': None}
Action: EAST, Outcome: {'position': [7, 7], 'collected-gold-at': None}
Action: SOUTH, Outcome: {'position': [7, 8], 'collected-gold-at': None}
Action: SOUTH, Outcome: {'position': [7, 9], 'collected-gold-at': None}
Action: EXIT, Outcome: {'position': [7, 9], 'collected-gold-at': None}
ERROR:client:run 6373796: Bad action {'action': 'EXIT'}
```
