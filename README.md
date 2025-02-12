# Reinforcement Learning Agent for Grid Navigation

This project implements a reinforcement learning agent that navigates a grid-based environment to collect gold and exit the cave. The agent uses **Policy Iteration**, a reinforcement learning algorithm, to determine the optimal policy for maximizing rewards.

---

## **Code Overview**

### **1. Imports and Constants**
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
- Generates all possible subsets of an iterable (e.g., gold locations).
- Used to represent all possible states of collected gold.

#### **`parse_map(raw_map)`**
- Parses the raw map string into a 2D grid.
- Extracts the positions of gold (`G`) and the starting position (`S`).

#### **`get_walkable_positions(grid)`**
- Returns a list of all walkable positions (non-wall cells) in the grid.

#### **`move_agent(position, action, grid)`**
- Moves the agent based on the chosen action.
- If the action is `EXIT`, the agent stays in place.
- Otherwise, the agent moves in the specified direction if the new position is walkable.

#### **`is_position_walkable(position, grid)`**
- Checks if a position is within the grid bounds and not a wall.

#### **`get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)`**
- Computes the reward for a given transition:
  - Small penalty for each step.
  - Additional penalty for hitting a wall.
  - Reward for collecting gold.
  - Reward for exiting the cave with collected gold.

#### **`get_possible_next_positions(position, action, grid)`**
- Returns all possible next positions for a given action.
- For `EXIT`, the agent can only stay at the current position if it's at the stairs (`S`).

#### **`get_transition_prob(position, action, next_position, grid)`**
- Computes the transition probability for a given action and next position.
- Since the environment is deterministic, the probability is either `1.0` or `0.0`.

---

### **3. Policy Iteration**
- **Policy Iteration**:
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
```
XXXXXXXXXXXXXX

CURRENT POSITION: (7, 9)
ERROR:client:run 6373796: Bad action {'action': 'EXIT'}
