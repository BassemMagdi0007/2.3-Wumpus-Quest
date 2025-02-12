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
