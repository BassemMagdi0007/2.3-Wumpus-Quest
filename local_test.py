import random
import logging
import sys
from itertools import chain, combinations

# Constants
GAMMA = 0.95  # Discount factor
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT"]

# Helper functions

def powerset(iterable):
    """Generate all possible subsets of a given iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def parse_map(raw_map):
    """Parse the map into a 2D list and extract key locations."""
    grid = [list(row) for row in raw_map.split('\n') if row.strip()]
    gold_locations = []
    start_pos = None
    for row_idx, line in enumerate(grid):
        for col_idx, cell in enumerate(line):
            if cell == 'G':
                gold_locations.append((col_idx, row_idx))  # Store as (column, row)
            elif cell == 'S':
                start_pos = (col_idx, row_idx)  # Store as (column, row)
    return grid, gold_locations, start_pos

def get_walkable_positions(grid):
    """Return a list of all coordinates (column, row) in the grid that are walkable."""
    walkable_positions = []
    for row_idx in range(len(grid)):
        for col_idx in range(len(grid[0])):
            if grid[row_idx][col_idx] != 'X':  # Walkable if not a wall
                walkable_positions.append((col_idx, row_idx))  # Store as (column, row)
    return walkable_positions

def move_agent(position, action, grid):
    """Move the agent based on the chosen action (deterministic)."""
    if action == "EXIT":
        return position  # EXIT is not a movement; return current position
    
    col, row = position  # Server uses (column, row)
    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    dc, dr = directions[action]

    new_col, new_row = col + dc, row + dr
    if is_position_walkable((new_col, new_row), grid):  # Use (column, row)
        return (new_col, new_row)
    return position  # Stay in place if movement is blocked

def is_position_walkable(position, grid):
    """Check if a position is within bounds and not a wall."""
    col, row = position
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != 'X':
        return True
    return False

def get_reward(position, action, next_position, gold_collected, gold_locations, start_pos):
    """Compute the reward for a given transition."""
    reward = -0.01  # Small penalty for each step

    # Penalty for hitting a wall
    if next_position == position:
        reward -= 0.1  # Additional penalty for hitting a wall

    # Reward for collecting gold
    if next_position in gold_locations and next_position not in gold_collected:
        reward += 1  # +1 for each gold collected

    # Reward for exiting the cave
    if action == "EXIT" and next_position == start_pos:
        reward += len(gold_collected)  # +1 for each gold carried out

    return reward

def get_possible_next_positions(position, action, grid):
    """Get all possible next positions given an action (deterministic)."""
    if action == "EXIT":
        # EXIT action only allows staying at the current position if it's the stairs
        if grid[position[1]][position[0]] == 'S':
            return {position}
        else:
            return set()  # EXIT is invalid if not at the stairs

    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    dc, dr = directions[action]
    new_col = position[0] + dc
    new_row = position[1] + dr
    new_pos = (new_col, new_row)
    if is_position_walkable(new_pos, grid):
        return {new_pos}
    else:
        return {position}  # Stay in current position if movement is blocked

def get_transition_prob(position, action, next_position, grid):
    """Compute the transition probability for a given action and next position (deterministic)."""
    if action == "EXIT":
        if grid[position[1]][position[0]] == 'S' and next_position == position:
            return 1.0
        else:
            return 0.0

    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    dc, dr = directions[action]
    new_col = position[0] + dc
    new_row = position[1] + dr
    new_pos = (new_col, new_row)
    actual_next_pos = new_pos if is_position_walkable(new_pos, grid) else position
    return 1.0 if next_position == actual_next_pos else 0.0

def policy_iteration(grid, gold_locations, start_pos):
    """Perform Policy Iteration to compute the optimal policy."""
    walkable_positions = get_walkable_positions(grid)
    states = [(pos, frozenset(gold_collected)) for pos in walkable_positions for gold_collected in powerset(gold_locations)]
    
    # Initialize policy and value function
    policy = {state: random.choice(ACTIONS) for state in states}  # Random initial policy
    V = {state: 0 for state in states}  # Initialize value function

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for state in states:
                position, gold_collected = state
                v = V[state]
                action = policy[state]
                total = 0
                for next_position in get_possible_next_positions(position, action, grid):
                    next_gold_collected = set(gold_collected)
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)
                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)
                    total += get_transition_prob(position, action, next_position, grid) * (reward + GAMMA * V[(next_position, frozenset(next_gold_collected))])
                V[state] = total
                delta = max(delta, abs(v - V[state]))
            if delta < EPSILON:
                break

        # Policy Improvement
        policy_stable = True
        for state in states:
            position, gold_collected = state
            old_action = policy[state]
            best_action = None
            best_value = -float('inf')
            for action in ACTIONS:
                total = 0
                for next_position in get_possible_next_positions(position, action, grid):
                    next_gold_collected = set(gold_collected)
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)
                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)
                    total += get_transition_prob(position, action, next_position, grid) * (reward + GAMMA * V[(next_position, frozenset(next_gold_collected))])
                if total > best_value:
                    best_value = total
                    best_action = action
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return policy

def print_grid(grid, agent_position):
    """Print the grid with the agent's position."""
    grid_copy = [row[:] for row in grid]
    col, row = agent_position
    grid_copy[row][col] = 'A'  # Mark the agent's position with 'A'
    for row in grid_copy:
        print(''.join(row))
    print()

def agent_function(request_data, request_info):
    """Main decision-making function for the agent."""
    print('_________________________________________________________')

    # Parse game state
    game_map = request_data.get('map', '')
    grid, gold_locations, start_pos = parse_map(game_map)
    free_skill_points = request_data.get("free-skill-points", 0)
    history = request_data.get("history", [])

    # Print the history in a readable format
    print("\nHISTORY:")
    for event in history:
        action = event.get('action')
        outcome = event.get('outcome')
        print(f"Action: {action}, Outcome: {outcome}")
        
    print('\nGAME MAP:\n', game_map)

    # Allocate skill points if needed
    if free_skill_points > 0:
        skill_allocation = {"agility": free_skill_points, "fighting": 0}
        return {"action": skill_allocation}

    # Extract current position and gold collected from history
    current_position = start_pos
    gold_collected = set()

    if history:
        for event in history:
            outcome = event.get('outcome', {})
            # Update current position based on outcome
            if 'position' in outcome:
                current_position = tuple(outcome['position'])
            # Update collected gold based on outcome
            if 'collected-gold-at' in outcome:
                gold_pos = tuple(outcome['collected-gold-at'])
                gold_collected.add(gold_pos)

    print("\nCURRENT POSITION:", current_position)
    print_grid(grid, current_position)  # Print the grid with the agent's position

    # Compute the optimal policy using Policy Iteration
    policy = policy_iteration(grid, gold_locations, start_pos)

    # Follow the policy and visualize the agent's movements
    while True:
        state = (current_position, frozenset(gold_collected))
        action = policy.get(state, "EXIT")  # Default to EXIT if no policy found

        print(f"\nChosen action for state {state}: {action}")

        # Move the agent based on the chosen action
        next_position = move_agent(current_position, action, grid)
        if next_position == current_position and action != "EXIT":
            print(f"Invalid move detected: {action} from {current_position}")
            break

        print(f"Agent moved to: {next_position}")
        current_position = next_position

        # Update collected gold if the agent moves to a position with gold
        if current_position in gold_locations:
            gold_collected.add(current_position)

        print_grid(grid, current_position)  # Print the grid with the agent's new position

        if action == "EXIT":
            break

    print('CURRENT POSITION:', current_position)

    return action

def main():
    # Hardcoded environment map
    game_map = """
    XXXXXXXXXXXXXX
    XXXXXXXXXXXXXX
    XXXXXXX   XXXX
    XXXXXXX X XXXX
    XXXXXXX X XXXX
    XXXXXXX X XXXX
    XXXXXXX X XXXX
    XX G      XXXX
    XX XXXX XXXXXX
    XXG    SXXXXXX
    XXXXXXXXXXXXXX
    XXXXXXXXXXXXXX
    """

    # Simulate request data
    request_data = {
        "map": game_map,
        "free-skill-points": 0,
        "history": []
    }

    # Call the agent function with the simulated data
    agent_function(request_data, None)

if __name__ == '__main__':
    main()