import random
import logging
import sys
from itertools import chain, combinations
from client import run  # Assuming this is provided by the server protocol

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
    """Move the agent based on the chosen action with a 10% chance of deviation."""
    col, row = position  # Server uses (column, row)
    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    dc, dr = directions[action]

    # Introduce randomness (10% chance to move left or right)
    if random.random() < 0.1:  # 10% chance to move left
        dc, dr = (dr, -dc) if dr != 0 else (-1, 0)
    elif random.random() < 0.2:  # Additional 10% chance to move right
        dc, dr = (-dr, dc) if dr != 0 else (1, 0)

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


def value_iteration(grid, gold_locations, start_pos):
    """Perform Value Iteration to compute the optimal policy."""
    walkable_positions = get_walkable_positions(grid)
    states = [(pos, frozenset(gold_collected)) for pos in walkable_positions for gold_collected in powerset(gold_locations)]
    V = {state: 0 for state in states}  # Initialize value function

    while True:
        delta = 0
        for state in states:
            position, gold_collected = state
            v = V[state]
            max_value = -float('inf')
            for action in ACTIONS:
                total = 0
                for next_position in get_possible_next_positions(position, action, grid):
                    next_gold_collected = set(gold_collected)
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)
                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)
                    total += get_transition_prob(position, action, next_position, grid) * (reward + GAMMA * V[(next_position, frozenset(next_gold_collected))])
                if total > max_value:
                    max_value = total
            V[state] = max_value
            delta = max(delta, abs(v - V[state]))
        if delta < EPSILON:
            break

    # Extract the optimal policy
    policy = {}
    for state in states:
        position, gold_collected = state
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

    return policy


def get_possible_next_positions(position, action, grid):
    """Get all possible next positions given an action."""
    if action == "EXIT":
        # EXIT action only allows staying at the current position if it's the stairs
        if grid[position[1]][position[0]] == 'S':  # Check if the current position is the stairs
            return {position}
        else:
            return set()  # EXIT is invalid if not at the stairs

    next_positions = set()
    col, row = position
    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }

    # Delta change in position based on the action
    dc, dr = directions[action]

    # Intended direction
    new_col, new_row = col + dc, row + dr
    if is_position_walkable((new_col, new_row), grid):
        next_positions.add((new_col, new_row))

    # Left deviation (10% chance)
    dc_left, dr_left = (dr, -dc) if dr != 0 else (-1, 0)
    new_col_left, new_row_left = col + dc_left, row + dr_left
    if is_position_walkable((new_col_left, new_row_left), grid):
        next_positions.add((new_col_left, new_row_left))

    # Right deviation (10% chance)
    dc_right, dr_right = (-dr, dc) if dr != 0 else (1, 0)
    new_col_right, new_row_right = col + dc_right, row + dr_right
    if is_position_walkable((new_col_right, new_row_right), grid):
        next_positions.add((new_col_right, new_row_right))

    return next_positions


def get_transition_prob(position, action, next_position, grid):
    """Compute the transition probability for a given action and next position."""
    if action == "EXIT":
        # EXIT action only allows staying at the current position if it's the stairs
        if grid[position[1]][position[0]] == 'S' and next_position == position:
            return 1.0  # 100% chance of staying at the stairs
        else:
            return 0.0  # EXIT is invalid if not at the stairs or not staying

    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    dc, dr = directions[action]

    # Intended direction
    new_col, new_row = position[0] + dc, position[1] + dr
    if (new_col, new_row) == next_position:
        return 0.8  # 80% chance of moving in the intended direction

    # Left deviation (10% chance)
    dc_left, dr_left = (dr, -dc) if dr != 0 else (-1, 0)
    new_col_left, new_row_left = position[0] + dc_left, position[1] + dr_left
    if (new_col_left, new_row_left) == next_position:
        return 0.1

    # Right deviation (10% chance)
    dc_right, dr_right = (-dr, dc) if dr != 0 else (1, 0)
    new_col_right, new_row_right = position[0] + dc_right, position[1] + dr_right
    if (new_col_right, new_row_right) == next_position:
        return 0.1

    return 0.0  # No other possible transitions


def agent_function(request_data, request_info):
    """Main decision-making function for the agent."""
    print('_________________________________________________________')
    print('\nREQUEST:')
    print(request_data)

    # Parse game state
    game_map = request_data.get('map', '')
    grid, gold_locations, start_pos = parse_map(game_map)
    free_skill_points = request_data.get("free-skill-points", 0)
    history = request_data.get("history", [])

    print('\nSTART:\n', start_pos)
    print('\nGAME MAP:\n', game_map)

    # Allocate skill points if needed
    if free_skill_points > 0:
        skill_allocation = {"agility": free_skill_points, "fighting": 0}
        return {"action": skill_allocation}

    # Extract current position
    if history:
        for event in reversed(history):
            if 'outcome' in event and 'position' in event['outcome']:
                current_position = tuple(event['outcome']['position'])  # Expecting (column, row)
                break
    else:
        current_position = start_pos

    # Track collected gold
    gold_collected = set()
    for event in history:
        if 'outcome' in event and 'collected-gold-at' in event['outcome']:
            gold_collected.add(tuple(event['outcome']['collected-gold-at']))

    # Compute the optimal policy using Value Iteration
    policy = value_iteration(grid, gold_locations, start_pos)

    # Choose the best action based on the current state
    state = (current_position, frozenset(gold_collected))
    action = policy.get(state, "EXIT")  # Default to EXIT if no policy found

    return {"action": action}


if __name__ == '__main__':
    import sys
    import logging
    from client import run

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the agent
    run(
        agent_config_file=sys.argv[1],
        agent=agent_function,
        parallel_runs=False,
        run_limit=1000  # Stop after 1000 runs
    )