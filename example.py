import random
import logging
import heapq
from itertools import chain, combinations

#TODO: Allow the agent to pass the S when it reaches it and still gold remaining 
#TODO: Agent falls into pits 
#TODO: Agent falls into pit to cross the bridge

# Constants
GAMMA = 0.99  # Discount factor
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT"]
REWARD_GOLD = 10  # Increased reward for collecting gold


# Helper functions
"""Generate all possible subsets of a given iterable."""
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


"""Parse the map into a 2D list and extract key locations."""
def parse_map(raw_map):
    grid = [list(row) for row in raw_map.split('\n') if row]
    gold_locations = []
    start_pos = None
    for row_idx, line in enumerate(grid):
        for col_idx, cell in enumerate(line):
            if cell == 'G':
                gold_locations.append((col_idx, row_idx))  # Store as (column, row)
            elif cell == 'S':
                start_pos = (col_idx, row_idx)  # Store as (column, row)
    return grid, gold_locations, start_pos


"""Return a list of all coordinates (column, row) in the grid that are walkable."""
def get_walkable_positions(grid):
    walkable_positions = []
    for row_idx in range(len(grid)):
        for col_idx in range(len(grid[0])):
            if grid[row_idx][col_idx] != 'X':  # Walkable if not a wall
                walkable_positions.append((col_idx, row_idx))  # Store as (column, row)
    return walkable_positions


"""Check if a position is within bounds and not a wall."""
def is_position_walkable(position, grid):
    col, row = position
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != 'X' and grid[row][col] != 'P':
        return True
    return False


"""Compute the reward for a given transition."""
def get_reward(position, action, next_position, gold_collected, gold_locations, start_pos):
    reward = -0.001  # Reduced penalty per step

    if next_position == position and action != "EXIT":
        reward -= 0.01  # Slightly higher penalty for hitting a wall

    if next_position in gold_locations and next_position not in gold_collected:
        reward += REWARD_GOLD  # Higher reward for collecting gold

    if action == "EXIT" and next_position == start_pos:
        reward += len(gold_collected) * 2  # Bonus for exiting with gold

    return reward


"""Get all possible next positions given an action (deterministic)."""
def get_possible_next_positions(position, action, grid):
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


"""Compute the transition probability for a given action and next position (deterministic)."""
def get_transition_prob(position, action, next_position, grid):
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


"""Perform Policy Iteration to compute the optimal policy."""
def policy_iteration(grid, gold_locations, start_pos):
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


"""Determine if the agent successfully crosses the bridge."""
def cross_bridge(agility_skill, bridge_position):
    dice_rolls = [random.randint(1, 6) for _ in range(agility_skill)]
    dice_rolls.sort(reverse=True)
    score = sum(dice_rolls[:3])
    # print(f"Dice Rolls: {dice_rolls}, Score: {score}")
    if score >= 12:
        print(f"Agent successfully crosses the bridge at {bridge_position}.")
    return score >= 12


"""Print the grid with the agent's position."""
def print_grid(grid, agent_position):
    grid_copy = [row[:] for row in grid]
    col, row = agent_position
    grid_copy[row][col] = 'A'  # Mark the agent's position with 'A'
    for row in grid_copy:
        print(''.join(row))
    print()


"""A* pathfinding algorithm to find the shortest path to the nearest gold."""
def a_star_search(grid, start, goal, agility_skill):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for action in ACTIONS[:-1]:  # Exclude "EXIT" action
            for next_position in get_possible_next_positions(current, action, grid):
                if not is_position_walkable(next_position, grid):
                    continue
                if grid[next_position[1]][next_position[0]] == 'B' and not cross_bridge(agility_skill, next_position):
                    continue  # Skip this position if it's a bridge and the agent can't cross it
                tentative_g_score = g_score[current] + 1  # Assume cost of 1 for each move
                if next_position not in g_score or tentative_g_score < g_score[next_position]:
                    came_from[next_position] = current
                    g_score[next_position] = tentative_g_score
                    f_score[next_position] = tentative_g_score + heuristic(next_position, goal)
                    heapq.heappush(open_set, (f_score[next_position], next_position))

    return []  # Return empty path if no path found


"""Agent function."""
def agent_function(request_data, request_info):
    print('_________________________________________________________')

    # Parse game state
    game_map = request_data.get('map', '')
    grid, gold_locations, start_pos = parse_map(game_map)
    free_skill_points = request_data.get("free-skill-points", 0)
    history = request_data.get("history", [])

    # Print the history in a readable format
    for event in history:
        action = event.get('action')
        outcome = event.get('outcome')
        print(f"Action: {action}, Outcome: {outcome}")

    # Allocate skill points if needed (first action)
    if free_skill_points > 0:
        skill_allocation = {"agility": free_skill_points, "fighting": 0}
        return skill_allocation  # Return JSON object for skill allocation

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

    print_grid(grid, current_position)  # Print the grid with the agent's position

    # Print the amount of collected gold
    print(f"COLLECTED GOLD: {len(gold_collected)}")

    # Check if the agent is on the stairs and has collected gold
    if grid[current_position[1]][current_position[0]] == 'S' and gold_collected:
        # Agent is on the stairs and has collected gold, EXIT is valid
        return "EXIT"  # Return plain string for EXIT action

    # Check if the agent is on a bridge and needs to cross it
    if grid[current_position[1]][current_position[0]] == 'B':
        agility_skill = request_data.get("skill-points", {}).get("agility", 0)
        for attempt in range(agility_skill): 
            if cross_bridge(agility_skill, current_position):
                print("Agent successfully crosses the bridge.")
                break
            else:
                print(f"Attempt {attempt + 1}: Agent failed to cross the bridge.")
        else:
            print("Agent failed to cross the bridge after 10 attempts and falls into the pit.")
            return "EXIT"  # Exit the cave if the agent fails after 10 attempts

    # Find the nearest gold using A* search
    nearest_gold = None
    shortest_path = []
    agility_skill = request_data.get("skill-points", {}).get("agility", 0)
    for gold_pos in gold_locations:
        if gold_pos not in gold_collected:
            path = a_star_search(grid, current_position, gold_pos, agility_skill)
            if not shortest_path or (path and len(path) < len(shortest_path)):
                shortest_path = path
                nearest_gold = gold_pos

    # Follow the shortest path to the nearest gold
    if shortest_path:
        next_position = shortest_path[0]
        for action in ACTIONS[:-1]:  # Exclude "EXIT" action
            if next_position in get_possible_next_positions(current_position, action, grid):
                return action

    # Compute the optimal policy using Policy Iteration
    policy = policy_iteration(grid, gold_locations, start_pos)

    # Follow the policy and choose the next action
    state = (current_position, frozenset(gold_collected))
    action = policy.get(state, "NORTH")  # Default to NORTH if no policy found

    # Prioritize collecting gold, even if it requires crossing a bridge
    if action == "EXIT" and not gold_collected:
        for gold_pos in gold_locations:
            if gold_pos not in gold_collected:
                action = "NORTH"  # Change action to move towards gold
                break

    return action  # Return plain string for the chosen action


"""Main function."""
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
        parallel_runs=True,
        run_limit=100000000  # Stop after 1000 runs
    )