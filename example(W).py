import random
import logging
from itertools import chain, combinations

# Constants
GAMMA = 0.99  # Increased discount factor to prioritize future rewards
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT", "FIGHT"]  # Added FIGHT action

# Helper functions remain the same until get_reward
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def parse_map(raw_map):
    grid = [list(row) for row in raw_map.split('\n') if row.strip()]
    gold_locations = []
    start_pos = None
    wumpus_locations = []  # Track Wumpus locations
    for row_idx, line in enumerate(grid):
        for col_idx, cell in enumerate(line):
            if cell == 'G':
                gold_locations.append((col_idx, row_idx))
            elif cell == 'S':
                start_pos = (col_idx, row_idx)
            elif cell == 'W':
                wumpus_locations.append((col_idx, row_idx))
    return grid, gold_locations, start_pos, wumpus_locations

def get_walkable_positions(grid):
    walkable_positions = []
    for row_idx in range(len(grid)):
        for col_idx in range(len(grid[0])):
            if grid[row_idx][col_idx] != 'X':
                walkable_positions.append((col_idx, row_idx))
    return walkable_positions

def is_position_walkable(position, grid):
    col, row = position
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != 'X':
        return True
    return False

def get_reward(position, action, next_position, gold_collected, gold_locations, start_pos, killed_wumpus):
    reward = -0.1  # Base step penalty
    
    # Penalty for hitting a wall
    if next_position == position and action != "FIGHT":
        reward -= 0.5

    # Reward for collecting gold
    if next_position in gold_locations and next_position not in gold_collected:
        reward += 10

    # Reward for killing Wumpus
    if action == "FIGHT" and next_position in killed_wumpus:
        reward += 15  # Significant reward for eliminating a threat

    # Reward for exiting with gold
    if action == "EXIT" and next_position == start_pos:
        total_gold = len(gold_collected)
        exit_reward = total_gold * 10
        if total_gold == len(gold_locations):
            exit_reward += 100
        reward += exit_reward

    return reward

def get_possible_next_positions(position, action, grid, killed_wumpus):
    if action == "EXIT":
        if grid[position[1]][position[0]] == 'S':
            return {position}
        return set()
    
    if action == "FIGHT":
        if grid[position[1]][position[0]] == 'W' and position not in killed_wumpus:
            return {position}  # Stay in place when fighting
        return set()

    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    
    if action in directions:
        dc, dr = directions[action]
        new_col = position[0] + dc
        new_row = position[1] + dr
        new_pos = (new_col, new_row)
        
        # Check if moving toward Wumpus (only allow if Wumpus is killed)
        if grid[new_row][new_col] == 'W' and (new_col, new_row) not in killed_wumpus:
            return {position}  # Can't move onto live Wumpus
            
        if is_position_walkable(new_pos, grid):
            # Calculate probabilities for movement uncertainty
            possible_positions = set()
            possible_positions.add(new_pos)  # Main direction
            
            # Check left movement possibility (10% chance)
            left_directions = {"NORTH": "WEST", "WEST": "SOUTH", "SOUTH": "EAST", "EAST": "NORTH"}
            if action in left_directions:
                left_dc, left_dr = directions[left_directions[action]]
                left_pos = (position[0] + left_dc, position[1] + left_dr)
                if is_position_walkable(left_pos, grid):
                    possible_positions.add(left_pos)
            
            # Check right movement possibility (10% chance)
            right_directions = {"NORTH": "EAST", "EAST": "SOUTH", "SOUTH": "WEST", "WEST": "NORTH"}
            if action in right_directions:
                right_dc, right_dr = directions[right_directions[action]]
                right_pos = (position[0] + right_dc, position[1] + right_dr)
                if is_position_walkable(right_pos, grid):
                    possible_positions.add(right_pos)
            
            return possible_positions
    
    return {position}

def get_transition_prob(position, action, next_position, grid, killed_wumpus):
    if action == "EXIT":
        if grid[position[1]][position[0]] == 'S' and next_position == position:
            return 1.0
        return 0.0
    
    if action == "FIGHT":
        if grid[position[1]][position[0]] == 'W' and position not in killed_wumpus and next_position == position:
            return 1.0
        return 0.0

    possible_positions = get_possible_next_positions(position, action, grid, killed_wumpus)
    if next_position not in possible_positions:
        return 0.0
    
    # Calculate transition probabilities based on movement uncertainty
    if len(possible_positions) == 1:
        return 1.0 if next_position in possible_positions else 0.0
    elif len(possible_positions) == 2:
        return 0.8 if next_position == list(possible_positions)[0] else 0.2
    else:  # 3 possible positions
        return 0.8 if next_position == list(possible_positions)[0] else 0.1

def policy_iteration(grid, gold_locations, start_pos, killed_wumpus):
    walkable_positions = get_walkable_positions(grid)
    states = [(pos, frozenset(gold_collected), frozenset(killed_wumpus)) 
              for pos in walkable_positions 
              for gold_collected in powerset(gold_locations)
              for killed_wumpus in powerset([(x, y) for x, y in walkable_positions if grid[y][x] == 'W'])]
    
    policy = {state: random.choice(ACTIONS) for state in states}
    V = {state: 0 for state in states}

    while True:
        # Policy Evaluation
        for _ in range(1000):
            delta = 0
            for state in states:
                position, gold_collected, current_killed_wumpus = state
                v = V[state]
                action = policy[state]
                total = 0
                
                for next_position in get_possible_next_positions(position, action, grid, current_killed_wumpus):
                    next_gold_collected = set(gold_collected)
                    next_killed_wumpus = set(current_killed_wumpus)
                    
                    # Update gold collection
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)
                    
                    # Update killed Wumpus
                    if action == "FIGHT" and grid[position[1]][position[0]] == 'W':
                        next_killed_wumpus.add(position)
                    
                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos, current_killed_wumpus)
                    prob = get_transition_prob(position, action, next_position, grid, current_killed_wumpus)
                    next_state = (next_position, frozenset(next_gold_collected), frozenset(next_killed_wumpus))
                    total += prob * (reward + GAMMA * V[next_state])
                
                V[state] = total
                delta = max(delta, abs(v - V[state]))
            
            if delta < EPSILON:
                break

        # Policy Improvement
        policy_stable = True
        for state in states:
            position, gold_collected, current_killed_wumpus = state
            old_action = policy[state]
            best_action = None
            best_value = float('-inf')
            
            for action in ACTIONS:
                total = 0
                for next_position in get_possible_next_positions(position, action, grid, current_killed_wumpus):
                    next_gold_collected = set(gold_collected)
                    next_killed_wumpus = set(current_killed_wumpus)
                    
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)
                    
                    if action == "FIGHT" and grid[position[1]][position[0]] == 'W':
                        next_killed_wumpus.add(position)
                    
                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos, current_killed_wumpus)
                    prob = get_transition_prob(position, action, next_position, grid, current_killed_wumpus)
                    next_state = (next_position, frozenset(next_gold_collected), frozenset(next_killed_wumpus))
                    total += prob * (reward + GAMMA * V[next_state])
                
                if total > best_value:
                    best_value = total
                    best_action = action
            
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return policy

def simulate_fight(fighting_skill):
    dice_rolls = [random.randint(1, 6) for _ in range(fighting_skill)]
    dice_rolls.sort(reverse=True)
    score = sum(dice_rolls[:3])
    return score >= 13  # Need at least 13 to defeat Wumpus

def cross_bridge(agility_skill):
    dice_rolls = [random.randint(1, 6) for _ in range(agility_skill)]
    dice_rolls.sort(reverse=True)
    score = sum(dice_rolls[:3])
    return score >= 12

def print_grid(grid, agent_position):
    grid_copy = [row[:] for row in grid]
    col, row = agent_position
    grid_copy[row][col] = 'A'
    for row in grid_copy:
        print(''.join(row))
    print()

# ...existing code...

def agent_function(request_data, request_info):
    print('_________________________________________________________')
    print("Received request:", request_data)
    
    # Parse game state
    game_map = request_data.get('map', '')
    grid, gold_locations, start_pos, wumpus_locations = parse_map(game_map)
    free_skill_points = request_data.get("free-skill-points", 0)
    history = request_data.get("history", [])

    # Count obstacles
    num_wumpus = sum(row.count('W') for row in grid)
    num_bridges = sum(row.count('B') for row in grid)

    # Allocate skill points
    if free_skill_points > 0:
        if num_wumpus == 0 and num_bridges == 0:
            # Default allocation if no obstacles
            return {"agility": free_skill_points // 2, "fighting": free_skill_points // 2}
        else:
            # Calculate weights
            total_obstacles = num_wumpus + num_bridges
            weight_fighting = num_wumpus / total_obstacles
            weight_navigation = num_bridges / total_obstacles

            # Allocate points
            fighting_points = round(free_skill_points * weight_fighting)
            navigation_points = free_skill_points - fighting_points

            return {"agility": navigation_points, "fighting": fighting_points}

    # Extract current state from history
    current_position = start_pos
    gold_collected = set()
    killed_wumpus = set()

    if history:
        for event in history:
            outcome = event.get('outcome', {})
            if 'position' in outcome:
                current_position = tuple(outcome['position'])
            if 'collected-gold-at' in outcome:
                gold_pos = tuple(outcome['collected-gold-at'])
                gold_collected.add(gold_pos)
            if 'killed-wumpus-at' in outcome:
                wumpus_pos = tuple(outcome['killed-wumpus-at'])
                killed_wumpus.add(wumpus_pos)

    print_grid(grid, current_position)  # Print the grid with the agent's position

    # Print the amount of collected gold
    print(f"COLLECTED GOLD: {len(gold_collected)}")

    # Check if we're at a Wumpus
    if grid[current_position[1]][current_position[0]] == 'W' and current_position not in killed_wumpus:
        print(f"Agent encountered a Wumpus at {current_position}. Initiating fight.")
        return "FIGHT"

    # Check for bridge crossing
    if grid[current_position[1]][current_position[0]] == 'B':
        agility_skill = request_data.get("skill-points", {}).get("agility", 0)
        if not cross_bridge(agility_skill):
            print("Agent failed to cross the bridge.")

    # Check if we should exit
    if grid[current_position[1]][current_position[0]] == 'S' and gold_collected:
        return "EXIT"

    # Get optimal policy
    policy = policy_iteration(grid, gold_locations, start_pos, killed_wumpus)
    state = (current_position, frozenset(gold_collected), frozenset(killed_wumpus))
    action = policy.get(state, "NORTH")

    # Override EXIT unless all gold is collected
    if action == "EXIT" and len(gold_collected) < len(gold_locations):
        uncollected = [g for g in gold_locations if g not in gold_collected]
        if uncollected:
            nearest = min(uncollected, key=lambda g: abs(g[0]-current_position[0]) + abs(g[1]-current_position[1]))
            dx = nearest[0] - current_position[0]
            dy = nearest[1] - current_position[1]
            if abs(dx) > abs(dy):
                action = "EAST" if dx > 0 else "WEST"
            else:
                action = "SOUTH" if dy > 0 else "NORTH"

    print(f"Agent decided to {action} from position {current_position}.")
    return action

# ...existing code...

if __name__ == '__main__':
    import sys
    import logging
    from client import run

    logging.basicConfig(level=logging.INFO)

    # Run the agent
    run(
        agent_config_file=sys.argv[1],
        agent=agent_function,
        parallel_runs=True,
        run_limit=100000000  # Stop after 1000 runs
    )