import numpy as np

obstacles = [(1,1)]
actions = ['L', 'R', 'U', 'D']
final_states = ((1,3),(2,3))

action_probabilities = {'L': 0.25, 'R': 0.25, 'U': 0.25, 'D': 0.25}
left_env = {'L': 'D', 'R': 'U', 'U': 'L', 'D': 'R'}
right_env = {'L': 'U', 'R': 'D', 'U': 'R', 'D': 'L'}

def get_new_position(action, x, y):
    if action == 'L':
        return x, y-1
    elif action == 'R':
        return x, y+1
    elif action == 'U':
        return x+1, y
    else:
        return x-1, y

def is_within_bounds(x, y):
    return (x, y) not in obstacles and 0 <= x < 3 and 0 <= y < 4

def print_result(values):
    for row in range(2, -1, -1):
        print("--- --- --- --- --- --- ---")
        for col in range(4):
            val = values[row][col]
            print(f" {val:.2f}|" if val >= 0 else f"{val:.2f}|", end="")
        print("")

def compute_state_value(x, y, reward, rewards, tolerance=1):
    total_value = 0
    for action in actions:
        new_x, new_y = get_new_position(action, x, y)
        if is_within_bounds(new_x, new_y):
            value = rewards[new_x][new_y] + tolerance * current_values[new_x][new_y]
        else:
            value = rewards[x][y] + tolerance * current_values[x][y]

        left_x, left_y = get_new_position(left_env[action], x, y)
        if is_within_bounds(left_x, left_y):
            left_value = rewards[left_x][left_y] + tolerance * current_values[left_x][left_y]
        else:
            left_value = rewards[x][y] + tolerance * current_values[x][y]

        right_x, right_y = get_new_position(right_env[action], x, y)
        if is_within_bounds(right_x, right_y):
            right_value = rewards[right_x][right_y] + tolerance * current_values[right_x][right_y]
        else:
            right_value = rewards[x][y] + tolerance * current_values[x][y]

        action_value = 0.8 * value + 0.1 * left_value + 0.1 * right_value
        total_value += action_probabilities[action] * action_value

    return total_value

def policy_evaluation(iter_count, threshold, reward, rewards, values):
    while True:
        max_diff = 0
        for row in range(3):
            for col in range(4):
                if (row, col) in final_states or (row, col) in obstacles:
                    continue
                old_value = values[row][col]
                values[row][col] = compute_state_value(row, col, reward, rewards)
                max_diff = max(max_diff, abs(old_value - values[row][col]))

        iter_count += 1
        if max_diff < threshold:
            print(f"Iterations: {iter_count}")
            break

    print_result(values)

def initialize_rewards(reward_value):
    grid = [[reward_value for _ in range(4)] for _ in range(3)]
    grid[2][3] = 1
    grid[1][3] = -1
    return grid

def setup_values():
    return [[0 for _ in range(4)] for _ in range(3)]

reward_list = [-0.04, -2, 0.1, 0.02, 1]

if __name__ == "__main__":
    for r in reward_list:
        print(f"Reward at S: {r}")
        reward_grid = initialize_rewards(r)
        current_values = setup_values()
        policy_evaluation(0, 1e-7, r, reward_grid, current_values)
        print("\n************************************\n")
