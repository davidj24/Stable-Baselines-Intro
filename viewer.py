import gymnasium as gym
import json
import time

REPLAY_FILE = "replay.json"
SPEED_MULTIPLIER = 1.0

print(f"Loadng replay from {REPLAY_FILE}")

try:
    with open(REPLAY_FILE, 'r') as f:
        replay_data = json.load(f)

except FileNotFoundError:
    print(f"Error: Couldn't find file {REPLAY_FILE}")
    print("Run train.py first to generate it")
    exit()


seed = replay_data['seed']
actions = replay_data['actions']
env_id = replay_data['env_id']

print(f"Replaying episode from environment {env_id} with seed {seed}")

env = gym.make(env_id, render_mode='human')

obs, info = env.reset(seed=seed)

for i, action in enumerate(actions):
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01 / SPEED_MULTIPLIER)

    if terminated or truncated:
        break

print("Replay complete.")
env.close()