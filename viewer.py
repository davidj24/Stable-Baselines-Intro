import gymnasium as gym
from stable_baselines3 import PPO
import json
import time
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    model_path: str
    num_episodes: int = 5



def main(args: Args):
    print(f"Loading model from: {args.model_path}")
    SPEED_MULTIPLIER = 1.0

    try:
        model = PPO.load(args.model_path)
        env_id = "LunarLander-v3"
    except Exception as e:
        print(f"Error: Couldn't load model from {args.model_path}")
        print(f"Full error is: {e}")
        exit()



    print(f"Successfully loaded model. Creating environment: {env_id}")
    env = gym.make(env_id, render_mode='human')

    for episode in range(args.num_episodes):
        print(f"Starting evaluation episode {episode + 1}/{args.num_episodes}")
        obs, info = env.reset()
        
        while True:
            action_arr, _states = model.predict(obs, deterministic=True)
            action = int(action_arr)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Episode completed. Resetting")
                time.sleep(1)
                break


    print("Visual Inference complete.")
    env.close()



if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)