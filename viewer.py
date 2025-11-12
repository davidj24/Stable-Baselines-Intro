import gymnasium as gym
from stable_baselines3 import PPO
import json
import time
import tyro


@dataclass
class Args:
    model_path: str

    

def main():
    print(f"Loading model from: {args.model_path}")
    SPEED_MULTIPLIER = 1.0

    model = PPO.load(args.model_path)
    env_id = model.env.unwrapped.spec.id



    print("Successfully loaded model. Creating environment: {env_id}")
    env = gym.make(env_id, render_mode='human')

    for episode in range(5):
        print(f"Starting evaluation episode {episode + 1}/5")
        obs, info = env.reset()
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Episode completed. Resetting")
                time.sleep(1)
                break


    print("Visual Inference complete.")
    env.close()



if __name__ == '__main__':
    args = tyro.cli(Args)
    main()