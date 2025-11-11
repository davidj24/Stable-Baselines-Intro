import gymnasium as gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "env_id": "LunarLander-v2",
    "total_timesteps": 100000,
    "learning_rate": 0.0003,
    "n_steps": 2048,
}


run = wandb.init(
    project="sb3-lunar-lander",
    config=config,
    sync_tensorboard=True,
)

print("Creating env")
env = gym.make(config["env_id"])

print("Creating model")
model = PPO(
    config["policy_type"],
    env,
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    verbose=1,
    tensorboard_log=f"wandb_logs/{run.id}"
)



callback = WandbCallback()

print(f"Started training for {config['total_timesteps']} timesteps")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callback
)
print("Training complete.")

run.finish()
env.close()

print("Run completed and env closed.")