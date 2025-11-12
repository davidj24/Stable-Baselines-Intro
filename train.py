import gymnasium as gym
from stable_baselines3 import PPO
import wandb
import os
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback








config = {
    "policy_type": "MlpPolicy",
    "env_id": "LunarLander-v3",
    "total_timesteps": 100000,
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "save_freq": 10000,
    "save_path": "./models/"
}


run = wandb.init(
    project="sb3-lunar-lander",
    config=config,
    sync_tensorboard=True,
)

print("Creating env")
env = gym.make(config["env_id"])
os.makedirs(config["save_path"], exist_ok=True)


print("Creating model")
model = PPO(
    config["policy_type"],
    env,
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    verbose=1,
    tensorboard_log=f"wandb_logs/{run.id}"
)



callbacks = [
    WandbCallback(),
    CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["save_path"],
        name_prefix="PPO"
    )
]

print(f"Started training for {config['total_timesteps']} timesteps")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callbacks
)
print("Training complete.")

run.finish()
env.close()

print("Run completed and env closed.")