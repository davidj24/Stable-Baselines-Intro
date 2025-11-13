import gymnasium as gym
from stable_baselines3 import PPO
import wandb
import os
import subprocess
import sys
import tyro
from dataclasses import dataclass
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


@dataclass
class Args:
    name_prefix: str = "m"
    viewer: bool = True
    wandb_log: bool = True

    # Training params
    policy_type: str = "MlpPolicy"
    env_id: str = "LunarLander-v3"
    total_timesteps: int = 200000
    learning_rate: float = 0.0003
    n_steps: int = 2048

    # Checkpoint params
    save_freq: int = 10000
    save_path: str = "./models/"

    wandb_project: str = "sb3-lunar-lander"


class LiveViewerCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose=0):
        super(LiveViewerCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.viewer_process = None
        self.log_file = "viewer.log"

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps.zip")
            self.model.save(path)
            print(f"Model saved to {path}")

            if self.viewer_process and self.viewer_process.poll() is None:
                print(f"Terminating old viewer")
                self.viewer_process.terminate()
                self.viewer_process.wait()

            command = [
                sys.executable,
                "viewer.py", "--model-path", path,
                "--num-episodes", "3"
            ]


            try:
                print(f"Launching new viewer using command: {' '.join(command)}. Output in {self.log_file}")
                self.logs = open(self.log_file, "w")
                self.viewer_process = subprocess.Popen(
                    command,
                    stdout=self.logs,
                    stderr=subprocess.STDOUT
                )
            except Exception as e:
                print(f"Failed to launch viewer. Full error: {e}")
                self.viewer_process = None
                
        return True
    


    def _on_training_end(self):
        if self.viewer_process and self.viewer_process.poll() is None:
            print("Training has ended. Closing viewer.")
            self.viewer_process.terminate()
            self.viewer_process.wait()
        if hasattr(self, 'logs'):
            self.logs.close()




def main(args: Args):
    if args.wandb_log:
        run = wandb.init(
            project="sb3-lunar-lander",
            config=vars(args),
            sync_tensorboard=True,
        )

    print("Creating env")
    env = gym.make(args.env_id)
    os.makedirs(args.save_path, exist_ok=True)


    print("Creating model")
    if args.wandb_log:
        model = PPO(
            args.policy_type,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            verbose=1,
            tensorboard_log=f"wandb_logs/{run.id}"
        )
    else:
        model = PPO(
            args.policy_type,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            verbose=1,
        )

    
    callbacks = []

    if args.wandb_log:
        callbacks.append(WandbCallback())

    if args.viewer:
        print(f">>>>>>> Live viewer ENABLED <<<<<<<")
        callbacks.append(
            LiveViewerCallback(
                save_freq=args.save_freq,
                save_path=args.save_path,
                name_prefix=args.name_prefix,
                verbose=0
            )
        )

    else:
        print(f">>>>>>> Live viewer DISABLED <<<<<<<")
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=args.save_path,
                name_prefix=args.name_prefix,
                verbose=0
            )
        )





    print(f"Started training for {args.total_timesteps} timesteps")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks
    )
    print("Training complete.")

    if args.wandb_log:
        run.finish()
    env.close()

    print("Run completed and env closed.")



if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)