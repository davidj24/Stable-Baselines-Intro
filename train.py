import gymnasium as gym
from stable_baselines3 import PPO
import wandb
import os
import subprocess
import sys
import tyro
from dataclasses import dataclass
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback


@dataclass
class Args:
    name_prefix: str = ""
    viewer: bool = True

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


class LiveViewerCallback(CheckpointCallback):
    def __init__(self, *args, **kwargs):
        super(LiveViewerCallback, self).__init__(*args, **kwargs)
        self.viewer_process = None
        self.newest_path = ""

    def _on_step(self) -> bool:
        continue_training = super(LiveViewerCallback, self)._on_step()
        if hasattr(self, 'last_save_path') and self.last_save_path != self.newest_path:
            print(f"Parent has saved new model to {self.last_save_path}")
            self.newest_path = self.last_save_path

            if self.viewer_process and self.viewer_process.poll() is None:
                print(f"Terminating old viewer")
                self.viewer_process.terminate()
                self.viewer_process.wait()

            command = [
                sys.executable,
                "viewer.py",
                "--model-path",
                self.last_save_path
            ]

            print(f"Launching new viewer using command: {' '.join(command)}")
            self.viewer_process = subprocess.Popen(command)
        return continue_training
    
    def _on_training_end(self):
        super(LiveViewerCallback, self)._on_training_end()
        if self.viewer_process and self.viewer_process.poll() is None:
            print("Training has ended. Closing viewer.")
            self.viewer_process.terminate()
            self.viewer_process.wait()




def main(args: Args):
    run = wandb.init(
        project="sb3-lunar-lander",
        config=vars(args),
        sync_tensorboard=True,
    )

    print("Creating env")
    env = gym.make(args.env_id)
    os.makedirs(args.save_path, exist_ok=True)


    print("Creating model")
    model = PPO(
        args.policy_type,
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        verbose=1,
        tensorboard_log=f"wandb_logs/{run.id}"
    )
    
    callbacks = [WandbCallback()]

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

    run.finish()
    env.close()

    print("Run completed and env closed.")



if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)