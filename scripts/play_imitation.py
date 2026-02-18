import argparse
import sys
import os
import imageio

# ==========================================
# SETUP & IMPORTS
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate robomimic policy.")
parser.add_argument(
    "--task", type=str, default="Isaac-Stretch-Cabinet-v0", help="Name of the task."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pth checkpoint."
)
parser.add_argument("--horizon", type=int, default=500, help="Horizon.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from isaaclab_tasks.utils import parse_env_cfg

# ==========================================
# 2. REGISTER CUSTOM ENVIRONMENT
# ==========================================
task_config_path = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if task_config_path not in sys.path:
    sys.path.append(task_config_path)

from stretch_env_cfg import StretchEnvCfg

if "Isaac-Stretch-Cabinet-v0" not in gym.envs.registry:
    gym.register(
        id="Isaac-Stretch-Cabinet-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": StretchEnvCfg},
    )


# ==========================================
# 3. ROLLOUT LOGIC
# ==========================================
def rollout(policy, env, horizon, device, video_path):
    """Play one episode and save video."""
    print(f"[INFO] Recording video to: {video_path}")

    policy.start_episode()
    obs_dict, _ = env.reset()

    frames = []

    for i in range(horizon):
        if i % 2 == 0:
            frame = env.render()
            if frame is not None:
                if isinstance(frame, list):
                    frame = frame[0]
                if isinstance(frame, np.ndarray) and frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(frame)

        obs_tensor = obs_dict["policy"]
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.squeeze(0)
        robomimic_obs = {"policy": obs_tensor}
        actions = policy(robomimic_obs)
        actions = torch.from_numpy(actions).to(device=device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if i % 50 == 0:
            print(f"Step {i}/{horizon}")

        if terminated or truncated:
            break

    if len(frames) > 0:
        print(f"[INFO] Saving {len(frames)} frames...")
        imageio.mimsave(video_path, frames, fps=60)
        print("✅ Video saved successfully!")
    else:
        print("❌ No frames captured (Check if cameras are enabled).")


def main():
    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "play_videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "rollout.mp4")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint, device=device, verbose=True
    )
    with torch.inference_mode():
        rollout(policy, env, args_cli.horizon, device, video_path)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
