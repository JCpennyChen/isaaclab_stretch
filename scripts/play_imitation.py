"""
Play a trained imitation learning policy in Isaac Lab and record video.
"""

import argparse
import sys

# 1. Setup Argument Parser FIRST
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play imitation policy")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pth model file"
)

# Add AppLauncher args (allows us to set camera/headless settings)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

# 2. FORCE CAMERA AND HEADLESS SETTINGS
# This fixes the "NO_GUI_OR_RENDERING" error!
args.enable_cameras = True
args.headless = True

# 3. Launch the App
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("[Script] App launched with cameras enabled. Importing libraries...")

# 4. Imports that require Isaac Sim to be running
import gymnasium as gym
import torch
import numpy as np
import os
import imageio
import stretch.tasks  # Registers Template-Stretch-v0
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def main():
    # Load trained policy
    print(f"[Script] Loading checkpoint: {args.checkpoint}")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=True
    )
    policy.start_episode()

    # Create Environment
    print(f"[Script] Creating environment: {args.task}")

    # Manually instantiate config to satisfy ManagerBasedRLEnv
    spec = gym.spec(args.task)
    if "env_cfg_entry_point" in spec.kwargs:
        cfg_cls = spec.kwargs["env_cfg_entry_point"]
        cfg = cfg_cls()
        # Create env with the config
        env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    else:
        env = gym.make(args.task, render_mode="rgb_array")

    # Setup Video Path
    video_dir = os.path.join(os.path.dirname(args.checkpoint), "play_videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "rollout.mp4")
    print(f"[Script] Video will be saved to: {video_path}")

    frames = []

    # Simulation Loop
    print("[Script] Resetting environment...")
    obs, _ = env.reset()

    print("[Script] Starting simulation loop (Limit: 400 steps)...")
    try:
        for step in range(400):
            # Capture frame
            if step % 2 == 0:
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    if frame.shape[-1] == 4:  # RGBA -> RGB
                        frame = frame[..., :3]
                    frames.append(frame)

            # Prepare policy input
            # FIX 1: Ensure observation has batch dimension [1, 32]
            if isinstance(obs["policy"], torch.Tensor):
                policy_obs = obs["policy"].to(device)
            else:
                policy_obs = torch.tensor(
                    obs["policy"], device=device, dtype=torch.float32
                )

            # If 1D [32], make it 2D [1, 32]
            if policy_obs.dim() == 1:
                policy_obs = policy_obs.unsqueeze(0)

            obs_dict = {"policy": policy_obs}

            # Get action
            action = policy(obs_dict)
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(device)

            # FIX 2: Ensure action has batch dimension [1, 12]
            # Isaac Lab crashes if this is 1D!
            if action.dim() == 1:
                action = action.unsqueeze(0)

            # Step environment
            obs, _, _, _, _ = env.step(action)

            if step % 50 == 0:
                print(f"Step {step}/400")

    except Exception as e:
        print(f"[Error] Simulation failed: {e}")
        import traceback

        traceback.print_exc()

    # Save Video
    print(f"[Script] Saving video with {len(frames)} frames...")
    if len(frames) > 0:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"✅ Video saved: {video_path}")
    else:
        print("❌ No frames captured. Check env.render()")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
