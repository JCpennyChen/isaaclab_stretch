"""
Play a trained imitation learning policy in Isaac Lab and record video.
"""

import argparse
import sys
import os

# 1. Setup Argument Parser FIRST
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play imitation policy")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the .pth model file"
)

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 2. FORCE CAMERA AND HEADLESS SETTINGS
args.enable_cameras = True
args.headless = True

# 3. Launch the App
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("[Script] App launched with cameras enabled. Importing libraries...")

# Add path to config if needed
task_config_path = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if task_config_path not in sys.path:
    sys.path.append(task_config_path)

# 4. Imports that require Isaac Sim to be running
import gymnasium as gym
import torch
import numpy as np
import imageio
import stretch.tasks  # Registers Template-Stretch-v0
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from isaaclab.actuators import ImplicitActuatorCfg

# [FIX] Import Isaac Lab modules to modify config
from isaaclab.envs import mdp as isaac_mdp


def main():
    # Load trained policy
    print(f"[Script] Loading checkpoint: {args.checkpoint}")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=device, verbose=True
    )
    policy.start_episode()

    spec = gym.spec(args.task)
    if "env_cfg_entry_point" in spec.kwargs:
        cfg_cls = spec.kwargs["env_cfg_entry_point"]
        cfg = cfg_cls()

        print("[Fix] Overriding environment actions AND ACTUATORS...")

        # 1. Clear default actions
        cfg.actions.base = None
        cfg.actions.lift_velocity = None
        cfg.actions.arm_velocity = None
        cfg.actions.wrist_velocity = None
        cfg.actions.gripper = None

        # 2. Add the "God Mode" direct joint control (16 dim)
        cfg.actions.joint_pos_direct = isaac_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=1.0,
            use_default_offset=False,
        )

        # 3. [CRITICAL FIX] Force High Stiffness (Match curobo_record.py)
        # Without this, the arm is too weak to pull the drawer open.
        strong_drive = ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=4000.0,  # High stiffness
            damping=100.0,
            effort_limit=20000.0,
            velocity_limit=100.0,
        )
        # Overwrite the robot's default actuator settings
        cfg.scene.robot.actuators = {"god_mode_drive": strong_drive}

        env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")

    # Setup Video Path
    video_dir = os.path.join(os.path.dirname(args.checkpoint), "play_videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "rollout.mp4")
    print(f"[Script] Video will be saved to: {video_path}")

    frames = []

    print("[Script] Resetting environment...")
    obs, _ = env.reset()

    print("[Script] Starting simulation loop (Limit: 400 steps)...")
    try:
        for step in range(400):
            if step % 2 == 0:
                frame = env.render()
                if frame is not None and isinstance(frame, np.ndarray):
                    if frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    frames.append(frame)

            # Process Observation
            if isinstance(obs, dict) and "policy" in obs:
                current_obs = obs["policy"]
            else:
                current_obs = obs

            if not isinstance(current_obs, torch.Tensor):
                current_obs = torch.tensor(
                    current_obs, device=device, dtype=torch.float32
                )
            else:
                current_obs = current_obs.to(device)

            if current_obs.dim() == 1:
                current_obs = current_obs.unsqueeze(0)

            # Get Action
            action = policy({"policy": current_obs})

            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(device)

            if action.dim() == 1:
                action = action.unsqueeze(0)

            # Step Environment
            obs, _, _, _, _ = env.step(action)

            if step % 50 == 0:
                print(f"Step {step}/400")

    except Exception as e:
        print(f"[Error] Simulation failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"[Script] Saving video with {len(frames)} frames...")
    if len(frames) > 0:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"✅ Video saved: {video_path}")
    else:
        print("❌ No frames captured.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
