import argparse
import sys
import os
import imageio

# ==========================================
# SETUP & IMPORTS
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate two-phase robomimic policy (reach + pull)."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Stretch-Cabinet-v0", help="Name of the task."
)
parser.add_argument(
    "--reach_checkpoint",
    type=str,
    required=True,
    help="Path to the reach phase .pth checkpoint.",
)
parser.add_argument(
    "--pull_checkpoint",
    type=str,
    required=True,
    help="Path to the pull phase .pth checkpoint.",
)
parser.add_argument("--horizon", type=int, default=500, help="Max steps per rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--switch_distance",
    type=float,
    default=0.05,
    help="EEF-to-handle distance threshold to trigger phase switch (meters).",
)
parser.add_argument(
    "--gripper_lock_steps",
    type=int,
    default=30,
    help="Steps to hold position while gripper closes before switching to pull.",
)
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
# REGISTER CUSTOM ENVIRONMENT
# ==========================================
task_config_path = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if task_config_path not in sys.path:
    sys.path.append(task_config_path)

from stretch_bc_rnn_cfg import StretchEnvCfg

if "Isaac-Stretch-Cabinet-v0" not in gym.envs.registry:
    gym.register(
        id="Isaac-Stretch-Cabinet-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": StretchEnvCfg},
    )


# ==========================================
# PHASE SWITCHING HELPERS
# ==========================================
GRIPPER_CLOSE_POS = -0.2


def get_eef_handle_distance(scene):
    """Compute distance between end-effector and drawer handle from the scene."""
    robot = scene["robot"]
    cabinet = scene["cabinet"]

    eef_idx = robot.find_bodies("link_grasp_center")[0][0]
    handle_idx = cabinet.find_bodies("drawer_handle_top")[0][0]

    eef_pos = robot.data.body_pos_w[0, eef_idx]
    handle_pos = cabinet.data.body_pos_w[0, handle_idx]

    return torch.norm(handle_pos - eef_pos).item()


def get_gripper_joint_indices(env):
    """Find the action-space indices corresponding to gripper joints."""
    unwrapped = env.unwrapped
    action_manager = unwrapped.action_manager

    gripper_indices = []
    offset = 0
    for name, term in action_manager._terms.items():
        dim = term.action_dim
        if "gripper" in name:
            gripper_indices.extend(range(offset, offset + dim))
        offset += dim
    return gripper_indices


# ==========================================
# ROLLOUT LOGIC
# ==========================================
def rollout(reach_policy, pull_policy, env, horizon, device, video_path, args):
    """Play one episode using reach→pull skill chain and save video."""
    print(f"[INFO] Recording video to: {video_path}")
    print(f"[INFO] Switch distance: {args.switch_distance}m")
    print(f"[INFO] Gripper lock steps: {args.gripper_lock_steps}")

    scene = env.unwrapped.scene
    gripper_indices = get_gripper_joint_indices(env)
    action_dim = env.unwrapped.action_manager.total_action_dim  # <-- ADD THIS
    print(f"[INFO] Environment action dim: {action_dim}")

    # Phase tracking
    current_phase = "reach"
    gripper_lock_timer = 0
    locking_gripper = False  # True during the transition hold period
    last_reach_actions = None

    reach_policy.start_episode()
    pull_policy.start_episode()
    obs_dict, _ = env.reset()

    frames = []

    for i in range(horizon):
        # --- Capture video frames ---
        if i % 2 == 0:
            frame = env.render()
            if frame is not None:
                if isinstance(frame, list):
                    frame = frame[0]
                if isinstance(frame, np.ndarray) and frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(frame)

        # --- Prepare observation for robomimic ---
        obs_tensor = obs_dict["policy"]
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.squeeze(0)
        robomimic_obs = {"policy": obs_tensor}

        # --- Phase switching logic ---
        if current_phase == "reach" and not locking_gripper:
            # Run reach policy
            actions = reach_policy(robomimic_obs)
            actions = torch.from_numpy(actions).to(device=device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            actions = actions[:, :action_dim]

            # Check if EEF is close enough to handle to start gripper lock
            distance = get_eef_handle_distance(scene)
            if distance < args.switch_distance:
                print(
                    f"[Step {i}] EEF within {distance:.3f}m of handle — locking gripper..."
                )
                locking_gripper = True
                gripper_lock_timer = 0
                last_reach_actions = actions.clone()

        if locking_gripper:
            # Hold the last reach position but force gripper closed
            actions = last_reach_actions.clone()
            for idx in gripper_indices:
                actions[0, idx] = GRIPPER_CLOSE_POS
            gripper_lock_timer += 1

            if gripper_lock_timer >= args.gripper_lock_steps:
                print(f"[Step {i}] Gripper locked! Switching to PULL policy.")
                current_phase = "pull"
                locking_gripper = False

        elif current_phase == "pull":
            # Run pull policy
            actions = pull_policy(robomimic_obs)
            actions = torch.from_numpy(actions).to(device=device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            actions = actions[:, :action_dim]

            # Keep gripper closed during pull
            for idx in gripper_indices:
                actions[0, idx] = GRIPPER_CLOSE_POS

        # --- Step environment ---
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if i % 50 == 0:
            dist = get_eef_handle_distance(scene)
            print(
                f"Step {i}/{horizon} | Phase: {current_phase} | EEF-Handle dist: {dist:.3f}m"
            )

        if terminated or truncated:
            break

    # --- Save video ---
    if len(frames) > 0:
        print(f"[INFO] Saving {len(frames)} frames...")
        imageio.mimsave(video_path, frames, fps=60)
        print("Video saved successfully!")
    else:
        print("No frames captured (check if cameras are enabled).")


def main():
    video_dir = os.path.join(os.path.dirname(args_cli.reach_checkpoint), "play_videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "rollout_two_phase.mp4")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load both policies
    print("[INFO] Loading REACH policy...")
    reach_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.reach_checkpoint, device=device, verbose=True
    )
    print("[INFO] Loading PULL policy...")
    pull_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.pull_checkpoint, device=device, verbose=True
    )

    with torch.inference_mode():
        for rollout_idx in range(args_cli.num_rollouts):
            print(f"\n{'='*50}")
            print(f"Rollout {rollout_idx + 1}/{args_cli.num_rollouts}")
            print(f"{'='*50}")

            vid_path = (
                video_path.replace(".mp4", f"_{rollout_idx}.mp4")
                if args_cli.num_rollouts > 1
                else video_path
            )
            rollout(
                reach_policy,
                pull_policy,
                env,
                args_cli.horizon,
                device,
                vid_path,
                args_cli,
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
