"""
student_policy_record.py

Records distillation data using the BC-RNN teacher policies.
Sequence matches bc_rnn_play.py exactly:

  Phase 0: Head camera alignment
           (no recording, no image capture)

  --- RECORDING START + CAMERA STARTS ---

  Phase 1: Gripper open + Reach policy
           (approach safe spot → insert to target handle)
           → recorded to reach collector

  Phase 2: Gripper close (GRIPPER_LOCK_STEPS steps, arm/base zeroed)
           → recorded to pull collector

  Phase 3: Pull policy (robot pulls back, gripper closed)
           → recorded to pull collector

  --- RECORDING END + CAMERA STOPS ---

Student observations (recorded every step):
  - proprio/arm_joint_pos:  (8D)  arm/wrist joint positions
  - proprio/base_pos:       (3D)  base joint positions
  - proprio/joint_vel:      (11D) arm/wrist/base velocities
  - proprio/gripper_state:  (2D)  gripper finger positions
  - proprio/head_pos:       (2D)  head pan/tilt joint positions
  - image:                  (H, W, 3) head camera RGB
  Total proprio: 26D

Actions (13D): [arm_delta(8), base_delta(3), gripper_delta(2)]
Head tracking is an independent controller — not part of the action space.
"""

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import torch
import glob
import argparse
import time
import numpy as np

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distillation data recorder")
parser.add_argument(
    "--reach_ckpt",
    type=str,
    default=None,
    help="Path to the reach phase checkpoint (.pth). Auto-detected if not provided.",
)
parser.add_argument(
    "--pull_ckpt",
    type=str,
    default=None,
    help="Path to the pull phase checkpoint (.pth). Auto-detected if not provided.",
)
parser.add_argument("--num_demos", type=int, default=10, help="Target successful demos")
parser.add_argument("--ratio", type=float, default=0.1, help="Validation split ratio")
parser.add_argument(
    "--max_reach_steps", type=int, default=500, help="Max reach phase steps"
)
parser.add_argument(
    "--max_pull_steps", type=int, default=400, help="Max pull phase steps"
)
parser.add_argument(
    "--noise_range",
    type=float,
    default=0.0,
    help="Cabinet position randomization range (meters)",
)
parser.add_argument("--filename", type=str, default="distillation")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/home/johnchen/SharedSSD/JohnChen/stretch/datasets",
)
parser.add_argument(
    "--image_dir",
    type=str,
    default="/home/johnchen/SharedSSD/JohnChen/stretch/distillation_images",
    help="Directory to save inspection images",
)
parser.add_argument(
    "--save_every_n",
    type=int,
    default=25,
    help="Save inspection PNG every N steps (1 = every step)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
tools_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/scripts/tools"
sys.path.append(target_config_dir)
sys.path.append(tools_dir)

from bc_rnn_stretch_cfg import (
    StretchEnvCfg,
    ARM_JOINT_NAMES,
    BASE_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    HEAD_JOINT_NAMES,
    compute_head_look_at,
)
from robomimic_collector import RobomimicDataCollector
import robomimic
from isaaclab.envs import ManagerBasedRLEnv
from isaacsim.core.prims import XFormPrim

import robomimic.utils.file_utils as FileUtils
import imageio

# ==========================================
# CONSTANTS  (match bc_rnn_play.py exactly)
# ==========================================
TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"

REACH_DONE_THRESHOLD = 0.03  # metres — proximity triggers reach→pull switch
GRIPPER_OPEN_POS = 0.1
GRIPPER_CLOSE_POS = -0.1
GRIPPER_LOCK_STEPS = 30  # steps of gripper-close dead-zone
DRAWER_OPEN_THRESHOLD = 0.38  # metres — success criterion
HANDLE_DISPLACEMENT_THRESHOLD = 0.05


# ==========================================
# HELPERS
# ==========================================
def load_policy(ckpt_path, device):
    print(f"[Policy] Loading: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=True
    )
    policy.start_episode()
    return policy


def capture_camera_image(head_cam):
    """Return the head camera frame as a uint8 (H, W, 3) numpy array."""
    rgb = head_cam.data.output["rgb"][0]
    rgb_np = rgb.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[:, :, :3]
    return rgb_np


def randomize_cabinet(
    cabinet_view, cabinet_articulation, default_pos, default_rot, device, noise_range
):
    new_pos = default_pos.clone()
    new_pos[0] += (torch.rand(1, device=device) * 2 - 1).item() * noise_range
    new_pos[1] += (torch.rand(1, device=device) * 2 - 1).item() * noise_range
    cabinet_view.set_world_poses(
        positions=new_pos.unsqueeze(0), orientations=default_rot
    )
    root_state = cabinet_articulation.data.default_root_state.clone()
    root_state[:, :3] = new_pos
    cabinet_articulation.write_root_pose_to_sim(root_state[:, :7])
    cabinet_articulation.write_root_velocity_to_sim(root_state[:, 7:])


# ==========================================
# CHECKPOINT AUTO-DETECTION
# ==========================================
def find_best_checkpoint(experiment_name):
    """
    Search for the best-validation checkpoint for a given experiment name
    in the robomimic-default output location.
    """
    robomimic_root = os.path.join(robomimic.__path__[0], "..")
    base_dir = os.path.join(robomimic_root, "bc_rnn_trained_models", experiment_name)
    base_dir = os.path.normpath(base_dir)

    pattern = os.path.join(base_dir, "*", "models", "*best_validation*.pth")
    matches = glob.glob(pattern)

    if matches:

        def extract_loss(path):
            fname = os.path.basename(path)
            loss_str = fname.split("best_validation_")[1].replace(".pth", "")
            return float(loss_str)

        return min(matches, key=extract_loss)

    # Fall back to the last epoch checkpoint from the most recent run
    fallback = os.path.join(base_dir, "*", "models", "model_epoch_*.pth")
    fallback_matches = glob.glob(fallback)
    if fallback_matches:
        return max(fallback_matches, key=os.path.getmtime)

    return None


def resolve_checkpoint(user_path, experiment_name, label):
    if user_path is not None:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Checkpoint not found: {user_path}")
        return user_path

    ckpt = find_best_checkpoint(experiment_name)
    if ckpt is None:
        raise FileNotFoundError(
            f"Could not auto-detect a checkpoint for '{experiment_name}'. "
            f"Train the {label} phase first or pass --{label}_ckpt."
        )
    return ckpt


# ==========================================
# MAIN
# ==========================================
def main():

    # --- Resolve checkpoints (auto-detect best validation if not provided) ---
    reach_ckpt = resolve_checkpoint(args_cli.reach_ckpt, "bc_rnn_reach_phase", "reach")
    pull_ckpt = resolve_checkpoint(args_cli.pull_ckpt, "bc_rnn_pull_phase", "pull")

    # --- Load policies before sim so CUDA context is shared ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reach_policy = load_policy(reach_ckpt, device)
    pull_policy = load_policy(pull_ckpt, device)

    env_cfg = StretchEnvCfg()
    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Scene references
    robot_entity = env.scene["robot"]
    cabinet_entity = env.scene["cabinet"]
    head_cam = env.scene["head_camera"]

    handle_body_idx = cabinet_entity.data.body_names.index("drawer_handle_top")
    grasp_center_idx = robot_entity.data.body_names.index("link_grasp_center")
    head_body_idx = robot_entity.data.body_names.index("link_head")
    head_joint_ids = robot_entity.find_joints(HEAD_JOINT_NAMES)[0]

    arm_joint_ids = robot_entity.find_joints(ARM_JOINT_NAMES)[0]
    base_joint_ids = robot_entity.find_joints(BASE_JOINT_NAMES)[0]
    gripper_joint_ids = robot_entity.find_joints(GRIPPER_JOINT_NAMES)[0]
    all_vel_ids = robot_entity.find_joints(ARM_JOINT_NAMES + BASE_JOINT_NAMES)[0]

    # Cabinet prims
    cabinet_view = XFormPrim("/World/envs/env_0/Cabinet", name="cabinet")
    target_frame_view = XFormPrim(TARGET_FRAME_PATH, name="target_frame")
    default_cabinet_pos, default_cabinet_rot = cabinet_view.get_world_poses()
    default_cabinet_pos = default_cabinet_pos[0].clone()

    def get_proprioception():
        return {
            "arm_joint_pos": robot_entity.data.joint_pos[:, arm_joint_ids],   # 8D
            "base_pos": robot_entity.data.joint_pos[:, base_joint_ids],        # 3D
            "joint_vel": robot_entity.data.joint_vel[:, all_vel_ids],          # 11D
            "gripper_state": robot_entity.data.joint_pos[:, gripper_joint_ids],  # 2D
            "head_pos": robot_entity.data.joint_pos[:, head_joint_ids],        # 2D (pan, tilt)
        }

    def compute_head_command():
        return compute_head_look_at(
            robot_entity, cabinet_entity, head_body_idx, handle_body_idx, head_joint_ids
        )

    def command_head_joints(head_target):
        robot_entity.set_joint_position_target(
            head_target.unsqueeze(0), joint_ids=head_joint_ids
        )

    def get_handle_dist():
        """EE (link_grasp_center) → drawer_handle_frame distance, matching bc_rnn_play.py."""
        pos, _ = target_frame_view.get_world_poses()
        handle_pos = torch.tensor(pos[0], device=env.device, dtype=torch.float32)
        grasp_pos = robot_entity.data.body_pos_w[0, grasp_center_idx]
        return torch.norm(handle_pos - grasp_pos).item()

    # ==========================================
    # Data Collectors
    # ==========================================
    os.makedirs(args_cli.dataset_dir, exist_ok=True)
    os.makedirs(args_cli.image_dir, exist_ok=True)

    num_valid = int(args_cli.num_demos * args_cli.ratio)
    val_indices = list(range(args_cli.num_demos - num_valid, args_cli.num_demos))
    print(
        f"[Split] {args_cli.num_demos} demos: "
        f"{args_cli.num_demos - num_valid} train, {num_valid} valid"
    )

    collector_reach = RobomimicDataCollector(
        env_name="Isaac-Stretch-Cabinet-v0",
        directory_path=args_cli.dataset_dir,
        filename=args_cli.filename + "_reach",
        num_demos=args_cli.num_demos,
        val_indices=val_indices,
    )
    collector_pull = RobomimicDataCollector(
        env_name="Isaac-Stretch-Cabinet-v0",
        directory_path=args_cli.dataset_dir,
        filename=args_cli.filename + "_pull",
        num_demos=args_cli.num_demos,
        val_indices=val_indices,
    )

    # ==========================================
    # Episode Loop
    # ==========================================
    successes = 0
    demos_failed = 0
    max_attempts = args_cli.num_demos * 5
    sim_start_time = time.time()

    for ep_idx in range(max_attempts):
        if collector_reach.is_stopped() and collector_pull.is_stopped():
            break

        print(f"\n{'='*60}")
        print(f"  EPISODE {ep_idx + 1}  (collected: {successes}/{args_cli.num_demos})")
        print(f"{'='*60}")

        obs, _ = env.reset()
        randomize_cabinet(
            cabinet_view,
            cabinet_entity,
            default_cabinet_pos,
            default_cabinet_rot,
            env.device,
            args_cli.noise_range,
        )
        initial_handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx].clone()

        reach_policy.start_episode()
        pull_policy.start_episode()

        ep_image_dir = os.path.join(args_cli.image_dir, f"episode_{ep_idx:03d}")
        os.makedirs(ep_image_dir, exist_ok=True)

        zero_action = torch.zeros(1, 13, device=env.device)
        step_count = 0

        # ======================================================
        # PHASE 0: HEAD CAMERA ALIGNMENT
        # No recording. No image capture.
        # ======================================================
        print("[Phase 0] Aligning head camera...")
        head_settle_timer = 0
        while head_settle_timer < 10:
            head_target, _ = compute_head_command()
            command_head_joints(head_target)
            head_current = robot_entity.data.joint_pos[0, head_joint_ids]
            head_error = torch.norm(head_target - head_current).item()
            if head_error < 0.05:
                head_settle_timer += 1
            else:
                head_settle_timer = 0
            obs, _, _, _, _ = env.step(zero_action)
            step_count += 1
        print(f"  Head aligned ({step_count} steps).")
        print("  --> RECORDING START | Camera capturing begins.")

        # ======================================================
        # PHASE 1: OPEN GRIPPER + REACH POLICY
        # Gripper stays open. Reach policy drives arm/base.
        # (safe spot → insert to target handle)
        # Recorded to collector_reach.
        # ======================================================
        print("[Phase 1] Gripper open | Reach policy: safe spot → handle...")
        reach_steps = 0
        reach_by_proximity = False

        for _ in range(args_cli.max_reach_steps):
            obs_dict = {"policy": obs["policy"][0].cpu().numpy()}
            action = reach_policy(obs_dict)

            # Arm + base from policy; gripper explicitly kept open
            arm_act = torch.tensor(
                action[: len(arm_joint_ids)], device=env.device, dtype=torch.float32
            ).unsqueeze(0)
            base_act = torch.tensor(
                action[len(arm_joint_ids) : len(arm_joint_ids) + len(base_joint_ids)],
                device=env.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            gripper_current = robot_entity.data.joint_pos[:, gripper_joint_ids]
            grip_act = (
                torch.tensor([[GRIPPER_OPEN_POS, GRIPPER_OPEN_POS]], device=env.device)
                - gripper_current
            )
            env_action = torch.cat([arm_act, base_act, grip_act], dim=-1)

            head_target, _ = compute_head_command()
            command_head_joints(head_target)

            # Capture image BEFORE step (current state)
            image_np = capture_camera_image(head_cam)
            student_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(image_np),
            }

            obs, rew, terminated, truncated, _ = env.step(env_action)
            step_count += 1
            reach_steps += 1

            # Capture image AFTER step (next state)
            next_image_np = capture_camera_image(head_cam)
            student_next_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(next_image_np),
            }

            collector_reach.add("obs", student_obs)
            collector_reach.add("actions", env_action)
            collector_reach.add("rewards", rew)
            collector_reach.add("dones", terminated | truncated)
            collector_reach.add("next_obs", student_next_obs)

            if reach_steps % args_cli.save_every_n == 0:
                imageio.imwrite(
                    os.path.join(ep_image_dir, f"reach_{reach_steps:04d}.png"), image_np
                )

            dist = get_handle_dist()
            if reach_steps % 10 == 0:
                print(f"  [Reach {reach_steps:3d}] EE→Handle: {dist:.4f}m")

            if dist < REACH_DONE_THRESHOLD:
                reach_by_proximity = True
                break

        reason = "proximity" if reach_by_proximity else "timeout"
        print(
            f"  Reach done ({reason}, dist={get_handle_dist():.3f}m, steps={reach_steps})."
        )

        # Reset pull RNN now — matches bc_rnn_play.py
        pull_policy.start_episode()

        # ======================================================
        # PHASE 2: GRIPPER CLOSE
        # Arm and base zeroed; gripper closes over GRIPPER_LOCK_STEPS steps.
        # Recorded to collector_pull (matches bc_rnn_record.py split boundary).
        # ======================================================
        print(f"[Phase 2] Closing gripper ({GRIPPER_LOCK_STEPS} steps)...")
        for grip_step in range(GRIPPER_LOCK_STEPS):
            arm_act = torch.zeros(1, len(arm_joint_ids), device=env.device)
            base_act = torch.zeros(1, len(base_joint_ids), device=env.device)
            gripper_current = robot_entity.data.joint_pos[:, gripper_joint_ids]
            grip_act = (
                torch.tensor(
                    [[GRIPPER_CLOSE_POS, GRIPPER_CLOSE_POS]], device=env.device
                )
                - gripper_current
            )
            env_action = torch.cat([arm_act, base_act, grip_act], dim=-1)

            head_target, _ = compute_head_command()
            command_head_joints(head_target)

            image_np = capture_camera_image(head_cam)
            student_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(image_np),
            }

            obs, rew, terminated, truncated, _ = env.step(env_action)
            step_count += 1

            next_image_np = capture_camera_image(head_cam)
            student_next_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(next_image_np),
            }

            collector_pull.add("obs", student_obs)
            collector_pull.add("actions", env_action)
            collector_pull.add("rewards", rew)
            collector_pull.add("dones", terminated | truncated)
            collector_pull.add("next_obs", student_next_obs)

            if (grip_step + 1) % args_cli.save_every_n == 0:
                imageio.imwrite(
                    os.path.join(ep_image_dir, f"grip_{grip_step:04d}.png"), image_np
                )

        print("  Gripper locked.")

        # ======================================================
        # PHASE 3: PULL POLICY
        # Gripper held closed; pull policy drives arm/base.
        # Recorded to collector_pull.
        # ======================================================
        print("[Phase 3] Pull policy: robot pulls back...")
        pull_steps = 0
        episode_success = False

        for _ in range(args_cli.max_pull_steps):
            obs_dict = {"policy": obs["policy"][0].cpu().numpy()}
            action = pull_policy(obs_dict)

            # Arm + base from policy; gripper explicitly kept closed
            arm_act = torch.tensor(
                action[: len(arm_joint_ids)], device=env.device, dtype=torch.float32
            ).unsqueeze(0)
            base_act = torch.tensor(
                action[len(arm_joint_ids) : len(arm_joint_ids) + len(base_joint_ids)],
                device=env.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            gripper_current = robot_entity.data.joint_pos[:, gripper_joint_ids]
            grip_act = (
                torch.tensor(
                    [[GRIPPER_CLOSE_POS, GRIPPER_CLOSE_POS]], device=env.device
                )
                - gripper_current
            )
            env_action = torch.cat([arm_act, base_act, grip_act], dim=-1)

            head_target, _ = compute_head_command()
            command_head_joints(head_target)

            image_np = capture_camera_image(head_cam)
            student_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(image_np),
            }

            obs, rew, terminated, truncated, _ = env.step(env_action)
            step_count += 1
            pull_steps += 1

            next_image_np = capture_camera_image(head_cam)
            student_next_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(next_image_np),
            }

            collector_pull.add("obs", student_obs)
            collector_pull.add("actions", env_action)
            collector_pull.add("rewards", rew)
            collector_pull.add("dones", terminated | truncated)
            collector_pull.add("next_obs", student_next_obs)

            if pull_steps % args_cli.save_every_n == 0:
                imageio.imwrite(
                    os.path.join(ep_image_dir, f"pull_{pull_steps:04d}.png"), image_np
                )

            drawer_max = cabinet_entity.data.joint_pos[0].max().item()
            if pull_steps % 10 == 0:
                print(f"  [Pull  {pull_steps:3d}] Drawer: {drawer_max:.4f}m")

            if drawer_max > DRAWER_OPEN_THRESHOLD:
                episode_success = True
                print(f"  Drawer open ({drawer_max:.3f}m) at pull step {pull_steps}!")
                break

        # ======================================================
        # RECORDING END + CAMERA STOPS
        # ======================================================
        final_handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
        displacement = torch.norm(final_handle_pos - initial_handle_pos).item()
        episode_success = displacement > HANDLE_DISPLACEMENT_THRESHOLD

        elapsed = time.time() - sim_start_time
        status = "SUCCESS" if episode_success else "FAIL"

        if episode_success:
            collector_reach.flush()
            collector_pull.flush()
            successes += 1
        else:
            collector_reach.reset_buffer()
            collector_pull.reset_buffer()
            demos_failed += 1

        print("\n  --> RECORDING END | Camera capturing stops.")
        print(f"  Result:       {status}")
        print(f"  Drawer:       {cabinet_entity.data.joint_pos[0].max().item():.3f}m")
        print(f"  Displacement: {displacement:.3f}m")
        print(
            f"  Steps:        {step_count} total (reach={reach_steps}, pull={pull_steps})"
        )
        print(
            f"  [{successes}/{args_cli.num_demos} collected | "
            f"{demos_failed} failed | {elapsed:.0f}s]"
        )

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    total_time = time.time() - sim_start_time
    print(f"\n{'='*60}")
    print("  DISTILLATION RECORDING COMPLETE")
    print(f"{'='*60}")
    print(f"  Demos collected: {successes}/{args_cli.num_demos}")
    print(f"  Failed attempts: {demos_failed}")
    print(f"  Total time:      {total_time:.0f}s")
    print(f"  HDF5 datasets:   {args_cli.dataset_dir}/")
    print(f"  Images:          {args_cli.image_dir}/")
    print(f"{'='*60}")

    env.close()
    collector_reach.close()
    collector_pull.close()


if __name__ == "__main__":
    main()
