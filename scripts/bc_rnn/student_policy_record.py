"""
student_policy_record.py

Run trained BC-RNN teacher policies and record distillation data.

The teacher has privileged access to full sim state (handle position,
drawer state) which the student will NOT have. Instead, the student
learns from what a real robot can observe:

  Student observations (recorded):
    - proprio/arm_joint_pos:  (8D)  arm/wrist joint positions
    - proprio/base_pos:       (3D)  base joint positions
    - proprio/joint_vel:      (11D) arm/wrist/base velocities
    - proprio/gripper_state:  (2D)  gripper finger positions
    - image:                  (H, W, 3) tracking camera RGB

  NOT recorded (privileged, sim-only):
    - handle_rel:  gripper-to-handle vector (3D)
    - drawer_pos:  drawer joint position (1D)

  NOT recorded (independent controller):
    - head_pos:  pan/tilt are driven by a separate tracking controller,
                 not by the student policy

  Actions (13D):
    [arm_delta(8), base_delta(3), gripper_delta(2)]
    Head tracking is an independent controller — not part of the action space.

Saves HDF5 datasets (reach + pull) for student distillation training,
and saves camera images to a separate directory for visual inspection.

Usage:
    ../IsaacLab/isaaclab.sh -p scripts/bc_rnn/student_policy_record.py \
        --reach_ckpt path/to/reach/best.pth \
        --pull_ckpt path/to/pull/best.pth \
        --num_demos 50
"""

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import torch
import argparse
import time
import numpy as np

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distillation data recorder")
parser.add_argument(
    "--reach_ckpt", type=str, required=True, help="Path to reach policy checkpoint"
)
parser.add_argument(
    "--pull_ckpt", type=str, default=None, help="Path to pull policy checkpoint"
)
parser.add_argument("--num_demos", type=int, default=10, help="Target successful demos")
parser.add_argument("--ratio", type=float, default=0.1, help="Validation split ratio")
parser.add_argument(
    "--reach_steps", type=int, default=350, help="Max reach phase steps"
)
parser.add_argument("--pull_steps", type=int, default=150, help="Max pull phase steps")
parser.add_argument(
    "--drawer_threshold",
    type=float,
    default=0.35,
    help="Drawer position for success",
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
    help="Directory to save camera images for inspection",
)
parser.add_argument(
    "--save_every_n",
    type=int,
    default=25,
    help="Save inspection image every N steps (1 = every step)",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=30,
    help="FPS denominator used for video time display in debug output",
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
from isaaclab.envs import ManagerBasedRLEnv

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import imageio


# ==========================================
# HELPERS
# ==========================================
def load_policy(ckpt_path, device):
    """Load a trained robomimic policy from a checkpoint."""
    print(f"[Policy] Loading: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=True
    )
    print(f"  Algorithm: {ckpt_dict.get('algo_name', 'unknown')}")
    print(f"  Epoch: {ckpt_dict.get('epoch', 'unknown')}")
    policy.start_episode()
    return policy


def capture_camera_image(head_cam):
    """Get RGB image from the head camera as a uint8 numpy array (H, W, 3)."""
    rgb = head_cam.data.output["rgb"][0]
    rgb_np = rgb.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[:, :, :3]
    return rgb_np


# ==========================================
# MAIN
# ==========================================
def main():
    env_cfg = StretchEnvCfg()
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Load teacher policies
    reach_policy = load_policy(args_cli.reach_ckpt, device)
    pull_policy = None
    if args_cli.pull_ckpt:
        pull_policy = load_policy(args_cli.pull_ckpt, device)

    # Scene references
    cabinet_entity = env.scene["cabinet"]
    robot_entity = env.scene["robot"]
    head_cam = env.scene["head_camera"]

    handle_body_idx = cabinet_entity.data.body_names.index("drawer_handle_top")
    head_body_idx = robot_entity.data.body_names.index("link_head")
    head_joint_ids = robot_entity.find_joints(HEAD_JOINT_NAMES)[0]

    arm_joint_ids = robot_entity.find_joints(ARM_JOINT_NAMES)[0]
    base_joint_ids = robot_entity.find_joints(BASE_JOINT_NAMES)[0]
    gripper_joint_ids = robot_entity.find_joints(GRIPPER_JOINT_NAMES)[0]
    all_vel_ids = robot_entity.find_joints(ARM_JOINT_NAMES + BASE_JOINT_NAMES)[0]

    def get_proprioception():
        return {
            "arm_joint_pos": robot_entity.data.joint_pos[:, arm_joint_ids],
            "base_pos": robot_entity.data.joint_pos[:, base_joint_ids],
            "joint_vel": robot_entity.data.joint_vel[:, all_vel_ids],
            "gripper_state": robot_entity.data.joint_pos[:, gripper_joint_ids],
        }

    def compute_head_command():
        return compute_head_look_at(
            robot_entity, cabinet_entity, head_body_idx, handle_body_idx, head_joint_ids
        )

    def command_head_joints(head_target):
        robot_entity.set_joint_position_target(
            head_target.unsqueeze(0), joint_ids=head_joint_ids
        )

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
    collector_pull = None
    if pull_policy is not None:
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
    max_attempts = args_cli.num_demos * 3
    sim_start_time = time.time()

    for ep_idx in range(max_attempts):
        all_done = collector_reach.is_stopped() and (
            collector_pull is None or collector_pull.is_stopped()
        )
        if all_done:
            break

        print(f"\n{'='*60}")
        print(f"  EPISODE {ep_idx + 1} (collected: {successes}/{args_cli.num_demos})")
        print(f"{'='*60}")

        obs, _ = env.reset()
        reach_policy.start_episode()
        if pull_policy is not None:
            pull_policy.start_episode()

        ep_image_dir = os.path.join(args_cli.image_dir, f"episode_{ep_idx}")
        os.makedirs(ep_image_dir, exist_ok=True)

        step_count = 0
        current_phase = "reach"
        episode_success = False

        # ------------------------------------------
        # PHASE 0: HEAD CAMERA ALIGNMENT (warm-up)
        # Steps with zero action until head settles — matches training data distribution
        # so the BC-RNN hidden state is properly initialized before the policy runs.
        # ------------------------------------------
        print("[Phase 0] Aligning head camera...")
        zero_action = torch.zeros(1, 13, device=env.device)
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
        print(f"  [Phase 0] Head aligned in {step_count} steps.")

        # Print initial state
        grasp_center_idx = robot_entity.data.body_names.index("link_grasp_center")
        ee_pos = robot_entity.data.body_pos_w[0, grasp_center_idx]
        handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
        dist = torch.norm(ee_pos - handle_pos).item()
        print(f"  [INIT] EE pos:     {ee_pos.tolist()}")
        print(f"  [INIT] Handle pos: {handle_pos.tolist()}")
        print(f"  [INIT] EE→Handle:  {dist:.4f}m")

        # ------------------------------------------
        # REACH PHASE
        # ------------------------------------------
        print("[Phase] Running REACH policy...")
        for step in range(args_cli.reach_steps):
            obs_dict = {"policy": obs["policy"][0].cpu().numpy()}
            action = reach_policy(obs_dict)
            env_action = torch.tensor(action, device=env.device).unsqueeze(0)[:, :13]

            # Head tracks handle analytically (not part of policy)
            head_target, _ = compute_head_command()
            command_head_joints(head_target)

            # Capture obs BEFORE step
            image_np = capture_camera_image(head_cam)
            student_obs = {
                "proprio": get_proprioception(),
                "image": torch.from_numpy(image_np),
            }

            next_obs, rew, terminated, truncated, _ = env.step(env_action)
            step_count += 1

            # Capture next_obs AFTER step
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

            if step_count % args_cli.save_every_n == 0:
                imageio.imwrite(
                    os.path.join(ep_image_dir, f"reach_{step_count:04d}.png"),
                    image_np,
                )

            if step_count % 10 == 0:
                ee_pos = robot_entity.data.body_pos_w[0, grasp_center_idx]
                handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
                dist = torch.norm(ee_pos - handle_pos).item()
                video_time = step_count / args_cli.video_fps
                root_pos = robot_entity.data.root_state_w[0, :3]
                root_quat = robot_entity.data.root_state_w[0, 3:7]
                joint_pos = robot_entity.data.joint_pos[0]
                print(
                    f"  [DEBUG] Step {step_count} | Phase: {current_phase} | Video: {video_time:.2f}s\n"
                    f"    Action(13D):{env_action[0].tolist()}\n"
                    f"    Robot pos:  {root_pos.tolist()}\n"
                    f"    Robot quat: {root_quat.tolist()}\n"
                    f"    Joint pos:  {joint_pos.tolist()}\n"
                    f"    EE pos:     {ee_pos.tolist()}\n"
                    f"    Handle pos: {handle_pos.tolist()}\n"
                    f"    EE→Handle:  {dist:.4f}m"
                )

            obs = next_obs

        print(f"  Reach phase completed ({step + 1} steps)")

        # ------------------------------------------
        # PULL PHASE
        # ------------------------------------------
        if pull_policy is not None:
            current_phase = "pull"
            print("[Phase] Running PULL policy...")

            for step in range(args_cli.pull_steps):
                obs_dict = {"policy": obs["policy"][0].cpu().numpy()}
                action = pull_policy(obs_dict)
                env_action = torch.tensor(action, device=env.device).unsqueeze(0)[
                    :, :13
                ]

                # Head tracks handle analytically (not part of policy)
                head_target, _ = compute_head_command()
                command_head_joints(head_target)

                # Capture obs BEFORE step
                image_np = capture_camera_image(head_cam)
                student_obs = {
                    "proprio": get_proprioception(),
                    "image": torch.from_numpy(image_np),
                }

                next_obs, rew, terminated, truncated, _ = env.step(env_action)
                step_count += 1

                # Capture next_obs AFTER step
                next_image_np = capture_camera_image(head_cam)
                student_next_obs = {
                    "proprio": get_proprioception(),
                    "image": torch.from_numpy(next_image_np),
                }

                if collector_pull is not None:
                    collector_pull.add("obs", student_obs)
                    collector_pull.add("actions", env_action)
                    collector_pull.add("rewards", rew)
                    collector_pull.add("dones", terminated | truncated)
                    collector_pull.add("next_obs", student_next_obs)

                pull_step = step_count
                if pull_step % args_cli.save_every_n == 0:
                    imageio.imwrite(
                        os.path.join(ep_image_dir, f"pull_{step_count:04d}.png"),
                        image_np,
                    )

                drawer_max = cabinet_entity.data.joint_pos[0].max().item()

                if step_count % 10 == 0:
                    ee_pos = robot_entity.data.body_pos_w[0, grasp_center_idx]
                    handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
                    dist = torch.norm(ee_pos - handle_pos).item()
                    video_time = step_count / args_cli.video_fps
                    root_pos = robot_entity.data.root_state_w[0, :3]
                    root_quat = robot_entity.data.root_state_w[0, 3:7]
                    joint_pos = robot_entity.data.joint_pos[0]
                    print(
                        f"  [DEBUG] Step {step_count} | Phase: {current_phase} | Video: {video_time:.2f}s\n"
                        f"    Action(13D):{env_action[0].tolist()}\n"
                        f"    Robot pos:  {root_pos.tolist()}\n"
                        f"    Robot quat: {root_quat.tolist()}\n"
                        f"    Joint pos:  {joint_pos.tolist()}\n"
                        f"    EE pos:     {ee_pos.tolist()}\n"
                        f"    Handle pos: {handle_pos.tolist()}\n"
                        f"    EE→Handle:  {dist:.4f}m\n"
                        f"    Drawer:     {drawer_max:.4f}m"
                    )

                if drawer_max > args_cli.drawer_threshold:
                    print(
                        f"  --> Drawer open ({drawer_max:.3f}m) at step {step_count}!"
                    )
                    episode_success = True
                    break

                obs = next_obs

            print(f"  Pull phase completed ({step + 1} steps)")

        # ------------------------------------------
        # Episode Summary
        # ------------------------------------------
        final_drawer = cabinet_entity.data.joint_pos[0].max().item()
        if final_drawer > args_cli.drawer_threshold:
            episode_success = True

        status = "SUCCESS" if episode_success else "FAIL"

        if episode_success:
            collector_reach.flush()
            if collector_pull is not None:
                collector_pull.flush()
            successes += 1
        else:
            collector_reach.reset_buffer()
            if collector_pull is not None:
                collector_pull.reset_buffer()
            demos_failed += 1

        elapsed = time.time() - sim_start_time
        print(f"\n  Result: {status}")
        print(f"  Final drawer position: {final_drawer:.3f}m")
        print(f"  Total steps: {step_count}")
        print(
            f"  [{successes}/{args_cli.num_demos} collected | {demos_failed} failed | {elapsed:.0f}s]"
        )

    # ==========================================
    # Final Summary
    # ==========================================
    total_time = time.time() - sim_start_time
    print(f"\n{'='*60}")
    print("  DISTILLATION RECORDING COMPLETE")
    print(f"{'='*60}")
    print(f"  Demos collected: {successes}/{args_cli.num_demos}")
    print(f"  Failed attempts: {demos_failed}")
    print(f"  Total time:      {total_time:.0f}s")
    print(f"  HDF5 datasets:   {args_cli.dataset_dir}/")
    print(f"  Camera images:   {args_cli.image_dir}/")
    print(f"{'='*60}")

    env.close()
    collector_reach.close()
    if collector_pull is not None:
        collector_pull.close()


if __name__ == "__main__":
    main()
