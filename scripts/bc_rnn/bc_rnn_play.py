import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
import glob
import torch
import argparse
import time

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC-RNN Policy Playback")
AppLauncher.add_app_launcher_args(parser)
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
parser.add_argument(
    "--num_episodes", type=int, default=10, help="Number of episodes to evaluate"
)
parser.add_argument(
    "--noise_range",
    type=float,
    default=0.0,
    help="Cabinet position randomization range (meters)",
)
parser.add_argument(
    "--max_reach_steps",
    type=int,
    default=500,
    help="Max steps for the reach phase before switching to pull",
)
parser.add_argument(
    "--max_pull_steps",
    type=int,
    default=400,
    help="Max steps for the pull phase before timing out",
)
parser.add_argument(
    "--record_video",
    action="store_true",
    help="Record a video for each episode and save to --video_dir",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="video_demos",
    help="Directory to save episode videos (default: video_demos)",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(target_config_dir)

from bc_rnn_stretch_cfg import (
    StretchEnvCfg,
    ARM_JOINT_NAMES,
    BASE_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    HEAD_JOINT_NAMES,
    compute_head_look_at,
)
from isaaclab.envs import ManagerBasedRLEnv
from isaacsim.core.prims import XFormPrim

# ==========================================
# ROBOMIMIC POLICY IMPORTS
# ==========================================
import robomimic
import robomimic.utils.file_utils as FileUtils

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(os.path.join(SCRIPT_DIR, "..", "tools"))
from video_recorder import VideoRecorder

TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"

GRIPPER_OPEN_POS = 0.1
GRIPPER_CLOSE_POS = -0.1
DRAWER_OPEN_THRESHOLD = 0.38
HANDLE_DISPLACEMENT_THRESHOLD = 0.05
GRIPPER_LOCK_STEPS = 30

# Distance from gripper to handle (in world frame) that triggers the reach→pull switch
REACH_DONE_THRESHOLD = 0.03


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
# OBS CONVERSION
# ==========================================
def env_obs_to_robomimic(obs):
    """
    Convert the IsaacLab obs dict (tensors, batch_dim=1) to a robomimic-compatible
    dict of 1-D numpy arrays (no batch dim).
    """
    return {k: v[0].detach().cpu().numpy() for k, v in obs.items()}


# ==========================================
# CABINET HELPERS  (mirrors record script)
# ==========================================
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


def reset_episode_state():
    return {
        "step_count": 0,
        "head_tracking_done": False,
        "head_settle_timer": 0,
        "reach_done": False,
        "gripper_timer": 0,
        "pull_done": False,
        "reach_steps": 0,
        "pull_steps": 0,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    # --- Resolve checkpoints ---
    reach_ckpt = resolve_checkpoint(args_cli.reach_ckpt, "bc_rnn_reach_phase", "reach")
    pull_ckpt = resolve_checkpoint(args_cli.pull_ckpt, "bc_rnn_pull_phase", "pull")
    print(f"[Policy] Loading reach: {reach_ckpt}")
    print(f"[Policy] Loading pull:  {pull_ckpt}")

    # --- Load policies (before sim so CUDA context is shared) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reach_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=reach_ckpt, device=device, verbose=True
    )
    pull_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=pull_ckpt, device=device, verbose=True
    )

    # --- Environment ---
    env_cfg = StretchEnvCfg()
    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

    # --- Scene references ---
    robot_entity = env.scene["robot"]
    cabinet_entity = env.scene["cabinet"]

    arm_joint_ids_isaac = robot_entity.find_joints(ARM_JOINT_NAMES)[0]
    base_joint_ids_isaac = robot_entity.find_joints(BASE_JOINT_NAMES)[0]
    gripper_joint_ids_isaac = robot_entity.find_joints(GRIPPER_JOINT_NAMES)[0]

    handle_body_idx = cabinet_entity.data.body_names.index("drawer_handle_top")
    grasp_body_idx = robot_entity.data.body_names.index("link_grasp_center")
    head_body_idx = robot_entity.data.body_names.index("link_head")
    head_joint_ids = robot_entity.find_joints(HEAD_JOINT_NAMES)[0]

    def compute_head_command():
        return compute_head_look_at(
            robot_entity, cabinet_entity, head_body_idx, handle_body_idx, head_joint_ids
        )

    # --- Cabinet scene prims ---
    cabinet_view = XFormPrim("/World/envs/env_0/Cabinet", name="cabinet")
    target_frame_view = XFormPrim(
        TARGET_FRAME_PATH, name="target_frame"
    )  # actual grasp point (matches record script)
    default_cabinet_pos, default_cabinet_rot = cabinet_view.get_world_poses()
    default_cabinet_pos = default_cabinet_pos[0].clone()

    randomize_cabinet(
        cabinet_view,
        cabinet_entity,
        default_cabinet_pos,
        default_cabinet_rot,
        env.device,
        args_cli.noise_range,
    )
    initial_handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx].clone()

    # --- Video recorder ---
    video_recorder = None
    if args_cli.record_video:
        video_dir = os.path.join(os.getcwd(), args_cli.video_dir)
        video_recorder = VideoRecorder(video_dir=video_dir, fps=30)
        print(f"[Video] Recording enabled. Saving to: {video_dir}")

    # --- Episode tracking ---
    ep = reset_episode_state()
    episodes_done = 0
    successes = 0
    sim_start = time.time()

    reach_policy.start_episode()
    pull_policy.start_episode()

    print(">>> Starting Playback Loop...")
    while simulation_app.is_running():
        if episodes_done >= args_cli.num_episodes:
            break

        # ==========================================
        # PHASE 0: HEAD CAMERA ALIGNMENT
        # ==========================================
        head_target, _ = compute_head_command()
        robot_entity.set_joint_position_target(
            head_target.unsqueeze(0), joint_ids=head_joint_ids
        )

        if not ep["head_tracking_done"]:
            head_current = robot_entity.data.joint_pos[0, head_joint_ids]
            head_error = torch.norm(head_target - head_current).item()
            if head_error < 0.05:
                ep["head_settle_timer"] += 1
                if ep["head_settle_timer"] >= 10:
                    print("--> [Phase 0] Head aligned. Starting policy inference...")
                    ep["head_tracking_done"] = True
            else:
                ep["head_settle_timer"] = 0

        # ==========================================
        # POLICY INFERENCE
        # ==========================================
        if ep["head_tracking_done"]:
            rob_obs = env_obs_to_robomimic(obs)

            if not ep["reach_done"]:
                # --- Phase 1: Reach ---
                predicted_action = reach_policy(rob_obs)
                arm_action = torch.tensor(
                    predicted_action[: len(arm_joint_ids_isaac)],
                    device=env.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                base_action = torch.tensor(
                    predicted_action[
                        len(arm_joint_ids_isaac) : len(arm_joint_ids_isaac)
                        + len(base_joint_ids_isaac)
                    ],
                    device=env.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                # Gripper open during reach
                gripper_current = robot_entity.data.joint_pos[
                    :, gripper_joint_ids_isaac
                ]
                gripper_action = (
                    torch.tensor(
                        [[GRIPPER_OPEN_POS, GRIPPER_OPEN_POS]], device=env.device
                    )
                    - gripper_current
                )

                ep["reach_steps"] += 1

                # Check reach completion: handle close enough OR timeout
                # Use target_frame_view (drawer_handle_frame) to match the record script grasp point
                handle_pos_w, _ = target_frame_view.get_world_poses()
                handle_pos_w = torch.tensor(
                    handle_pos_w[0], device=env.device, dtype=torch.float32
                )
                grasp_pos_w = robot_entity.data.body_pos_w[0, grasp_body_idx]
                handle_dist = torch.norm(handle_pos_w - grasp_pos_w).item()

                if (
                    handle_dist < REACH_DONE_THRESHOLD
                    or ep["reach_steps"] >= args_cli.max_reach_steps
                ):
                    reason = (
                        "proximity" if handle_dist < REACH_DONE_THRESHOLD else "timeout"
                    )
                    print(
                        f"--> Reach done ({reason}, dist={handle_dist:.3f}m, "
                        f"steps={ep['reach_steps']}). Closing gripper..."
                    )
                    ep["reach_done"] = True
                    ep["gripper_timer"] = 0
                    pull_policy.start_episode()

            elif ep["gripper_timer"] < GRIPPER_LOCK_STEPS:
                # --- Gripper close transition ---
                arm_action = torch.zeros(1, len(arm_joint_ids_isaac), device=env.device)
                base_action = torch.zeros(
                    1, len(base_joint_ids_isaac), device=env.device
                )
                gripper_current = robot_entity.data.joint_pos[
                    :, gripper_joint_ids_isaac
                ]
                gripper_action = (
                    torch.tensor(
                        [[GRIPPER_CLOSE_POS, GRIPPER_CLOSE_POS]], device=env.device
                    )
                    - gripper_current
                )
                ep["gripper_timer"] += 1
                if ep["gripper_timer"] == GRIPPER_LOCK_STEPS:
                    print("--> Gripper locked. Starting pull phase...")

            else:
                # --- Phase 2: Pull ---
                predicted_action = pull_policy(rob_obs)
                arm_action = torch.tensor(
                    predicted_action[: len(arm_joint_ids_isaac)],
                    device=env.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                base_action = torch.tensor(
                    predicted_action[
                        len(arm_joint_ids_isaac) : len(arm_joint_ids_isaac)
                        + len(base_joint_ids_isaac)
                    ],
                    device=env.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                # Keep gripper closed
                gripper_current = robot_entity.data.joint_pos[
                    :, gripper_joint_ids_isaac
                ]
                gripper_action = (
                    torch.tensor(
                        [[GRIPPER_CLOSE_POS, GRIPPER_CLOSE_POS]], device=env.device
                    )
                    - gripper_current
                )

                ep["pull_steps"] += 1

                # Check pull completion
                drawer_pos = cabinet_entity.data.joint_pos[0].max().item()
                if drawer_pos > DRAWER_OPEN_THRESHOLD:
                    print(f"--> Drawer open ({drawer_pos:.3f}m). Pull done!")
                    ep["pull_done"] = True

                if ep["pull_steps"] >= args_cli.max_pull_steps and not ep["pull_done"]:
                    print(
                        f"--> Pull phase timeout at {ep['pull_steps']} steps. "
                        f"Drawer at {drawer_pos:.3f}m."
                    )
                    ep["pull_done"] = True

        else:
            # Still in head alignment — zero actions
            arm_action = torch.zeros(1, len(arm_joint_ids_isaac), device=env.device)
            base_action = torch.zeros(1, len(base_joint_ids_isaac), device=env.device)
            gripper_current = robot_entity.data.joint_pos[:, gripper_joint_ids_isaac]
            gripper_action = (
                torch.tensor([[GRIPPER_OPEN_POS, GRIPPER_OPEN_POS]], device=env.device)
                - gripper_current
            )

        env_actions = torch.cat([arm_action, base_action, gripper_action], dim=-1)

        # ==========================================
        # DEBUG PRINTS (every 10 steps)
        # ==========================================
        if ep["step_count"] % 10 == 0 and ep["head_tracking_done"]:
            arm_pos = robot_entity.data.joint_pos[0, arm_joint_ids_isaac]
            base_pos = robot_entity.data.joint_pos[0, base_joint_ids_isaac]
            grip_pos = robot_entity.data.joint_pos[0, gripper_joint_ids_isaac]
            handle_pos_dbg, _ = target_frame_view.get_world_poses()
            handle_pos_dbg = torch.tensor(
                handle_pos_dbg[0], device=env.device, dtype=torch.float32
            )
            grasp_pos_dbg = robot_entity.data.body_pos_w[0, grasp_body_idx]
            ee_dist = torch.norm(handle_pos_dbg - grasp_pos_dbg).item()
            phase = (
                "reach"
                if not ep["reach_done"]
                else (
                    "gripper_close"
                    if ep["gripper_timer"] < GRIPPER_LOCK_STEPS
                    else "pull"
                )
            )
            print(
                f"\n[Step {ep['step_count']} | phase={phase} | EE→handle dist={ee_dist:.4f}m] --- Debug ---"
            )
            print("  Actions:")
            for name, val in zip(ARM_JOINT_NAMES, arm_action[0].tolist()):
                print(f"    arm   {name}: {val:+.4f}")
            for name, val in zip(BASE_JOINT_NAMES, base_action[0].tolist()):
                print(f"    base  {name}: {val:+.4f}")
            for name, val in zip(GRIPPER_JOINT_NAMES, gripper_action[0].tolist()):
                print(f"    grip  {name}: {val:+.4f}")
            print("  Joint Positions:")
            for name, val in zip(ARM_JOINT_NAMES, arm_pos.tolist()):
                print(f"    arm   {name}: {val:+.4f}")
            for name, val in zip(BASE_JOINT_NAMES, base_pos.tolist()):
                print(f"    base  {name}: {val:+.4f}")
            for name, val in zip(GRIPPER_JOINT_NAMES, grip_pos.tolist()):
                print(f"    grip  {name}: {val:+.4f}")

        next_obs, _, terminated, truncated, _ = env.step(env_actions)
        obs = next_obs
        ep["step_count"] += 1

        if video_recorder is not None and ep["head_tracking_done"]:
            video_recorder.capture_frame()

        # ==========================================
        # EPISODE DONE CHECK
        # ==========================================
        if ep["pull_done"]:
            final_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
            displacement = torch.norm(final_pos - initial_handle_pos).item()
            success = displacement > HANDLE_DISPLACEMENT_THRESHOLD

            episodes_done += 1
            if success:
                successes += 1

            elapsed = time.time() - sim_start
            print(
                f"[Episode {episodes_done}/{args_cli.num_episodes}] "
                f"{'SUCCESS' if success else 'FAIL'} | "
                f"Displacement: {displacement:.3f}m | "
                f"Reach: {ep['reach_steps']} steps | Pull: {ep['pull_steps']} steps | "
                f"{elapsed:.0f}s elapsed"
            )

            if video_recorder is not None:
                result_tag = "success" if success else "fail"
                video_recorder.save(f"bc_rnn_ep{episodes_done:02d}_{result_tag}.mp4")

            # --- Reset for next episode ---
            obs, _ = env.reset()
            randomize_cabinet(
                cabinet_view,
                cabinet_entity,
                default_cabinet_pos,
                default_cabinet_rot,
                env.device,
                args_cli.noise_range,
            )
            initial_handle_pos = cabinet_entity.data.body_pos_w[
                0, handle_body_idx
            ].clone()
            ep = reset_episode_state()
            reach_policy.start_episode()
            pull_policy.start_episode()

    # ==========================================
    # SUMMARY
    # ==========================================
    total_time = time.time() - sim_start
    print(f"\n{'='*50}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Episodes:  {episodes_done}")
    print(f"  Successes: {successes}")
    print(f"  Rate:      {successes / max(episodes_done, 1) * 100:.1f}%")
    print(f"  Time:      {total_time:.0f}s")
    print(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()
