import os
import sys
import torch
import argparse
import time

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained Diffusion Policy")
parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to trained Diffusion Policy checkpoint (.pth)",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1,
    help="Number of episodes to run",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=600,
    help="Max steps per episode",
)
parser.add_argument(
    "--action_horizon",
    type=int,
    default=4,
    help="Number of actions to execute per inference (action chunking)",
)
parser.add_argument(
    "--drawer_threshold",
    type=float,
    default=0.30,
    help="Drawer position threshold to consider success",
)
parser.add_argument(
    "--record_video",
    action="store_true",
    default=False,
    help="Record video of each episode",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="/home/johnchen/SharedSSD/JohnChen/stretch/diffusion_videos",
    help="Directory to save recorded videos",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=60,
    help="Frames per second for recorded video",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(target_config_dir)

from delta_action_diff_stretch_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, quat_inv, quat_apply
from isaacsim.core.prims import XFormPrim

TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"
PULL_OFFSET = [0.45, 0.0, 0.03]  # must match dp_record.py PHASE_OFFSETS["pull"]

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import numpy as np
import imageio


def compute_pull_goal(target_frame_view, cabinet_view, robot_entity, device):
    """Compute pull goal in robot base frame — identical to dp_record.py compute_phase_targets."""
    curr_pos, _ = target_frame_view.get_world_poses()
    _, cabinet_rot = cabinet_view.get_world_poses()

    robot_pos_w = robot_entity.data.root_state_w[0:1, :3]
    robot_quat_w = robot_entity.data.root_state_w[0:1, 3:7]

    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    offset_t = torch.tensor([PULL_OFFSET], device=device)

    pull_pos_w, _ = combine_frame_transforms(
        curr_pos[0:1], cabinet_rot, offset_t, identity_quat
    )
    pull_pos_r = quat_apply(quat_inv(robot_quat_w), pull_pos_w - robot_pos_w)
    return pull_pos_r[0].cpu().numpy()  # (3,)


# ==========================================
# VIDEO RECORDER
# ==========================================
class VideoRecorder:
    """Video recorder using omni.replicator annotator for Isaac Sim."""

    def __init__(self, video_dir, fps=60, camera_prim="/OmniverseKit_Persp"):
        self.video_dir = video_dir
        self.fps = fps
        self.frames = []
        os.makedirs(video_dir, exist_ok=True)

        import omni.replicator.core as rep

        self.render_product = rep.create.render_product(
            camera_prim,
            resolution=(1280, 720),
        )
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

    def capture_frame(self):
        try:
            data = self.rgb_annotator.get_data()
            if data is not None and data.size > 0:
                frame = np.array(data)
                if frame.ndim == 3:
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    self.frames.append(frame)
            elif len(self.frames) == 0:
                print(f"[Video] Warning: annotator returned empty data (frame {len(self.frames)})")
        except Exception as e:
            print(f"[Video] Warning: frame capture failed: {e}")

    def save(self, filename):
        if len(self.frames) == 0:
            print("[Video] No frames captured, skipping save.")
            return None
        filepath = os.path.join(self.video_dir, filename)
        print(f"[Video] Saving {len(self.frames)} frames to {filepath}")
        imageio.mimwrite(filepath, self.frames, fps=self.fps)
        self.frames = []
        return filepath

    def reset(self):
        self.frames = []


# ==========================================
# POLICY LOADING
# ==========================================
def load_policy(ckpt_path, device):
    """Load a trained robomimic policy from a checkpoint."""
    print(f"[Policy] Loading: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=True,
    )

    algo_name = ckpt_dict.get("algo_name", "unknown")
    epoch = ckpt_dict.get("epoch", "unknown")
    print(f"  Algorithm: {algo_name}")
    print(f"  Epoch: {epoch}")

    policy.start_episode()
    return policy


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    env_cfg = StretchEnvCfg()
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Load single policy (full task)
    policy = load_policy(args_cli.ckpt, device)

    # Scene references
    cabinet_entity = env.scene["cabinet"]
    robot_entity = env.scene["robot"]

    # Joint indices for building current_pos (13D: 8 arm + 3 base + 2 gripper)
    arm_joint_names = [
        "joint_lift",
        "joint_arm_l0",
        "joint_arm_l1",
        "joint_arm_l2",
        "joint_arm_l3",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]
    base_joint_names = ["joint_x", "joint_y", "joint_rot_z"]
    gripper_joint_names = ["joint_gripper_finger_left", "joint_gripper_finger_right"]

    arm_joint_ids = robot_entity.find_joints(arm_joint_names)[0]
    base_joint_ids = robot_entity.find_joints(base_joint_names)[0]
    gripper_joint_ids = robot_entity.find_joints(gripper_joint_names)[0]

    # XFormPrim views for pull goal computation (matches dp_record.py exactly)
    target_frame_view = XFormPrim(TARGET_FRAME_PATH, name="target_frame")
    cabinet_view = XFormPrim("/World/envs/env_0/Cabinet", name="cabinet")

    # Video recorder
    recorder = None
    if args_cli.record_video:
        print(f"[Video] Recording enabled. Saving to: {args_cli.video_dir}")
        recorder = VideoRecorder(args_cli.video_dir, fps=args_cli.video_fps)

    # ==========================================
    # Episode Loop
    # ==========================================
    successes = 0
    total_episodes = args_cli.num_episodes
    sim_start_time = time.time()

    for ep_idx in range(total_episodes):
        print(f"\n{'='*60}")
        print(f"  EPISODE {ep_idx + 1}/{total_episodes}")
        print(f"{'='*60}")

        obs, _ = env.reset()
        policy.start_episode()
        if recorder is not None:
            recorder.reset()

        # Compute pull goal ONCE at episode start (matches dp_record.py behavior)
        episode_pull_goal = compute_pull_goal(target_frame_view, cabinet_view, robot_entity, env.device)
        print(f"  [INIT] Pull goal (robot frame): {episode_pull_goal}")

        step_count = 0
        episode_success = False
        action_queue = []  # buffered actions from action chunking

        # Print initial state
        ee_pos = robot_entity.data.body_pos_w[
            0, robot_entity.data.body_names.index("link_grasp_center")
        ]
        handle_pos = cabinet_entity.data.body_pos_w[
            0, cabinet_entity.data.body_names.index("drawer_handle_top")
        ]
        dist = torch.norm(ee_pos - handle_pos).item()
        print(f"  [INIT] EE pos:     {ee_pos.tolist()}")
        print(f"  [INIT] Handle pos: {handle_pos.tolist()}")
        print(f"  [INIT] EE→Handle:  {dist:.4f}m")

        # ------------------------------------------
        # Main Loop with Action Chunking
        # ------------------------------------------
        while step_count < args_cli.max_steps:

            # Get new action chunk if queue is empty
            if len(action_queue) == 0:
                obs_np = obs["policy"][0].cpu().numpy()  # shape (28,)
                obs_dict = {
                    "policy": obs_np[np.newaxis, :],
                    "handle_target": episode_pull_goal[np.newaxis, :],
                }
                action_chunk = policy(obs_dict)

                if step_count == 0:
                    print(f"  [POLICY] action_chunk shape: {action_chunk.shape}, ndim: {action_chunk.ndim}")
                    first = action_chunk[0] if action_chunk.ndim == 2 else action_chunk
                    print(f"  [POLICY] first action arm:{first[:8].round(4).tolist()} base:{first[8:11].round(4).tolist()} grip:{first[11:].round(4).tolist()}")

                # Diffusion Policy returns (prediction_horizon, action_dim)
                # Take the first action_horizon actions
                if action_chunk.ndim == 2:
                    for i in range(min(args_cli.action_horizon, len(action_chunk))):
                        action_queue.append(action_chunk[i])
                else:
                    # Single action (fallback for BC-RNN compatibility)
                    action_queue.append(action_chunk)

            # Pop next delta action from queue
            delta_action = action_queue.pop(0)
            delta_tensor = torch.tensor(delta_action, device=env.device).unsqueeze(0)

            # Convert delta to absolute: current_pos + delta
            current_pos = torch.cat(
                [
                    robot_entity.data.joint_pos[:, arm_joint_ids],
                    robot_entity.data.joint_pos[:, base_joint_ids],
                    robot_entity.data.joint_pos[:, gripper_joint_ids],
                ],
                dim=-1,
            )

            absolute_action = current_pos + delta_tensor

            # Step environment with absolute joint positions
            obs, _, _, _, _ = env.step(absolute_action)
            step_count += 1

            if recorder is not None:
                simulation_app.update()
                recorder.capture_frame()

            # Debug prints every 10 steps
            if step_count % 10 == 0:
                ee_pos = robot_entity.data.body_pos_w[
                    0, robot_entity.data.body_names.index("link_grasp_center")
                ]
                handle_pos = cabinet_entity.data.body_pos_w[
                    0, cabinet_entity.data.body_names.index("drawer_handle_top")
                ]
                dist = torch.norm(ee_pos - handle_pos).item()
                drawer_pos = cabinet_entity.data.joint_pos[0]
                drawer_max = drawer_pos.max().item()
                base_pos_rel = obs["policy"][0, 8:11].cpu().numpy()
                handle_rel = obs["policy"][0, 22:25].cpu().numpy()
                print(
                    f"  [DEBUG] Step {step_count}\n"
                    f"    arm:    {delta_tensor[0, 0:8].tolist()}\n"
                    f"    base:   {delta_tensor[0, 8:11].tolist()}  ← (x, y, rot)\n"
                    f"    gripper:{delta_tensor[0, 11:13].tolist()}\n"
                    f"    obs base_pos:   {base_pos_rel.round(4)}\n"
                    f"    obs handle_rel: {handle_rel.round(4)}\n"
                    f"    EE pos:       {ee_pos.tolist()}\n"
                    f"    EE→Handle:    {dist:.4f}m\n"
                    f"    Drawer:       {drawer_max:.4f}m"
                )

            # Check drawer
            drawer_max = cabinet_entity.data.joint_pos[0].max().item()
            if drawer_max > args_cli.drawer_threshold:
                print(f"  --> Drawer open ({drawer_max:.3f}m) at step {step_count}!")
                episode_success = True
                break

        # ------------------------------------------
        # Episode Summary
        # ------------------------------------------
        final_drawer = cabinet_entity.data.joint_pos[0].max().item()
        if final_drawer > args_cli.drawer_threshold:
            episode_success = True

        status = "SUCCESS" if episode_success else "FAIL"
        if episode_success:
            successes += 1

        print(f"\n  Result: {status}")
        print(f"  Final drawer position: {final_drawer:.3f}m")
        print(f"  Total steps: {step_count}")
        print(f"  [{successes}/{ep_idx + 1} successful]")

        if recorder is not None:
            video_name = f"episode_{ep_idx + 1}_{status.lower()}.mp4"
            recorder.save(video_name)

    # ==========================================
    # Final Summary
    # ==========================================
    total_time = time.time() - sim_start_time
    success_rate = successes / total_episodes * 100

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes:     {total_episodes}")
    print(f"  Successes:    {successes}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
