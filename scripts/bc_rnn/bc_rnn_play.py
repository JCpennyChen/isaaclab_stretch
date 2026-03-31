"""
play_imitation.py

Deploy trained BC-RNN policies in the Stretch cabinet environment.
Runs the reach policy, then switches to the pull policy.
Optionally records video of each episode.

Usage:
    ../IsaacLab/isaaclab.sh -p scripts/play_imitation.py \
        --reach_ckpt path/to/reach/best.pth \
        --pull_ckpt path/to/pull/best.pth

    # Record video
    ../IsaacLab/isaaclab.sh -p scripts/play_imitation.py \
        --reach_ckpt path/to/reach/best.pth \
        --pull_ckpt path/to/pull/best.pth \
        --record_video
"""

import os
import sys
import torch
import argparse
import time

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

# Add ALL custom arguments BEFORE AppLauncher args
parser = argparse.ArgumentParser(description="Play trained BC-RNN policies")
parser.add_argument(
    "--reach_ckpt",
    type=str,
    required=True,
    help="Path to trained reach phase checkpoint (.pth)",
)
parser.add_argument(
    "--pull_ckpt",
    type=str,
    default=None,
    help="Path to trained pull phase checkpoint (.pth). If omitted, only reach is run.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1,
    help="Number of episodes to run",
)
parser.add_argument(
    "--reach_steps",
    type=int,
    default=450,
    help="Max steps for reach phase before switching to pull",
)
parser.add_argument(
    "--pull_steps",
    type=int,
    default=150,
    help="Max steps for pull phase",
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
    default="/home/johnchen/SharedSSD/JohnChen/stretch/videos",
    help="Directory to save recorded videos",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=30,
    help="Frames per second for recorded video",
)

# Add AppLauncher args AFTER custom args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras if recording video
if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(target_config_dir)

from bc_rnn_stretch_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import numpy as np
import imageio


# ==========================================
# VIDEO RECORDER
# ==========================================
class VideoRecorder:
    """Video recorder using omni.replicator annotator for Isaac Sim."""

    def __init__(self, video_dir, fps=30):
        self.video_dir = video_dir
        self.fps = fps
        self.frames = []
        os.makedirs(video_dir, exist_ok=True)

        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport

        self.viewport = get_active_viewport()
        self.render_product = rep.create.render_product(
            self.viewport.get_active_camera(),
            resolution=(1280, 720),
        )
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

    def capture_frame(self):
        """Capture the current frame via replicator annotator."""
        try:
            data = self.rgb_annotator.get_data()
            if data is not None and data.size > 0:
                frame = np.array(data)
                if frame.ndim == 3:
                    # Drop alpha channel if present (RGBA -> RGB)
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    self.frames.append(frame)
        except Exception as e:
            if len(self.frames) == 0:
                print(f"[Video] Warning: frame capture failed: {e}")

    def save(self, filename):
        """Save captured frames to an mp4 file."""
        if len(self.frames) == 0:
            print("[Video] No frames captured, skipping save.")
            return None

        filepath = os.path.join(self.video_dir, filename)
        print(f"[Video] Saving {len(self.frames)} frames to {filepath}")
        imageio.mimwrite(filepath, self.frames, fps=self.fps)
        self.frames = []
        return filepath

    def reset(self):
        """Clear captured frames."""
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

    # Load policies
    reach_policy = load_policy(args_cli.reach_ckpt, device)

    pull_policy = None
    if args_cli.pull_ckpt is not None:
        pull_policy = load_policy(args_cli.pull_ckpt, device)

    # Scene references
    cabinet_entity = env.scene["cabinet"]
    robot_entity = env.scene["robot"]

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
        reach_policy.start_episode()
        if pull_policy is not None:
            pull_policy.start_episode()
        if recorder is not None:
            recorder.reset()

        step_count = 0
        current_phase = "reach"
        episode_success = False

        # Print initial state
        root_pos = robot_entity.data.root_state_w[0, :3]
        root_quat = robot_entity.data.root_state_w[0, 3:7]
        ee_pos = robot_entity.data.body_pos_w[
            0, robot_entity.data.body_names.index("link_grasp_center")
        ]
        handle_pos = cabinet_entity.data.body_pos_w[
            0, cabinet_entity.data.body_names.index("drawer_handle_top")
        ]
        dist = torch.norm(ee_pos - handle_pos).item()
        print(f"  [INIT] Robot pos:  {root_pos.tolist()}")
        print(f"  [INIT] Robot quat: {root_quat.tolist()}")
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
            action_tensor = torch.tensor(action, device=env.device).unsqueeze(0)

            obs, _, _, _, _ = env.step(action_tensor)
            step_count += 1

            if recorder is not None:
                recorder.capture_frame()

            if step_count % 10 == 0:
                root_pos = robot_entity.data.root_state_w[0, :3]
                root_quat = robot_entity.data.root_state_w[0, 3:7]
                joint_pos = robot_entity.data.joint_pos[0]
                ee_pos = robot_entity.data.body_pos_w[
                    0, robot_entity.data.body_names.index("link_grasp_center")
                ]
                handle_pos = cabinet_entity.data.body_pos_w[
                    0, cabinet_entity.data.body_names.index("drawer_handle_top")
                ]
                dist = torch.norm(ee_pos - handle_pos).item()
                print(
                    f"  [DEBUG] Step {step_count} | Phase: {current_phase}\n"
                    f"    Action:     {action_tensor[0].tolist()}\n"
                    f"    Robot pos:  {root_pos.tolist()}\n"
                    f"    Robot quat: {root_quat.tolist()}\n"
                    f"    Joint pos:  {joint_pos.tolist()}\n"
                    f"    EE pos:     {ee_pos.tolist()}\n"
                    f"    Handle pos: {handle_pos.tolist()}\n"
                    f"    EE→Handle:  {dist:.4f}m"
                )

        print(f"  Reach phase completed ({args_cli.reach_steps} steps)")

        # ------------------------------------------
        # PULL PHASE
        # ------------------------------------------
        if pull_policy is not None:
            current_phase = "pull"
            print("[Phase] Running PULL policy...")

            for step in range(args_cli.pull_steps):
                obs_dict = {"policy": obs["policy"][0].cpu().numpy()}

                action = pull_policy(obs_dict)
                action_tensor = torch.tensor(action, device=env.device).unsqueeze(0)

                obs, _, _, _, _ = env.step(action_tensor)
                step_count += 1

                if recorder is not None:
                    recorder.capture_frame()

                # Check drawer position
                drawer_pos = cabinet_entity.data.joint_pos[0]
                drawer_max = drawer_pos.max().item()

                if step_count % 10 == 0:
                    root_pos = robot_entity.data.root_state_w[0, :3]
                    root_quat = robot_entity.data.root_state_w[0, 3:7]
                    joint_pos = robot_entity.data.joint_pos[0]
                    ee_pos = robot_entity.data.body_pos_w[
                        0, robot_entity.data.body_names.index("link_grasp_center")
                    ]
                    handle_pos = cabinet_entity.data.body_pos_w[
                        0, cabinet_entity.data.body_names.index("drawer_handle_top")
                    ]
                    dist = torch.norm(ee_pos - handle_pos).item()
                    print(
                        f"  [DEBUG] Step {step_count} | Phase: {current_phase}\n"
                        f"    Action:     {action_tensor[0].tolist()}\n"
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

            print(f"  Pull phase completed ({step + 1} steps)")

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

        # Save episode video
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
