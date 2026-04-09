"""
student_policy_play.py

Deploy trained student policies (CNN+MLP) in the Stretch cabinet environment.
Runs the reach policy, then switches to the pull policy.
Optionally records video of each episode.

Usage:
    ../IsaacLab/isaaclab.sh -p scripts/bc_rnn/student_policy_play.py \
        --reach_ckpt checkpoints/student/reach/best.pth \
        --pull_ckpt checkpoints/student/pull/best.pth

    # Record video
    ../IsaacLab/isaaclab.sh -p scripts/bc_rnn/student_policy_play.py \
        --reach_ckpt checkpoints/student/reach/best.pth \
        --pull_ckpt checkpoints/student/pull/best.pth \
        --record_video
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained student policies")
parser.add_argument(
    "--reach_ckpt",
    type=str,
    required=True,
    help="Path to trained reach student checkpoint",
)
parser.add_argument(
    "--pull_ckpt",
    type=str,
    default=None,
    help="Path to trained pull student checkpoint",
)
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument(
    "--reach_steps", type=int, default=175, help="Max reach phase steps"
)
parser.add_argument("--pull_steps", type=int, default=150, help="Max pull phase steps")
parser.add_argument("--drawer_threshold", type=float, default=0.35)
parser.add_argument(
    "--image_h",
    type=int,
    default=120,
    help="Resized image height (must match training)",
)
parser.add_argument(
    "--image_w", type=int, default=160, help="Resized image width (must match training)"
)
parser.add_argument("--record_video", action="store_true", default=False)
parser.add_argument(
    "--video_dir",
    type=str,
    default="/home/johnchen/SharedSSD/JohnChen/stretch/videos",
)
parser.add_argument("--video_fps", type=int, default=30)

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
from video_recorder import VideoRecorder
from isaaclab.envs import ManagerBasedRLEnv


# ==========================================
# Model (must match student_policy_train.py)
# ==========================================
class StudentPolicy(nn.Module):
    """Simple CNN image encoder + MLP policy."""

    def __init__(self, proprio_dim=24, action_dim=13, image_size=(120, 160)):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out_dim = 64 * 4 * 4

        mlp_in = cnn_out_dim + proprio_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, proprio, image):
        img_feat = self.cnn(image)
        x = torch.cat([img_feat, proprio], dim=-1)
        return self.mlp(x)


def load_student_policy(ckpt_path, device, image_size):
    """Load a trained student policy checkpoint."""
    print(f"[Policy] Loading: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    # Infer dims from checkpoint weights so this always matches training
    cnn_out_dim = 64 * 4 * 4  # 1024, matches the CNN architecture
    proprio_dim = ckpt["model"]["mlp.0.weight"].shape[1] - cnn_out_dim
    action_dim = ckpt["model"]["mlp.6.weight"].shape[0]
    model = StudentPolicy(
        proprio_dim=proprio_dim, action_dim=action_dim, image_size=image_size
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  Val loss: {ckpt.get('val_loss', 'unknown')}")
    return model


def prepare_image(head_cam, image_size, device):
    """Capture camera RGB and prepare as model input tensor."""
    rgb = head_cam.data.output["rgb"][0]  # (H, W, 3) or (H, W, 4)
    rgb_np = rgb.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[:, :, :3]

    # (H, W, 3) uint8 -> (1, 3, h, w) float
    img = rgb_np.astype(np.float32) / 255.0
    img = torch.tensor(img, device=device).permute(2, 0, 1).unsqueeze(0)
    img = nn.functional.interpolate(
        img, size=image_size, mode="bilinear", align_corners=False
    )
    return img


# ==========================================
# MAIN
# ==========================================
def main():
    env_cfg = StretchEnvCfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    image_size = (args_cli.image_h, args_cli.image_w)

    # Load student policies
    reach_policy = load_student_policy(args_cli.reach_ckpt, device, image_size)
    pull_policy = None
    if args_cli.pull_ckpt is not None:
        pull_policy = load_student_policy(args_cli.pull_ckpt, device, image_size)

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
        """Extract proprio as a single (1, 26) tensor — matches student_policy_train.py."""
        arm = robot_entity.data.joint_pos[:, arm_joint_ids]  # 8D
        base = robot_entity.data.joint_pos[:, base_joint_ids]  # 3D
        vel = robot_entity.data.joint_vel[:, all_vel_ids]  # 11D
        grip = robot_entity.data.joint_pos[:, gripper_joint_ids]  # 2D
        head = robot_entity.data.joint_pos[:, head_joint_ids]  # 2D (pan, tilt)
        return torch.cat([arm, base, vel, grip, head], dim=-1)  # (1, 26)

    # Video recorder
    recorder = None
    if args_cli.record_video:
        print(f"[Video] Recording enabled. Saving to: {args_cli.video_dir}")
        recorder = VideoRecorder(args_cli.video_dir, fps=args_cli.video_fps)

    # ==========================================
    # Episode Loop
    # ==========================================
    successes = 0
    sim_start = time.time()

    for ep_idx in range(args_cli.num_episodes):
        print(f"\n{'='*60}")
        print(f"  EPISODE {ep_idx + 1}/{args_cli.num_episodes}")
        print(f"{'='*60}")

        obs, _ = env.reset()
        if recorder is not None:
            recorder.reset()

        step_count = 0
        episode_success = False

        # ------------------------------------------
        # PHASE 0: HEAD CAMERA ALIGNMENT (no policy)
        # Match student_policy_record.py: settle head before policy runs
        # ------------------------------------------
        print("[Phase 0] Aligning head camera...")
        zero_action = torch.zeros(1, 13, device=device)
        head_settle_timer = 0
        while head_settle_timer < 10:
            head_target, _ = compute_head_look_at(
                robot_entity,
                cabinet_entity,
                head_body_idx,
                handle_body_idx,
                head_joint_ids,
            )
            robot_entity.set_joint_position_target(
                head_target.unsqueeze(0), joint_ids=head_joint_ids
            )
            head_current = robot_entity.data.joint_pos[0, head_joint_ids]
            head_error = torch.norm(head_target - head_current).item()
            if head_error < 0.05:
                head_settle_timer += 1
            else:
                head_settle_timer = 0
            obs, _, _, _, _ = env.step(zero_action)
            step_count += 1
        print(f"  Head aligned ({step_count} steps).")

        # ------------------------------------------
        # REACH PHASE
        # ------------------------------------------
        print("[Phase] Running REACH student policy...")
        for step in range(args_cli.reach_steps):
            # Build student observation
            proprio = get_proprioception()  # (1, 24)
            image = prepare_image(head_cam, image_size, device)  # (1, 3, H, W)

            with torch.no_grad():
                env_action = reach_policy(proprio, image)  # (1, 13)

            # Head tracks handle analytically (independent controller, not part of policy)
            head_target, _ = compute_head_look_at(
                robot_entity,
                cabinet_entity,
                head_body_idx,
                handle_body_idx,
                head_joint_ids,
            )
            robot_entity.set_joint_position_target(
                head_target.unsqueeze(0), joint_ids=head_joint_ids
            )

            obs, _, _, _, _ = env.step(env_action)
            step_count += 1

            if recorder is not None:
                recorder.capture_frame()

            if step_count % 50 == 0:
                ee_idx = robot_entity.data.body_names.index("link_grasp_center")
                ee_pos = robot_entity.data.body_pos_w[0, ee_idx]
                handle_pos = cabinet_entity.data.body_pos_w[0, handle_body_idx]
                dist = torch.norm(ee_pos - handle_pos).item()
                print(f"  [Step {step_count}] EE->Handle: {dist:.4f}m")

        print(f"  Reach phase completed ({step + 1} steps)")

        # ------------------------------------------
        # PULL PHASE
        # ------------------------------------------
        if pull_policy is not None:
            print("[Phase] Running PULL student policy...")

            for step in range(args_cli.pull_steps):
                proprio = get_proprioception()
                image = prepare_image(head_cam, image_size, device)

                with torch.no_grad():
                    env_action = pull_policy(proprio, image)  # (1, 13)

                # Head tracks handle analytically (independent controller, not part of policy)
                head_target, _ = compute_head_look_at(
                    robot_entity,
                    cabinet_entity,
                    head_body_idx,
                    handle_body_idx,
                    head_joint_ids,
                )
                robot_entity.set_joint_position_target(
                    head_target.unsqueeze(0), joint_ids=head_joint_ids
                )

                obs, _, _, _, _ = env.step(env_action)
                step_count += 1

                if recorder is not None:
                    recorder.capture_frame()

                drawer_max = cabinet_entity.data.joint_pos[0].max().item()

                if step_count % 50 == 0:
                    print(f"  [Step {step_count}] Drawer: {drawer_max:.4f}m")

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

        if recorder is not None:
            video_name = f"student_ep{ep_idx + 1}_{status.lower()}.mp4"
            recorder.save(video_name)

    # ==========================================
    # Final Summary
    # ==========================================
    total_time = time.time() - sim_start
    success_rate = successes / args_cli.num_episodes * 100

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes:     {args_cli.num_episodes}")
    print(f"  Successes:    {successes}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
