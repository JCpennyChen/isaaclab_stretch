"""
test_curobo.py — CuRobo Phase 1 (Safe Spot) Path Visualizer for Stretch

Plans a single motion to the "safe spot" in front of the drawer handle
(Phase 1 from curobo_stretch.py) and visualizes the planned EEF path
using USD sphere prims.

Usage:
  python test_curobo.py
  python test_curobo.py --playback
  python test_curobo.py --headless
"""

import os
import sys
import torch
import numpy as np
import argparse

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stretch + CuRobo Phase 1 Visualizer")
parser.add_argument(
    "--headless", action="store_true", default=False, help="Force display off"
)
parser.add_argument(
    "--playback",
    action="store_true",
    default=False,
    help="Play the trajectory on the robot after visualizing",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if target_config_dir not in sys.path:
    sys.path.append(target_config_dir)

from isaaclab.envs import ManagerBasedRLEnv
from stretch_bc_rnn_cfg import StretchEnvCfg
from isaacsim.core.prims import XFormPrim
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_inv,
    quat_apply,
    quat_mul,
    axis_angle_from_quat,
)

from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import load_yaml
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

# ==========================================
# PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CUROBO_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "assets", "robot_configs", "stretch_fake_joint.yml"
)
ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets")

# Phase 1 target: same prim paths and offset as curobo_stretch.py
TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"
CABINET_PRIM_PATH = "/World/envs/env_0/Cabinet"
SAFE_SPOT_OFFSET = [0.05, 0.0, 0.03]


# ==========================================
# FK PATH COMPUTATION
# ==========================================
def compute_eef_path_from_trajectory(motion_gen, trajectory):
    """
    Run FK on every trajectory waypoint to get the EEF path in robot base frame.

    IMPORTANT DETAILS:
    1) get_full_js() returns 15 joints (11 active + 4 locked), but
       kinematics.forward() expects only the 11 active joints.
       We must extract only the active joint columns.
    2) CuRobo's FK reuses internal CUDA buffers, so each call to forward()
       overwrites the previous output. We must .clone() the results.
    """
    curobo_joint_names = motion_gen.kinematics.joint_names  # 11 active joints
    traj_joint_names = trajectory.joint_names  # 15 joints (active + locked)

    # Build index mapping: for each CuRobo active joint, find its column
    # in the full trajectory
    joint_idx_map = [traj_joint_names.index(c) for c in curobo_joint_names]

    positions = trajectory.position  # (T, 15)
    active_positions = positions[:, joint_idx_map]  # (T, 11)

    eef_pos_list = []
    eef_quat_list = []

    for t in range(active_positions.shape[0]):
        fk_out = motion_gen.kinematics.forward(active_positions[t : t + 1])
        # fk_out is a tuple of 7 elements:
        #   [0] = ee position  (1, 3)
        #   [1] = ee quaternion (1, 4)
        #   [2-6] = other data (link poses, collision spheres, etc.)
        #
        # .clone() is CRITICAL: CuRobo reuses internal CUDA buffers,
        # so without clone, all entries point to the same (last) value.
        eef_pos_list.append(fk_out[0][0].clone())
        eef_quat_list.append(fk_out[1][0].clone())

    eef_positions = torch.stack(eef_pos_list, dim=0)  # (T, 3)
    eef_quats = torch.stack(eef_quat_list, dim=0)  # (T, 4)

    print(f"  [FK] EEF pos[0]:   {eef_positions[0].cpu().numpy().round(4)}")
    print(
        f"  [FK] EEF pos[mid]: "
        f"{eef_positions[len(eef_positions)//2].cpu().numpy().round(4)}"
    )
    print(f"  [FK] EEF pos[-1]:  {eef_positions[-1].cpu().numpy().round(4)}")

    return eef_positions, eef_quats


# ==========================================
# MAIN
# ==========================================
def main():
    # ------------------------------------------------------------------
    # STEP 1: Create the Stretch environment
    # ------------------------------------------------------------------
    env_cfg = StretchEnvCfg()
    env_cfg.viewer.eye = (2.0, 2.0, 2.0)
    env_cfg.episode_length_s = 10000.0

    print("=" * 60)
    print("[Step 1] Creating Stretch environment")
    print("=" * 60)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

    robot_entity = env.scene["robot"]
    eef_idx = robot_entity.find_bodies("link_grasp_center")[0][0]
    base_joint_ids_isaac = robot_entity.find_joints(
        ["joint_x", "joint_y", "joint_rot_z"]
    )[0]

    # ------------------------------------------------------------------
    # STEP 2: Initialize CuRobo MotionGen
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 2] Initializing CuRobo MotionGen")
    print("=" * 60)

    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType(device=env.device)

    robot_cfg = load_yaml(CUROBO_CONFIG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSET_ROOT

    dummy_world = WorldConfig(
        cuboid=[
            Cuboid(
                "startup_dummy",
                pose=[0, 0, -10.0, 1, 0, 0, 0],
                dims=[1.0, 1.0, 1.0],
            )
        ]
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        dummy_world,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        optimize_dt=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        trajopt_tsteps=40,
        interpolation_dt=0.0167,
    )
    motion_gen = MotionGen(motion_gen_config)

    if (
        hasattr(motion_gen, "self_collision_checker")
        and motion_gen.self_collision_checker is not None
    ):
        motion_gen.self_collision_checker.min_dist = 0.005

    print("  Warming up planner...")
    motion_gen.warmup(enable_graph=True)
    print("  Warmup complete!")

    # ------------------------------------------------------------------
    # STEP 3: Compute Phase 1 target from the drawer handle
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 3] Computing Phase 1 target (safe spot near drawer handle)")
    print("=" * 60)

    # Let the sim settle
    for _ in range(10):
        zero_action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
        env.step(zero_action)

    # Read handle and cabinet poses from the USD stage
    target_frame_view = XFormPrim(TARGET_FRAME_PATH, name="target_frame")
    cabinet_view = XFormPrim(CABINET_PRIM_PATH, name="cabinet")

    handle_pos_w, _ = target_frame_view.get_world_poses()
    _, cabinet_rot_w = cabinet_view.get_world_poses()

    print(f"  Handle world position:  {handle_pos_w}")
    print(f"  Cabinet world rotation: {cabinet_rot_w}")

    # Apply the safe spot offset in the cabinet's local frame
    front_offset = torch.tensor([SAFE_SPOT_OFFSET], device=env.device)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)

    target_pos_w, _ = combine_frame_transforms(
        handle_pos_w[0:1], cabinet_rot_w, front_offset, identity_quat
    )

    # Fixed gripper orientation (same as curobo_stretch.py)
    target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=env.device)

    print(f"  Phase 1 target (world): {target_pos_w}")

    # Transform into robot base frame
    robot_pos_w = robot_entity.data.root_state_w[0:1, :3]
    robot_quat_w = robot_entity.data.root_state_w[0:1, 3:7]

    rel_pos = target_pos_w - robot_pos_w
    target_pos_r = quat_apply(quat_inv(robot_quat_w), rel_pos)
    target_quat_r = quat_mul(quat_inv(robot_quat_w), target_quat_w)

    print(f"  Robot base (world):     {robot_pos_w}")
    print(f"  Target pos (robot):     {target_pos_r}")
    print(f"  Target quat (robot):    {target_quat_r}")

    target_pose = Pose(position=target_pos_r, quaternion=target_quat_r)

    # ------------------------------------------------------------------
    # STEP 4: Build start state and plan
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 4] Planning Phase 1 (safe spot approach)")
    print("=" * 60)

    start_joint_pos = robot_entity.data.joint_pos[0].clone()
    start_joint_vel = robot_entity.data.joint_vel[0].clone()

    cu_js = JointState(
        position=start_joint_pos.unsqueeze(0),
        velocity=start_joint_vel.unsqueeze(0) * 0.0,
        acceleration=start_joint_vel.unsqueeze(0) * 0.0,
        joint_names=robot_entity.joint_names,
    ).get_ordered_joint_state(motion_gen.kinematics.joint_names)

    print(f"  Start joints (CuRobo order): {cu_js.position.cpu().numpy().round(4)}")

    # Phase 1 uses the full planner (not linear mode)
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_finetune_trajopt=True,
        max_attempts=10,
        time_dilation_factor=0.5,
        enable_graph_attempt=2,
    )

    result = motion_gen.plan_single(cu_js, target_pose, plan_config)

    if not result.success.item():
        print(f"\n  *** PHASE 1 PLANNING FAILED ***")
        print(f"  Status: {result.status}")
        if "COLLISION" in str(result.status):
            print("  The robot is likely starting in collision.")
        print("\n  Keeping sim alive for inspection...")
        while simulation_app.is_running():
            zero_action = torch.zeros(
                (1, env.action_space.shape[-1]), device=env.device
            )
            env.step(zero_action)
        env.close()
        return

    trajectory = result.get_interpolated_plan()
    trajectory = motion_gen.get_full_js(trajectory)
    traj_len = trajectory.position.shape[0]

    print(f"\n  PHASE 1 PLAN SUCCESS!")
    print(f"  Trajectory waypoints: {traj_len}")

    # ------------------------------------------------------------------
    # STEP 5: Visualize the planned EEF path
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 5] Visualizing planned path (USD spheres)")
    print("=" * 60)

    eef_positions, eef_quats = compute_eef_path_from_trajectory(motion_gen, trajectory)

    # Convert from robot base frame -> world frame for visualization
    eef_positions_world = []
    for i in range(eef_positions.shape[0]):
        p_r = eef_positions[i].unsqueeze(0)
        p_w = quat_apply(robot_quat_w, p_r) + robot_pos_w
        eef_positions_world.append(p_w.squeeze(0).cpu().numpy())
    eef_positions_world = np.array(eef_positions_world)

    # Target and start in world frame
    target_w_np = target_pos_w.squeeze(0).cpu().numpy()
    start_eef_w = robot_entity.data.body_pos_w[0, eef_idx].cpu().numpy()

    print(
        f"  Start EEF (world): "
        f"[{start_eef_w[0]:.3f}, {start_eef_w[1]:.3f}, {start_eef_w[2]:.3f}]"
    )
    print(
        f"  Target    (world): "
        f"[{target_w_np[0]:.3f}, {target_w_np[1]:.3f}, {target_w_np[2]:.3f}]"
    )

    # --- Debug: print path samples ---
    print(f"\n  --- EEF Path Debug (every 25th waypoint) ---")
    print(f"  {'Idx':>5s}  {'X':>8s}  {'Y':>8s}  {'Z':>8s}")
    print(f"  {'-'*35}")
    for i in range(0, len(eef_positions_world), 25):
        p = eef_positions_world[i]
        print(f"  {i:5d}  {p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}")
    p = eef_positions_world[-1]
    print(f"  {len(eef_positions_world)-1:5d}  {p[0]:8.4f}  {p[1]:8.4f}  {p[2]:8.4f}")

    # --- Sanity check ---
    path_start = eef_positions_world[0]
    path_end = eef_positions_world[-1]
    dist_start_to_eef = np.linalg.norm(path_start - start_eef_w)
    dist_end_to_target = np.linalg.norm(path_end - target_w_np)
    total_path_length = sum(
        np.linalg.norm(eef_positions_world[i + 1] - eef_positions_world[i])
        for i in range(len(eef_positions_world) - 1)
    )
    print(f"\n  --- Path Sanity Check ---")
    print(f"  Path start vs actual EEF:   {dist_start_to_eef:.4f} m (should be ~0)")
    print(f"  Path end vs target:         {dist_end_to_target:.4f} m (should be ~0)")
    print(f"  Total path length:          {total_path_length:.4f} m")
    print(
        f"  Straight-line distance:     "
        f"{np.linalg.norm(target_w_np - start_eef_w):.4f} m"
    )

    # --- Create USD sphere visualization ---
    create_path_spheres(
        eef_positions_world, name_prefix="path", radius=0.015, color=(0, 1, 0)
    )
    create_marker_sphere(
        start_eef_w, name="start_marker", radius=0.03, color=(0, 0.3, 1)
    )
    create_marker_sphere(
        target_w_np, name="target_marker", radius=0.03, color=(1, 0, 0)
    )

    print("\n  Legend:")
    print("    GREEN spheres = planned EEF path")
    print("    BLUE  sphere  = start EEF")
    print("    RED   sphere  = safe spot target (near handle)")

    # ------------------------------------------------------------------
    # STEP 7: Keep sim alive
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Done] Phase 1 visualization complete.")
    print("       Close the window or Ctrl+C to exit.")
    print("=" * 60)

    while simulation_app.is_running():
        base_action = robot_entity.data.joint_pos[:, base_joint_ids_isaac]
        arm_action = torch.zeros((1, 6), device=env.device)
        gripper_action = torch.tensor([[0.1, 0.1]], device=env.device)
        env_actions = torch.cat([arm_action, base_action, gripper_action], dim=-1)
        env.step(env_actions)

    env.close()


if __name__ == "__main__":
    main()
