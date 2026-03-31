import os
import sys
import torch
import argparse
import time

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stretch + CuRobo Integration")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
sys.path.append(target_config_dir)

# ==========================================
# ISAAC SIM IMPORTS
# ==========================================
from bc_rnn_stretch_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaacsim.core.prims import XFormPrim
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_inv,
    quat_apply,
    quat_mul,
)

# ==========================================
# CUROBO IMPORTS
# ==========================================
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
# GLOBAL CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CUROBO_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "assets", "robot_configs", "stretch_fake_joint.yml"
)
ASSET_ROOT = os.path.join(PROJECT_ROOT, "assets")

TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"

PLANNER_CONFIG = {
    "max_attempts": 10,
    "time_dilation_factor": 0.5,
    "enable_graph_attempt": 2,
}

PHASE_OFFSETS = {
    "safe_spot": [0.05, 0.0, 0.03],
    "insert": [-0.05, 0.0, 0.03],
    "pull": [0.45, 0.0, 0.03],
}

LINEAR_PLAN_CONFIG = MotionGenPlanConfig(
    enable_graph=False,
    enable_finetune_trajopt=True,
    max_attempts=1,
    time_dilation_factor=1.0,
)

TRANSITION_WAIT_STEPS = 30
GRIPPER_LOCK_STEPS = 30
GRIPPER_OPEN_POS = 0.1
GRIPPER_CLOSE_POS = -0.1
DRAWER_OPEN_THRESHOLD = 0.38


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    env_cfg = StretchEnvCfg()

    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType(device=env.device)

    robot_cfg = load_yaml(CUROBO_CONFIG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSET_ROOT

    dummy_world = WorldConfig(
        cuboid=[
            Cuboid(
                "startup_dummy", pose=[0, 0, -10.0, 1, 0, 0, 0], dims=[1.0, 1.0, 1.0]
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

    print("[CuRobo] Warming up...")
    motion_gen.warmup(enable_graph=True)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_finetune_trajopt=True,
        **PLANNER_CONFIG,
    )

    # ==========================================
    # State Variables
    # ==========================================
    trajectory = None
    traj_idx = 0
    step_count = 0

    phase_one_done = False
    phase_two_done = False
    phase_three_done = False

    transition_timer = 0
    gripper_timer = 0

    robot_entity = env.scene["robot"]

    # Resolve Isaac Lab joint indices
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

    arm_joint_ids_isaac = robot_entity.find_joints(arm_joint_names)[0]
    base_joint_ids_isaac = robot_entity.find_joints(base_joint_names)[0]

    curobo_names = motion_gen.kinematics.joint_names
    arm_ids_curobo = [curobo_names.index(n) for n in arm_joint_names]
    base_ids_curobo = [curobo_names.index(n) for n in base_joint_names]

    # ==========================================
    # Precompute all phase target poses
    # ==========================================
    target_frame_view = XFormPrim(TARGET_FRAME_PATH, name="target_frame")
    cabinet_view = XFormPrim("/World/envs/env_0/Cabinet", name="cabinet")

    curr_pos, _ = target_frame_view.get_world_poses()
    _, cabinet_rot = cabinet_view.get_world_poses()
    robot_pos_w = robot_entity.data.root_state_w[0:1, :3]
    robot_quat_w = robot_entity.data.root_state_w[0:1, 3:7]
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=env.device)
    target_quat_r = quat_mul(quat_inv(robot_quat_w), target_quat_w)

    phase_targets = {}
    for phase_name, offset in PHASE_OFFSETS.items():
        offset_t = torch.tensor([offset], device=env.device)
        target_pos_w, _ = combine_frame_transforms(
            curr_pos[0:1], cabinet_rot, offset_t, identity_quat
        )
        target_pos_r = quat_apply(quat_inv(robot_quat_w), target_pos_w - robot_pos_w)
        phase_targets[phase_name] = Pose(
            position=target_pos_r, quaternion=target_quat_r
        )
        print(f"  [{phase_name}] Target pos (robot frame): {target_pos_r}")

    current_phase_name = "safe_spot"
    target_pose = phase_targets[current_phase_name]

    print(">>> Starting Simulation Loop...")
    sim_start_time = time.time()
    while simulation_app.is_running():

        # ==========================================
        # PLANNER TRIGGER
        # ==========================================
        robot_velocity = torch.sum(torch.abs(robot_entity.data.joint_vel[0]))
        vel_threshold = 2.0
        is_static = robot_velocity < vel_threshold

        if trajectory is None and step_count > 5 and is_static and not phase_three_done:
            if phase_two_done:
                current_phase = "3"
            elif phase_one_done:
                current_phase = "2"
            else:
                current_phase = "1"

            print(f"[CuRobo] Planning Phase {current_phase}...")

            start_state = robot_entity.data.joint_pos[0]

            cu_js = JointState(
                position=start_state.unsqueeze(0),
                velocity=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                acceleration=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                joint_names=robot_entity.joint_names,
            ).get_ordered_joint_state(motion_gen.kinematics.joint_names)

            if current_phase in ["2", "3"]:
                print(f" -> Using LINEAR MODE for Phase {current_phase}")
                active_config = LINEAR_PLAN_CONFIG
            else:
                active_config = plan_config

            result = motion_gen.plan_single(cu_js, target_pose, active_config)

            if result.success.item():
                traj_len = result.optimized_plan.position.shape[1]
                print(f"\nPhase {current_phase} PLAN SUCCESS!")
                print(f" -> Generated {traj_len} steps.")

                if traj_len < 5:
                    print(" -> WARNING: Path is extremely short!")

                trajectory = result.get_interpolated_plan()
                trajectory = motion_gen.get_full_js(trajectory)

                if trajectory.joint_names is not None:
                    print(f"  CuRobo full-JS names: {trajectory.joint_names}")
                    print(f"  IsaacLab joint names: {robot_entity.joint_names}")

                traj_idx = 0
            else:
                print(f"\n[DEBUG] Phase {current_phase} PLAN FAILED!")
                print(f" -> Status: {result.status}")
                if "COLLISION" in str(result.status):
                    print(" -> CAUSE: The robot is likely starting in collision.")

        # ==========================================
        # Execution Logic (Joint Position)
        # ==========================================
        if trajectory is not None:
            if traj_idx >= len(trajectory.position):
                # End of trajectory: Hold the last position
                target_state = trajectory[-1]

                if not phase_one_done:
                    transition_timer += 1
                    if transition_timer > TRANSITION_WAIT_STEPS:
                        print("--> Phase 1 Done. Switching to Phase 2...")
                        phase_one_done = True
                        trajectory = None
                        current_phase_name = "insert"
                        target_pose = phase_targets[current_phase_name]
                        traj_idx = 0
                        transition_timer = 0

                elif not phase_two_done:
                    gripper_timer += 1
                    if gripper_timer > GRIPPER_LOCK_STEPS:
                        print("--> Gripper Locked! Switching to Phase 3...")
                        phase_two_done = True
                        trajectory = None
                        current_phase_name = "pull"
                        target_pose = phase_targets[current_phase_name]
                        traj_idx = 0
                        gripper_timer = 0

            else:
                # Mid-trajectory: Advance when close enough (compare arm joints only)
                target_state = trajectory[traj_idx]
                joint_error = torch.norm(
                    target_state.position[arm_ids_curobo]
                    - robot_entity.data.joint_pos[0, arm_joint_ids_isaac]
                ).item()
                if step_count % 25 == 0:
                    print(
                        f"[DEBUG Traj] Step {step_count} | "
                        f"traj_idx: {traj_idx}/{len(trajectory.position)} | "
                        f"joint_error: {joint_error:.4f} | "
                        f"threshold: 0.05"
                    )
                if joint_error < 0.1:
                    traj_idx += 1

            # --- Phase 3 drawer check (runs every step, independent of traj progress) ---
            if phase_two_done and not phase_three_done:
                cabinet = env.scene["cabinet"]
                drawer_pos = cabinet.data.joint_pos[0]

                if step_count % 10 == 0:
                    print(
                        f"[DEBUG Drawer] Step {step_count} | "
                        f"joint_names: {cabinet.joint_names} | "
                        f"joint_pos: {drawer_pos.tolist()} | "
                        f"max: {drawer_pos.max().item():.4f} | "
                        f"threshold: {DRAWER_OPEN_THRESHOLD}"
                    )

                if drawer_pos.max().item() > DRAWER_OPEN_THRESHOLD:
                    print(
                        f"--> Drawer open ({drawer_pos.max().item():.3f}m). Phase 3 Done!"
                    )
                    phase_three_done = True
                    trajectory = None
                    break

            # Send joint positions directly (cuRobo indices for trajectory, Isaac indices for robot)
            if trajectory is not None:
                arm_action = target_state.position[arm_ids_curobo].unsqueeze(0)
                base_action = target_state.position[base_ids_curobo].unsqueeze(0)
            else:
                # Phase just ended — hold current position
                arm_action = robot_entity.data.joint_pos[:, arm_joint_ids_isaac]
                base_action = robot_entity.data.joint_pos[:, base_joint_ids_isaac]

        else:
            # No active trajectory — hold current position
            arm_action = robot_entity.data.joint_pos[:, arm_joint_ids_isaac]
            base_action = robot_entity.data.joint_pos[:, base_joint_ids_isaac]

        # ==========================================
        # GRIPPER LOGIC
        # ==========================================
        should_close = phase_two_done or phase_three_done or (gripper_timer > 0)
        gripper_cmd = GRIPPER_CLOSE_POS if should_close else GRIPPER_OPEN_POS
        gripper_action = torch.tensor([[gripper_cmd, gripper_cmd]], device=env.device)

        env_actions = torch.cat([arm_action, base_action, gripper_action], dim=-1)

        obs, _, _, _, _ = env.step(env_actions)
        step_count += 1

        # ==========================================
        # DEBUG: Print obs and actions every 25 steps
        # ==========================================
        if step_count % 25 == 0:
            elapsed = time.time() - sim_start_time
            sim_time = step_count * env_cfg.sim.dt * env_cfg.decimation
            print(
                f"\n[DEBUG] Step {step_count} | Sim time: {sim_time:.2f}s | Wall time: {elapsed:.1f}s"
            )
            print(f"  Phase: {current_phase_name}")
            print(f"  Actions:      {env_actions}")
            print(f"  Obs (concat): {obs['policy']}")

    total_time = time.time() - sim_start_time
    total_sim_time = step_count * env_cfg.sim.dt * env_cfg.decimation
    print(
        f"\n[DONE] {step_count} steps | Sim time: {total_sim_time:.2f}s | Wall time: {total_time:.1f}s"
    )
    env.close()


if __name__ == "__main__":
    main()
