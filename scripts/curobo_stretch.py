import os
import sys
import torch
import argparse

# ==========================================
# ISAAC SIM INITIALIZATION
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stretch + CuRobo Integration")
parser.add_argument(
    "--headless", action="store_true", default=False, help="Force display off"
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

from stretch_bc_rnn_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, quat_inv, quat_apply, quat_mul
from isaacsim.core.prims import XFormPrim

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
ASSET_ROOT = PROJECT_ROOT

TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"

MOTION_GEN_PARAMS = {
    "num_trajopt_seeds": 12,
    "num_graph_seeds": 12,
    "trajopt_tsteps": 40,
    "interpolation_dt": 0.0167,
}

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
GRIPPER_OPEN_POS = 0.3
GRIPPER_CLOSE_POS = -0.2


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    env_cfg = StretchEnvCfg()
    env_cfg.viewer.eye = (2.0, 2.0, 2.0)
    env_cfg.episode_length_s = 10000.0

    print("[IsaacLab] Creating environment")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

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
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.0167,
        optimize_dt=True,
        trajopt_tsteps=32,
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

    # --- State Variables ---
    trajectory = None
    traj_idx = 0
    step_count = 0
    target_pose = None

    target_frame_view = XFormPrim(TARGET_FRAME_PATH, name="target_frame")
    phase_one_done = False
    phase_two_done = False
    phase_three_done = False

    transition_timer = 0
    gripper_timer = 0
    hold_joints = None

    print(">>> Starting Simulation Loop...")
    while simulation_app.is_running():
        # --- Determine Target Pose ---
        if target_pose is None:
            curr_pos, _ = target_frame_view.get_world_poses()

            cabinet_prim_path = "/World/envs/env_0/Cabinet"
            cabinet_view = XFormPrim(cabinet_prim_path, name="cabinet")
            _, cabinet_rot = cabinet_view.get_world_poses()

            if not phase_one_done:
                print(
                    f"[Logic] Phase 1: Planning to Safe Spot {PHASE_OFFSETS['safe_spot']}..."
                )
                front_offset = torch.tensor(
                    [PHASE_OFFSETS["safe_spot"]], device=env.device
                )
            elif not phase_two_done:
                print(f"[Logic] Phase 2: Planning Insert {PHASE_OFFSETS['insert']}...")
                front_offset = torch.tensor(
                    [PHASE_OFFSETS["insert"]], device=env.device
                )
            else:
                print(f"[Logic] Phase 3: Planning Pull {PHASE_OFFSETS['pull']}...")
                front_offset = torch.tensor([PHASE_OFFSETS["pull"]], device=env.device)

            target_pos_w, _ = combine_frame_transforms(
                curr_pos[0:1], cabinet_rot, front_offset
            )
            target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=env.device)

            robot = env.scene["robot"]
            robot_pos_w = robot.data.root_state_w[0:1, :3]
            robot_quat_w = robot.data.root_state_w[0:1, 3:7]

            rel_pos = target_pos_w - robot_pos_w

            target_pos_r = quat_apply(quat_inv(robot_quat_w), rel_pos)
            target_quat_r = quat_mul(quat_inv(robot_quat_w), target_quat_w)

            target_pose = Pose(position=target_pos_r, quaternion=target_quat_r)

        # ==========================================
        # PLANNER TRIGGER
        # ==========================================
        robot_entity = env.scene["robot"]
        body_joint_names = [
            "joint_x",
            "joint_y",
            "joint_rot_z",
            "joint_lift",
            "joint_wrist_.*",
            "joint_gripper_.*",
            "joint_arm_.*",
            "joint_head_.*",
        ]
        body_indices, _ = robot_entity.find_joints(body_joint_names)
        robot_velocity = torch.sum(torch.abs(robot_entity.data.joint_vel[0]))
        vel_threshold = 2.0 if phase_two_done else 0.5
        is_static = robot_velocity < vel_threshold

        if trajectory is None and step_count > 5 and is_static:
            current_phase = "1"
            if phase_one_done:
                current_phase = "2"
            if phase_two_done:
                current_phase = "3"

            print(f"[CuRobo] Planning Phase {current_phase}...")

            start_state = (
                hold_joints[0]
                if hold_joints is not None
                else robot_entity.data.joint_pos[0]
            )

            cu_js = JointState(
                position=start_state.unsqueeze(0),
                velocity=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                acceleration=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                joint_names=robot_entity.joint_names,
            ).get_ordered_joint_state(motion_gen.kinematics.joint_names)

            if current_phase == "3":
                print(" -> Using LINEAR MODE for Phase 3 Pull")
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
                traj_idx = 0
            else:
                print(f"\n[DEBUG] Phase {current_phase} PLAN FAILED!")
                print(f" -> Status: {result.status}")
                if "COLLISION" in str(result.status):
                    print(" -> CAUSE: The robot is likely starting in collision.")

        # --- Execution Logic ---
        actions = robot_entity.data.joint_pos.clone()

        if hold_joints is not None:
            actions = hold_joints.clone()
        else:
            actions = robot_entity.data.joint_pos.clone()

        if trajectory is not None:
            if traj_idx >= len(trajectory.position):
                target_state = trajectory[-1]

                if not phase_one_done:
                    transition_timer += 1
                    if transition_timer > TRANSITION_WAIT_STEPS:
                        print("--> Phase 1 Done. Switching to Phase 2...")
                        hold_joints = actions.clone()
                        phase_one_done = True
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        transition_timer = 0

                elif not phase_two_done:
                    gripper_timer += 1
                    if gripper_timer > GRIPPER_LOCK_STEPS:
                        print("--> Gripper Locked! Switching to Phase 3...")
                        phase_two_done = True
                        hold_joints = actions.clone()
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        gripper_timer = 0

                else:
                    phase_three_done = True

            else:
                target_state = trajectory[traj_idx]
                traj_idx += 1

            if trajectory is not None:
                flat_pos = target_state.position.view(-1)
                target_pos_dict = {
                    name: flat_pos[i] for i, name in enumerate(trajectory.joint_names)
                }
                for i, name in enumerate(env.scene["robot"].joint_names):
                    if name in target_pos_dict:
                        actions[0, i] = target_pos_dict[name]

        # ==========================================
        # GRIPPER LOGIC
        # ==========================================
        gripper_idx = -1
        should_close = phase_two_done or phase_three_done or (gripper_timer > 0)

        if should_close:
            actions[0, gripper_idx] = GRIPPER_CLOSE_POS
        else:
            actions[0, gripper_idx] = GRIPPER_OPEN_POS

        full_action = actions[0]
        body_vals = full_action[body_indices]
        env_actions = torch.cat([body_vals]).unsqueeze(0)

        obs, rew, terminated, truncated, extras = env.step(env_actions)
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
