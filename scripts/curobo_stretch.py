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
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# PATH SETUP & IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if target_config_dir not in sys.path:
    sys.path.append(target_config_dir)

from stretch_env_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.utils.math import combine_frame_transforms
from isaacsim.core.prims import XFormPrim

from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
CUROBO_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "assets", "robot_configs", "stretch_fake_joint.yml"
)
ASSET_ROOT = PROJECT_ROOT

TARGET_FRAME_PATH = "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"

MOTION_GEN_PARAMS = {
    "num_trajopt_seeds": 12,
    "num_graph_seeds": 12,
    "trajopt_tsteps": 40,
    "interpolation_dt": 0.02,
}

PLANNER_CONFIG = {
    "max_attempts": 10,
    "time_dilation_factor": 0.5,
    "enable_graph_attempt": 2,
}

PHASE_OFFSETS = {
    "safe_spot": [-0.05, 0.05, 0.0],
    "insert": [0.03, 0.05, 0.0],
    "pull": [-0.45, 0.05, 0.0],
}

TRANSITION_WAIT_STEPS = 30
GRIPPER_LOCK_STEPS = 60
GRIPPER_OPEN_POS = 0.3
GRIPPER_CLOSE_POS = -0.2


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Environment Configuration
    env_cfg = StretchEnvCfg()
    env_cfg.viewer.eye = (2.0, 2.0, 2.0)
    env_cfg.episode_length_s = 10000.0

    # Action Setup
    env_cfg.actions.base = None
    env_cfg.actions.lift_velocity = None
    env_cfg.actions.arm_velocity = None
    env_cfg.actions.wrist_velocity = None
    env_cfg.actions.gripper = None

    env_cfg.actions.joint_pos_direct = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )

    print("[IsaacLab] Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

    # CuRobo Setup
    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType(device=env.device)

    if not os.path.exists(CUROBO_CONFIG_PATH):
        raise FileNotFoundError(f"Could not find CuRobo config at {CUROBO_CONFIG_PATH}")

    robot_cfg = load_yaml(CUROBO_CONFIG_PATH)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSET_ROOT

    dummy_world = WorldConfig(
        cuboid=[
            Cuboid(
                "startup_dummy", pose=[0, 0, -10.0, 1, 0, 0, 0], dims=[1.0, 1.0, 1.0]
            )
        ]
    )

    # 1. SETUP MOTION GEN (For Phase 1 Planning)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        dummy_world,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        optimize_dt=True,
        **MOTION_GEN_PARAMS,
    )
    motion_gen = MotionGen(motion_gen_config)

    # 2. SETUP IK SOLVER
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        dummy_world,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_seeds=1,  # Keep 1 for smooth sequential movement
    )
    ik_solver = IKSolver(ik_config)

    print("[CuRobo] Warming up...")
    motion_gen.warmup(enable_graph=True)

    usd_help = UsdHelper()
    usd_help.load_stage(env.sim.stage)

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
            curr_pos, curr_quat = target_frame_view.get_world_poses()

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

            target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=env.device)
            target_pos_w, _ = combine_frame_transforms(
                curr_pos[0:1], target_quat_w, front_offset
            )
            target_pose = Pose(position=target_pos_w, quaternion=target_quat_w)

        # --- Obstacle Update ---
        if step_count % 60 == 0 and step_count > 0:
            base_env_path = "/World/envs/env_0"
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=[
                    f"{base_env_path}/Cabinet",
                    f"{base_env_path}/ObstacleCube",
                ],
                ignore_substring=[f"{base_env_path}/Robot"],
            ).get_collision_check_world()

            motion_gen.update_world(obstacles)

            # --- FIX: Stop updating IK obstacles during pulling ---
            # This prevents the solver from seeing the handle grasp as a collision
            if not phase_two_done:
                ik_solver.update_world(obstacles)

        # ==========================================
        # PLANNER TRIGGER
        # ==========================================
        robot_entity = env.scene["robot"]
        robot_velocity = torch.sum(torch.abs(robot_entity.data.joint_vel[0]))
        is_static = robot_velocity < 0.5

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

            # -------------------------------------------------------
            # PHASE 2 & 3: FORCED STRAIGHT LINE (Sequential IK)
            # -------------------------------------------------------
            if phase_one_done:
                print(
                    f"[Logic] Phase {current_phase}: Generating Straight-Line Path..."
                )

                # --- FIX: Increase steps for Phase 3 to ensure smooth base movement ---
                steps = 50
                if phase_two_done:  # Phase 3 (Pull)
                    steps = 80  # Finer resolution for the long pull

                # Get current Gripper Pose
                curr_pose = motion_gen.kinematics.compute_kinematics(cu_js)
                curr_pos_w = curr_pose.ee_position
                curr_quat_w = curr_pose.ee_quaternion

                local_shift = torch.zeros((1, 3), device=env.device)

                if not phase_two_done:
                    dist = PHASE_OFFSETS["insert"][0] - PHASE_OFFSETS["safe_spot"][0]
                    local_shift[0, 0] = dist
                else:
                    dist = PHASE_OFFSETS["pull"][0] - PHASE_OFFSETS["insert"][0]
                    local_shift[0, 0] = dist

                goal_pos_w, _ = combine_frame_transforms(
                    curr_pos_w, curr_quat_w, local_shift
                )

                # Linear Interpolation
                traj_pos_w = torch.zeros((steps, 3), device=env.device)
                for i in range(steps):
                    alpha = float(i) / (steps - 1)
                    traj_pos_w[i] = (1 - alpha) * curr_pos_w + alpha * goal_pos_w

                traj_quat_w = curr_quat_w.repeat(steps, 1)

                # --- SEQUENTIAL SOLVER LOOP ---
                solutions = []
                current_seed = cu_js.position.unsqueeze(0)
                success_all = True

                for i in range(steps):
                    single_goal = Pose(
                        position=traj_pos_w[i].unsqueeze(0),
                        quaternion=traj_quat_w[i].unsqueeze(0),
                    )

                    ik_out = ik_solver.solve_batch(
                        goal_pose=single_goal, seed_config=current_seed
                    )

                    if not ik_out.success.item():
                        print(f"[CuRobo] IK Stuck at step {i}/{steps}")
                        success_all = False
                        break

                    solutions.append(ik_out.solution)
                    current_seed = ik_out.solution

                if success_all:
                    print(f"[CuRobo] Cartesian Path Success! Steps: {steps}")
                    full_traj = torch.cat(solutions, dim=0).squeeze(1)

                    trajectory = JointState(
                        position=full_traj,
                        joint_names=motion_gen.kinematics.joint_names,
                    )
                    trajectory.velocity = torch.zeros_like(trajectory.position)
                    trajectory.acceleration = torch.zeros_like(trajectory.position)
                    trajectory = motion_gen.get_full_js(trajectory)
                    traj_idx = 0
                else:
                    print("[CuRobo] Cartesian Fail: IK could not trace full path.")

            # -------------------------------------------------------
            # PHASE 1: STANDARD PLANNER
            # -------------------------------------------------------
            else:
                result = motion_gen.plan_single(cu_js, target_pose, plan_config)
                if result.success.item():
                    print(
                        f"[CuRobo] Planner Success! Steps: {result.optimized_plan.position.shape[1]}"
                    )
                    trajectory = result.get_interpolated_plan()
                    trajectory = motion_gen.get_full_js(trajectory)
                    traj_idx = 0
                else:
                    print(f"[CuRobo] Fail: {result.status}")

        # --- Execution Logic ---
        actions = robot_entity.data.joint_pos.clone()

        if hold_joints is not None:
            actions = hold_joints.clone()
        else:
            actions = robot_entity.data.joint_pos.clone()

        if trajectory is not None:
            if traj_idx >= len(trajectory.position):
                # --- Transition Logic ---
                if not phase_one_done:
                    transition_timer += 1
                    target_state = trajectory[-1]
                    if transition_timer > TRANSITION_WAIT_STEPS:
                        print("--> Phase 1 Done. Switching to Phase 2...")
                        hold_joints = actions.clone()
                        phase_one_done = True
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        transition_timer = 0

                elif not phase_two_done:
                    target_state = trajectory[-1]
                    gripper_timer += 1
                    if gripper_timer > GRIPPER_LOCK_STEPS:
                        print("--> Gripper Locked! Switching to Phase 3 (Pull)...")
                        phase_two_done = True
                        hold_joints = actions.clone()
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        gripper_timer = 0

                else:
                    target_state = trajectory[-1]
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

        # --- Gripper Logic ---
        gripper_idx = -1
        if phase_three_done:
            actions[0, gripper_idx] = GRIPPER_OPEN_POS
        elif not phase_two_done:
            if (
                phase_one_done
                and trajectory is not None
                and traj_idx >= len(trajectory.position)
            ):
                actions[0, gripper_idx] = GRIPPER_CLOSE_POS
                if gripper_timer % 30 == 0:
                    print("Clamping Gripper...")
            else:
                actions[0, gripper_idx] = GRIPPER_OPEN_POS
        else:
            actions[0, gripper_idx] = GRIPPER_CLOSE_POS

        obs, rew, terminated, truncated, extras = env.step(actions)
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
