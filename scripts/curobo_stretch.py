import os
import sys
import torch
import argparse

# ==========================================
# 1. ISAAC SIM INITIALIZATION
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
# 2. ENVIRONMENT & UTILITY IMPORTS
# ==========================================
target_config_dir = "/home/johnchen/SharedSSD/JohnChen/stretch/source/stretch/stretch/tasks/manager_based/stretch"
if target_config_dir not in sys.path:
    sys.path.append(target_config_dir)

from stretch_env_cfg import StretchEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import combine_frame_transforms
from isaacsim.core.prims import XFormPrim

# ==========================================
# 3. CUROBO IMPORTS
# ==========================================
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.util_file import load_yaml
from curobo.geom.sdf.world import CollisionCheckerType


def main():
    # ==========================================
    # 4. ENVIRONMENT CONFIGURATION
    # ==========================================
    env_cfg = StretchEnvCfg()
    env_cfg.viewer.eye = (2.0, 2.0, 2.0)
    env_cfg.episode_length_s = 10000.0

    # High stiffness for precise trajectory following
    print("[Fix] forcing High Stiffness (4000.0) on all joints...")
    strong_drive = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=4000.0,
        damping=100.0,
        effort_limit=20000.0,
        velocity_limit=100.0,
    )
    env_cfg.scene.robot.actuators = {"god_mode_drive": strong_drive}

    # ==========================================
    # 5. ACTION SETUP
    # ==========================================
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

    # ==========================================
    # 6. CUROBO MOTION GENERATOR SETUP
    # ==========================================
    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType(device=env.device)

    curobo_config_path = "/home/johnchen/SharedSSD/JohnChen/stretch/assets/configs/robot_configs/stretch_joint.yml"
    if not os.path.exists(curobo_config_path):
        raise FileNotFoundError(f"Could not find CuRobo config at {curobo_config_path}")

    robot_cfg = load_yaml(curobo_config_path)["robot_cfg"]

    # Initializing a dummy world for the solver warmup
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
        interpolation_dt=0.05,
        optimize_dt=True,
        trajopt_tsteps=32,
    )

    motion_gen = MotionGen(motion_gen_config)
    print("[CuRobo] Warming up...")
    motion_gen.warmup(enable_graph=True)

    usd_help = UsdHelper()
    usd_help.load_stage(env.sim.stage)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=10,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    # ==========================================
    # 7. STATE VARIABLES & TARGETS
    # ==========================================
    trajectory = None
    traj_idx = 0
    step_count = 0
    target_pose = None

    target_frame_path = (
        "/World/envs/env_0/Cabinet/drawer_handle_top/drawer_handle_frame"
    )
    target_frame_view = XFormPrim(target_frame_path, name="target_frame")
    phase_one_done = False
    phase_two_done = False
    phase_three_done = False

    transition_timer = 0
    gripper_timer = 0
    hold_joints = None

    # ==========================================
    # 8. MAIN SIMULATION LOOP
    # ==========================================
    print(">>> Starting Simulation Loop...")
    while simulation_app.is_running():
        # --- DETERMINE TARGET POSE ---
        if target_pose is None:
            curr_pos, curr_quat = target_frame_view.get_world_poses()

            # PHASE 1: SAFE SPOT
            if not phase_one_done:
                print("[Logic] Phase 1: Planning to Safe Spot (-0.08)...")
                front_offset = torch.tensor([[-0.08, 0.03, 0.0]], device=env.device)

            # PHASE 2: INSERT
            elif not phase_two_done:
                print("[Logic] Phase 2: Planning Insert (0.1)...")
                front_offset = torch.tensor([[0.1, 0.03, 0.0]], device=env.device)

            # PHASE 3: PULL BACK
            else:
                print("[Logic] Phase 3: Planning Pull (-0.45)...")
                front_offset = torch.tensor([[-0.45, 0.03, 0.0]], device=env.device)

            target_quat_w = torch.tensor([[0.5, 0.5, -0.5, -0.5]], device=env.device)
            target_pos_w, _ = combine_frame_transforms(
                curr_pos[0:1], target_quat_w, front_offset
            )
            target_pose = Pose(position=target_pos_w, quaternion=target_quat_w)

        # --- OBSTACLE UPDATE ---
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

        # --- PLANNER ---
        robot_entity = env.scene["robot"]
        robot_velocity = torch.sum(torch.abs(robot_entity.data.joint_vel[0]))
        is_static = robot_velocity < 0.5

        if trajectory is None and step_count > 50 and is_static:
            print(f"[CuRobo] Planning path...")
            cu_js = JointState(
                position=robot_entity.data.joint_pos[0].unsqueeze(0),
                velocity=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                acceleration=robot_entity.data.joint_vel[0].unsqueeze(0) * 0.0,
                joint_names=robot_entity.joint_names,
            ).get_ordered_joint_state(motion_gen.kinematics.joint_names)

            result = motion_gen.plan_single(cu_js, target_pose, plan_config)

            if result.success.item():
                print(
                    f"[CuRobo] Success! Steps: {result.optimized_plan.position.shape[1]}"
                )
                trajectory = result.get_interpolated_plan()
                trajectory = motion_gen.get_full_js(trajectory)
                traj_idx = 0
            else:
                print(f"[CuRobo] Fail: {result.status}")

        # --- TRAJECTORY EXECUTION ---
        actions = robot_entity.data.joint_pos.clone()

        if hold_joints is not None:
            actions = hold_joints.clone()
        else:
            actions = robot_entity.data.joint_pos.clone()

        if trajectory is not None:
            if traj_idx >= len(trajectory.position):
                # --- TRANSITION LOGIC ---
                if not phase_one_done:
                    transition_timer += 1
                    target_state = trajectory[-1]
                    if transition_timer > 30:  # Wait 0.5s
                        print("--> Phase 1 Done. Switching to Phase 2...")
                        phase_one_done = True
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        transition_timer = 0

                # END OF PHASE 2
                elif not phase_two_done:
                    target_state = trajectory[-1]
                    gripper_timer += 1

                    # Wait 60 steps (1 second) for gripper to close fully
                    if gripper_timer > 60:
                        print("--> Gripper Locked! Switching to Phase 3 (Pull)...")
                        phase_two_done = True
                        hold_joints = actions.clone()
                        trajectory = None
                        target_pose = None
                        traj_idx = 0
                        gripper_timer = 0

                # END OF PHASE 3 (Pull)
                else:
                    target_state = trajectory[-1]
                    phase_three_done = True

            else:
                target_state = trajectory[traj_idx]
                traj_idx += 1

            # Map Joints
            if trajectory is not None:
                flat_pos = target_state.position.view(-1)
                target_pos_dict = {
                    name: flat_pos[i] for i, name in enumerate(trajectory.joint_names)
                }
                for i, name in enumerate(env.scene["robot"].joint_names):
                    if name in target_pos_dict:
                        actions[0, i] = target_pos_dict[name]

        # ==========================================
        # Gripper Logic
        # ==========================================
        gripper_idx = -1

        # Priority 1: If Phase 3 is totally finished -> Release Handle
        if phase_three_done:
            actions[0, gripper_idx] = 0.2

        # Priority 2: During Phase 1 & 2 Approach & Insert
        elif not phase_two_done:
            # Check if we are currently waiting for the grip (Phase 2 finished, waiting on timer)
            if (
                phase_one_done
                and trajectory is not None
                and traj_idx >= len(trajectory.position)
            ):
                actions[0, gripper_idx] = -0.5  # Close
                if gripper_timer % 30 == 0:
                    print("Clamping Gripper...")
            else:
                actions[0, gripper_idx] = 0.2  # Open

        # Priority 3: During Phase 3 Pulling
        else:
            actions[0, gripper_idx] = -0.5

        # --- STEP SIMULATION ---
        obs, rew, terminated, truncated, extras = env.step(actions)
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
